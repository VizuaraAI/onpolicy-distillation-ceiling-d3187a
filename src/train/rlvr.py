"""Arm 3: reinforcement learning with a verifiable reward.

Same student rollouts as arm 2, but the only signal is the task
verifier's scalar reward. No teacher distributions. This isolates how
much of any gain comes from dense teacher supervision rather than from
on-policy exploration alone, which is the distinction the
reweighting-versus-acquisition question turns on.

The update is a group-relative policy gradient: sample a group of
completions per prompt, standardize the rewards within the group as the
advantage, and add a KL penalty toward the frozen reference policy.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .common import Budget, apply_parameterization, build_optimizer, run_arm

logger = logging.getLogger(__name__)


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Standardize rewards within each prompt's group of rollouts."""
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((grouped - mean) / std).view(-1)


def sequence_logprob(logits: torch.Tensor, ids: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Summed log-probability of the generated tokens."""
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=1)


def train(student, reference, capability, cfg, *, seed: int, run_dir):
    """Run arm 3 on one capability.

    ``reference`` is a frozen copy of the base student, used only for
    the KL penalty. The teacher is not used in this arm at all.
    """
    from .onpolicy_distill import sample_rollouts

    model = apply_parameterization(student.model, cfg)
    budget = Budget.from_cfg(cfg)
    optimizer, scheduler = build_optimizer(model, budget)

    group_size = int(cfg.get("arms.rlvr.rollouts_per_prompt", 8))
    kl_coef = float(cfg.get("arms.rlvr.kl_coef", 0.02))
    max_new = int(cfg.get("battery.max_new_tokens", 256))

    pool = capability.sample(
        int(cfg.get("battery.n_train_per_capability", 4000)), seed=seed
    )
    generator = torch.Generator().manual_seed(seed)

    def step_fn(step: int) -> dict:
        idx = torch.randint(0, len(pool), (budget.batch_size,),
                            generator=generator)
        batch = [pool[int(i)] for i in idx]

        ids, mask = sample_rollouts(
            student, [ex.prompt for ex in batch], n_per_prompt=group_size,
            temperature=1.0, max_new_tokens=max_new,
        )
        texts = student.tokenizer.batch_decode(ids, skip_special_tokens=True)
        rewards = torch.tensor(
            [capability.verify(text, ex.answer)
             for ex in batch for text in texts[:group_size]],
            device=student.device, dtype=torch.float32,
        )[:ids.shape[0]]
        advantages = group_advantages(rewards, group_size)

        logits = model(ids).logits[:, :-1]
        float_mask = mask[:, 1:].float()
        logp = sequence_logprob(logits, ids[:, 1:], float_mask)
        with torch.no_grad():
            ref_logits = reference.model(ids).logits[:, :-1]
            ref_logp = sequence_logprob(ref_logits, ids[:, 1:], float_mask)

        policy_loss = -(advantages * logp).mean()
        kl = (logp - ref_logp).mean()
        loss = policy_loss + kl_coef * kl

        (loss / budget.grad_accum).backward()
        if (step + 1) % budget.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                budget.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return {"loss": loss.item(), "reward": rewards.mean().item(),
                "kl": kl.item(), "tokens": int(float_mask.sum().item())}

    return run_arm("rlvr", step_fn, model=model, cfg=cfg,
                   capability=capability, seed=seed, run_dir=run_dir)
