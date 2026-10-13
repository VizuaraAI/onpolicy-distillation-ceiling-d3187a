"""Arm 2: on-policy distillation. This is the arm under test.

The student generates, the teacher scores the student's own tokens, and
the student is updated toward the teacher's distribution on those
tokens. Reverse KL by default, following the argument that it keeps the
student from spreading mass over teacher modes it cannot support.

Two things to watch:

- The teacher distribution is truncated to the top-k entries for memory.
  That truncation level is a real hyperparameter and deserves an
  ablation row, not a silent default.
- Teacher and student must tokenize identically. ``_check_tokenizers``
  fails loudly rather than letting a misalignment corrupt every number
  downstream.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .common import (Budget, apply_parameterization, build_optimizer,
                     run_arm, teacher_accuracy)

logger = logging.getLogger(__name__)


def _check_tokenizers(student, teacher) -> None:
    probe = "The student generates and the teacher scores those tokens."
    if student.tokenizer(probe)["input_ids"] != teacher.tokenizer(probe)["input_ids"]:
        raise ValueError(
            "student and teacher tokenize differently. On-policy "
            "distillation needs teacher log-probs on student tokens, so "
            "either keep both models in the same family or implement an "
            "explicit alignment in _align_tokenizers before continuing."
        )


@torch.no_grad()
def sample_rollouts(student, prompts: list[str], *, n_per_prompt: int,
                    temperature: float, max_new_tokens: int):
    """Sample student continuations, returning padded ids and a mask."""
    tokenizer = student.tokenizer
    repeated = [p for p in prompts for _ in range(n_per_prompt)]
    encoded = tokenizer(
        repeated, return_tensors="pt", padding=True, truncation=True,
        max_length=512,
    ).to(student.device)
    generated = student.model.generate(
        **encoded, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = encoded["input_ids"].shape[1]
    # Only the generated continuation carries the distillation signal.
    response_mask = torch.zeros_like(generated, dtype=torch.bool)
    response_mask[:, prompt_len:] = generated[:, prompt_len:] != tokenizer.pad_token_id
    return generated, response_mask


def gkd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
             mask: torch.Tensor, *, beta: float = 1.0,
             topk: int = 128) -> torch.Tensor:
    """Generalized KD loss on student-generated tokens.

    ``beta=1`` is reverse KL (student-weighted), ``beta=0`` is forward
    KL (teacher-weighted), and intermediate values interpolate through
    the generalized Jensen-Shannon family.
    """
    if mask.sum() == 0:
        return student_logits.sum() * 0.0

    # Truncate to the teacher's top-k support to bound memory.
    topk = min(topk, teacher_logits.shape[-1])
    _, indices = teacher_logits.topk(topk, dim=-1)
    student_sub = student_logits.gather(-1, indices)
    teacher_sub = teacher_logits.gather(-1, indices)

    student_logp = F.log_softmax(student_sub, dim=-1)
    teacher_logp = F.log_softmax(teacher_sub, dim=-1)

    forward = F.kl_div(student_logp, teacher_logp, log_target=True,
                       reduction="none").sum(-1)
    reverse = F.kl_div(teacher_logp, student_logp, log_target=True,
                       reduction="none").sum(-1)
    per_token = beta * reverse + (1.0 - beta) * forward
    return (per_token * mask).sum() / mask.sum()


def train(student, teacher, capability, cfg, *, seed: int, run_dir,
          min_teacher_accuracy: float | None = None):
    """Run arm 2 on one capability."""
    _check_tokenizers(student, teacher)

    threshold = (min_teacher_accuracy
                 if min_teacher_accuracy is not None
                 else float(cfg.get("models.min_teacher_accuracy", 0.6)))
    accuracy = teacher_accuracy(teacher, capability, cfg, seed=seed)
    skip = ""
    if accuracy < threshold:
        skip = (f"teacher accuracy {accuracy:.2f} is below the "
                f"{threshold:.2f} gate")

    model = apply_parameterization(student.model, cfg)
    budget = Budget.from_cfg(cfg)
    optimizer, scheduler = build_optimizer(model, budget)

    beta = float(cfg.get("arms.onpolicy_distill.beta_divergence", 1.0))
    topk = int(cfg.get("arms.onpolicy_distill.teacher_topk", 128))
    rollouts = int(cfg.get("arms.onpolicy_distill.student_rollouts_per_prompt", 4))
    temperature = float(cfg.get("arms.onpolicy_distill.rollout_temperature", 1.0))
    max_new = int(cfg.get("battery.max_new_tokens", 256))

    pool = capability.sample(
        int(cfg.get("battery.n_train_per_capability", 4000)), seed=seed
    )
    generator = torch.Generator().manual_seed(seed)

    def step_fn(step: int) -> dict:
        idx = torch.randint(0, len(pool), (budget.batch_size,),
                            generator=generator)
        prompts = [pool[int(i)].prompt for i in idx]

        ids, mask = sample_rollouts(
            student, prompts, n_per_prompt=rollouts,
            temperature=temperature, max_new_tokens=max_new,
        )
        with torch.no_grad():
            teacher_logits = teacher.model(ids).logits[:, :-1]
        student_logits = model(ids).logits[:, :-1]

        loss = gkd_loss(student_logits, teacher_logits, mask[:, 1:].float(),
                        beta=beta, topk=topk)
        (loss / budget.grad_accum).backward()

        if (step + 1) % budget.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                budget.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return {"loss": loss.item(), "tokens": int(mask.sum().item()),
                "teacher_accuracy": accuracy}

    return run_arm("onpolicy_distill", step_fn, model=model, cfg=cfg,
                   capability=capability, seed=seed, run_dir=run_dir,
                   skip_reason=skip)
