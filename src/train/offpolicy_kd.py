"""Arm 1: off-policy sequence-level knowledge distillation.

The teacher samples, the student imitates those samples with ordinary
cross entropy. This is the classic baseline and the control that
isolates what on-policy rollouts actually buy: the two arms see the same
teacher and the same budget, and differ only in who generated the data.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .common import (Budget, apply_parameterization, build_optimizer,
                     run_arm, teacher_accuracy)

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample_teacher(teacher, prompts: list[str], *, n_per_prompt: int,
                   temperature: float, max_new_tokens: int) -> list[str]:
    """Teacher continuations, as plain text, for the student to imitate."""
    tokenizer = teacher.tokenizer
    repeated = [p for p in prompts for _ in range(n_per_prompt)]
    encoded = tokenizer(
        repeated, return_tensors="pt", padding=True, truncation=True,
        max_length=512,
    ).to(teacher.device)
    generated = teacher.model.generate(
        **encoded, do_sample=temperature > 0,
        temperature=max(temperature, 1e-5), max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def train(student, teacher, capability, cfg, *, seed: int, run_dir):
    """Run arm 1 on one capability."""
    threshold = float(cfg.get("models.min_teacher_accuracy", 0.6))
    accuracy = teacher_accuracy(teacher, capability, cfg, seed=seed)
    skip = ""
    if accuracy < threshold:
        skip = (f"teacher accuracy {accuracy:.2f} is below the "
                f"{threshold:.2f} gate")

    model = apply_parameterization(student.model, cfg)
    budget = Budget.from_cfg(cfg)
    optimizer, scheduler = build_optimizer(model, budget)

    samples_per = int(cfg.get("arms.offpolicy_kd.teacher_samples_per_prompt", 4))
    temperature = float(cfg.get("arms.offpolicy_kd.teacher_temperature", 0.8))
    max_new = int(cfg.get("battery.max_new_tokens", 256))

    pool = capability.sample(
        int(cfg.get("battery.n_train_per_capability", 4000)), seed=seed
    )
    generator = torch.Generator().manual_seed(seed)

    def step_fn(step: int) -> dict:
        idx = torch.randint(0, len(pool), (budget.batch_size,),
                            generator=generator)
        prompts = [pool[int(i)].prompt for i in idx]

        texts = sample_teacher(
            teacher, prompts, n_per_prompt=samples_per,
            temperature=temperature, max_new_tokens=max_new,
        )
        encoded = student.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=768,
        ).to(student.device)
        ids, attention = encoded["input_ids"], encoded["attention_mask"]

        logits = model(ids, attention_mask=attention).logits[:, :-1]
        targets = ids[:, 1:]
        mask = attention[:, 1:].float()
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        loss = (per_token * mask).sum() / mask.sum().clamp_min(1)

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

    return run_arm("offpolicy_kd", step_fn, model=model, cfg=cfg,
                   capability=capability, seed=seed, run_dir=run_dir,
                   skip_reason=skip)
