"""Arm 4: continued pretraining. The positive control.

Plain next-token prediction on an in-domain corpus for the capability.
This arm exists to prove the ABSENT tasks are learnable by a model of
this size at this budget. If it fails, the whole experiment is
uninterpretable, because a flat result everywhere would be explained by
the task being too hard rather than by any ceiling.

Build this arm second, right after arm 2, not last.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .common import Budget, apply_parameterization, build_optimizer, run_arm

logger = logging.getLogger(__name__)


def build_corpus(capability, n_tokens: int, *, seed: int,
                 chars_per_token: int = 4) -> list[str]:
    """An in-domain corpus of fully worked examples.

    For the generated ABSENT tasks this is prompt plus correct answer,
    which is exactly the signal a model would have had in pretraining if
    the symbol system had been in its data. For the REPRESENTED tasks it
    plays the same role.
    """
    target_chars = n_tokens * chars_per_token
    documents: list[str] = []
    total = 0
    batch = 2000
    offset = 0
    while total < target_chars:
        examples = capability.sample(batch, seed=seed + offset)
        for example in examples:
            document = f"{example.prompt}{example.answer}\n\n"
            documents.append(document)
            total += len(document)
            if total >= target_chars:
                break
        offset += 1
    return documents


def train(student, capability, cfg, *, seed: int, run_dir):
    """Run arm 4 on one capability. No teacher is involved."""
    model = apply_parameterization(student.model, cfg)
    budget = Budget.from_cfg(cfg)
    # Continued pretraining usually wants a larger learning rate than
    # the distillation arms; the token budget stays identical.
    budget.lr = float(cfg.get("arms.continued_pretrain.lr", budget.lr))
    optimizer, scheduler = build_optimizer(model, budget)

    corpus = build_corpus(
        capability,
        int(cfg.get("arms.continued_pretrain.corpus_tokens_per_capability",
                    2_000_000)),
        seed=seed,
    )
    logger.info("built a %d document corpus for %s", len(corpus),
                capability.name)
    generator = torch.Generator().manual_seed(seed)

    def step_fn(step: int) -> dict:
        idx = torch.randint(0, len(corpus), (budget.batch_size,),
                            generator=generator)
        texts = [corpus[int(i)] for i in idx]
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

        return {"loss": loss.item(), "tokens": int(mask.sum().item())}

    return run_arm("continued_pretrain", step_fn, model=model, cfg=cfg,
                   capability=capability, seed=seed, run_dir=run_dir)
