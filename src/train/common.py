"""Shared machinery for all four training arms.

Everything that makes the arms comparable lives here: the budget, the
optimizer, the teacher competence gate, and the run-directory metadata.
Arm modules supply only their own loss.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

logger = logging.getLogger(__name__)


@dataclass
class Budget:
    """The equal-cost contract every arm signs."""

    token_budget: int
    optimizer_steps: int
    batch_size: int
    grad_accum: int
    lr: float
    warmup_steps: int
    max_grad_norm: float

    @classmethod
    def from_cfg(cls, cfg) -> "Budget":
        return cls(
            token_budget=int(cfg.get("budget.token_budget", 8_000_000)),
            optimizer_steps=int(cfg.get("budget.optimizer_steps", 2000)),
            batch_size=int(cfg.get("budget.batch_size", 8)),
            grad_accum=int(cfg.get("budget.grad_accum", 4)),
            lr=float(cfg.get("budget.lr", 1e-5)),
            warmup_steps=int(cfg.get("budget.warmup_steps", 100)),
            max_grad_norm=float(cfg.get("budget.max_grad_norm", 1.0)),
        )


@dataclass
class ArmResult:
    arm: str
    capability: str
    seed: int
    steps_run: int
    tokens_seen: int
    final_loss: float
    skipped_reason: str = ""
    history: list[dict] = field(default_factory=list)

    @property
    def skipped(self) -> bool:
        return bool(self.skipped_reason)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def prepare_run_dir(root: Path, name: str) -> Path:
    """A fresh run directory. Never overwrites an existing run."""
    path = root / name
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / f"{name}_{suffix}"
    path.mkdir(parents=True)
    return path


def write_meta(run_dir: Path, cfg, extra: dict | None = None) -> None:
    meta = {"config": cfg.raw, "config_path": cfg.path, "git_sha": git_sha()}
    meta.update(extra or {})
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )


def apply_parameterization(model, cfg):
    """Full fine-tuning by default, LoRA when the config asks for it."""
    mode = cfg.get("budget.parameterization", "full")
    if mode == "full":
        model.requires_grad_(True)
        return model
    if mode != "lora":
        raise ValueError(f"unrecognized parameterization {mode!r}")

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=int(cfg.get("budget.lora.r", 32)),
        lora_alpha=int(cfg.get("budget.lora.alpha", 64)),
        lora_dropout=float(cfg.get("budget.lora.dropout", 0.05)),
        target_modules=list(cfg.get("budget.lora.target_modules", [])),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def build_optimizer(model, budget: Budget):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=budget.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, budget.warmup_steps)),
    )
    return optimizer, scheduler


@torch.no_grad()
def teacher_accuracy(teacher, capability, cfg, n: int = 100,
                     seed: int = 0) -> float:
    """Can the teacher do this task at all?

    A teacher that cannot do base-7 arithmetic tells us nothing about
    whether the student could have learned it, so arms on that
    capability are skipped rather than recorded as a zero.
    """
    from ..eval.passk import generate

    examples = capability.sample(n, seed=seed)
    outputs = generate(
        teacher, [ex.prompt for ex in examples], n_samples=1,
        temperature=0.0,
        max_new_tokens=int(cfg.get("battery.max_new_tokens", 256)),
    )
    correct = sum(
        capability.verify(samples[0], ex.answer)
        for samples, ex in zip(outputs, examples)
    )
    return correct / max(1, len(examples))


def run_arm(name: str, step_fn: Callable[[int], dict], *, model, cfg,
            capability, seed: int, run_dir: Path,
            skip_reason: str = "") -> ArmResult:
    """Drive one arm for the shared budget.

    ``step_fn(step)`` performs one optimizer step and returns a dict of
    scalars including at least ``loss`` and ``tokens``. Keeping the loop
    here is what guarantees the arms are budget-matched.
    """
    budget = Budget.from_cfg(cfg)
    if skip_reason:
        logger.warning("skipping %s on %s: %s", name, capability.name,
                       skip_reason)
        return ArmResult(name, capability.name, seed, 0, 0, float("nan"),
                         skipped_reason=skip_reason)

    history: list[dict] = []
    tokens_seen = 0
    last_loss = float("nan")

    for step in range(budget.optimizer_steps):
        stats = step_fn(step)
        tokens_seen += int(stats.get("tokens", 0))
        last_loss = float(stats.get("loss", last_loss))
        if step % 50 == 0:
            logger.info("%s step %d/%d: %s", name, step,
                        budget.optimizer_steps, stats)
            history.append({"step": step, **stats})
        if tokens_seen >= budget.token_budget:
            logger.info("%s hit the token budget at step %d", name, step)
            break

    result = ArmResult(name, capability.name, seed, step + 1, tokens_seen,
                       last_loss, history=history)
    (run_dir / f"{name}_{capability.name}_seed{seed}.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    return result
