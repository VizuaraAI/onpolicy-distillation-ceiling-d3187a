"""R(c): how well does the base student already represent capability c?

R(c) = (probe component) x (SAE coverage component), both in [0, 1].

This is the only number in the project that must be computed before any
training arm runs. Once written, ``representability.json`` is frozen.
If you recompute it after seeing training results, the ceiling test is
no longer falsifiable, and a reviewer will be right to say so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.capabilities import Capability
from ..models.registry import LoadedModel, collect_activations
from ..models.sae import BatchTopKSAE, SAEConfig, latent_detector_f1, train_sae
from .probes import ProbeResult, probe_capability, probe_component, resolve_layers

logger = logging.getLogger(__name__)


@dataclass
class CapabilityAudit:
    name: str
    expected_group: str
    probe_score: float
    coverage_score: float
    representability: float
    n_variables: int
    leaked_variables: list[str]


def sae_coverage(sae: BatchTopKSAE, activations: torch.Tensor, examples,
                 *, threshold: float = 0.5, device: str = "cuda") -> float:
    """Fraction of task-critical variables some latent detects.

    A variable counts as covered if at least one dictionary latent is a
    usable binary detector for it. This is the dictionary-side evidence
    that the capability's ingredients exist in the base model.
    """
    from .probes import _binarize

    variables = sorted({key for ex in examples for key in ex.variables})
    if not variables:
        return 0.0

    covered = 0
    for variable in variables:
        raw = [ex.variables.get(variable) for ex in examples]
        if any(value is None for value in raw):
            continue
        labels = _binarize(raw)
        if labels is None:
            continue
        f1 = latent_detector_f1(
            sae, activations, torch.from_numpy(labels), device=device
        )
        if float(f1.max()) >= threshold:
            covered += 1
    return covered / len(variables)


def audit_capability(student: LoadedModel, capability: Capability, cfg, *,
                     n_examples: int = 500, seed: int = 0) -> CapabilityAudit:
    """Score one capability's representability in the base student."""
    examples = capability.sample(n_examples, seed=seed)
    prompts = [ex.prompt for ex in examples]
    device = cfg.get("models.device", "cuda")

    layers = resolve_layers(student, cfg.get("audit.probe_layers", "all"))
    probes: list[ProbeResult] = probe_capability(
        student, examples, layers=layers,
        l2=float(cfg.get("audit.probe_l2", 1.0)),
        test_frac=float(cfg.get("audit.probe_test_frac", 0.2)),
        seed=seed,
        shuffle_control=bool(cfg.get("audit.probe_shuffle_control", True)),
        max_length=int(cfg.get("battery.max_prompt_tokens", 512)),
    )
    probe_score = probe_component(probes)

    # Dictionary side: train an SAE on this capability's activations and
    # ask whether any latent detects each task-critical variable.
    sae_layer = cfg.get("audit.sae.layer", "middle")
    token_acts = collect_activations(
        student, prompts, layer=sae_layer, pool="none",
        max_length=int(cfg.get("battery.max_prompt_tokens", 512)),
    )
    sae = train_sae(
        token_acts, SAEConfig.from_cfg(cfg, student.d_model),
        steps=int(cfg.get("audit.sae.steps", 20000)),
        lr=float(cfg.get("audit.sae.lr", 3e-4)),
        batch_tokens=int(cfg.get("audit.sae.batch_tokens", 4096)),
        device=device,
    )
    pooled_acts = collect_activations(
        student, prompts, layer=sae_layer, pool="last",
        max_length=int(cfg.get("battery.max_prompt_tokens", 512)),
    )
    coverage = sae_coverage(
        sae, pooled_acts, examples,
        threshold=float(cfg.get("audit.coverage_f1_threshold", 0.5)),
        device=device,
    )

    return CapabilityAudit(
        name=capability.name,
        expected_group=capability.expected_group,
        probe_score=probe_score,
        coverage_score=coverage,
        representability=probe_score * coverage,
        n_variables=len(probes),
        leaked_variables=[p.variable for p in probes if p.leaked],
    )


def split_by_tercile(audits: list[CapabilityAudit]) -> dict[str, list[str]]:
    """Lowest tercile is ABSENT, highest is REPRESENTED, middle is held.

    The middle tercile is kept out of the headline contrast and used
    only for the R(c) correlation analysis, which is the part of the
    result that generalizes beyond these specific tasks.
    """
    if not audits:
        return {"absent": [], "middle": [], "represented": []}
    scores = np.array([a.representability for a in audits])
    low, high = np.quantile(scores, [1 / 3, 2 / 3])
    groups: dict[str, list[str]] = {"absent": [], "middle": [], "represented": []}
    for audit in audits:
        if audit.representability <= low:
            groups["absent"].append(audit.name)
        elif audit.representability >= high:
            groups["represented"].append(audit.name)
        else:
            groups["middle"].append(audit.name)
    return groups


def write_audit(path: Path, cfg, audits: list[CapabilityAudit]) -> dict:
    """Write the frozen audit record, refusing to clobber an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. The audit is frozen once written. "
            "Bump audit_version in the config and use a new output path "
            "if you really need to redo it."
        )
    record = {
        "audit_version": cfg.get("audit_version", "v1"),
        "student": cfg.get("models.student"),
        "split_mode": cfg.get("audit.split_mode", "tercile"),
        "capabilities": [asdict(a) for a in audits],
        "split": split_by_tercile(audits),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("wrote frozen audit to %s", path)
    return record


def read_audit(path: Path) -> dict:
    """Load the frozen audit. Everything downstream reads, never writes."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.cli audit` first: no "
            "training arm should run before the split is frozen."
        )
    return json.loads(path.read_text(encoding="utf-8"))
