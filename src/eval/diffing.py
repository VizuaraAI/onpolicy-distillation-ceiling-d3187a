"""The representational readout: what did training change inside?

Train a crosscoder on paired activations from the base student and a
trained student, then count the latents that are genuinely new after
the latent-scaling correction. Reporting ``nlf_raw`` alone is wrong;
both numbers always travel together.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models.crosscoder import (CrosscoderConfig, novel_latent_fraction,
                                 train_crosscoder)
from ..models.registry import collect_paired_activations

logger = logging.getLogger(__name__)


def diff_models(base, trained, capability, cfg, *, seed: int = 0,
                n_examples: int = 500) -> dict:
    """Crosscoder diff of one trained student against the base student."""
    examples = capability.sample(n_examples, seed=seed)
    prompts = [ex.prompt for ex in examples]
    device = cfg.get("models.device", "cuda")
    layer = cfg.get("diffing.crosscoder.layer", "middle")
    max_length = int(cfg.get("battery.max_prompt_tokens", 512))

    base_acts, trained_acts = collect_paired_activations(
        base, trained, prompts, layer=layer, max_length=max_length
    )
    logger.info("collected %d paired activation vectors", base_acts.shape[0])

    crosscoder = train_crosscoder(
        base_acts, trained_acts,
        CrosscoderConfig.from_cfg(cfg, base.d_model),
        steps=int(cfg.get("diffing.crosscoder.steps", 30000)),
        lr=float(cfg.get("diffing.crosscoder.lr", 3e-4)),
        batch_tokens=int(cfg.get("diffing.crosscoder.batch_tokens", 4096)),
        device=device,
    )

    if not cfg.get("diffing.latent_scaling", True):
        raise ValueError(
            "diffing.latent_scaling is disabled. Novel-latent counts "
            "without the correction are inflated by the sparsity penalty "
            "and must not be reported."
        )

    stats = novel_latent_fraction(
        crosscoder, base_acts, trained_acts,
        ratio=float(cfg.get("diffing.novelty_decoder_norm_ratio", 0.9)),
        device=device,
    )
    stats.update(reweighting_statistics(crosscoder, base_acts, trained_acts,
                                        device=device))
    stats["capability"] = capability.name
    return {"crosscoder": crosscoder, "stats": stats,
            "activations": (base_acts, trained_acts)}


@torch.no_grad()
def reweighting_statistics(crosscoder, base_acts, trained_acts, *,
                           device: str = "cuda",
                           chunk: int = 8192) -> dict[str, float]:
    """How much did the *shared* latents change?

    If the ceiling hypothesis holds, this is where all the action is:
    near-zero novel latents, but large shifts in how often and how
    strongly the existing ones fire.
    """
    base_norm, variant_norm = crosscoder.decoder_norms()
    cosine = F.cosine_similarity(
        crosscoder.decoder_base.weight, crosscoder.decoder_variant.weight, dim=0
    ).cpu()

    fire_counts = torch.zeros(crosscoder.config.d_hidden, device=device)
    magnitude = torch.zeros(crosscoder.config.d_hidden, device=device)
    n_rows = 0
    for start in range(0, base_acts.shape[0], chunk):
        base = base_acts[start:start + chunk].to(device)
        trained = trained_acts[start:start + chunk].to(device)
        latents = crosscoder.encode(base, trained)
        fire_counts += (latents > 0).float().sum(dim=0)
        magnitude += latents.sum(dim=0)
        n_rows += base.shape[0]

    shared = (variant_norm / (base_norm + variant_norm).clamp_min(1e-8))
    shared_mask = (shared > 0.2) & (shared < 0.8)
    return {
        "shared_latent_frac": shared_mask.float().mean().item(),
        "mean_decoder_cosine": cosine.mean().item(),
        "shared_decoder_cosine": cosine[shared_mask].mean().item()
        if shared_mask.any() else float("nan"),
        "mean_firing_rate": (fire_counts / max(1, n_rows)).mean().item(),
        "mean_latent_magnitude": (magnitude / max(1, n_rows)).mean().item(),
    }


def write_report(run_dir: Path, stats_by_capability: dict[str, dict]) -> Path:
    """Write the diffing report that feeds the paper's results table."""
    path = run_dir / "diffing.json"
    path.write_text(
        json.dumps(stats_by_capability, indent=2, default=float),
        encoding="utf-8",
    )
    for name, stats in stats_by_capability.items():
        logger.info(
            "%s: nlf_raw=%.4f nlf_corrected=%.4f (%.0f%% of apparent novelty "
            "was a sparsity artifact)", name, stats["nlf_raw"],
            stats["nlf_corrected"], 100 * stats["artifact_frac"],
        )
    return path
