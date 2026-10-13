"""Sparse crosscoder and the latent-scaling correction.

A crosscoder learns one sparse dictionary that reconstructs the
activations of *both* models from a shared latent code, using a separate
decoder per model. A latent whose base-side decoder norm is near zero
while its variant-side norm is large looks like a feature the fine-tune
introduced.

That naive reading is the trap. The sparsity penalty splits shared
features, so many apparently variant-only latents are really shared
latents in disguise. ``latent_scaling`` re-fits, per latent, the scalar
that best explains the base activations from the variant decoder
direction; latents that are genuinely new have a base-side scale near
zero, and artifacts do not. Always report the corrected number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CrosscoderConfig:
    d_model: int
    d_hidden: int
    k: int = 32

    @classmethod
    def from_cfg(cls, cfg, d_model: int) -> "CrosscoderConfig":
        return cls(
            d_model=d_model,
            d_hidden=int(cfg.get("diffing.crosscoder.width_multiple", 16)) * d_model,
            k=int(cfg.get("diffing.crosscoder.k", 32)),
        )


class Crosscoder(nn.Module):
    """One encoder over concatenated activations, two decoders."""

    def __init__(self, config: CrosscoderConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(2 * config.d_model, config.d_hidden)
        self.decoder_base = nn.Linear(config.d_hidden, config.d_model, bias=True)
        self.decoder_variant = nn.Linear(config.d_hidden, config.d_model, bias=True)

    def encode(self, base: torch.Tensor, variant: torch.Tensor) -> torch.Tensor:
        pre = F.relu(self.encoder(torch.cat([base, variant], dim=-1)))
        n_keep = max(1, self.config.k * pre.shape[0])
        flat = pre.flatten()
        n_keep = min(n_keep, flat.numel())
        threshold = torch.topk(flat, n_keep, sorted=False).values.min()
        return pre * (pre >= threshold)

    def forward(self, base: torch.Tensor, variant: torch.Tensor):
        latents = self.encode(base, variant)
        return self.decoder_base(latents), self.decoder_variant(latents), latents

    def loss(self, base: torch.Tensor, variant: torch.Tensor):
        recon_base, recon_variant, latents = self(base, variant)
        loss = F.mse_loss(recon_base, base) + F.mse_loss(recon_variant, variant)
        with torch.no_grad():
            live = (latents > 0).any(dim=0).float().mean().item()
        return loss, {"loss": loss.item(), "live_latent_frac": live}

    @torch.no_grad()
    def decoder_norms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-latent decoder norms for the base and variant sides."""
        return (
            self.decoder_base.weight.norm(dim=0).cpu(),
            self.decoder_variant.weight.norm(dim=0).cpu(),
        )


def train_crosscoder(base_acts: torch.Tensor, variant_acts: torch.Tensor,
                     config: CrosscoderConfig, *, steps: int = 30000,
                     lr: float = 3e-4, batch_tokens: int = 4096,
                     device: str = "cuda", log_every: int = 1000) -> Crosscoder:
    """Train a crosscoder on paired activations from two models."""
    if base_acts.shape != variant_acts.shape:
        raise ValueError(
            "base and variant activations must be paired and identically "
            "shaped; see collect_paired_activations"
        )
    model = Crosscoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = base_acts.shape[0]

    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_tokens, n),))
        loss, stats = model.loss(
            base_acts[idx].to(device), variant_acts[idx].to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log_every and step % log_every == 0:
            logger.info("crosscoder step %d: %s", step, stats)
    return model


@torch.no_grad()
def raw_novel_mask(model: Crosscoder, ratio: float = 0.9) -> torch.Tensor:
    """Latents that *look* variant-only from decoder norms alone.

    This is the uncorrected number. Report it, but never report it
    alone: it is the quantity the sparsity artifact inflates.
    """
    base_norm, variant_norm = model.decoder_norms()
    total = (base_norm + variant_norm).clamp_min(1e-8)
    return (variant_norm / total) > ratio


@torch.no_grad()
def latent_scaling(model: Crosscoder, base_acts: torch.Tensor,
                   variant_acts: torch.Tensor, *, device: str = "cuda",
                   chunk: int = 8192) -> torch.Tensor:
    """Per-latent base-side scale, used to strip sparsity artifacts.

    For each latent j we fit the scalar beta_j minimizing
    ``|| base_acts - beta_j * a_j * d_j ||`` where ``a_j`` is the latent
    activation and ``d_j`` the variant decoder direction. The
    closed-form solution is the ratio of two accumulated inner products.

    A latent that is genuinely absent from the base model explains none
    of the base activations, so beta_j is near zero. An artifact (a
    shared feature split by sparsity) has a clearly nonzero beta_j.
    """
    direction = F.normalize(model.decoder_variant.weight, dim=0).to(device)
    numerator = torch.zeros(model.config.d_hidden, device=device)
    denominator = torch.zeros(model.config.d_hidden, device=device)

    for start in range(0, base_acts.shape[0], chunk):
        base = base_acts[start:start + chunk].to(device)
        variant = variant_acts[start:start + chunk].to(device)
        latents = model.encode(base, variant)
        # projection of the base activation onto each latent direction
        projection = base @ direction
        numerator += (latents * projection).sum(dim=0)
        denominator += (latents * latents).sum(dim=0)

    return (numerator / denominator.clamp_min(1e-8)).cpu()


@torch.no_grad()
def novel_latent_fraction(model: Crosscoder, base_acts: torch.Tensor,
                          variant_acts: torch.Tensor, *, ratio: float = 0.9,
                          beta_threshold: float = 0.1,
                          device: str = "cuda") -> dict[str, float]:
    """Both the raw and the corrected novel-latent fraction.

    The corrected number is the one that belongs in the paper. The raw
    number belongs next to it, so a reader can see how much of the
    apparent novelty was an artifact.
    """
    raw_mask = raw_novel_mask(model, ratio=ratio)
    beta = latent_scaling(model, base_acts, variant_acts, device=device)
    corrected_mask = raw_mask & (beta.abs() < beta_threshold)
    d_hidden = float(model.config.d_hidden)
    return {
        "nlf_raw": raw_mask.float().mean().item(),
        "nlf_corrected": corrected_mask.float().mean().item(),
        "n_raw": float(raw_mask.sum().item()),
        "n_corrected": float(corrected_mask.sum().item()),
        "d_hidden": d_hidden,
        "artifact_frac": 1.0 - (
            corrected_mask.sum().item() / max(1.0, raw_mask.sum().item())
        ),
    }
