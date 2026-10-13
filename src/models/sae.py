"""A BatchTopK sparse autoencoder over residual-stream activations.

BatchTopK keeps the k*batch largest activations across the whole batch
rather than exactly k per example, which avoids the dead-latent and
shrinkage problems of an L1 penalty. That matters here: the entire
capability-ceiling claim is about counting latents, so an architecture
whose sparsity mechanism distorts the count is not acceptable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    d_model: int
    d_hidden: int
    k: int = 32

    @classmethod
    def from_cfg(cls, cfg, d_model: int) -> "SAEConfig":
        return cls(
            d_model=d_model,
            d_hidden=int(cfg.get("audit.sae.width_multiple", 16)) * d_model,
            k=int(cfg.get("audit.sae.k", 32)),
        )


class BatchTopKSAE(nn.Module):
    """Encoder, BatchTopK gate, tied-init decoder."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.d_model, config.d_hidden)
        self.decoder = nn.Linear(config.d_hidden, config.d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(config.d_model))

        # Tied initialization, then unit-normalize decoder columns so
        # decoder norms are comparable across latents.
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu(self.encoder(x - self.pre_bias))
        return self._batch_topk(pre)

    def _batch_topk(self, pre: torch.Tensor) -> torch.Tensor:
        """Keep the ``k * batch`` largest activations in the batch."""
        n_keep = max(1, self.config.k * pre.shape[0])
        flat = pre.flatten()
        n_keep = min(n_keep, flat.numel())
        threshold = torch.topk(flat, n_keep, sorted=False).values.min()
        return pre * (pre >= threshold)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        recon = self.decoder(latents) + self.pre_bias
        return recon, latents

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        recon, latents = self(x)
        mse = F.mse_loss(recon, x)
        with torch.no_grad():
            variance = x.var(dim=0).sum().clamp_min(1e-8)
            explained = 1.0 - (x - recon).var(dim=0).sum() / variance
            live = (latents > 0).any(dim=0).float().mean()
        return mse, {
            "mse": mse.item(),
            "explained_variance": explained.item(),
            "live_latent_frac": live.item(),
        }


def train_sae(activations: torch.Tensor, config: SAEConfig, *,
              steps: int = 20000, lr: float = 3e-4,
              batch_tokens: int = 4096, device: str = "cuda",
              log_every: int = 1000) -> BatchTopKSAE:
    """Train an SAE on a stack of activation vectors."""
    if activations.ndim != 2:
        raise ValueError("expected activations of shape (n_tokens, d_model)")
    sae = BatchTopKSAE(config).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n = activations.shape[0]

    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_tokens, n),))
        batch = activations[idx].to(device)
        loss, stats = sae.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()
        if log_every and step % log_every == 0:
            logger.info("sae step %d: %s", step, stats)
    return sae


@torch.no_grad()
def latent_detector_f1(sae: BatchTopKSAE, activations: torch.Tensor,
                       labels: torch.Tensor, device: str = "cuda",
                       chunk: int = 8192) -> torch.Tensor:
    """Best-threshold F1 of every latent as a detector for ``labels``.

    Returns a vector of length ``d_hidden``. The audit's SAE coverage is
    the fraction of task-critical variables for which some latent beats
    the configured F1 threshold.
    """
    labels = labels.bool().to(device)
    n_pos = labels.sum().clamp_min(1)
    tp = torch.zeros(sae.config.d_hidden, device=device)
    fp = torch.zeros(sae.config.d_hidden, device=device)

    for start in range(0, activations.shape[0], chunk):
        batch = activations[start:start + chunk].to(device)
        fired = sae.encode(batch) > 0
        batch_labels = labels[start:start + chunk].unsqueeze(1)
        tp += (fired & batch_labels).sum(dim=0)
        fp += (fired & ~batch_labels).sum(dim=0)

    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / n_pos
    return (2 * precision * recall / (precision + recall).clamp_min(1e-8)).cpu()
