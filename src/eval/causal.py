"""The causal readout: are the candidate new latents load-bearing?

A latent that survives the latent-scaling correction is only evidence of
acquisition if removing it costs the model the capability. This module
ablates and steers each candidate at the layer it was found and
remeasures accuracy.

A latent counts as acquired only if it is both novel after correction
and causally necessary. That conjunction is what makes a refutation of
the ceiling hypothesis credible rather than suggestive.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch

from .passk import generate

logger = logging.getLogger(__name__)


@contextmanager
def latent_intervention(loaded, crosscoder, latent_index: int, *,
                        layer: str | int = "middle", coefficient: float = 0.0):
    """Temporarily add ``coefficient`` times a latent direction.

    ``coefficient=0`` with ``ablate=True`` semantics is handled by
    projecting the direction out instead of adding to it, which is what
    :func:`ablate_latent` does.
    """
    index = loaded.layer_index(layer)
    direction = crosscoder.decoder_variant.weight[:, latent_index].detach()
    direction = (direction / direction.norm().clamp_min(1e-8)).to(loaded.device)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        shifted = hidden + coefficient * direction
        if isinstance(output, tuple):
            return (shifted,) + output[1:]
        return shifted

    handle = loaded.model.model.layers[index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_latent(loaded, crosscoder, latent_index: int, *,
                  layer: str | int = "middle"):
    """Project a latent direction out of the residual stream."""
    index = loaded.layer_index(layer)
    direction = crosscoder.decoder_variant.weight[:, latent_index].detach()
    direction = (direction / direction.norm().clamp_min(1e-8)).to(loaded.device)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        projection = (hidden @ direction).unsqueeze(-1) * direction
        cleaned = hidden - projection
        if isinstance(output, tuple):
            return (cleaned,) + output[1:]
        return cleaned

    handle = loaded.model.model.layers[index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def _accuracy(loaded, capability, examples, cfg) -> float:
    outputs = generate(
        loaded, [ex.prompt for ex in examples], n_samples=1, temperature=0.0,
        max_new_tokens=int(cfg.get("battery.max_new_tokens", 256)),
    )
    correct = sum(
        capability.verify(texts[0], ex.answer)
        for texts, ex in zip(outputs, examples)
    )
    return correct / max(1, len(examples))


def test_candidates(loaded, crosscoder, candidate_indices, capability, cfg, *,
                    seed: int = 0, n_examples: int = 200) -> list[dict]:
    """Ablate and steer each candidate new latent, then remeasure."""
    examples = capability.sample(n_examples, seed=seed)
    layer = cfg.get("diffing.crosscoder.layer", "middle")
    coefficients = list(cfg.get("diffing.causal.steer_coefficients",
                                [-2.0, -1.0, 1.0, 2.0]))
    baseline = _accuracy(loaded, capability, examples, cfg)
    logger.info("baseline accuracy on %s: %.3f", capability.name, baseline)

    results = []
    for latent_index in candidate_indices:
        record = {"latent": int(latent_index), "baseline": baseline}

        if cfg.get("diffing.causal.ablate", True):
            with ablate_latent(loaded, crosscoder, latent_index, layer=layer):
                record["ablated"] = _accuracy(loaded, capability, examples, cfg)
            record["ablation_drop"] = baseline - record["ablated"]

        for coefficient in coefficients:
            with latent_intervention(loaded, crosscoder, latent_index,
                                     layer=layer, coefficient=coefficient):
                record[f"steer_{coefficient}"] = _accuracy(
                    loaded, capability, examples, cfg
                )

        # Necessary means ablating it costs a meaningful share of the
        # capability, not merely a measurable one.
        record["necessary"] = record.get("ablation_drop", 0.0) > 0.5 * baseline
        results.append(record)
        logger.info("latent %d: %s", latent_index, record)

    return results


def causal_novel_fraction(results: list[dict], d_hidden: int) -> float:
    """Fraction of all latents that are both novel and necessary."""
    if not d_hidden:
        return 0.0
    return sum(1 for r in results if r.get("necessary")) / d_hidden
