"""The behavioral readout: pass@1 and pass@k.

The pass@k curve is what separates sharpening from expansion. If
training only reweights what the base model already does, pass@1 rises
while pass@k at large k stays flat or falls: the model got better at
picking among solutions it could already reach. If training genuinely
added a capability, the whole coverage curve lifts.

This readout needs no interpretability tooling, which is the point. It
is an independent check on the crosscoder analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for ``c`` correct out of ``n`` samples.

    The naive estimator (did any of the first k succeed) is biased and
    noisy. This is the standard combinatorial correction.
    """
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@torch.no_grad()
def generate(loaded, prompts: list[str], *, n_samples: int = 1,
             temperature: float = 0.8, top_p: float = 0.95,
             max_new_tokens: int = 256,
             batch_size: int = 8) -> list[list[str]]:
    """Sample ``n_samples`` continuations for each prompt."""
    tokenizer = loaded.tokenizer
    outputs: list[list[str]] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        ).to(loaded.device)
        generated = loaded.model.generate(
            **encoded, do_sample=temperature > 0,
            temperature=max(temperature, 1e-5), top_p=top_p,
            num_return_sequences=n_samples, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )
        for i in range(len(batch)):
            outputs.append(decoded[i * n_samples:(i + 1) * n_samples])

    return outputs


def evaluate(loaded, capability, cfg, *, seed: int = 0,
             n_examples: int | None = None) -> dict[str, float]:
    """pass@1 and the full pass@k coverage curve for one capability."""
    n_examples = n_examples or int(cfg.get("battery.n_eval_per_capability", 500))
    ks = list(cfg.get("eval.pass_at_k", [1, 4, 16, 64, 256]))
    n_samples = int(cfg.get("eval.n_samples_for_passk", 256))
    if n_samples < max(ks):
        raise ValueError(
            f"n_samples_for_passk ({n_samples}) must be at least "
            f"max(pass_at_k) ({max(ks)}), or pass@k is not estimable"
        )

    examples = capability.sample(n_examples, seed=seed)
    samples = generate(
        loaded, [ex.prompt for ex in examples], n_samples=n_samples,
        temperature=float(cfg.get("eval.sample_temperature", 0.8)),
        top_p=float(cfg.get("eval.sample_top_p", 0.95)),
        max_new_tokens=int(cfg.get("battery.max_new_tokens", 256)),
    )

    correct_counts = [
        sum(capability.verify(text, ex.answer) for text in texts)
        for texts, ex in zip(samples, examples)
    ]
    results = {"n_examples": float(len(examples)), "n_samples": float(n_samples)}
    for k in ks:
        results[f"pass@{k}"] = float(np.mean([
            pass_at_k(n_samples, int(c), k) for c in correct_counts
        ]))
    results["partial"] = float(np.mean([
        capability.partial(texts[0], ex.answer)
        for texts, ex in zip(samples, examples)
    ]))
    return results


def coverage_signature(base: dict[str, float],
                       trained: dict[str, float]) -> dict[str, float]:
    """Does the solution boundary move, or does it only sharpen?

    ``verdict`` is ``"sharpening"`` when pass@1 improves but high-k
    coverage does not, which is the reweighting signature, and
    ``"expansion"`` when high-k coverage lifts too.
    """
    ks = sorted(
        int(key.split("@")[1]) for key in base if key.startswith("pass@")
    )
    if not ks:
        return {}
    low, high = ks[0], ks[-1]
    delta_low = trained[f"pass@{low}"] - base[f"pass@{low}"]
    delta_high = trained[f"pass@{high}"] - base[f"pass@{high}"]
    return {
        f"delta_pass@{low}": delta_low,
        f"delta_pass@{high}": delta_high,
        "verdict": "expansion" if delta_high > 0.01 else "sharpening",
    }
