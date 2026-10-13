"""Loading students and teachers, and pulling activations out of them.

Keep teacher and student in the same model family unless you have a
reason not to. On-policy distillation needs teacher log-probs on
*student* tokens, and cross-family tokenizer alignment is the single
most common source of silently wrong results in this kind of project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch

logger = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedModel:
    """A model plus its tokenizer and a few resolved conveniences."""

    name: str
    model: object
    tokenizer: object
    device: str
    n_layers: int
    d_model: int

    def layer_index(self, spec: str | int) -> int:
        """Resolve ``"middle"`` / ``"last"`` / an int to a layer index."""
        if isinstance(spec, int):
            return spec % self.n_layers
        if spec == "middle":
            return self.n_layers // 2
        if spec == "last":
            return self.n_layers - 1
        raise ValueError(f"unrecognized layer spec {spec!r}")


def load_model(name: str, *, dtype: str = "bfloat16", device: str = "cuda",
               train: bool = False) -> LoadedModel:
    """Load a causal LM and its tokenizer from the Hugging Face hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _DTYPES.get(dtype, torch.bfloat16)
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable, falling back to cpu")
        device = "cpu"
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch_dtype, device_map=None
    ).to(device)
    model.train(train)
    if not train:
        model.requires_grad_(False)

    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None) or config.n_layer
    d_model = getattr(config, "hidden_size", None) or config.n_embd
    logger.info("loaded %s: %d layers, d_model=%d", name, n_layers, d_model)
    return LoadedModel(name, model, tokenizer, device, n_layers, d_model)


def load_student_and_teacher(cfg):
    """Load both models as configured, teacher frozen."""
    student = load_model(
        cfg.get("models.student"), dtype=cfg.get("models.dtype", "bfloat16"),
        device=cfg.get("models.device", "cuda"), train=True,
    )
    teacher = load_model(
        cfg.get("models.teacher"), dtype=cfg.get("models.dtype", "bfloat16"),
        device=cfg.get("models.device", "cuda"), train=False,
    )
    return student, teacher


@torch.no_grad()
def collect_activations(loaded: LoadedModel, texts: Iterable[str], *,
                        layer: str | int = "middle", max_length: int = 512,
                        batch_size: int = 8,
                        pool: str = "last") -> torch.Tensor:
    """Residual-stream activations at one layer.

    ``pool="last"`` returns one vector per sequence (the final
    non-padding position), which is what the probes want.
    ``pool="none"`` returns every non-padding token position stacked,
    which is what the SAE and crosscoder want.
    """
    texts = list(texts)
    index = loaded.layer_index(layer)
    tokenizer, model = loaded.tokenizer, loaded.model
    chunks: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        ).to(loaded.device)
        out = model(**encoded, output_hidden_states=True)
        # hidden_states[0] is the embedding output, so layer i is at i+1.
        hidden = out.hidden_states[index + 1]
        mask = encoded["attention_mask"].bool()

        if pool == "last":
            lengths = mask.sum(dim=1) - 1
            picked = hidden[torch.arange(hidden.size(0)), lengths]
            chunks.append(picked.float().cpu())
        elif pool == "none":
            chunks.append(hidden[mask].float().cpu())
        else:
            raise ValueError(f"unrecognized pool {pool!r}")

    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


@torch.no_grad()
def collect_paired_activations(base: LoadedModel, variant: LoadedModel,
                               texts: Iterable[str], **kwargs):
    """Activations from two models on identical inputs.

    The crosscoder is only meaningful if both models see byte-identical
    prompts, so this is the only supported way to build its training
    data. Do not collect the two sides separately with different
    formatting.
    """
    texts = list(texts)
    a = collect_activations(base, texts, pool="none", **kwargs)
    b = collect_activations(variant, texts, pool="none", **kwargs)
    if a.shape != b.shape:
        raise ValueError(
            f"paired activation shape mismatch {tuple(a.shape)} vs "
            f"{tuple(b.shape)}: the two models tokenized the inputs "
            "differently, so the diff would be meaningless"
        )
    return a, b
