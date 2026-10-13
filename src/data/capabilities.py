"""The capability battery.

A capability bundles a generator, a strict verifier, a partial-credit
scorer, and a declared expectation about whether the base student
already represents it. The expectation is a prior only: the audit in
``src/audit`` is what actually decides the split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

from .loaders import REPRESENTED_GENERATORS
from .synthetic import ABSENT_GENERATORS, Example


@dataclass(frozen=True)
class Capability:
    """One task in the battery."""

    name: str
    generate: Callable[..., list[Example]]
    verify: Callable[[str, str], float]
    partial: Callable[[str, str], float]
    expected_group: str  # "represented" or "absent", a prior, not the split

    def sample(self, n: int, seed: int = 0) -> list[Example]:
        return self.generate(n, random.Random(seed))


def build_registry() -> dict[str, Capability]:
    """All capabilities, keyed by name."""
    registry: dict[str, Capability] = {}
    for name, (gen, verify, partial) in REPRESENTED_GENERATORS.items():
        registry[name] = Capability(name, gen, verify, partial, "represented")
    for name, (gen, verify, partial) in ABSENT_GENERATORS.items():
        registry[name] = Capability(name, gen, verify, partial, "absent")
    return registry


REGISTRY = build_registry()


def select(names: Iterable[str]) -> list[Capability]:
    """Look up capabilities by name, erroring loudly on typos."""
    chosen = []
    for name in names:
        if name not in REGISTRY:
            raise KeyError(
                f"unknown capability {name!r}; available: {sorted(REGISTRY)}"
            )
        chosen.append(REGISTRY[name])
    return chosen


def from_config(cfg) -> list[Capability]:
    """Every capability named in the config's battery block."""
    names = list(cfg.get("battery.represented", [])) + list(
        cfg.get("battery.absent", [])
    )
    return select(names)


def score(capability: Capability, predictions: list[str],
          answers: list[str]) -> dict[str, float]:
    """Strict and partial accuracy for one capability."""
    if len(predictions) != len(answers):
        raise ValueError("predictions and answers must be the same length")
    if not answers:
        return {"accuracy": 0.0, "partial": 0.0, "n": 0}
    strict = sum(capability.verify(p, a) for p, a in zip(predictions, answers))
    soft = sum(capability.partial(p, a) for p, a in zip(predictions, answers))
    n = len(answers)
    return {"accuracy": strict / n, "partial": soft / n, "n": float(n)}
