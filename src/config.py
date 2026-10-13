"""Config loading. One YAML in, one nested dict-like object out.

Experiment parameters live in the YAML only. Nothing in this project
should read an environment variable to decide an experimental setting,
because then the run metadata no longer describes the run.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class Section(dict):
    """A dict that also supports attribute access, for readability.

    ``cfg.budget.token_budget`` reads better than
    ``cfg["budget"]["token_budget"]`` in training loops.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Section(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


@dataclass
class Config:
    """Resolved experiment configuration plus the path it came from."""

    raw: dict = field(default_factory=dict)
    path: str = ""

    def __getattr__(self, name: str) -> Any:
        raw = self.__dict__.get("raw", {})
        if name in raw:
            value = raw[name]
            return Section(value) if isinstance(value, dict) else value
        raise AttributeError(name)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested key, for example ``cfg.get("budget.lr")``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def run_dir(self, name: str) -> Path:
        return Path(self.raw.get("output_root", "runs")) / name

    def to_json(self) -> str:
        return json.dumps(self.raw, indent=2, sort_keys=True, default=str)


def load_config(path: str | Path) -> Config:
    """Read a YAML config file into a :class:`Config`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return Config(raw=raw, path=str(path))


def _parse_scalar(text: str) -> Any:
    """Parse a CLI value, with a fix for exponent notation.

    YAML 1.1 only recognizes a float exponent when it has a decimal
    point, so ``yaml.safe_load("3e-5")`` returns the *string* ``"3e-5"``.
    That would sail straight into an optimizer as a learning rate, so
    try float first for anything that looks like a number.
    """
    stripped = text.strip()
    if stripped:
        try:
            value = float(stripped)
        except ValueError:
            pass
        else:
            return int(stripped) if stripped.lstrip("+-").isdigit() else value
    return yaml.safe_load(text)


def override(cfg: Config, dotted_assignments: list[str]) -> Config:
    """Apply ``a.b=value`` CLI overrides, returning a new Config.

    Values are parsed as scalars, so ``budget.lr=3e-5`` becomes a float
    and ``diffing.latent_scaling=false`` becomes a bool.
    """
    raw = copy.deepcopy(cfg.raw)
    for assignment in dotted_assignments:
        if "=" not in assignment:
            raise ValueError(f"expected key=value, got {assignment!r}")
        dotted, _, text = assignment.partition("=")
        parts = dotted.split(".")
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into {dotted!r}")
        node[parts[-1]] = _parse_scalar(text)
    return Config(raw=raw, path=cfg.path)
