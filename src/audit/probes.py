"""Linear probes on frozen base-student activations.

A probe answers one question: is this task-critical variable linearly
recoverable from the base model's residual stream? If it is not, the
model probably does not represent it, which is exactly the condition
under which the capability-ceiling hypothesis predicts distillation
will fail.

The shuffled-label control is not optional. A probe that beats chance on
shuffled labels means the pipeline is leaking, and every downstream
number is then meaningless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..models.registry import LoadedModel, collect_activations

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    variable: str
    best_auc: float
    best_layer: int
    shuffled_auc: float
    per_layer_auc: dict[int, float] = field(default_factory=dict)

    @property
    def leaked(self) -> bool:
        """True if the shuffled control did better than chance."""
        return self.shuffled_auc > 0.60


def _binarize(values: list) -> np.ndarray | None:
    """Turn a task variable into a binary target.

    Integers are split at the median, strings are turned into a
    one-versus-rest target on the most common value. Variables with only
    one distinct value carry no signal and are skipped.
    """
    unique = sorted({str(v) for v in values})
    if len(unique) < 2:
        return None
    if all(isinstance(v, (int, float, bool)) for v in values):
        array = np.asarray(values, dtype=float)
        threshold = float(np.median(array))
        labels = (array > threshold).astype(int)
        if labels.min() == labels.max():
            labels = (array >= threshold).astype(int)
        return labels if labels.min() != labels.max() else None
    most_common = max(unique, key=lambda u: sum(str(v) == u for v in values))
    return np.asarray([int(str(v) == most_common) for v in values])


def _fit_auc(features: np.ndarray, labels: np.ndarray, *, l2: float,
             test_frac: float, seed: int) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_frac, random_state=seed,
        stratify=labels,
    )
    probe = LogisticRegression(
        C=1.0 / max(l2, 1e-8), max_iter=2000, solver="lbfgs"
    )
    probe.fit(x_train, y_train)
    scores = probe.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, scores))


def probe_capability(student: LoadedModel, examples, *, layers: list[int],
                     l2: float = 1.0, test_frac: float = 0.2,
                     seed: int = 0, shuffle_control: bool = True,
                     max_length: int = 512,
                     batch_size: int = 8) -> list[ProbeResult]:
    """Probe every task-critical variable of one capability.

    Returns one :class:`ProbeResult` per variable, each holding the best
    cross-layer AUC. The capability's probe component is the mean of
    those best AUCs.
    """
    prompts = [ex.prompt for ex in examples]
    variables = sorted({key for ex in examples for key in ex.variables})
    results: list[ProbeResult] = []

    cached: dict[int, np.ndarray] = {}
    for layer in layers:
        activations = collect_activations(
            student, prompts, layer=layer, max_length=max_length,
            batch_size=batch_size, pool="last",
        )
        cached[layer] = activations.numpy()

    rng = np.random.default_rng(seed)
    for variable in variables:
        raw = [ex.variables.get(variable) for ex in examples]
        if any(value is None for value in raw):
            logger.info("skipping %s: not present on every example", variable)
            continue
        labels = _binarize(raw)
        if labels is None:
            logger.info("skipping %s: no label variation", variable)
            continue

        per_layer = {}
        for layer, features in cached.items():
            try:
                per_layer[layer] = _fit_auc(
                    features, labels, l2=l2, test_frac=test_frac, seed=seed
                )
            except ValueError as exc:
                logger.info("probe failed at layer %d for %s: %s",
                            layer, variable, exc)

        if not per_layer:
            continue
        best_layer = max(per_layer, key=per_layer.__getitem__)

        shuffled_auc = 0.5
        if shuffle_control:
            permuted = labels[rng.permutation(len(labels))]
            try:
                shuffled_auc = _fit_auc(
                    cached[best_layer], permuted, l2=l2,
                    test_frac=test_frac, seed=seed,
                )
            except ValueError:
                shuffled_auc = 0.5

        result = ProbeResult(
            variable=variable, best_auc=per_layer[best_layer],
            best_layer=best_layer, shuffled_auc=shuffled_auc,
            per_layer_auc=per_layer,
        )
        if result.leaked:
            logger.warning(
                "shuffled-label control AUC %.3f for %s: investigate a leak "
                "before trusting this probe", shuffled_auc, variable,
            )
        results.append(result)

    return results


def resolve_layers(student: LoadedModel, spec) -> list[int]:
    """Turn the config's ``probe_layers`` into concrete layer indices."""
    if spec == "all":
        # Every fourth layer keeps the audit affordable on one GPU while
        # still covering early, middle and late representations.
        return list(range(0, student.n_layers, max(1, student.n_layers // 8)))
    if isinstance(spec, int):
        return [student.layer_index(spec)]
    return [student.layer_index(int(s)) for s in spec]


@torch.no_grad()
def probe_component(results: list[ProbeResult]) -> float:
    """The probe half of R(c): mean best AUC, rescaled to [0, 1]."""
    usable = [r for r in results if not r.leaked]
    if not usable:
        return 0.0
    mean_auc = float(np.mean([r.best_auc for r in usable]))
    return max(0.0, min(1.0, (mean_auc - 0.5) * 2.0))
