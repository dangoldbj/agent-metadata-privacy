"""Evaluation: chance baselines, metrics, and the leakage table.

Leakage is reported as a classifier's out-of-fold accuracy (a certified lower
bound on I(task class ; observation), see ``classify.py``) against an explicit
*chance* baseline, with a bootstrap 95% CI. The headline artifact is the leakage
table across adversary vantage points -- crucially including the **label-blind
network** row above chance.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .adversary import build_feature_space, extract_features, labels_array
from .classify import CVResult, cross_validate
from .config import ExperimentConfig
from .trace import Trace
from .workflows import WorkflowModel, generate_dataset

Transform = Callable[[list[Trace]], list[Trace]]


def _identity(traces: list[Trace]) -> list[Trace]:
    return traces


def chance_accuracy(y: np.ndarray) -> dict[str, float]:
    """Uniform, majority, and stratified-random chance accuracies."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return {
        "uniform": 1.0 / len(p),
        "majority": float(p.max()),
        "stratified": float((p**2).sum()),
    }


def bootstrap_accuracy_ci(
    y_true: np.ndarray, y_pred: np.ndarray, *, n: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for accuracy over the out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    correct = (y_true == y_pred).astype(float)
    m = len(correct)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        means[i] = correct[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_cv(cv: CVResult, chance: dict[str, float], *, ci_n: int, seed: int) -> dict:
    acc = float((cv.y_true == cv.y_pred).mean())
    lo, hi = bootstrap_accuracy_ci(cv.y_true, cv.y_pred, n=ci_n, seed=seed)
    macro_f1 = float(f1_score(cv.y_true, cv.y_pred, average="macro", labels=cv.labels))
    return {
        "accuracy": acc,
        "acc_ci_lo": lo,
        "acc_ci_hi": hi,
        "macro_f1": macro_f1,
        "fold_mean": float(np.mean(cv.fold_accuracy)),
        "fold_std": float(np.std(cv.fold_accuracy)),
        "chance_uniform": chance["uniform"],
        "chance_majority": chance["majority"],
        "lift_over_chance": acc / chance["uniform"],
    }


def evaluate_views(
    traces: list[Trace],
    model: WorkflowModel,
    cfg: ExperimentConfig,
    *,
    views: tuple[str, ...] | None = None,
    classifier: str = "rf",
    transform: Transform | None = None,
    protection_label: str = "none",
) -> pd.DataFrame:
    """Leakage for each adversary view under one (optional) protection transform."""
    views = views or cfg.eval.views
    transform = transform or _identity
    fs = build_feature_space(model)
    obs = transform(traces)
    y = labels_array(obs)
    chance = chance_accuracy(y)

    rows = []
    for view in views:
        X = extract_features(obs, view, fs)
        cv = cross_validate(
            X, y, classifier=classifier, folds=cfg.eval.cv_folds, seed=cfg.seed
        )
        row = {"view": view, "protection": protection_label, "classifier": classifier}
        row.update(summarize_cv(cv, chance, ci_n=cfg.eval.bootstrap_ci, seed=cfg.seed))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_prefix(
    traces: list[Trace],
    model: WorkflowModel,
    cfg: ExperimentConfig,
    *,
    views: tuple[str, ...] | None = None,
    classifier: str = "rf",
) -> pd.DataFrame:
    """Prospectivity: leakage as a function of the observed prefix fraction.

    For each fraction ``p`` we truncate every trace to its first ``p`` of stages and
    re-measure recovery -- how well an observer predicts the *pending* task class
    from only the opening of the workflow, before it completes.
    """
    views = views or cfg.eval.views
    fs = build_feature_space(model)
    rows = []
    for p in cfg.eval.prefix_grid:
        prefixed = [tr.prefix(p) for tr in traces]
        y = labels_array(prefixed)
        chance = chance_accuracy(y)
        for view in views:
            X = extract_features(prefixed, view, fs)
            cv = cross_validate(
                X, y, classifier=classifier, folds=cfg.eval.cv_folds, seed=cfg.seed
            )
            row = {"prefix": p, "view": view}
            row.update(summarize_cv(cv, chance, ci_n=cfg.eval.bootstrap_ci, seed=cfg.seed))
            rows.append(row)
    return pd.DataFrame(rows)


def run_leakage(cfg: ExperimentConfig, *, classifier: str = "rf") -> pd.DataFrame:
    """Result 1: generate the dataset and report leakage across vantage points."""
    traces, model = generate_dataset(cfg.generator, cfg.seed)
    return evaluate_views(traces, model, cfg, classifier=classifier)


def run_prospectivity(cfg: ExperimentConfig, *, classifier: str = "rf") -> pd.DataFrame:
    """Result 2: the accuracy-vs-prefix prospectivity curve."""
    traces, model = generate_dataset(cfg.generator, cfg.seed)
    return evaluate_prefix(traces, model, cfg, classifier=classifier)


def evaluate_protection(
    traces: list[Trace],
    model: WorkflowModel,
    cfg: ExperimentConfig,
    *,
    views: tuple[str, ...] | None = None,
    classifier: str = "rf",
) -> pd.DataFrame:
    """Result 3: leakage per view under each §5 protection transform."""
    from .protection import get_protection  # local import avoids a cycle at module load

    views = views or cfg.eval.views
    frames = []
    for protection in cfg.eval.protections:
        transform = get_protection(protection)
        frames.append(
            evaluate_views(
                traces, model, cfg, views=views, classifier=classifier,
                transform=transform, protection_label=protection,
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_protection(cfg: ExperimentConfig, *, classifier: str = "rf") -> pd.DataFrame:
    """Result 3: generate the dataset and sweep the protections."""
    traces, model = generate_dataset(cfg.generator, cfg.seed)
    return evaluate_protection(traces, model, cfg, classifier=classifier)
