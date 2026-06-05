"""Classifiers and the cross-validation harness.

The classifier is deliberately *simple and unoptimized*. We read its above-chance
accuracy as a **certified lower bound** on the leakage I(task class ; observation)
(Fano's inequality: a decoder that recovers the label well above chance proves the
observation carries information about it). An unoptimized model can only
*understate* leakage, never overstate it -- which is exactly why we do not chase
classifier performance.

The default is a random forest (robust, scale-free, handles the mixed sparse-bag /
dense-scalar feature matrix); logistic regression is available as a linear
robustness echo. Both honor the sequential structure only through n-gram features,
not through a bespoke sequence model -- enough for a lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier(name: str, seed: int):
    """Return a fresh sklearn estimator."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed, max_features="sqrt"
        )
    if name == "logreg":
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0),
        )
    raise ValueError(f"unknown classifier: {name!r}")


@dataclass
class CVResult:
    """Out-of-fold predictions and per-fold accuracies for one (view, protection)."""

    y_true: np.ndarray
    y_pred: np.ndarray
    fold_accuracy: list[float]
    labels: list[str]


def cross_validate(
    X: np.ndarray, y: np.ndarray, *, classifier: str, folds: int, seed: int
) -> CVResult:
    """Stratified k-fold; returns out-of-fold predictions over the whole dataset."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y)
    fold_acc: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        clf = build_classifier(classifier, seed)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        y_pred[test_idx] = pred
        fold_acc.append(float((pred == y[test_idx]).mean()))
    labels = sorted(set(y.tolist()))
    return CVResult(y_true=y, y_pred=y_pred, fold_accuracy=fold_acc, labels=labels)


def cross_validate_proba(
    X: np.ndarray, y: np.ndarray, *, classifier: str, folds: int, seed: int
) -> tuple[list[str], np.ndarray]:
    """Out-of-fold class posteriors over the whole dataset.

    Returns ``(classes, proba)`` where ``classes`` is the sorted class vocabulary
    and ``proba[i, c]`` is the held-out probability that sample ``i`` belongs to
    class ``classes[c]``. Each row is scored by a model trained on the *other*
    folds, so there is no train/test leakage -- the same discipline as
    :func:`cross_validate`. This is what an adversary that must *rank and act on*
    workflows (rather than merely label them) needs: a calibrated-enough score per
    workflow under the same simple, unoptimized decoder.
    """
    classes = sorted(set(y.tolist()))
    col = {c: j for j, c in enumerate(classes)}
    proba = np.zeros((len(y), len(classes)), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        clf = build_classifier(classifier, seed)
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])
        # Map this fold's class order onto the global column order (a fold need not
        # see every class, though stratification makes that the norm here).
        for local_j, c in enumerate(clf.classes_):
            proba[test_idx, col[c]] = p[:, local_j]
    return classes, proba
