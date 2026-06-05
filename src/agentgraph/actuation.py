"""Actuation: the *value* of acting on the metadata leak (the integrity axis).

The leakage and prospectivity results (``evaluate.py``) measure an
*information-theoretic* quantity: how well an observer can **recover** the task
class, and how **early**. That is the privacy story, and it is essentially
website-fingerprinting territory. Actuation asks a different, **decision-theoretic**
question: what is recoverability *worth* to an adversary that must **act** under a
budget? This is the integrity story, and it is what the paper's "workflow-integrity,
not only privacy" thesis rests on.

The actuation game
------------------
A population of ``N`` concurrent workflows. Each carries an adversary **value**
``v(w) >= 0``. A metadata-only, on-path adversary must, by an early **decision
deadline** ``f`` (it has seen only the opening fraction ``f`` of every workflow),
commit a **budget** ``B`` of interventions -- choosing *which* ``B`` workflows to act
on -- to maximise ``J = sum_{w in chosen} v(w)``. It never reads content; it ranks
workflows using only the label-blind *network* view of the observed prefix.

We compare three policies under an identical budget:

* **informed**  -- rank by the classifier's out-of-fold posterior on the target,
  take the top ``B``;
* **blind**     -- choose ``B`` at random (closed-form expectation ``J_blind``);
* **oracle**    -- choose ``B`` by true value (upper bound ``J_oracle``).

The central quantity is the **Value of Metadata**

    VoM(B, f) = J_informed - J_blind ,

the decision-theoretic *value of information* made operational, and its normalised,
dimensionless form, the **capture ratio**

    kappa = (J_informed - J_blind) / (J_oracle - J_blind) ,

normalised so that 0 is the blind baseline and 1 the oracle. A ranking worse than
random can push kappa slightly below 0; in our experiments it lies in [0, 1]. It is
the share of the *available* selection leverage the adversary realises from metadata
alone -- which workflows to act on -- the precondition for changing outcomes rather
than a demonstration that outcomes change. ``kappa`` is the headline number.

Why this is a *separate* axis (not "better inference"). ``VoM > 0`` needs the
*product* of two independent ingredients: inference signal in the opening
(prospectivity) **and** intervention efficacy under the budget. With ``B = 0`` you
can know everything yet ``VoM = 0``; with chance-level inference you can act yet
``VoM = 0``. Neither implies the other -- so actuation does not reduce to the leakage
number, and website-fingerprinting (which bounds the inference term only) says
nothing about it.

Instantiation here (Instantiation 1, *budgeted prospective targeting*): the minimal
construction that crosses from *knowing* to *doing*. One task class is the
high-value target, ``v(w) = 1`` iff ``w`` is of that class. Then ``J_informed`` is
just the count of true target-class workflows among the top-``B`` by prefix-``f``
posterior; ``J_oracle = min(B, n_target)`` and ``J_blind = B * n_target / N`` in
closed form. No new generative dynamics, so the counterfactual is clean. (Read
equivalently as precision@B / recall of the target under an early deadline.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .adversary import build_feature_space, extract_features, labels_array
from .classify import cross_validate_proba
from .config import ExperimentConfig
from .protection import get_protection
from .workflows import generate_dataset

# Decision deadlines (fraction of the workflow observed before committing the budget)
# and intervention budgets (fraction of the N workflows the adversary may act on).
DEADLINES: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0)
BUDGET_FRACS: tuple[float, ...] = (0.05, 0.125, 0.25, 0.5)
# Headline cell used for the protection figure / summary: an *early* deadline and a
# budget equal to one class's mass (1/K at K=8 => 0.125).
HEADLINE_DEADLINE = 0.2
HEADLINE_BUDGET = 0.125


def _capture(score: np.ndarray, is_target: np.ndarray, B: int, N: int) -> dict[str, float]:
    """Capture metrics for one (score, target, budget). ``B = 0`` is well defined."""
    n_target = float(is_target.sum())
    j_blind = B * n_target / N if N else 0.0
    j_oracle = float(min(B, n_target))
    if B <= 0:
        j_informed = 0.0
    elif B >= len(score):
        j_informed = n_target
    else:
        top = np.argpartition(score, -B)[-B:]  # exact top-B workflows by score
        j_informed = float(is_target[top].sum())
    vom = j_informed - j_blind
    denom = j_oracle - j_blind
    kappa = vom / denom if denom > 1e-12 else 0.0
    return {
        "n_target": n_target,
        "budget_B": float(B),
        "j_informed": j_informed,
        "j_blind": j_blind,
        "j_oracle": j_oracle,
        "vom": vom,
        "capture_ratio": kappa,
        "precision_at_b": j_informed / B if B > 0 else 0.0,
        "recall": j_informed / n_target if n_target > 0 else 0.0,
    }


def run_actuation(
    cfg: ExperimentConfig,
    *,
    classifier: str = "rf",
    deadlines: tuple[float, ...] = DEADLINES,
    budget_fracs: tuple[float, ...] = BUDGET_FRACS,
    view: str = "network",
) -> pd.DataFrame:
    """Result 4: the value of acting on the leak, swept over protections, deadlines,
    budgets, and target classes.

    For each (protection, deadline ``f``) we fit the label-blind decoder once on the
    protected prefix and read out-of-fold posteriors; every (target class, budget)
    cell is then a cheap re-ranking of those posteriors. The same §5 property set that
    collapses inference is expected to drive ``kappa`` toward the blind baseline (0),
    and -- mirroring the protection result -- only the *full* set should do so.
    """
    traces, model = generate_dataset(cfg.generator, cfg.seed)
    fs = build_feature_space(model)
    N = len(traces)
    rows: list[dict] = []
    for protection in cfg.eval.protections:
        obs = get_protection(protection)(traces)
        for f in deadlines:
            prefixed = [tr.prefix(f) for tr in obs]
            y = labels_array(prefixed)
            X = extract_features(prefixed, view, fs)
            classes, proba = cross_validate_proba(
                X, y, classifier=classifier, folds=cfg.eval.cv_folds, seed=cfg.seed
            )
            for j, t in enumerate(classes):
                score = proba[:, j]
                is_target = (y == t).astype(float)
                for bfrac in budget_fracs:
                    B = max(1, round(bfrac * N))
                    m = _capture(score, is_target, B, N)
                    rows.append({
                        "protection": protection,
                        "deadline_f": f,
                        "budget_frac": bfrac,
                        "target_class": t,
                        "view": view,
                        **m,
                    })
    return pd.DataFrame(rows)


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Capture ratio per protection at the headline (deadline, budget), averaged over
    target classes -- the slice the protection figure and summary use.

    The error band ``cap_ci_lo/hi`` is the mean $\\pm 1.96$ standard errors *across
    target classes*: the relevant uncertainty for a headline that averages over which
    class is the adversary's target (it answers whether the leverage is robust to that
    choice rather than driven by one easy class), and it brackets the point estimate by
    construction.
    """
    cell = df[
        (np.isclose(df["deadline_f"], HEADLINE_DEADLINE))
        & (np.isclose(df["budget_frac"], HEADLINE_BUDGET))
    ]

    def _row(g: pd.DataFrame) -> pd.Series:
        k = g["capture_ratio"].to_numpy()
        se = k.std(ddof=1) / np.sqrt(len(k)) if len(k) > 1 else 0.0
        mean = float(k.mean())
        return pd.Series({
            "capture_ratio": mean,
            "vom": float(g["vom"].mean()),
            "cap_ci_lo": mean - 1.96 * se,
            "cap_ci_hi": mean + 1.96 * se,
        })

    return cell.groupby("protection").apply(_row, include_groups=False).reset_index()
