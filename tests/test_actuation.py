"""Structural tests for the actuation result: the value of acting on the leak.

We verify the invariants that make the capture ratio / VoM a meaningful quantity
(ordering, the two separation edges) and the qualitative integrity claims (leverage
exists with no protection; the combined wire properties collapse it), in the repo's
"assert structure, not exact numbers" style.
"""

from __future__ import annotations

import numpy as np

from agentgraph.actuation import _capture, run_actuation
from agentgraph.classify import cross_validate_proba
from agentgraph.config import DEFAULT


def _small_cfg():
    # tiny but balanced; enough signal for the structural claims, fast to fit.
    return DEFAULT.with_generator(n_classes=4, n_traces_per_class=60)


def test_capture_invariants_and_separation_edges() -> None:
    rng = np.random.default_rng(0)
    N = 200
    is_target = (rng.random(N) < 0.25).astype(float)

    # A perfectly informative score recovers all available leverage: kappa == 1.
    perfect = _capture(is_target.copy(), is_target, B=50, N=N)
    assert perfect["capture_ratio"] > 0.99
    # An uninformative score realises ~none of it: kappa ~ 0 (blind baseline).
    blind = _capture(rng.random(N), is_target, B=50, N=N)
    assert -0.3 < blind["capture_ratio"] < 0.3
    # Separation edge 1 -- no budget: you may know everything yet VoM == 0.
    nobudget = _capture(is_target.copy(), is_target, B=0, N=N)
    assert nobudget["capture_ratio"] == 0.0 and nobudget["vom"] == 0.0

    # Ordering invariant: blind <= informed <= oracle, always.
    for m in (perfect, blind, nobudget):
        assert m["j_blind"] <= m["j_informed"] + 1e-9
        assert m["j_informed"] <= m["j_oracle"] + 1e-9


def test_out_of_fold_proba_is_a_distribution() -> None:
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0.0, 1.0, (40, 3)), rng.normal(6.0, 1.0, (40, 3))])
    y = np.array(["a"] * 40 + ["b"] * 40)
    classes, proba = cross_validate_proba(X, y, classifier="rf", folds=5, seed=0)
    assert classes == ["a", "b"]
    assert proba.shape == (80, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_leverage_exists_and_collapses_under_wire_properties() -> None:
    cfg = _small_cfg()
    df = run_actuation(cfg, deadlines=(0.2, 1.0), budget_fracs=(0.25,))

    # The ordering invariant holds on every cell of the real grid.
    assert (df["j_blind"] <= df["j_informed"] + 1e-9).all()
    assert (df["j_informed"] <= df["j_oracle"] + 1e-9).all()

    # On the full workflow, leverage is real with no protection ...
    full = df[np.isclose(df["deadline_f"], 1.0)]
    by_prot = full.groupby("protection")["capture_ratio"].mean()
    assert by_prot["none"] > 0.3
    # ... the combined wire properties collapse it sharply ...
    assert by_prot["both"] < by_prot["none"] - 0.15
    # ... and a single wire property leaves more leverage than the two together
    # (the "only as a set" claim).
    assert by_prot["unlinkability"] > by_prot["both"] - 0.05

    # Prospectivity feeds actuation: deciding earlier yields no more leverage than
    # deciding on the full workflow, under no protection.
    early = df[np.isclose(df["deadline_f"], 0.2) & (df["protection"] == "none")]
    assert early["capture_ratio"].mean() <= by_prot["none"] + 1e-6
