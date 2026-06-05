"""Fast smoke test that the structural claims hold on a tiny sensitivity grid."""

from __future__ import annotations

from agentgraph.sensitivity import run_sensitivity


def test_structural_claims_hold() -> None:
    df = run_sensitivity({"n_classes": [4, 8]})
    assert {"network_none", "network_prefix", "network_both", "chance"} <= set(df.columns)
    for _, r in df.iterrows():
        # leakage exists: label-blind network well above chance
        assert r["network_none"] > 2 * r["chance"]
        # prospectivity: a short prefix already leaks above chance
        assert r["network_prefix"] > 2 * r["chance"]
        # protection works: both wire properties collapse recovery sharply
        assert r["network_both"] < r["network_none"] - 0.15
