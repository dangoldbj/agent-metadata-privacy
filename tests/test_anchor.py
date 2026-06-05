"""The generator must stay structurally faithful to the real A2A capture.

Uses the committed ``results/anchor_capture.json`` (a real ``a2a-sdk`` lifecycle).
If it is absent, the test skips -- the core experiment does not depend on the
anchor capture being regenerated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentgraph.anchor import compare, generator_summary, load_capture, real_summary
from agentgraph.config import DEFAULT

CAPTURE = Path(__file__).resolve().parents[1] / "results" / "anchor_capture.json"


@pytest.mark.skipif(not CAPTURE.exists(), reason="no committed A2A capture")
def test_generator_matches_real_capture_in_shape_and_scale() -> None:
    real = real_summary(load_capture(CAPTURE))
    gen = generator_summary(DEFAULT)
    df = compare(real, gen)
    # every key per-delegation metric must sit within an order of magnitude
    assert df["same_order_of_magnitude"].all(), df.to_string(index=False)
    # the real lifecycle has exactly one discovery fetch per run; the generator
    # places discovery before every delegation (>= 1 per workflow).
    assert real["discovery_per_run"] >= 1
    assert gen["discovery_per_run"] >= 1


def test_compare_flags_order_of_magnitude_gap() -> None:
    real = {"request_bytes": 100, "update_bytes": 100,
            "updates_per_delegation": 3, "interarrival_s": 0.1}
    gen = {"request_bytes": 100, "update_bytes": 100_000,  # 1000x -> out of range
           "updates_per_delegation": 3, "interarrival_s": 0.1}
    df = compare(real, gen).set_index("metric")
    assert df.loc["request_bytes", "same_order_of_magnitude"]
    assert not df.loc["update_bytes", "same_order_of_magnitude"]
