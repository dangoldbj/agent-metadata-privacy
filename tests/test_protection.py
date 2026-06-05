"""Verification that each §5 transform provably closes the channel it targets.

We plant a *single* leakage channel (endpoint identity, message length, or
capability label) so the class is recoverable only through that channel, then check
the matching protection collapses recovery to chance while the others do not. This
is the unit-level proof behind the headline protection-collapse result.
"""

from __future__ import annotations

import numpy as np

from agentgraph.adversary import _NETWORK_SCALARS, FeatureSpace, extract_features
from agentgraph.classify import cross_validate
from agentgraph.protection import discovery_privacy, metadata_min, unlinkability
from agentgraph.trace import Direction, Message, StepType, Trace


def _fs(capabilities: list[str], endpoints: list[str]) -> FeatureSpace:
    return FeatureSpace(
        capabilities=capabilities,
        endpoints=endpoints,
        cap_index={c: i for i, c in enumerate(capabilities)},
        endpoint_index={e: i for i, e in enumerate(endpoints)},
    )


def _one_stage_trace(tid: int, cls: str, provider: str, length: int, cap: str) -> Trace:
    m = Message(
        src="client", dst=provider, t=0.0, length=length, direction=Direction.C2S,
        stage_idx=0, step_type=StepType.REQUEST, capability=cap, label_visible=True,
    )
    return Trace(trace_id=tid, task_class=cls, client_id="client", n_stages=1, messages=(m,))


def _accuracy(traces: list[Trace], view: str, fs: FeatureSpace) -> float:
    X = extract_features(traces, view, fs)
    y = np.array([t.task_class for t in traces])
    cv = cross_validate(X, y, classifier="rf", folds=3, seed=0)
    return float((cv.y_true == cv.y_pred).mean())


def test_unlinkability_zeroes_the_endpoint_bag() -> None:
    # Class lives only in the provider identity; identical length/timing/cap.
    fs = _fs(capabilities=["x"], endpoints=["client", "agent-A", "agent-B"])
    traces = [
        _one_stage_trace(i, "A" if i % 2 == 0 else "B",
                         "agent-A" if i % 2 == 0 else "agent-B", 1000, "x")
        for i in range(120)
    ]
    assert _accuracy(traces, "network", fs) > 0.9  # endpoint id is a perfect tell
    # after unlinkability every token is fresh => out-of-vocabulary => bag is empty
    prot = unlinkability(traces)
    Xp = extract_features(prot, "network", fs)
    assert Xp[:, : len(fs.endpoints)].sum() == 0.0
    assert _accuracy(prot, "network", fs) < 0.62  # near chance (0.5)


def test_metadata_min_constant_volume_and_timing() -> None:
    # Class lives only in message length; same provider, same timing.
    fs = _fs(capabilities=["x"], endpoints=["client", "agent-S"])
    traces = [
        _one_stage_trace(i, "A" if i % 2 == 0 else "B", "agent-S",
                         500 if i % 2 == 0 else 5000, "x")
        for i in range(120)
    ]
    assert _accuracy(traces, "network", fs) > 0.9  # length is a perfect tell
    prot = metadata_min(traces)
    # padded length and constant cadence => len_std and iat_std are exactly zero
    X = extract_features(prot, "network", fs)
    base = len(fs.endpoints) + 4 + 8
    len_std = X[:, base + _NETWORK_SCALARS.index("len_std")]
    assert np.allclose(len_std, 0.0)
    assert _accuracy(prot, "network", fs) < 0.62  # near chance


def test_discovery_privacy_removes_the_label_channel() -> None:
    # Class lives only in the capability label seen by the registry view.
    fs = _fs(capabilities=["alpha", "beta"], endpoints=["client", "agent-S"])
    traces = [
        _one_stage_trace(i, "A" if i % 2 == 0 else "B", "agent-S", 1000,
                         "alpha" if i % 2 == 0 else "beta")
        for i in range(120)
    ]
    assert _accuracy(traces, "registry", fs) > 0.9  # label is a perfect tell
    prot = discovery_privacy(traces)
    X = extract_features(prot, "registry", fs)
    assert X.sum() == 0.0  # no labels visible => empty registry features
    assert _accuracy(prot, "registry", fs) < 0.62  # near chance


def test_transport_protections_do_not_touch_the_label_channel() -> None:
    # Faithfulness: unlinkability/metadata_min leave the registry (label) view intact.
    fs = _fs(capabilities=["alpha", "beta"], endpoints=["client", "agent-S"])
    traces = [
        _one_stage_trace(i, "A" if i % 2 == 0 else "B", "agent-S", 1000,
                         "alpha" if i % 2 == 0 else "beta")
        for i in range(120)
    ]
    assert _accuracy(unlinkability(traces), "registry", fs) > 0.9
    assert _accuracy(metadata_min(traces), "registry", fs) > 0.9
