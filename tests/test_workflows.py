"""Sanity tests for the generative workflow model.

These assert the *structural* properties the experiment relies on -- balanced
classes, genuine capability overlap, correct lifecycle ordering, and
reproducibility -- not any accuracy number.
"""

from __future__ import annotations

import numpy as np

from agentgraph.config import GeneratorConfig
from agentgraph.trace import Direction, StepType
from agentgraph.workflows import WorkflowModel, generate_dataset


def _small_cfg(**overrides) -> GeneratorConfig:
    base = dict(n_classes=5, n_traces_per_class=40)
    base.update(overrides)
    return GeneratorConfig(**base)


def test_dataset_is_balanced() -> None:
    cfg = _small_cfg()
    traces, model = generate_dataset(cfg, seed=1)
    assert len(traces) == cfg.n_classes * cfg.n_traces_per_class
    counts = {name: 0 for name in model.class_names}
    for tr in traces:
        counts[tr.task_class] += 1
    assert set(counts.values()) == {cfg.n_traces_per_class}
    # chance accuracy is exactly 1 / n_classes for a balanced set
    assert abs(1 / cfg.n_classes - 1 / len(counts)) < 1e-12


def test_capability_overlap_exists() -> None:
    # With overlap and a shared common pool, classes must not be capability-disjoint,
    # otherwise the capability *set* alone would trivially reveal the class. We check
    # that most class pairs share at least one capability (genuine overlap), so
    # recovery has to lean on sequence / timing / volume too.
    cfg = _small_cfg(overlap=0.6)
    model = WorkflowModel(cfg, seed=2)
    cap_sets = [set(s.capability for s in tc.stages) for tc in model.classes]
    pairs = [
        (i, j) for i in range(len(cap_sets)) for j in range(i + 1, len(cap_sets))
    ]
    sharing = sum(1 for i, j in pairs if cap_sets[i] & cap_sets[j])
    assert sharing / len(pairs) >= 0.7, "expected most class pairs to share a capability"


def test_lifecycle_ordering_within_stage() -> None:
    # Each stage must open with a discovery query and contain a request; updates
    # precede the final response; timestamps are non-decreasing overall.
    cfg = _small_cfg()
    model = WorkflowModel(cfg, seed=3)
    tr = model.generate_trace(class_idx=0, trace_id=0)
    times = [m.t for m in tr.messages]
    assert times == sorted(times), "timestamps must be non-decreasing"

    by_stage: dict[int, list] = {}
    for m in tr.messages:
        by_stage.setdefault(m.stage_idx, []).append(m)
    for _si, msgs in by_stage.items():
        steps = [m.step_type for m in msgs]
        assert steps[0] is StepType.DISCOVERY_QUERY
        assert StepType.REQUEST in steps
        assert steps[-1] is StepType.RESPONSE
        # no UPDATE may appear after the final RESPONSE
        last_resp = max(i for i, s in enumerate(steps) if s is StepType.RESPONSE)
        assert all(s is not StepType.UPDATE for s in steps[last_resp + 1 :])


def test_obs_descriptor_only_labels_on_discovery() -> None:
    # The semantic capability label is observable (label_visible) only on discovery
    # queries -- the channel the registry adversary uses.
    model = WorkflowModel(_small_cfg(), seed=4)
    tr = model.generate_trace(0, 0)
    for m in tr.messages:
        if m.label_visible:
            assert m.step_type is StepType.DISCOVERY_QUERY
        assert m.direction in (Direction.C2S, Direction.S2C)
        assert m.length >= 1


def test_prefix_is_a_leading_subtrace() -> None:
    model = WorkflowModel(_small_cfg(), seed=5)
    tr = model.generate_trace(0, 0)
    half = tr.prefix(0.5)
    assert half.n_stages <= tr.n_stages
    assert len(half.messages) <= len(tr.messages)
    # prefix keeps a leading run of stages
    if half.messages:
        assert max(m.stage_idx for m in half.messages) <= max(m.stage_idx for m in tr.messages)
    assert tr.prefix(1.0) is tr


def test_reproducible_under_seed() -> None:
    a, _ = generate_dataset(_small_cfg(), seed=7)
    b, _ = generate_dataset(_small_cfg(), seed=7)
    assert len(a) == len(b)
    # same client + class + message count sequence => deterministic
    sig_a = [(t.task_class, t.client_id, len(t.messages)) for t in a]
    sig_b = [(t.task_class, t.client_id, len(t.messages)) for t in b]
    assert sig_a == sig_b


def test_multiskill_providers_break_id_capability_bijection() -> None:
    # At least one provider agent must serve more than one capability, so an opaque
    # endpoint id is not a 1:1 stand-in for a capability label.
    model = WorkflowModel(_small_cfg(multiskill_prob=0.9), seed=8)
    skills: dict[str, set[str]] = {}
    for cap, provs in model.population.providers.items():
        for p in provs:
            skills.setdefault(p, set()).add(cap)
    assert any(len(caps) > 1 for caps in skills.values())


def test_rng_is_isolated_to_model() -> None:
    # Global numpy RNG state must not affect generation (we use a private Generator).
    np.random.seed(123)
    a, _ = generate_dataset(_small_cfg(), seed=11)
    np.random.seed(999)
    b, _ = generate_dataset(_small_cfg(), seed=11)
    assert [t.task_class for t in a] == [t.task_class for t in b]
