"""Generative model of agent workflows and the communication graph they induce.

No public corpus of A2A traces exists, so we *generate* traces from a principled
stochastic model of the A2A task lifecycle (discovery -> delegation -> tool calls
-> streamed updates -> completion) and, separately, calibrate/validate it against a
small real capture (``anchor.py``). The model is built so that the experiment's
qualitative claims are *structural*:

* **Task classes** are stochastic processes over capability-typed stages. They are
  composed from a shared capability vocabulary with a tunable ``overlap`` so that
  the capability *set* alone is not a giveaway -- recovery must use sequence,
  timing, and volume.
* **Capabilities are served by multiple provider agents**, and agents may be
  multi-skill, so a transport-visible (opaque) endpoint id is *not* a relabeled
  capability. This keeps the label-blind network view honestly distinct from the
  registry (semantic-label) view, while still letting *persistent* ids leak class
  through repeated-endpoint patterns (paper §6) -- a channel that the unlinkability
  property later removes.
* **Per-class timing and size signatures** give orthogonal signal in the
  descriptors ``(t, length)`` that the network observer can exploit and that
  metadata minimization later removes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeneratorConfig
from .trace import Direction, Message, StepType, Trace

# A realistic capability vocabulary, deliberately weighted toward the
# cross-border-transaction example used in the paper (§6). The first
# ``n_capabilities`` entries are used; order is fixed for reproducibility.
CAPABILITY_VOCAB: tuple[str, ...] = (
    "sanctions-screening",
    "payments",
    "contract-review",
    "kyc",
    "fx-conversion",
    "credit-check",
    "pricing",
    "logistics",
    "inventory",
    "invoicing",
    "tax-calc",
    "compliance-audit",
    "translation",
    "summarization",
    "scheduling",
    "notification",
    "identity-proofing",
    "risk-scoring",
    "document-extraction",
    "settlement",
    "escrow",
    "dispute-resolution",
    "data-enrichment",
    "fraud-detection",
)

REGISTRY_ID = "registry"


@dataclass(frozen=True, slots=True)
class Stage:
    """One capability invocation in a workflow skeleton."""

    capability: str
    optional: bool


@dataclass(frozen=True, slots=True)
class TaskClass:
    """A workflow template: just an ordered skeleton of capability stages.

    No per-class timing/size signature is injected: all class signal flows through
    *which* capabilities are used and *in what order*. Timing and size profiles live
    on the capabilities (see ``CapabilityProfiles``) and are shared across classes.
    """

    name: str
    stages: tuple[Stage, ...]


@dataclass(frozen=True, slots=True)
class CapabilityProfiles:
    """Per-capability service-time and message-size profiles, shared by all classes."""

    time_scale: dict[str, float]  # multiplies baseline inter-arrival for this capability
    request_bytes: dict[str, float]  # mean request size for this capability
    update_bytes: dict[str, float]  # mean update/response size for this capability


class AgentPopulation:
    """Agents, the capabilities they provide, and the clients that orchestrate."""

    def __init__(self, cfg: GeneratorConfig, rng: np.random.Generator) -> None:
        if cfg.n_capabilities > len(CAPABILITY_VOCAB):
            raise ValueError(
                f"n_capabilities={cfg.n_capabilities} exceeds vocabulary "
                f"({len(CAPABILITY_VOCAB)})"
            )
        self.capabilities: list[str] = list(CAPABILITY_VOCAB[: cfg.n_capabilities])

        # Each capability is served by several provider agents; some providers are
        # multi-skill (also serve a second, random capability). Provider ids are
        # opaque tokens -- not the capability name.
        self.providers: dict[str, list[str]] = {c: [] for c in self.capabilities}
        pid = 0
        for cap in self.capabilities:
            for _ in range(cfg.providers_per_capability):
                token = f"agent-{pid:04d}"
                pid += 1
                self.providers[cap].append(token)
                if rng.random() < cfg.multiskill_prob:
                    other = self.capabilities[rng.integers(len(self.capabilities))]
                    self.providers[other].append(token)

        self.clients: list[str] = [f"client-{i:03d}" for i in range(cfg.n_clients)]
        self.registry: str = REGISTRY_ID


class WorkflowModel:
    """Builds the task classes and samples traces from them."""

    def __init__(self, cfg: GeneratorConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.population = AgentPopulation(cfg, self.rng)
        self.profiles = self._build_profiles()
        self.classes = self._build_classes()

    # -- capability profiles (shared across classes) ---------------------------
    def _build_profiles(self) -> CapabilityProfiles:
        cfg, rng = self.cfg, self.rng
        caps = self.population.capabilities
        return CapabilityProfiles(
            time_scale={c: float(rng.lognormal(0.0, cfg.cap_time_spread)) for c in caps},
            request_bytes={
                c: float(cfg.base_request_bytes * rng.lognormal(0.0, cfg.cap_size_spread))
                for c in caps
            },
            update_bytes={
                c: float(cfg.base_update_bytes * rng.lognormal(0.0, cfg.cap_size_spread))
                for c in caps
            },
        )

    # -- class construction ----------------------------------------------------
    def _build_classes(self) -> list[TaskClass]:
        cfg, rng = self.cfg, self.rng
        caps = self.population.capabilities
        n_common = max(1, round(cfg.common_pool_frac * len(caps)))
        common_pool = caps[:n_common]
        distinctive_pool = caps[n_common:] or caps  # fall back if no distinctive caps

        classes: list[TaskClass] = []
        for c in range(cfg.n_classes):
            length = int(rng.integers(cfg.workflow_len_min, cfg.workflow_len_max + 1))
            stages: list[Stage] = []
            for _ in range(length):
                # `overlap` controls how often a stage draws from the shared common
                # pool vs. the class-distinctive pool.
                if rng.random() < cfg.overlap:
                    cap = common_pool[rng.integers(len(common_pool))]
                else:
                    cap = distinctive_pool[rng.integers(len(distinctive_pool))]
                stages.append(Stage(cap, optional=bool(rng.random() < 0.30)))
            classes.append(TaskClass(f"class_{c}", tuple(stages)))
        return classes

    @property
    def class_names(self) -> list[str]:
        return [tc.name for tc in self.classes]

    # -- sampling --------------------------------------------------------------
    def _interarrival(self, cap: str) -> float:
        cfg = self.cfg
        scale = self.profiles.time_scale[cap]
        gap = cfg.base_interarrival * scale * self.rng.lognormal(0.0, cfg.timing_noise)
        return float(max(gap, 1e-4))

    def _size(self, mean_bytes: float) -> int:
        val = mean_bytes * self.rng.lognormal(0.0, self.cfg.size_noise)
        return int(max(round(val), 1))

    def generate_trace(self, class_idx: int, trace_id: int) -> Trace:
        cfg, rng = self.cfg, self.rng
        tc = self.classes[class_idx]
        client = self.population.clients[rng.integers(len(self.population.clients))]
        registry = self.population.registry

        msgs: list[Message] = []
        t = 0.0
        present_stage = 0
        for stage in tc.stages:
            if stage.optional and rng.random() >= cfg.optional_stage_prob:
                continue  # optional stage absent in this instance
            cap = stage.capability
            si = present_stage
            present_stage += 1
            req_bytes = self.profiles.request_bytes[cap]
            upd_bytes = self.profiles.update_bytes[cap]

            # discovery: client queries the registry, naming the capability
            t += self._interarrival(cap)
            msgs.append(
                Message(client, registry, t, self._size(120), Direction.C2S,
                        si, StepType.DISCOVERY_QUERY, cap, label_visible=True)
            )
            t += self._interarrival(cap)
            msgs.append(
                Message(registry, client, t, self._size(200), Direction.S2C,
                        si, StepType.DISCOVERY_RESULT, cap)
            )

            # delegation: client picks a provider for the capability and requests
            provs = self.population.providers[cap]
            provider = provs[rng.integers(len(provs))]
            t += self._interarrival(cap)
            msgs.append(
                Message(client, provider, t, self._size(req_bytes),
                        Direction.C2S, si, StepType.REQUEST, cap)
            )

            # streamed updates, then a final response
            n_updates = int(rng.poisson(cfg.updates_lambda))
            for _ in range(n_updates):
                t += self._interarrival(cap)
                msgs.append(
                    Message(provider, client, t, self._size(upd_bytes),
                            Direction.S2C, si, StepType.UPDATE, cap)
                )
            t += self._interarrival(cap)
            msgs.append(
                Message(provider, client, t, self._size(upd_bytes),
                        Direction.S2C, si, StepType.RESPONSE, cap)
            )

        return Trace(trace_id, tc.name, client, present_stage, tuple(msgs))

    def generate_dataset(self) -> list[Trace]:
        """Sample a balanced dataset: ``n_traces_per_class`` per class."""
        traces: list[Trace] = []
        tid = 0
        for class_idx in range(len(self.classes)):
            for _ in range(self.cfg.n_traces_per_class):
                traces.append(self.generate_trace(class_idx, tid))
                tid += 1
        return traces


def generate_dataset(cfg: GeneratorConfig, seed: int) -> tuple[list[Trace], WorkflowModel]:
    """Convenience: build a model and sample its dataset. Returns (traces, model)."""
    model = WorkflowModel(cfg, seed)
    return model.generate_dataset(), model
