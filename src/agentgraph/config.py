"""Central, seeded configuration for the graph-inference experiment.

Every knob the experiment depends on lives here, so that a single seed plus this
config deterministically reproduces every number and figure. The defaults define
the headline run; the sensitivity grid (``sensitivity.py``) sweeps the marked
``# [sensitivity]`` knobs to show the result is *structural*, not parameter-tuned.

The experiment backs the paper's §6 claim (``sec:different``) that agent
communication-graph metadata leaks *pending workflow intent* along three axes:
semanticity, prospectivity, actuation. See the repo README and ``paper/main.tex``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters of the generative workflow model (``workflows.py``).

    A *task class* is a stochastic process over the A2A task lifecycle
    (discovery -> delegation -> tool calls -> updates -> completion). Classes are
    composed procedurally from a shared capability vocabulary so that ``n_classes``
    and ``overlap`` are true dials; capabilities are served by *multiple* provider
    agents and agents may be *multi-skill*, so an opaque transport id is never a
    relabeled capability (this keeps the network view distinct from the registry
    view).
    """

    # --- task classes ---------------------------------------------------------
    n_classes: int = 8  # [sensitivity] number of distinct task classes (workflows)
    workflow_len_min: int = 4  # min number of capability stages in a workflow
    workflow_len_max: int = 9  # max number of capability stages in a workflow

    # --- capability vocabulary / agent population -----------------------------
    n_capabilities: int = 16  # size of the shared capability taxonomy
    providers_per_capability: int = 3  # distinct provider agents offering each capability
    multiskill_prob: float = 0.35  # prob. a provider agent also serves a second capability
    n_clients: int = 12  # pool of orchestrator/client agents that run workflows

    # --- overlap between classes ----------------------------------------------
    # Fraction of each class's stages drawn from a *shared* common capability pool
    # (vs. class-distinctive capabilities). High overlap => the capability *set* is
    # not a giveaway and recovery must lean on sequence / timing / volume.
    overlap: float = 0.5  # [sensitivity] in [0, 1]
    common_pool_frac: float = 0.5  # fraction of the vocabulary that is "common" to all
    optional_stage_prob: float = 0.25  # prob. an optional stage is present in an instance

    # --- timing (inter-arrival seconds) ---------------------------------------
    # Timing/size signatures are attached to *capabilities* (shared across classes:
    # a `payments` call is heavy whoever invokes it), NOT to classes directly. So
    # class signal flows only through which capabilities are used and in what order
    # -- the paper's semanticity + sequence thesis -- and the network observer must
    # recover it indirectly (persistent-id fingerprints, capability-correlated
    # timing/volume), not from a baked-in per-class scale.
    base_interarrival: float = 0.25  # baseline mean gap between messages
    cap_time_spread: float = 0.5  # per-capability service-time variation (signal via caps)
    timing_noise: float = 0.45  # [sensitivity] lognormal noise on inter-arrival (obscures)

    # --- volume (message length in bytes) -------------------------------------
    base_request_bytes: int = 800
    base_update_bytes: int = 1500
    cap_size_spread: float = 0.6  # per-capability message-size variation (signal via caps)
    size_noise: float = 0.4  # lognormal noise on sizes (obscures)
    updates_lambda: float = 2.5  # mean number of streamed updates per delegation (Poisson)

    # --- dataset size ---------------------------------------------------------
    n_traces_per_class: int = 400  # balanced => chance accuracy = 1 / n_classes


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol (``evaluate.py``)."""

    cv_folds: int = 5
    bootstrap_ci: int = 1000  # bootstrap resamples for 95% CIs on accuracy
    # Prospectivity: evaluate task-class recovery from a leading prefix of the
    # workflow (fraction of stages observed). 1.0 = the full completed workflow.
    prefix_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    # Adversary vantage points (feature views) to evaluate; see ``adversary.py``.
    views: tuple[str, ...] = ("registry", "network", "combined")
    # §5 protections to sweep; see ``protection.py``. The first four are the wire
    # properties; discovery_privacy is the bootstrap property; "all" composes them.
    protections: tuple[str, ...] = (
        "none", "unlinkability", "metadata_min", "both", "discovery_privacy", "all",
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config: one seed, one generator config, one eval config."""

    seed: int = 20260603  # global seed; everything derives from it
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def with_generator(self, **overrides: object) -> "ExperimentConfig":
        """Return a copy with generator knobs overridden (used by the sensitivity grid)."""
        return replace(self, generator=replace(self.generator, **overrides))


DEFAULT = ExperimentConfig()
"""The headline configuration. ``experiments/run_all.py`` uses this."""
