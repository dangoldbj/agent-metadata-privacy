"""Sensitivity analysis: are the three headline results *structural*?

We sweep the generator's main knobs (one at a time, around the default) and check
that the qualitative story holds everywhere, rather than at one lucky setting:

* the **label-blind network** observer recovers task class well above chance
  (leakage exists),
* it does so from only a short **prefix** of the workflow (prospectivity),
* and the two wire **protections together** collapse that recovery sharply
  (the defense works).

This is the mathematician's robustness check: we claim the *structure*, not the
exact number. To keep the grid fast we use a lighter evaluation config; the
qualitative conclusions are what matter.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .config import DEFAULT, EvalConfig, ExperimentConfig
from .evaluate import evaluate_views
from .protection import both
from .workflows import generate_dataset

# One-at-a-time sweep values for each [sensitivity] knob.
GRID: dict[str, list] = {
    "n_classes": [4, 6, 8, 12, 16],
    "overlap": [0.2, 0.35, 0.5, 0.65, 0.8],
    "timing_noise": [0.2, 0.45, 0.7, 1.0],
}

# A lighter, faster evaluation for the grid; the default run reports the precise CIs.
_LIGHT_EVAL = EvalConfig(cv_folds=3, bootstrap_ci=200)
_LIGHT_TRACES_PER_CLASS = 150
_PREFIX = 0.2  # the "short opening" point for the prospectivity check


def _light_cfg(**generator_overrides) -> ExperimentConfig:
    base = replace(
        DEFAULT.generator,
        n_traces_per_class=_LIGHT_TRACES_PER_CLASS,
        **generator_overrides,
    )
    return ExperimentConfig(seed=DEFAULT.seed, generator=base, eval=_LIGHT_EVAL)


def _acc(df: pd.DataFrame) -> float:
    return float(df.iloc[0]["accuracy"])


def headline_metrics(cfg: ExperimentConfig) -> dict[str, float]:
    """The three headline numbers for one configuration."""
    traces, model = generate_dataset(cfg.generator, cfg.seed)
    chance = 1.0 / cfg.generator.n_classes

    registry_none = _acc(evaluate_views(traces, model, cfg, views=("registry",)))
    network_none = _acc(evaluate_views(traces, model, cfg, views=("network",)))
    network_both = _acc(
        evaluate_views(traces, model, cfg, views=("network",), transform=both,
                       protection_label="both")
    )
    prefixed = [t.prefix(_PREFIX) for t in traces]
    network_prefix = _acc(evaluate_views(prefixed, model, cfg, views=("network",)))

    return {
        "chance": chance,
        "registry_none": registry_none,
        "network_none": network_none,
        "network_prefix": network_prefix,
        "network_both": network_both,
        "network_none_lift": network_none / chance,
        "network_prefix_lift": network_prefix / chance,
        "network_both_lift": network_both / chance,
    }


def run_sensitivity(grid: dict[str, list] | None = None) -> pd.DataFrame:
    """Sweep each knob one at a time; return a tidy table of headline metrics."""
    grid = grid or GRID
    rows = []
    for knob, values in grid.items():
        for v in values:
            cfg = _light_cfg(**{knob: v})
            row = {"knob": knob, "value": v}
            row.update(headline_metrics(cfg))
            rows.append(row)
    return pd.DataFrame(rows)
