"""Calibrate/validate the generator against a real A2A capture.

``scripts/capture_a2a.py`` records a real ``a2a-sdk`` task lifecycle. Here we
summarize it and compare the per-delegation structure -- request size, streamed
update size, updates per delegation, and update cadence -- against the generator.
The point is *structural validation*: the generator should reproduce the A2A
lifecycle shape (a request followed by several sub-second streamed updates, behind
a discovery step) and land within a sane order of magnitude on sizes and timing.

We are explicit that the captured agent is a trivial echo; its payloads are tiny,
whereas the generator deliberately models the richer payloads of real capability
work (contract review, settlement, ...). So we validate *shape and order of
magnitude*, not exact byte counts -- which is what calibration against a small
capture can honestly support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .trace import StepType
from .workflows import generate_dataset


def load_capture(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def real_summary(capture: dict) -> dict[str, float]:
    recs = capture["records"]
    posts = [r["bytes"] for r in recs if r["kind"] == "http_request" and r.get("method") == "POST"]
    gets = [r for r in recs if r["kind"] == "http_request" and r.get("method") == "GET"]
    by_task: dict[int, list[float]] = {}
    sizes: list[float] = []
    for r in recs:
        if r["kind"] == "stream_event":
            by_task.setdefault(r["task_no"], []).append(r["t"])
            sizes.append(r["bytes"])
    iats: list[float] = []
    for ts in by_task.values():
        ts = sorted(ts)
        iats += [b - a for a, b in zip(ts, ts[1:])]
    return {
        "discovery_per_run": float(len(gets)),
        "request_bytes": float(np.mean(posts)) if posts else float("nan"),
        "update_bytes": float(np.mean(sizes)) if sizes else float("nan"),
        "updates_per_delegation": float(np.mean([len(v) for v in by_task.values()])),
        "interarrival_s": float(np.mean(iats)) if iats else float("nan"),
    }


def generator_summary(cfg: ExperimentConfig, *, n_per_class: int = 50) -> dict[str, float]:
    light = cfg.with_generator(n_traces_per_class=n_per_class)
    traces, _ = generate_dataset(light.generator, light.seed)
    req_bytes: list[float] = []
    upd_bytes: list[float] = []
    updates_per_stage: list[int] = []
    iats: list[float] = []
    n_discovery = 0
    for tr in traces:
        # group messages by stage to recover per-delegation structure
        stages: dict[int, list] = {}
        for m in tr.messages:
            stages.setdefault(m.stage_idx, []).append(m)
        n_discovery += len(stages)  # one discovery per stage
        for msgs in stages.values():
            msgs = sorted(msgs, key=lambda m: m.t)
            n_upd = sum(1 for m in msgs if m.step_type is StepType.UPDATE)
            updates_per_stage.append(n_upd)
            for m in msgs:
                if m.step_type is StepType.REQUEST:
                    req_bytes.append(m.length)
                elif m.step_type in (StepType.UPDATE, StepType.RESPONSE):
                    upd_bytes.append(m.length)
            # cadence among the delegation messages (request -> updates -> response)
            deleg = [m for m in msgs if m.step_type in
                     (StepType.REQUEST, StepType.UPDATE, StepType.RESPONSE)]
            iats += [b.t - a.t for a, b in zip(deleg, deleg[1:])]
    return {
        "discovery_per_run": float(n_discovery / max(len(traces), 1)),
        "request_bytes": float(np.mean(req_bytes)),
        "update_bytes": float(np.mean(upd_bytes)),
        "updates_per_delegation": float(np.mean(updates_per_stage)),
        "interarrival_s": float(np.mean(iats)),
    }


def compare(real: dict[str, float], gen: dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric in ["request_bytes", "update_bytes", "updates_per_delegation", "interarrival_s"]:
        r, g = real[metric], gen[metric]
        ratio = g / r if r else float("nan")
        # structural match = same order of magnitude (within ~30x), i.e. |log10 ratio| < 1.5
        same_oom = bool(abs(np.log10(ratio)) < 1.5) if (r and g) else False
        rows.append({"metric": metric, "real": r, "generator": g,
                     "ratio_gen_over_real": ratio, "same_order_of_magnitude": same_oom})
    return pd.DataFrame(rows)


def run_anchor(
    cfg: ExperimentConfig, capture_path: str | Path, out_dir: str | Path
) -> pd.DataFrame:
    """Load the capture, compare to the generator, save and return the table."""
    capture = load_capture(capture_path)
    real = real_summary(capture)
    gen = generator_summary(cfg)
    df = compare(real, gen)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "anchor_comparison.csv", index=False)
    return df
