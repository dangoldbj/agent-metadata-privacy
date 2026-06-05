"""Trace and message data model.

A *trace* is the set of messages exchanged to complete one workflow instance --
the A2A task lifecycle of one logical exchange (paper §3.1). Each message carries
the paper's transport-visible descriptor

    obs(m) = (src, dst, t, length, direction),

i.e. endpoint identifiers, timestamp, length, and direction -- and *excludes*
content (Assumption 1, content is encrypted). The latent annotations
(``stage_idx``, ``step_type``, ``capability``) are ground truth used only for
evaluation and to model what each *adversary vantage point* can see; an adversary
operating on transport metadata does not get them, except that a registry observer
sees the capability named in a discovery query (``label_visible``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    """The transport-visible direction ``d`` of a message."""

    C2S = "c2s"  # client -> server (a request or a discovery query)
    S2C = "s2c"  # server -> client (a response, update, or discovery result)


class StepType(str, Enum):
    """Latent role of a message in the workflow (ground truth, not in ``obs(m)``)."""

    DISCOVERY_QUERY = "discovery_query"
    DISCOVERY_RESULT = "discovery_result"
    REQUEST = "request"
    UPDATE = "update"
    RESPONSE = "response"


@dataclass(frozen=True, slots=True)
class Message:
    """One message and its transport-visible descriptor ``obs(m)``.

    The first five fields are exactly ``obs(m)``. The remaining fields are latent
    ground truth; ``capability`` is observable as a *semantic label* only when
    ``label_visible`` is true (discovery queries, which name the capability to the
    registry).
    """

    # --- obs(m) = (src, dst, t, length, direction) ----------------------------
    src: str
    dst: str
    t: float
    length: int
    direction: Direction
    # --- latent ground truth (not part of obs(m)) -----------------------------
    stage_idx: int
    step_type: StepType
    capability: str | None = None
    label_visible: bool = False


@dataclass(frozen=True, slots=True)
class Trace:
    """All messages of one workflow instance, plus ground-truth labels."""

    trace_id: int
    task_class: str
    client_id: str
    n_stages: int
    messages: tuple[Message, ...]

    @property
    def duration(self) -> float:
        if not self.messages:
            return 0.0
        return self.messages[-1].t - self.messages[0].t

    def prefix(self, frac: float) -> "Trace":
        """Return the sub-trace observable after the first ``frac`` of stages.

        Prospectivity: an observer who has seen only the opening of a workflow.
        ``frac`` in (0, 1]; we keep messages whose ``stage_idx`` is among the first
        ``ceil(frac * n_stages)`` *present* stages (some optional stages may be
        absent, so we cut on the observed stage ordinals rather than raw indices).
        """
        if frac >= 1.0:
            return self
        present = sorted({m.stage_idx for m in self.messages})
        if not present:
            return self
        keep_count = max(1, math.ceil(frac * len(present)))
        keep = set(present[:keep_count])
        kept = tuple(m for m in self.messages if m.stage_idx in keep)
        return Trace(
            trace_id=self.trace_id,
            task_class=self.task_class,
            client_id=self.client_id,
            n_stages=len(keep),
            messages=kept,
        )
