"""The paper's §5 transport properties, realized as transforms on observed traffic.

Each transform rewrites the adversary-visible ``obs(m)`` of a dataset to model what
a metadata-protecting binding would expose. We then re-measure leakage on the
transformed traffic: a property "works" to the extent it drives recovery toward
chance.

* ``unlinkability`` (Def. 1) -- every interaction uses *fresh* identifiers
  unlinkable to an agent's others. We assign a fresh pseudonym per (trace, stage,
  role), so no token recurs across stages or across traces. This closes the
  persistent-id "bag of endpoints" channel: the new tokens are out-of-vocabulary
  for the adversary's fixed feature space and contribute nothing. Within-stage
  grouping is retained (a single delegation is still one queue), matching SMP's
  unidirectional-queue model rather than overclaiming.

* ``metadata_min`` (Def. 4) -- the descriptors ``(t, length)`` are reduced so they
  do not distinguish interactions: lengths are padded to a constant and timestamps
  are re-emitted on a constant cadence (batched/mixed), erasing the volume and
  timing channels. Per the definition this targets ``(t, ℓ, d)`` only; it does
  *not* hide message counts, the direction sequence, or identifiers.

* ``both`` -- unlinkability ∘ metadata_min.

The mechanistic point of the experiment: each property alone leaves one channel
open (ids, or the capability-correlated timing/volume fingerprint), so leakage
persists; only the two together collapse it toward chance. This is the paper's
argument that the *set* of properties matters and partial measures are insufficient.
"""

from __future__ import annotations

from collections.abc import Callable

from .trace import Message, Trace
from .workflows import REGISTRY_ID

Transform = Callable[[list[Trace]], list[Trace]]

PAD_BYTES = 4096  # constant padded message length under metadata minimization
SLOT_SECONDS = 1.0  # constant inter-message cadence under metadata minimization


def identity(traces: list[Trace]) -> list[Trace]:
    return traces


def unlinkability(traces: list[Trace]) -> list[Trace]:
    """Fresh per-(trace, stage, role) pseudonyms; closes the persistent-id channel."""
    out: list[Trace] = []
    for tr in traces:
        client = tr.client_id
        new_msgs: list[Message] = []
        for m in tr.messages:
            tag = f"{tr.trace_id}-{m.stage_idx}"
            if m.src == REGISTRY_ID:
                src = f"r-{tag}"
            elif m.src == client:
                src = f"c-{tag}"
            else:
                src = f"p-{tag}"
            if m.dst == REGISTRY_ID:
                dst = f"r-{tag}"
            elif m.dst == client:
                dst = f"c-{tag}"
            else:
                dst = f"p-{tag}"
            new_msgs.append(
                Message(src, dst, m.t, m.length, m.direction, m.stage_idx,
                        m.step_type, m.capability, m.label_visible)
            )
        out.append(
            Trace(tr.trace_id, tr.task_class, f"c-{tr.trace_id}", tr.n_stages, tuple(new_msgs))
        )
    return out


def metadata_min(traces: list[Trace]) -> list[Trace]:
    """Pad lengths to a constant and re-emit timestamps on a constant cadence."""
    out: list[Trace] = []
    for tr in traces:
        new_msgs: list[Message] = []
        # batched/mixed delivery: messages leave on a fixed grid in their order,
        # so every inter-arrival is identical and timing carries no signal.
        for k, m in enumerate(tr.messages):
            new_msgs.append(
                Message(m.src, m.dst, k * SLOT_SECONDS, PAD_BYTES, m.direction,
                        m.stage_idx, m.step_type, m.capability, m.label_visible)
            )
        out.append(
            Trace(tr.trace_id, tr.task_class, tr.client_id, tr.n_stages, tuple(new_msgs))
        )
    return out


def both(traces: list[Trace]) -> list[Trace]:
    """Apply metadata minimization then unlinkability (the two wire properties)."""
    return unlinkability(metadata_min(traces))


def discovery_privacy(traces: list[Trace]) -> list[Trace]:
    """Hide the capability named at discovery (Def. 5), a bootstrap-layer property.

    The transport properties above do not touch the semantic-label channel that the
    registry observer uses; only discovery privacy does. We model it by removing the
    ``label_visible`` flag, so the registry view extracts nothing.
    """
    out: list[Trace] = []
    for tr in traces:
        new_msgs = [
            Message(m.src, m.dst, m.t, m.length, m.direction, m.stage_idx,
                    m.step_type, m.capability, label_visible=False)
            for m in tr.messages
        ]
        out.append(
            Trace(tr.trace_id, tr.task_class, tr.client_id, tr.n_stages, tuple(new_msgs))
        )
    return out


def all_properties(traces: list[Trace]) -> list[Trace]:
    """Every property: discovery privacy + metadata minimization + unlinkability."""
    return unlinkability(metadata_min(discovery_privacy(traces)))


PROTECTIONS: dict[str, Transform] = {
    "none": identity,
    "unlinkability": unlinkability,
    "metadata_min": metadata_min,
    "both": both,
    "discovery_privacy": discovery_privacy,
    "all": all_properties,
}


def get_protection(name: str) -> Transform:
    try:
        return PROTECTIONS[name]
    except KeyError:
        raise ValueError(f"unknown protection: {name!r}") from None
