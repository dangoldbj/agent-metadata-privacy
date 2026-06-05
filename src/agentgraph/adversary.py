"""Adversary vantage points: feature extractors over a trace's ``obs(m)``.

Each view corresponds to an adversary class in the paper's Table 1 and projects a
trace onto exactly what that vantage point can see:

* ``registry``  -- the registry/discovery adversary G. Sees the *semantic
  capability labels* named in discovery queries. This is the **semanticity**
  channel; it is available even under fresh transport identifiers, because the
  query names the capability (only *discovery privacy*, a separate bootstrap
  property, removes it).
* ``network``   -- the passive network observer N (and on-path relay R). Sees only
  ``obs(m) = (src, dst, t, length, direction)`` with *opaque* endpoint ids: a
  persistent-id "bag of endpoints" channel, plus timing, volume, and
  direction-sequence structure. **No semantic labels.** This is the strong claim:
  a label-blind observer still recovers task class.
* ``combined``  -- a colluding adversary with both; an upper bound.

The feature *vocabulary* (capabilities, endpoint tokens, n-gram slots) is fixed up
front from the known population. Consequently, when a §5 protection rewrites
identifiers to fresh tokens, those tokens are out-of-vocabulary and contribute
nothing -- which is exactly how unlinkability is meant to close the persistent-id
channel. Structural/timing/volume features never reference identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import Direction, Trace
from .workflows import REGISTRY_ID, WorkflowModel

_DIR_IDX = {Direction.C2S: 0, Direction.S2C: 1}

# Fixed scalar feature names for the network view (all derived from obs(m) only,
# never from latent stage/step/capability fields).
_NETWORK_SCALARS: tuple[str, ...] = (
    "n_messages",
    "n_c2s",
    "n_s2c",
    "frac_c2s",
    "n_distinct_endpoints",
    "n_distinct_providers",
    "max_fanin_to_client",
    "duration",
    "msgs_per_sec",
    "iat_mean",
    "iat_std",
    "iat_p10",
    "iat_median",
    "iat_p90",
    "len_total",
    "len_mean",
    "len_std",
    "len_p10",
    "len_median",
    "len_p90",
    "bytes_c2s",
    "bytes_s2c",
    "byte_ratio_c2s",
    "max_s2c_burst",
)


@dataclass
class FeatureSpace:
    """Fixed feature vocabulary shared across every trace and every protection."""

    capabilities: list[str]
    endpoints: list[str]  # known provider + client + registry tokens
    cap_index: dict[str, int]
    endpoint_index: dict[str, int]

    @property
    def n_cap(self) -> int:
        return len(self.capabilities)

    def registry_dim(self) -> int:
        return self.n_cap + self.n_cap * self.n_cap  # unigram + bigram

    def network_dim(self) -> int:
        return len(self.endpoints) + 4 + 8 + len(_NETWORK_SCALARS)  # bag + dir 2/3-grams + scalars


def build_feature_space(model: WorkflowModel) -> FeatureSpace:
    """Fix the vocabulary from the known agent population."""
    caps = list(model.population.capabilities)
    endpoints: list[str] = []
    seen: set[str] = set()
    for token in (
        list(model.population.clients)
        + [p for provs in model.population.providers.values() for p in provs]
        + [REGISTRY_ID]
    ):
        if token not in seen:
            seen.add(token)
            endpoints.append(token)
    return FeatureSpace(
        capabilities=caps,
        endpoints=endpoints,
        cap_index={c: i for i, c in enumerate(caps)},
        endpoint_index={e: i for i, e in enumerate(endpoints)},
    )


def _registry_features(trace: Trace, fs: FeatureSpace, out: np.ndarray, base: int) -> None:
    """Capability unigram + ordered-bigram histogram over discovery queries."""
    n = fs.n_cap
    queried: list[int] = []
    for m in trace.messages:
        if m.label_visible and m.capability is not None:
            idx = fs.cap_index.get(m.capability)
            if idx is not None:
                queried.append(idx)
    for idx in queried:
        out[base + idx] += 1.0
    bigram_base = base + n
    for a, b in zip(queried, queried[1:]):
        out[bigram_base + a * n + b] += 1.0


def _network_features(trace: Trace, fs: FeatureSpace, out: np.ndarray, base: int) -> None:
    msgs = trace.messages
    n_endpoints = len(fs.endpoints)

    # 1) persistent-id "bag of endpoints": counts of known tokens as src/dst.
    #    Fresh (protected) tokens are out-of-vocabulary and silently dropped.
    for m in msgs:
        for token in (m.src, m.dst):
            j = fs.endpoint_index.get(token)
            if j is not None:
                out[base + j] += 1.0

    # 2) direction bigrams (4) and trigrams (8)
    bigram_base = base + n_endpoints
    trigram_base = bigram_base + 4
    dseq = [_DIR_IDX[m.direction] for m in msgs]
    for a, b in zip(dseq, dseq[1:]):
        out[bigram_base + a * 2 + b] += 1.0
    for a, b, c in zip(dseq, dseq[1:], dseq[2:]):
        out[trigram_base + a * 4 + b * 2 + c] += 1.0

    # 3) scalar structural / timing / volume features (obs(m) only)
    s = trigram_base + 8
    sc = {name: 0.0 for name in _NETWORK_SCALARS}
    n_msg = len(msgs)
    sc["n_messages"] = float(n_msg)
    if n_msg:
        times = np.array([m.t for m in msgs], dtype=float)
        times.sort()
        lengths = np.array([m.length for m in msgs], dtype=float)
        is_c2s = np.array([m.direction is Direction.C2S for m in msgs])
        n_c2s = float(is_c2s.sum())
        sc["n_c2s"] = n_c2s
        sc["n_s2c"] = float(n_msg - n_c2s)
        sc["frac_c2s"] = n_c2s / n_msg

        endpoints = {t for m in msgs for t in (m.src, m.dst) if t != REGISTRY_ID}
        providers = {t for m in msgs for t in (m.src, m.dst) if t.startswith("agent-")}
        sc["n_distinct_endpoints"] = float(len(endpoints))
        sc["n_distinct_providers"] = float(len(providers))
        # heaviest in-flow to a single client endpoint (burstiness of a delegation)
        fanin: dict[str, int] = {}
        for m in msgs:
            if m.direction is Direction.S2C:
                fanin[m.dst] = fanin.get(m.dst, 0) + 1
        sc["max_fanin_to_client"] = float(max(fanin.values()) if fanin else 0)

        duration = float(times[-1] - times[0])
        sc["duration"] = duration
        sc["msgs_per_sec"] = n_msg / duration if duration > 0 else 0.0
        if n_msg > 1:
            iat = np.diff(times)
            sc["iat_mean"] = float(iat.mean())
            sc["iat_std"] = float(iat.std())
            sc["iat_p10"] = float(np.percentile(iat, 10))
            sc["iat_median"] = float(np.median(iat))
            sc["iat_p90"] = float(np.percentile(iat, 90))

        sc["len_total"] = float(lengths.sum())
        sc["len_mean"] = float(lengths.mean())
        sc["len_std"] = float(lengths.std())
        sc["len_p10"] = float(np.percentile(lengths, 10))
        sc["len_median"] = float(np.median(lengths))
        sc["len_p90"] = float(np.percentile(lengths, 90))
        bytes_c2s = float(lengths[is_c2s].sum())
        bytes_s2c = float(lengths[~is_c2s].sum())
        sc["bytes_c2s"] = bytes_c2s
        sc["bytes_s2c"] = bytes_s2c
        sc["byte_ratio_c2s"] = bytes_c2s / (bytes_c2s + bytes_s2c) if (bytes_c2s + bytes_s2c) else 0.0
        # longest consecutive run of server->client messages (a streamed update burst)
        burst = best = 0
        for m in msgs:
            burst = burst + 1 if m.direction is Direction.S2C else 0
            best = max(best, burst)
        sc["max_s2c_burst"] = float(best)

    for k, name in enumerate(_NETWORK_SCALARS):
        out[s + k] = sc[name]


def extract_features(
    traces: list[Trace], view: str, fs: FeatureSpace
) -> np.ndarray:
    """Return the feature matrix ``X`` for ``view`` over ``traces``."""
    if view == "registry":
        dim = fs.registry_dim()
    elif view == "network":
        dim = fs.network_dim()
    elif view == "combined":
        dim = fs.registry_dim() + fs.network_dim()
    else:
        raise ValueError(f"unknown view: {view!r}")

    X = np.zeros((len(traces), dim), dtype=np.float32)
    for i, tr in enumerate(traces):
        if view == "registry":
            _registry_features(tr, fs, X[i], 0)
        elif view == "network":
            _network_features(tr, fs, X[i], 0)
        else:  # combined
            _registry_features(tr, fs, X[i], 0)
            _network_features(tr, fs, X[i], fs.registry_dim())
    return X


def labels_array(traces: list[Trace]) -> np.ndarray:
    return np.array([t.task_class for t in traces])
