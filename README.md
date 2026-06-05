# Agent Metadata Privacy

**Secure private transport for AI-agent interoperability.**

Agent-interoperability protocols such as [A2A](https://a2a-protocol.org/) and
[MCP](https://modelcontextprotocol.io/) standardize *what* agents say to one
another, but assume address-based transport over HTTP(S). Such transports protect
message **content** — increasingly with end-to-end encryption — yet expose the
**communication graph**: which agent contacts which, when, and how often.

This work treats that graph as a first-class threat. In agent systems the graph is
more revealing than a privacy framing suggests: because agent endpoints are
capability-labeled, their workflows structured and chained, and their interactions
action-coupled, an observer of the graph can infer not just past relationships but
**pending workflows** — and, at machine speed, can act on that inference before the
workflow completes. The threat is one of **workflow integrity**, not only privacy.

## The paper

The core contribution is a threat model and privacy analysis, transport- and
protocol-agnostic:

> **From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous
> Agent Interoperability**

It contributes (1) a threat model for the agent-interop communication graph as a
metadata surface distinct from payload confidentiality; (2) an account of why agent
metadata is distinctively revealing (semanticity, prospectivity, actuation); (3) a
transport- and bootstrap-layer privacy-property framework and an evaluation of
candidate transports (SimpleX/SMP, Tor, mixnets) against it; and (4) an A2A case
study showing a metadata-protecting binding is expressible but surfaces the
protocol's implicit identity assumptions.

The source is in [`paper/`](paper/) (`main.tex`, `references.bib`, built `main.pdf`).

- **Preprint (arXiv):** *forthcoming.*
- **Reference implementation (A2A over a metadata-protecting transport):**
  *forthcoming.*

## Empirical evaluation

The paper argues that the communication graph leaks *pending workflow intent*
along three axes — semanticity, prospectivity, actuation. The `src/agentgraph/`
experiment turns that argument into a measurement. Over a generative model of A2A
task workflows (capabilities served by multiple agents, so transport-visible ids
are not relabeled capabilities), an observer tries to recover the latent **task
class** from communication-graph metadata alone — no payloads. We read a simple
classifier's above-chance accuracy as a *lower bound* on the leakage.

Four results, one harness (run `uv run experiments/run_all.py`):

1. **Leakage exists.** A **label-blind network observer** — seeing only opaque
   endpoint ids, timing, volume, and direction — recovers the task class far above
   chance, not only the registry observer who sees capability labels.
2. **Prospectivity.** From only the first ~10–20% of a workflow, the observer
   already predicts its class at several times chance: *predictive leverage* over a
   task before it completes.
3. **Protection collapses leakage.** Applying the paper's §5 properties as
   transforms, each wire property *alone* barely dents the network observer
   (persistent ids and the timing/volume fingerprint are independent channels);
   only **unlinkability and metadata minimization together** collapse recovery
   toward chance. The semantic-label channel falls only to **discovery privacy** —
   empirically separating the wire and bootstrap properties.
4. **Actuation: the value of acting on the leak.** Beyond *recoverability* (an
   information-theoretic quantity), we measure what the leak is *worth* to an
   adversary that must act under a budget (a decision-theoretic one). Deciding from
   only a workflow's opening, a budgeted adversary that ranks workflows by the
   label-blind posterior captures **~90%** of the advantage a clairvoyant attacker
   would have over a metadata-blind one (the **capture ratio** κ). This **Value of
   Metadata** is the *product* of inference and budget — it vanishes if either is
   absent, so actuation is a genuinely separate axis, not a restatement of leakage —
   and the same §5 properties that collapse inference drive κ to the blind baseline.

A sensitivity sweep shows these conclusions are *structural*, holding as the number
of task classes, the capability overlap, and the timing noise vary. The generator
is **anchored to a real A2A capture**: `scripts/capture_a2a.py` drives a real
`a2a-sdk` task lifecycle (discovery → message/send → streamed updates → completion)
over HTTP and records the wire messages, and the generator is validated to match
that lifecycle's structure and scale. See
[`results/summary.md`](results/summary.md) for the generated tables and figures.

The evaluation uses simulated workflows; magnitude is generator-dependent (hence
the sensitivity analysis — we claim the *structure*, not the number). It measures
passive inference and in-model budgeted actuation *leverage* (which workflows an
adversary would select), not live workflow manipulation.

## Status

This repository hosts the paper and the empirical evaluation behind it. A reference
binding (A2A over a metadata-protecting transport) and measurements of its cost are
the next step.

## Repository layout

```
paper/                  threat model + privacy analysis (LaTeX source and PDF)
src/agentgraph/         the graph-inference experiment (library)
experiments/run_all.py  reproducible driver: regenerates every result and figure
scripts/capture_a2a.py  captures a real a2a-sdk lifecycle (the generator's anchor)
results/                generated tables, figures, and summary
tests/                  sanity + protection-collapse + anchor verification
```

## Running the experiment

```
uv sync                            # set up the environment (Python 3.13)
uv run experiments/run_all.py      # regenerate all results, figures, and summary
uv run pytest                      # sanity + verification tests
```

## License

Code in this repository is released under the [MIT License](LICENSE). The paper
text and figures are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
