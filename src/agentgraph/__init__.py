"""agentgraph -- empirical graph-inference experiment for the paper
*From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous
Agent Interoperability*.

It measures how much *pending workflow intent* an observer can recover from
agent communication-graph metadata alone (no payloads), how early in a workflow
(prospectivity), and how far the paper's §5 transport properties collapse that
leakage back to chance.
"""

from __future__ import annotations

from .config import DEFAULT, EvalConfig, ExperimentConfig, GeneratorConfig

__all__ = ["DEFAULT", "EvalConfig", "ExperimentConfig", "GeneratorConfig"]

__version__ = "0.1.0"
