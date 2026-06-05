"""Regenerate the paper's figures without embedded titles.

In the paper the LaTeX captions carry the description, so an embedded matplotlib
title would duplicate it. This reads the result CSVs from ``results/`` and writes
titleless PDFs into ``paper/figures/`` (the versions ``main.tex`` includes). The
``results/`` figures keep their titles for the README.

Run after ``experiments/run_all.py``:  uv run python scripts/paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agentgraph import figures
from agentgraph.config import DEFAULT

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"


def main() -> None:
    chance = 1.0 / DEFAULT.generator.n_classes
    leakage = pd.read_csv(RESULTS / "leakage.csv")
    prospect = pd.read_csv(RESULTS / "prospectivity.csv")
    protection = pd.read_csv(RESULTS / "protection.csv")
    actuation = pd.read_csv(RESULTS / "actuation.csv")

    figures.fig_leakage(leakage, chance, OUT, show_title=False)
    figures.fig_prospectivity(prospect, chance, OUT, show_title=False)
    figures.fig_protection(protection, chance, OUT, show_title=False)
    figures.fig_actuation(actuation, OUT, show_title=False)
    figures.fig_actuation_separation(actuation, OUT, show_title=False)

    # the paper includes only the PDFs; drop the PNG copies _save also emits
    for stem in ("fig_leakage", "fig_prospectivity", "fig_protection",
                 "fig_actuation", "fig_actuation_separation"):
        (OUT / f"{stem}.png").unlink(missing_ok=True)
    print(f"wrote titleless figures to {OUT}")


if __name__ == "__main__":
    main()
