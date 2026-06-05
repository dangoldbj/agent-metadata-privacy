"""Figures for the three results. Saved as both PDF (for the paper) and PNG.

All plots draw the chance baseline explicitly, since every claim is framed as a
lift over chance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_VIEW_ORDER = ["registry", "network", "combined"]
_VIEW_LABEL = {
    "registry": "registry\n(semantic label)",
    "network": "network\n(label-blind)",
    "combined": "combined",
}
_PROT_ORDER = ["none", "unlinkability", "metadata_min", "both", "discovery_privacy", "all"]
_PROT_LABEL = {
    "none": "none",
    "unlinkability": "unlink.",
    "metadata_min": "meta-min",
    "both": "both\n(wire)",
    "discovery_privacy": "disc.\nprivacy",
    "all": "all",
}


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def fig_leakage(df: pd.DataFrame, chance: float, out_dir: Path,
                show_title: bool = True) -> list[Path]:
    """Bar chart: task-class accuracy per adversary view, with chance line."""
    df = df.set_index("view").reindex(_VIEW_ORDER)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(df))
    err = np.vstack([df["accuracy"] - df["acc_ci_lo"], df["acc_ci_hi"] - df["accuracy"]])
    ax.bar(x, df["accuracy"], yerr=err, capsize=4, color="#3b6ea5", width=0.6)
    ax.axhline(chance, ls="--", color="crimson", lw=1.4, label=f"chance = {chance:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([_VIEW_LABEL[v] for v in df.index], fontsize=9)
    ax.set_ylabel("task-class accuracy")
    ax.set_ylim(0, 1.05)
    if show_title:
        ax.set_title("Leakage: task class recovered from graph metadata")
    ax.legend(loc="lower right", fontsize=9)
    for xi, a in zip(x, df["accuracy"]):
        ax.text(xi, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    return _save(fig, out_dir, "fig_leakage")


def fig_prospectivity(df: pd.DataFrame, chance: float, out_dir: Path,
                      show_title: bool = True) -> list[Path]:
    """Line plot: accuracy vs observed prefix fraction (predictive leverage)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    markers = {"registry": "o", "network": "s", "combined": "^"}
    for view in _VIEW_ORDER:
        sub = df[df["view"] == view].sort_values("prefix")
        ax.plot(sub["prefix"], sub["accuracy"], marker=markers[view],
                label=_VIEW_LABEL[view].replace("\n", " "))
    ax.axhline(chance, ls="--", color="crimson", lw=1.4, label=f"chance = {chance:.3f}")
    ax.set_xlabel("fraction of workflow observed")
    ax.set_ylabel("task-class accuracy")
    ax.set_ylim(0, 1.05)
    if show_title:
        ax.set_title("Prospectivity: predicting the pending task from its opening")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out_dir, "fig_prospectivity")


def fig_protection(df: pd.DataFrame, chance: float, out_dir: Path,
                   show_title: bool = True) -> list[Path]:
    """Heatmap: accuracy for each (protection, view); shows property -> adversary."""
    piv = (
        df.pivot(index="protection", columns="view", values="accuracy")
        .reindex(index=_PROT_ORDER, columns=_VIEW_ORDER)
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=chance, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(_VIEW_ORDER)))
    ax.set_xticklabels([_VIEW_LABEL[v] for v in _VIEW_ORDER], fontsize=9)
    ax.set_yticks(range(len(_PROT_ORDER)))
    ax.set_yticklabels([_PROT_LABEL[p] for p in _PROT_ORDER], fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="black")
    if show_title:
        ax.set_title(f"Protection: accuracy by property (chance = {chance:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="accuracy")
    return _save(fig, out_dir, "fig_protection")


def fig_actuation(df: pd.DataFrame, out_dir: Path,
                  show_title: bool = True) -> list[Path]:
    """Bar chart: capture ratio per §5 property at the headline (early deadline,
    one-class budget), averaged over targets. The integrity analogue of
    ``fig_protection``: leverage is high under ``none`` and collapses only under the
    full property set. Chance (the blind baseline) is 0."""
    from .actuation import headline_table

    agg = headline_table(df).set_index("protection").reindex(_PROT_ORDER)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(agg))
    kappa = agg["capture_ratio"].to_numpy()
    err = np.vstack([
        np.clip(kappa - agg["cap_ci_lo"].to_numpy(), 0, None),
        np.clip(agg["cap_ci_hi"].to_numpy() - kappa, 0, None),
    ])
    ax.bar(x, kappa, yerr=err, capsize=4, color="#3b6ea5", width=0.6)
    ax.axhline(0.0, ls="--", color="crimson", lw=1.4, label="blind baseline (chance) = 0")
    ax.set_xticks(x)
    ax.set_xticklabels([_PROT_LABEL[p] for p in agg.index], fontsize=9)
    ax.set_ylabel("capture ratio  $\\kappa$")
    ax.set_ylim(-0.05, 1.12)
    if show_title:
        ax.set_title("Actuation: value of acting on the leak, by property")
    ax.legend(loc="upper left", fontsize=9)
    for xi, a in zip(x, kappa):
        # Tall bars get a white in-bar label (keeps clear of the legend); short bars
        # get a label just above.
        if a > 0.5:
            ax.text(xi, a - 0.07, f"{a:.2f}", ha="center", fontsize=9, color="white")
        else:
            ax.text(xi, a + 0.04, f"{a:.2f}", ha="center", fontsize=9)
    return _save(fig, out_dir, "fig_actuation")


def fig_actuation_separation(df: pd.DataFrame, out_dir: Path,
                             show_title: bool = True) -> list[Path]:
    """Two panels evidencing that actuation is the *product* of inference and budget
    -- a separate axis, zero on either edge.

    Left: capture ratio vs decision deadline (budget = one-class mass) for no
    protection, unlinkability alone, and the combined wire properties. Leverage
    tracks prospectivity (rising with how much is observed, substantial even early)
    and the wire set collapses it to the blind baseline -- the inference edge: kill
    the (usable) signal and leverage vanishes even though a residual labeling channel
    survives. Right: Value of Metadata vs budget (no protection, full deadline). VoM
    grows from the no-budget edge and peaks where the budget is scarce relative to the
    target set -- the capacity edge, and the value-of-information signature."""
    from .actuation import HEADLINE_BUDGET

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # Left panel: kappa vs deadline, by protection (at the one-class-mass budget).
    cellL = df[np.isclose(df["budget_frac"], HEADLINE_BUDGET)]
    markers = {"none": "o", "unlinkability": "s", "both": "^"}
    labels = {"none": "no protection", "unlinkability": "unlinkability only",
              "both": "both (wire)"}
    for prot in ("none", "unlinkability", "both"):
        sub = (cellL[cellL["protection"] == prot]
               .groupby("deadline_f", as_index=False)["capture_ratio"].mean()
               .sort_values("deadline_f"))
        axL.plot(sub["deadline_f"], sub["capture_ratio"], marker=markers[prot],
                 label=labels[prot])
    axL.axhline(0.0, ls="--", color="crimson", lw=1.4, label="blind baseline = 0")
    axL.set_xlabel("decision deadline (fraction observed)")
    axL.set_ylabel("capture ratio  $\\kappa$")
    axL.set_ylim(-0.1, 1.05)
    axL.legend(loc="center right", fontsize=8)
    if show_title:
        axL.set_title("Leverage tracks inference; the wire set collapses it")

    # Right panel: VoM vs budget (no protection, full deadline) -- the capacity edge.
    cellR = (df[(df["protection"] == "none") & np.isclose(df["deadline_f"], 1.0)]
             .groupby("budget_frac", as_index=False)["vom"].mean()
             .sort_values("budget_frac"))
    axR.plot(cellR["budget_frac"], cellR["vom"], marker="o", color="#3b6ea5")
    axR.set_xlabel("budget (fraction of workflows)")
    axR.set_ylabel("Value of Metadata (workflows)")
    axR.set_ylim(bottom=0)
    if show_title:
        axR.set_title("Leverage needs budget to spend")

    fig.tight_layout()
    return _save(fig, out_dir, "fig_actuation_separation")


def fig_anchor(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Grouped bars (log scale): real A2A capture vs generator on key metrics."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["real"], w, label="real A2A capture", color="#9a6fb0")
    ax.bar(x + w / 2, df["generator"], w, label="generator", color="#3b6ea5")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df["metric"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("value (log scale)")
    ax.set_title("Anchor: generator vs a real A2A lifecycle (same order of magnitude)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "fig_anchor")


def fig_sensitivity(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Per-knob panels: network leakage (none / prefix / both) as a knob varies."""
    knobs = list(dict.fromkeys(df["knob"]))
    fig, axes = plt.subplots(1, len(knobs), figsize=(4.0 * len(knobs), 3.4), squeeze=False)
    for ax, knob in zip(axes[0], knobs):
        sub = df[df["knob"] == knob].sort_values("value")
        ax.plot(sub["value"], sub["network_none"], marker="o", label="none")
        ax.plot(sub["value"], sub["network_prefix"], marker="s", label="prefix 0.2")
        ax.plot(sub["value"], sub["network_both"], marker="^", label="both (wire)")
        ax.plot(sub["value"], sub["chance"], ls="--", color="crimson", label="chance")
        ax.set_title(f"vs {knob}")
        ax.set_xlabel(knob)
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("network task-class accuracy")
    axes[0][-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Sensitivity: the network-observer story is structural")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_sensitivity")
