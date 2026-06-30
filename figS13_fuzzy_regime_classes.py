"""Supplementary Figure S13. Converts hard dominant regimes into fuzzy regime classes based on regime shares. Shows that actors mostly bridge adjacent regions rather than spanning non-adjacent ones directly."""

from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

ROOT = Path(__file__).resolve().parent
ACTOR_FP = ROOT / "output/fig45_portfolio_space_ridgelines_actor_summary.csv"
OUT_PDF = ROOT / "figures/figS13_fuzzy_regime_classes.pdf"
OUT_PNG = ROOT / "figures/figS13_fuzzy_regime_classes.png"
OUT_CSV = ROOT / "output/fig33_fuzzy_regime_classes.csv"
TAU = 0.25
ORDER = ["1", "1+2", "2", "2+3", "3", "1+2+3"]
COLORS = {
    "1": "#1f77b4",
    "1+2": "#5b9bd5",
    "2": "#6f9f44",
    "2+3": "#d98c3f",
    "3": "#d62728",
    "1+2+3": "#7f7f7f",
}


def assign_fuzzy_class(row, tau=TAU):
    active = [str(i) for i in (1, 2, 3) if row[f"region_{i}_share"] >= tau]
    if not active:
        active = [str(int(row["dominant_region"]))]
    return "+".join(active)


def main():
    df = pd.read_csv(ACTOR_FP).copy()
    df["fuzzy_class"] = df.apply(assign_fuzzy_class, axis=1)
    df = df[df["fuzzy_class"].isin(ORDER)].copy()
    df.to_csv(OUT_CSV, index=False)

    summary = (
        df.groupby("fuzzy_class")
        .agg(
            n_actors=("actor", "size"),
            mean_k_active=("k_active", "mean"),
            mean_dominant_share=("dominant_region_share", "mean"),
            mean_entropy=("entropy_norm_support_raw_rca", "mean"),
        )
        .reindex(ORDER)
        .reset_index()
    )

    colors = [COLORS[c] for c in summary["fuzzy_class"]]

    uplt.rc["grid"] = True
    fig, axs = uplt.subplots(ncols=2, figsize=(9.2, 4.0), share=0)

    ax = axs[0]
    x = np.arange(len(summary))
    xpad = 0.55
    xlim = (-xpad, len(summary) - 1 + xpad)

    ax.bar(x, summary["n_actors"], color=colors, edgecolor="black", linewidth=0.6)
    ax.format(
        title="How many actors span adjacent modes?",
        xlabel="Fuzzy mode class",
        ylabel="Actors",
        xticks=x,
        xticklabels=summary["fuzzy_class"].tolist(),
        xlim=xlim,
    )

    ax = axs[1]
    ax.scatter(
        x,
        summary["mean_k_active"],
        s=85,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for xi, yi in zip(x, summary["mean_k_active"]):
        ax.plot([xi, xi], [0, yi], color="0.75", lw=1.2, zorder=1)
    ax.format(
        title="Bridge classes are broader, but not fully diffuse",
        xlabel="Fuzzy mode class",
        ylabel="Mean active topics",
        xticks=x,
        xticklabels=summary["fuzzy_class"].tolist(),
        xlim=xlim,
        ylim=(0, max(24, float(summary["mean_k_active"].max()) + 2)),
    )

    fig.text(
        0.99,
        -0.001,
        r"Class membership if region share $\geq 0.25$; $1+3$ does not appear at this threshold",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.save(OUT_PDF)
    fig.save(OUT_PNG, transparent=False)


if __name__ == "__main__":
    main()
