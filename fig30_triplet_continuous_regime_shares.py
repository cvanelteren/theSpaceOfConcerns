from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy.stats import spearmanr

ACTOR_SUMMARY_FP = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")
TRIPLET_FP = Path("output/fig28_regime_triplet_coverage_triplets.csv")

OUT_DATA = Path("output/fig30_triplet_continuous_regime_shares.csv")
OUT_SUMMARY = Path("output/fig30_triplet_continuous_regime_shares_summary.json")
OUT_PDF = Path("figures/fig30_triplet_continuous_regime_shares.pdf")
OUT_PNG = Path("figures/fig30_triplet_continuous_regime_shares.png")


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_center = x - x.mean()
    denom = float(np.dot(x_center, x_center))
    if denom == 0:
        return 0.0, float(np.mean(y))
    slope = float(np.dot(x_center, y - y.mean()) / denom)
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def main() -> None:
    actor_summary = pd.read_csv(ACTOR_SUMMARY_FP)
    actor_summary = actor_summary.set_index("actor")
    triplets = pd.read_csv(TRIPLET_FP)

    for region in (1, 2, 3):
        vals = []
        for row in triplets.itertuples(index=False):
            shares = actor_summary.loc[
                [row.actor_a, row.actor_b, row.actor_c],
                f"region_{region}_share",
            ].to_numpy(dtype=float)
            vals.append(float(np.mean(shares)))
        triplets[f"triplet_region_{region}_share_mean"] = vals

    # Focus the plot on the broad-middle-regime hypothesis, but keep all share summaries in the csv.
    x = triplets["triplet_region_2_share_mean"].to_numpy(dtype=float)
    y_cov = triplets["coverage_efficiency_resid"].to_numpy(dtype=float)
    y_ov = triplets["mean_pairwise_overlap_resid"].to_numpy(dtype=float)

    rho_cov = spearmanr(x, y_cov, nan_policy="omit")
    rho_ov = spearmanr(x, y_ov, nan_policy="omit")
    slope_cov, intercept_cov = fit_line(x, y_cov)
    slope_ov, intercept_ov = fit_line(x, y_ov)

    triplets.to_csv(OUT_DATA, index=False)
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "n_triplets": int(len(triplets)),
                "spearman_region2_vs_coverage_resid": {
                    "rho": float(rho_cov.statistic),
                    "pvalue": float(rho_cov.pvalue),
                },
                "spearman_region2_vs_overlap_resid": {
                    "rho": float(rho_ov.statistic),
                    "pvalue": float(rho_ov.pvalue),
                },
                "linear_region2_vs_coverage_resid": {
                    "slope": slope_cov,
                    "intercept": intercept_cov,
                },
                "linear_region2_vs_overlap_resid": {
                    "slope": slope_ov,
                    "intercept": intercept_ov,
                },
                "note": (
                    "Triplet residuals come from fig28. "
                    "The x-axis is mean Regime 2 share across the three actors in each triplet."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    color_map = {
        "1-1-1": "#7f7f7f",
        "1-1-2": "#4c78a8",
        "1-1-3": "#2a9d8f",
        "1-2-2": "#3a86c8",
        "1-2-3": "#1b9e77",
        "1-3-3": "#55a630",
        "2-2-2": "#f28e2b",
        "2-2-3": "#c9a227",
        "2-3-3": "#e15759",
        "3-3-3": "#8f63b8",
    }
    colors = [color_map.get(code, "gray6") for code in triplets["regime_triplet"]]

    fig, axs = uplt.subplots(ncols=2, refwidth=3.5, refaspect=1.0, share=False)
    axs.format(abc="[A]", grid=False)

    panels = [
        (
            axs[0],
            y_cov,
            slope_cov,
            intercept_cov,
            "Coverage vs size-matched null",
            float(rho_cov.statistic),
            float(rho_cov.pvalue),
        ),
        (
            axs[1],
            y_ov,
            slope_ov,
            intercept_ov,
            "Overlap vs size-matched null",
            float(rho_ov.statistic),
            float(rho_ov.pvalue),
        ),
    ]
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 150)
    for ax, y, slope, intercept, ylabel, rho, pval in panels:
        ax.scatter(
            x,
            y,
            c=colors,
            s=10,
            alpha=0.23,
            edgecolor="none",
            zorder=2,
        )
        ax.plot(x_grid, intercept + slope * x_grid, color="black", lw=1.3, zorder=3)
        ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.6, zorder=1)
        ax.format(
            xlabel="Mean Regime 2 share in triplet",
            ylabel=ylabel,
            xlim=(0.0, 1.0),
        )
        ax.text(
            0.03,
            0.97,
            f"Spearman rho = {rho:.2f}\np = {pval:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "pad": 2.0},
            zorder=4,
        )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=220)


if __name__ == "__main__":
    main()
