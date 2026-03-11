"""Supplementary Figure S21. Country-pair overlap versus best-match phi proximity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy.stats import pearsonr, spearmanr

PAIR_DATA_FP = Path("output/fig27_pairwise_portfolio_complementarity_pairs.csv")
PAIR_SUMMARY_FP = Path("output/fig27_pairwise_portfolio_complementarity_summary.json")

OUT_PDF = Path("figures/figS21_country_overlap_vs_phi.pdf")
OUT_PNG = Path("figures/figS21_country_overlap_vs_phi.png")
OUT_SUMMARY = Path("output/figS21_country_overlap_vs_phi_summary.json")


def _load_pair_df() -> tuple[pd.DataFrame, dict[str, object]]:
    if not PAIR_DATA_FP.exists():
        from fig27_pairwise_portfolio_complementarity import main as build_pair_data

        build_pair_data()

    pair_df = pd.read_csv(PAIR_DATA_FP)
    if pair_df.empty:
        raise RuntimeError("No pairwise complementarity data available.")

    summary: dict[str, object] = {}
    if PAIR_SUMMARY_FP.exists():
        summary = json.loads(PAIR_SUMMARY_FP.read_text(encoding="utf-8"))
    return pair_df, summary


def _quadrant_counts(
    x: np.ndarray, y: np.ndarray, x_med: float, y_med: float
) -> dict[str, int]:
    return {
        "low_overlap_high_phi": int(np.sum((x < x_med) & (y >= y_med))),
        "high_overlap_high_phi": int(np.sum((x >= x_med) & (y >= y_med))),
        "low_overlap_low_phi": int(np.sum((x < x_med) & (y < y_med))),
        "high_overlap_low_phi": int(np.sum((x >= x_med) & (y < y_med))),
    }


def _centered_limits(
    values: np.ndarray, center: float, pad_frac: float = 0.08
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    span = max(center - float(np.nanmin(values)), float(np.nanmax(values)) - center)
    span *= 1.0 + float(pad_frac)
    return center - span, center + span


def main() -> None:
    pair_df, prior_summary = _load_pair_df()
    pair_df = pair_df.dropna(
        subset=["overlap_jaccard_rpa", "exclusive_phi_proximity"]
    ).copy()

    x = pair_df["overlap_jaccard_rpa"].to_numpy(dtype=float)
    y = pair_df["exclusive_phi_proximity"].to_numpy(dtype=float)

    x_med = float(prior_summary.get("median_overlap_jaccard_rpa", np.median(x)))
    y_med = float(prior_summary.get("median_exclusive_phi_proximity", np.median(y)))

    complementary = (x < x_med) & (y >= y_med)
    sp = spearmanr(x, y)
    pr = pearsonr(x, y)
    qcounts = _quadrant_counts(x, y, x_med, y_med)

    summary = {
        "n_pairs": int(len(pair_df)),
        "median_overlap_jaccard_rpa": x_med,
        "median_exclusive_phi_proximity": y_med,
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "share_low_overlap_high_phi": float(np.mean(complementary)),
        "quadrant_counts": qcounts,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = uplt.subplots(refwidth=5.2, refaspect=0.8)

    ax.scatter(
        x[~complementary],
        y[~complementary],
        s=18,
        c="gray6",
        alpha=0.48,
        edgecolor="none",
        zorder=2,
    )
    ax.scatter(
        x[complementary],
        y[complementary],
        s=24,
        c="teal7",
        alpha=0.8,
        edgecolor="none",
        zorder=3,
    )

    ax.axvline(x_med, color="black", lw=1.0, ls="--", alpha=0.65, zorder=1)
    ax.axhline(y_med, color="black", lw=1.0, ls="--", alpha=0.65, zorder=1)

    xlim = _centered_limits(x, x_med)
    ylim = _centered_limits(y, y_med)

    ax.format(
        abc="[A]",
        xlabel=r"Country-country overlap (Jaccard on $RPA>1$ topics)",
        ylabel=r"Best-match $\phi$ of non-overlapping support",
        xlim=xlim,
        ylim=ylim,
    )
    ax.grid(alpha=0.16, color="black")

    ax.text(
        0.95,
        0.97,
        (
            f"n pairs = {len(pair_df)}\n"
            f"Pearson $r$ = {pr.statistic:.2f}\n"
            f"Spearman $\\rho$ = {sp.statistic:.2f}\n"
            f"Low-overlap / high-$\\phi$ = {np.mean(complementary):.1%}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
            "boxstyle": "round,pad=0.25",
        },
        zorder=5,
    )

    txt = ax.text(
        xlim[0] + 0.03 * (xlim[1] - xlim[0]),
        y_med + 0.03 * (ylim[1] - ylim[0]),
        "Low overlap,\nhigh $\\phi$",
        color="teal8",
        fontsize=12,
        weight="bold",
        ha="left",
        va="bottom",
        zorder=5,
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="k")])

    fig.format(
        suptitle="Country pairs with little direct overlap can still be close in the concern space"
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
