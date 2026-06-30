"""Slide-friendly version of Figure S21.

Country-pair overlap vs best-match phi proximity, formatted for presentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
import ultraplot as uplt

PAIR_DATA_FP = Path("output/fig27_pairwise_portfolio_complementarity_pairs.csv")
PAIR_SUMMARY_FP = Path("output/fig27_pairwise_portfolio_complementarity_summary.json")

OUT_PDF = Path("figures/figS21_country_overlap_vs_phi_slides.pdf")
OUT_PNG = Path("figures/figS21_country_overlap_vs_phi_slides.png")
OUT_BLANK_PDF = Path("figures/figS21_country_overlap_vs_phi_slides_blank.pdf")
OUT_BLANK_PNG = Path("figures/figS21_country_overlap_vs_phi_slides_blank.png")

# Label a handful of notable complementary pairs
HIGHLIGHT_PAIRS = {
    "United States - Norway",
    "Australia - Switzerland",
    "United States - Estonia",
    "New Zealand - Bulgaria",
    "Norway - Malaysia",
}


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


def _centered_limits(
    values: np.ndarray, center: float, pad_frac: float = 0.08
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    span = max(center - float(np.nanmin(values)), float(np.nanmax(values)) - center)
    span *= 1.0 + float(pad_frac)
    return center - span, center + span


def _format_ax(ax, xlim, ylim, x_med, y_med) -> None:
    ax.axvline(x_med, color="black", lw=1.2, ls="--", alpha=0.55, zorder=1)
    ax.axhline(y_med, color="black", lw=1.2, ls="--", alpha=0.55, zorder=1)
    ax.format(
        xlabel="Portfolio overlap between country pairs\n(Jaccard similarity on specialised topics)",
        ylabel="Proximity of non-overlapping topics\nin the concern space ($\\phi$)",
        xlim=xlim,
        ylim=ylim,
        fontsize=10,
        ticklabelsize=11,
        labelsize=10,
    )
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.grid(alpha=0.12, color="black")


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
    share_comp = float(np.mean(complementary))

    xlim = _centered_limits(x, x_med, pad_frac=0.12)
    ylim = _centered_limits(y, y_med, pad_frac=0.22)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # --- Frame 1: blank (all points grey, no annotations) ---
    fig, ax = uplt.subplots()
    ax.scatter(x, y, s=20, c="gray5", alpha=0.35, edgecolor="k", zorder=2)
    _format_ax(ax, xlim, ylim, x_med, y_med)
    fig.savefig(OUT_BLANK_PDF, bbox_inches="tight")
    fig.savefig(OUT_BLANK_PNG, dpi=180, bbox_inches="tight", transparent=True)
    print(f"Wrote {OUT_BLANK_PDF}")
    print(f"Wrote {OUT_BLANK_PNG}")

    # --- Frame 2: full reveal (complementary highlighted + annotations) ---
    fig, ax = uplt.subplots()
    ax.scatter(
        x[~complementary], y[~complementary],
        s=20, c="gray5", alpha=0.35, edgecolor="k", zorder=2,
    )
    ax.scatter(
        x[complementary], y[complementary],
        s=20, c="teal6", alpha=0.85, edgecolor="k", zorder=3,
    )
    _format_ax(ax, xlim, ylim, x_med, y_med)

    for _, row in pair_df[complementary].iterrows():
        if row["pair_label"] in HIGHLIGHT_PAIRS:
            ax.annotate(
                row["pair_label"],
                xy=(row["overlap_jaccard_rpa"], row["exclusive_phi_proximity"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8.5,
                color="teal9",
                zorder=6,
            )

    txt = ax.text(
        xlim[0] + 0.03 * (xlim[1] - xlim[0]),
        y_med + 0.06 + 0.04 * (ylim[1] - ylim[0]),
        f"Complementary pairs\n({share_comp:.0%} of all pairs)",
        color="teal8",
        fontsize=13,
        weight="bold",
        ha="left",
        va="bottom",
        zorder=5,
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground="white")])

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", transparent=True)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
