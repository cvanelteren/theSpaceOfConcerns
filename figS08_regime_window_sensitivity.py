"""Supplementary Figure S08. Compares regime persistence under annual and rolling five-year windows. Checks that the transition result is not a temporal-aggregation artifact."""

#!/usr/bin/env python3
from __future__ import annotations

# Plot 1y vs 5y regime-transition stability diagnostics for SI.

from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

OUT_PDF = Path("figures/figS08_regime_window_sensitivity.pdf")
OUT_PNG = Path("figures/figS08_regime_window_sensitivity.png")

MATRIX_PATHS = {
    1: Path("output/fig45_regime_transition_matrix_row_normalized_window1.csv"),
    5: Path("output/fig45_regime_transition_matrix_row_normalized_window5.csv"),
}
SUMMARY_PATHS = {
    1: Path("output/fig45_regime_transition_summary_window1.csv"),
    5: Path("output/fig45_regime_transition_summary_window5.csv"),
}


def _load_matrix(window_years: int) -> pd.DataFrame:
    path = MATRIX_PATHS[window_years]
    if not path.exists():
        raise FileNotFoundError(f"Missing matrix file: {path}")
    mat = pd.read_csv(path, index_col=0)
    mat.index = pd.to_numeric(mat.index, errors="coerce")
    mat.columns = pd.to_numeric(mat.columns, errors="coerce")
    mat = mat.reindex(index=[1, 2, 3], columns=[1, 2, 3])
    if mat.isna().all().all():
        raise RuntimeError(f"Matrix is empty after reindexing: {path}")
    return mat


def _load_summary(window_years: int) -> dict:
    path = SUMMARY_PATHS[window_years]
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Summary file is empty: {path}")
    return df.iloc[0].to_dict()


def _plot_transition_matrix(ax, mat: pd.DataFrame, title: str):
    vals = mat.to_numpy(dtype=float)
    img = ax.imshow(vals, vmin=0.0, vmax=1.0, cmap="fire")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            label = "NA" if not np.isfinite(v) else f"{v:.1%}"
            text_color = "white" if np.isfinite(v) and v >= 0.55 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=text_color)

    ax.format(
        xlabel="To regime",
        ylabel="From regime",
        title=title,
        xticks=[0, 1, 2],
        yticks=[0, 1, 2],
        xticklabels=["R1", "R2", "R3"],
        yticklabels=["R1", "R2", "R3"],
    )
    return img


def _plot_rate_comparison(ax, s1: dict, s5: dict):
    metric_keys = [
        "same_region_rate",
        "adjacent_or_same_rate",
        "far_jump_rate",
    ]
    metric_labels = ["Same regime", "Same/adjacent", "Far jump (1<->3)"]

    x = np.arange(len(metric_keys), dtype=float)
    width = 0.34
    y1 = np.asarray([float(s1[k]) for k in metric_keys], dtype=float)
    y5 = np.asarray([float(s5[k]) for k in metric_keys], dtype=float)

    bars1 = ax.bar(x - width / 2, y1, width=width, label="1-year window", color="gray6")
    bars5 = ax.bar(x + width / 2, y5, width=width, label="5-year window", color="teal7")

    label_offset = 0.018
    for bar in list(bars1) + list(bars5):
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + label_offset,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    n1 = int(s1.get("n_transitions", 0))
    n5 = int(s5.get("n_transitions", 0))
    max_rate = float(np.nanmax(np.r_[y1, y5])) if len(y1) else 1.0
    y_top = max(1.05, max_rate + 0.08)
    ax.format(
        xlabel="Transition statistic",
        ylabel="Rate",
        title=f"Window comparison (n1={n1}, n5={n5})",
        xticks=x,
        xticklabels=metric_labels,
        ylim=(0.0, y_top),
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(frameon=False, loc="r", ncols=1)


def main():
    mat1 = _load_matrix(1)
    mat5 = _load_matrix(5)
    s1 = _load_summary(1)
    s5 = _load_summary(5)

    fig, axs = uplt.subplots(ncols=3, share=0)
    axs.format(abc="[A]")
    ax1, ax2, ax3 = axs

    img = _plot_transition_matrix(ax1, mat1, "Regime transitions (1-year windows)")
    _plot_transition_matrix(ax2, mat5, "Regime transitions (5-year windows)")
    _plot_rate_comparison(ax3, s1, s5)

    cbar = ax2.colorbar(img, loc="r")
    cbar.set_label("Transition probability")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.12)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
