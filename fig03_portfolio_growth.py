"""Main-text Figure 3: how actors grow their portfolios in the space of concerns.

One claim, three panels, in the order the argument runs:
  A  new specialization concentrates near the existing portfolio (locality)
  B  once adopted, topics stay in the portfolio for years (persistence)
  C  only retain-and-adopt reproduces the observed local-entry rank (sufficiency)

The dominant-mode transition matrix moved to Figure 2, where it belongs with the
positioning claim. The retain-and-adopt schematic moved to Methods, next to the
model specification it illustrates. The breadth and popularity time series
remain in the appendix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ultraplot as uplt
from matplotlib.lines import Line2D

import fig03_local_portfolio_movement as f3
import fig04_split_support_validation as f4

OUT_PNG = Path("figures/fig03_portfolio_growth.png")
OUT_PDF = Path("figures/fig03_portfolio_growth.pdf")


def build_figure():
    # --- empirical panels (old Figure 3) ---------------------------------
    adoption_df, _ = f3._load_adoption_panel()
    hazard_meta = f3._load_json_or_empty(f3.HAZARD_META_PATH)

    adoption_df = adoption_df.copy()
    adoption_df["plot_distance"] = adoption_df["distance"].to_numpy(dtype=float)
    agg, event_rate = f3._binned_adoption_curve(adoption_df)

    hazard_time_unit = str(hazard_meta.get("time_unit", "year")).strip().lower()
    hazard_window_size = int(
        hazard_meta.get(
            "window_size", hazard_meta.get("window_years", f3.RETENTION_WINDOW_SIZE)
        )
    )
    retention_curves, retention_step_label = f3._compute_retention_curves(
        time_unit=hazard_time_unit, window_size=hazard_window_size
    )

    # --- transition matrix panel (moved from Figure 2) ---------------------
    regime_meta = f3._load_regime_transition_summary(f3.REGIME_SUMMARY_PATHS)
    regime_matrix, regime_matrix_source = f3._load_regime_transition_matrix(
        f3.REGIME_MATRIX_PATHS
    )
    regime_window_size = regime_meta.get(
        "window_size", regime_meta.get("window_years", np.nan)
    )
    regime_source = str(regime_meta.get("_source_path", regime_matrix_source))
    regime_time_unit = str(regime_meta.get("time_unit", "")).strip().lower()
    if regime_time_unit not in {"meeting", "year"}:
        regime_time_unit = "meeting" if "_meeting_" in regime_source else "year"

    # --- layout: square-ish 2×2 grid ------------------------------------
    fig, axs = uplt.subplots(nrows=2, ncols=2, share=0, figsize=(10.0, 9.8), wspace=4.0, hspace=4.0)
    ax_a, ax_b, ax_c, ax_d = axs

    # A: Transition matrix
    f3._plot_transition_matrix_panel(
        ax=ax_a, regime_transition_matrix=regime_matrix,
        window_size=regime_window_size, time_unit=regime_time_unit,
        regime_matrix_source=regime_matrix_source, xlabel_on_top=False,
    )
    ax_a.format(
        xticklabels=["Coord.", "Compliance", "Strategy"],
        yticklabels=["Coord.", "Compliance", "Strategy"],
    )
    # Larger labels, grid
    ax_a.xaxis.label.set_fontsize(10)
    ax_a.yaxis.label.set_fontsize(10)
    for label in ax_a.get_xticklabels() + ax_a.get_yticklabels():
        label.set_fontsize(9)
    ax_a.grid(True, alpha=0.1)

    # B: Local adoption
    f3._plot_adoption_panel(
        ax=ax_b, agg=agg, plot_distance=adoption_df["plot_distance"].to_numpy(float),
        event_rate=event_rate, show_legend=False,
    )
    ax_b.format(
        xlabel="Distance from existing portfolio",
        ylabel="Entry probability",
    )
    ax_b.xaxis.label.set_fontsize(10)
    ax_b.yaxis.label.set_fontsize(10)
    for label in ax_b.get_xticklabels() + ax_b.get_yticklabels():
        label.set_fontsize(9)
    ax_b.grid(True, alpha=0.1)

    # C: Retention persistence
    f3._plot_retention_panel(
        ax=ax_c, retention_curves=retention_curves, step_label=retention_step_label,
    )
    ax_c.format(
        xlabel=f"Time since entry ({retention_step_label})",
        ylabel="Prob. topic remains active",
    )
    ax_c.xaxis.label.set_fontsize(10)
    ax_c.yaxis.label.set_fontsize(10)
    for label in ax_c.get_xticklabels() + ax_c.get_yticklabels():
        label.set_fontsize(9)
    ax_c.grid(True, alpha=0.1)
    leg = ax_c.get_legend()
    if leg:
        leg.set_title("Gap tolerance")
        for text in leg.get_texts():
            text.set_fontsize(8)

    # D: Generative model sufficiency
    process_entry = f4._load_history(f4.PROCESS_ENTRY)
    f4.draw_entry_panel(ax_d, process_entry)

    # Use neutral comparison bars and reserve the accent for the mechanism
    comparison_colors = ["#B8BDC7", "#7B8794", "#009E73"]
    for patch, color in zip(ax_d.patches[:3], comparison_colors):
        patch.set_facecolor(color)

    ax_d.set_ylabel("Mean proximity rank")
    ax_d.yaxis.label.set_fontsize(10)
    for label in ax_d.get_xticklabels() + ax_d.get_yticklabels():
        label.set_fontsize(9)
    ax_d.grid(True, alpha=0.1, axis="y")
    
    ax_d.legend(
        handles=[
            Line2D([0], [0], color=f4.COLORS["observed"], lw=1.8),
            Line2D([0], [0], color=f4.COLORS["baseline"], lw=1.2, linestyle="--"),
        ],
        labels=["Observed", "Random"],
        loc="lower center", ncols=2, frame=False, fontsize=8,
    )

    # Panel letters placed manually
    for _ax, _lab in zip((ax_a, ax_b, ax_c, ax_d), ("A", "B", "C", "D")):
        _ax.text(
            0.025, 0.96, f"[{_lab}]", transform=_ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=11, zorder=10,
        )
    return fig


def main() -> None:
    fig = build_figure()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PNG, dpi=300)
    fig.save(OUT_PDF)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
