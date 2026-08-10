"""Main-text Figure 4: a evolving-portfolio rule is enough to reproduce the record.

  A  what the three candidate rules actually do to a portfolio between periods
  B  how far each one lands from the observed locality of real entries

The two panels sit side by side and share a vertical position: each bar in B is
drawn at the same height as the rule it scores in A, so the mapping from
mechanism to performance needs no reading of category labels. That alignment is
the point of the layout -- vertical bars under a stacked schematic forced the
reader to match "Fixed portfolio" in one panel to "Fixed portfolio" in another.

The figure used to open with two time series (breadth and topic popularity) and
carry the rules only as a caption description. That inverted the difficulty: a
reader cannot judge whether "fixed portfolio" undershooting breadth is damning
without first knowing what a support rule is. Those two series move to the
appendix, where they remain the evidence that the fit is not circular -- nothing
in the fitted rule sets breadth or popularity directly.

Panel B is a sufficiency check, not an independent test of locality: the
evolving-portfolio entry stage contains a phi-proximity term, so matching the
observed entry rank is the model doing what it was built to do. What the panel
adds is the contrast -- full reallocation overshoots and fixed portfolio
undershoots -- so the observed value is not trivially reachable by any rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ultraplot as uplt
from matplotlib.patches import Rectangle

import fig04_split_support_validation as f4
import figstyle

FOCUS_MODEL = "split_support"
# Order must match f4.ROW_Y, i.e. the row order drawn inside the schematic.
ROW_MODELS = ["one_stage", "two_stage", "split_support"]

X_MIN, X_MAX = 0.45, 0.76

OUT_PNG = Path("figures/fig04_retain_and_adopt.png")
OUT_PDF = Path("figures/fig04_retain_and_adopt.pdf")


def _draw_aligned_scores(ax, entry) -> None:
    """Horizontal bars placed at the schematic's own row heights.

    Bars are levelled with the `t+1` network states, not with f4.ROW_Y. The
    states live inside an inset, so their true height in the schematic axes is
    the inset origin plus its height times f4.ROW_INSET_Y -- about 0.04 below
    the rule rows. Deriving it rather than hard-coding keeps the alignment
    correct if the schematic's geometry is retuned.
    """
    row_y = [
        f4.FINAL_INSET_BOUNDS[1] + f4.FINAL_INSET_BOUNDS[3] * v
        for v in f4.ROW_INSET_Y
    ]
    observed = float(
        entry[entry["model"] == "observed"].iloc[0]["mean_entry_phi_rank_mean"]
    )

    ax.axvline(observed, color=f4.COLORS["observed"], lw=2.0, zorder=4)
    ax.axvline(0.5, color=f4.COLORS["baseline"], lw=1.2, linestyle="--", zorder=3)

    bar_h = 0.16
    for key, y in zip(ROW_MODELS, row_y):
        row = entry[entry["model"] == key].iloc[0]
        mid = float(row["mean_entry_phi_rank_mean"])
        lo = float(row["mean_entry_phi_rank_q05"])
        hi = float(row["mean_entry_phi_rank_q95"])
        # Drawn as a patch rather than via barh: ultraplot's wrapper maps
        # `height` to the bar's length along x, so every bar came out the same
        # size regardless of its value.
        ax.add_patch(
            Rectangle(
                (X_MIN, y - 0.5 * bar_h), mid - X_MIN, bar_h,
                facecolor=f4.COLORS[key], edgecolor="#333333",
                linewidth=0.6, zorder=2,
            )
        )
        ax.plot([lo, hi], [y, y], color="#222222", lw=1.1, zorder=5)
        for edge in (lo, hi):
            ax.plot([edge, edge], [y - 0.028, y + 0.028],
                    color="#222222", lw=1.1, zorder=5)
        # Value at the bar tip: the axis is short, so a tick read is imprecise.
        ax.text(hi + 0.006, y, f"{mid:.3f}", va="center", ha="left",
                fontsize=7.6, color="#222222", zorder=6)

    # These two markers are labelled in place, so the panel needs no legend.
    ax.text(observed, 1.01, f"Observed\n{observed:.3f}", ha="center", va="bottom",
            fontsize=figstyle.FS_ANNOT, color=f4.COLORS["observed"], zorder=6,
            linespacing=1.25)
    ax.text(0.5, 1.01, "Random\nchoice", ha="center", va="bottom",
            fontsize=figstyle.FS_ANNOT, color=f4.COLORS["baseline"], zorder=6,
            linespacing=1.25)

    ax.format(
        xlim=(X_MIN, X_MAX), ylim=(0.0, 1.0),
        xlabel="Mean proximity rank of newly entered topics",
        ylabel="", yticks=[], grid=False,
    )
    ax.grid(axis="x", alpha=0.18, linewidth=0.7)


def build_figure() -> uplt.Figure:
    entry = f4._load_history(f4.PROCESS_ENTRY)

    # Wide schematic, narrow score column. Geometry is explicit because the
    # schematic's internal layout assumes a wide, short axes -- refwidth alone
    # yields a portrait panel and stretches the diagram vertically.
    fig, axs = uplt.subplots(
        nrows=1, ncols=2, share=0,
        figsize=(11.4, 3.9), wratios=(2.35, 1.0), wspace=1.4,
    )
    ax_a, ax_b = axs

    # Short heading, and the standing note about what is held fixed moves to the
    # caption: at this density the panel should carry the diagram, not prose.
    f4.draw_schematic_panel(
        ax_a, title="Three allocation strategies", show_note=False,
    )
    _draw_aligned_scores(ax_b, entry)

    # The schematic axis is turned off, so its letter is placed manually.
    ax_a.text(0.0, 1.0, "a", transform=ax_a.transAxes, ha="left", va="top",
              fontweight="bold", fontsize=11, zorder=10)
    ax_b.text(0.03, 0.97, "b", transform=ax_b.transAxes, ha="left", va="top",
              fontweight="bold", fontsize=11, zorder=10)

    figstyle.apply_typography(ax_b)
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
