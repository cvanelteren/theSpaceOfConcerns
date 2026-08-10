"""Main-text Figure 2: actors hold distinct, complementary positions.

Displays that can fail:

  A  position against tenure and breadth -- the tenure gradient the text claims
     is shown, along with the breadth entanglement that qualifies it
  B  pairwise portfolios -- low topic overlap does not mean disconnected work

The dominant-mode transition chain that used to sit between these panels moved
to Figure 3, where it opens the locality evidence: stickiness and locality are
one claim, and showing them in two figures made each look like a restatement
of the other.

Colour is reserved: the orange/blue/green triple means "mode" throughout the
paper, so panel B uses a single off-palette accent rather than borrowing those
hues for unrelated categories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy import stats

import figstyle

ACTOR_CSV = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")
REGION_CSV = Path("output/fig45_portfolio_space_ridgelines_region_summary.csv")
TENURE_CSV = Path("output/fig05_pioneer_regime_position_data.csv")
PAIRS_CSV = Path("output/fig27_pairwise_portfolio_complementarity_pairs.csv")

OUT_PNG = Path("figures/fig02_where_actors_sit.png")
OUT_PDF = Path("figures/fig02_where_actors_sit.pdf")

REGION_COLORS = figstyle.MODE_COLORS
MODE_GLOSS = figstyle.MODE_GLOSS
PAIR_ACCENT = figstyle.ACCENT   # deliberately outside the mode palette
PAIR_MUTED = figstyle.ACCENT_MUTED
TEXT = figstyle.TEXT
LABEL_ACTORS = {
    "Malaysia", "Estonia", "Switzerland", "Ukraine", "Netherlands",
    "New Zealand", "Russian Federation", "Australia", "SCAR", "United States",
}


def _shade_regions(ax, regions: pd.DataFrame, *, label: bool = False) -> None:
    for _, row in regions.iterrows():
        rid = int(row["region_id"])
        ax.axvspan(
            float(row["boundary_left"]),
            float(row["boundary_right"]),
            facecolor=REGION_COLORS[rid],
            alpha=0.07,
            zorder=0,
        )
        if label:
            ax.text(
                0.5 * (float(row["boundary_left"]) + float(row["boundary_right"])),
                0.015,
                MODE_GLOSS[rid].replace("\n", " "),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=figstyle.FS_ANNOT,
                color=REGION_COLORS[rid],
            )


def _draw_position(ax) -> dict:
    actors = pd.read_csv(ACTOR_CSV)
    tenure = pd.read_csv(TENURE_CSV)[["country", "first_activity"]]
    regions = pd.read_csv(REGION_CSV)

    df = actors.merge(tenure, left_on="actor", right_on="country", how="inner")
    df = df.dropna(subset=["first_activity", "centroid_xplot_raw_rca"])
    x = df["centroid_xplot_raw_rca"].to_numpy(float)
    y = df["first_activity"].to_numpy(float)
    breadth = df["k_active"].to_numpy(float)

    _shade_regions(ax, regions, label=True)
    # Twenty-odd actors share 1961 as their first year, so ties would stack into
    # a single unreadable row on the axis floor.
    rng = np.random.default_rng(7)
    y_plot = y + rng.uniform(-0.9, 0.9, size=y.size)
    ax.scatter(
        x,
        y_plot,
        s=7.0 * breadth,
        c=[REGION_COLORS[int(r)] for r in df["dominant_region"]],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    # Label away from the nearest axis edge, so the crowded 1961 cohort does not
    # collide with the zone captions along the floor, then ladder labels upward
    # when neighbours are too close in x -- half the founding members share both
    # a first year and a similar position.
    y_mid = 0.5 * (float(y.min()) + float(y.max()))
    labelled = df[df["actor"].isin(LABEL_ACTORS)].sort_values("centroid_xplot_raw_rca")
    placed: list[tuple[float, float]] = []
    for _, row in labelled.iterrows():
        idx = df.index.get_loc(row.name)
        above = y_plot[idx] < y_mid
        lx = float(row["centroid_xplot_raw_rca"])
        ly = y_plot[idx] + (3.0 if above else -3.0)
        step = 4.4 if above else -4.4
        while any(abs(lx - px) < 0.18 and abs(ly - py) < 4.0 for px, py in placed):
            ly += step
        placed.append((lx, ly))
        # Centred text on an actor near x=0 or x=1 spills past the spine, so
        # anchor away from whichever edge is close. The axis is fixed to (0, 1).
        if lx < 0.16:
            ha, tx = "left", max(lx, 0.01)
        elif lx > 0.84:
            ha, tx = "right", min(lx, 0.99)
        else:
            ha, tx = "center", lx
        ax.text(
            tx,
            ly,
            str(row["actor"]).replace("Russian Federation", "Russia"),
            ha=ha,
            va="bottom" if above else "top",
            fontsize=figstyle.FS_ANNOT,
            color=TEXT,
            zorder=4,
        )

    rho_tenure = stats.spearmanr(x, y)
    rho_breadth = stats.spearmanr(x, breadth)
    ax.format(
        title="Position tracks tenure",
        xlabel="Position on the concern axis (RPA-weighted)",
        ylabel="First year active",
        xlim=(0, 1),
        ylim=(float(y.min()) - 9, float(y.max()) + 8),
    )
    return {
        "n_actors": int(len(df)),
        "rho_position_tenure": float(rho_tenure.statistic),
        "p_position_tenure": float(rho_tenure.pvalue),
        "rho_position_breadth": float(rho_breadth.statistic),
    }


def _draw_complementarity(ax) -> dict:
    pairs = pd.read_csv(PAIRS_CSV)
    # Draw the thresholds that define the categories, so the reader can see the
    # classification is a cut on two continuous axes rather than a clustering.
    complementary = pairs[pairs["pair_type"] == "Complementary"]
    aligned = pairs[pairs["pair_type"] == "Aligned"]
    if len(complementary) and len(aligned):
        ax.axhline(
            float(complementary["exclusive_phi_proximity"].min()),
            color="0.6", lw=0.8, ls="--", zorder=1,
        )
        x_threshold = 0.5 * (
            float(complementary["overlap_jaccard_rpa"].max())
            + float(aligned["overlap_jaccard_rpa"].min())
        )
        ax.axvline(x_threshold, color="0.6", lw=0.8, ls="--", zorder=1)
    is_complementary = pairs["pair_type"] == "Complementary"
    ax.scatter(
        pairs.loc[~is_complementary, "overlap_jaccard_rpa"],
        pairs.loc[~is_complementary, "exclusive_phi_proximity"],
        s=14, color=PAIR_MUTED, alpha=0.65, edgecolor="none", zorder=2,
        label=f"Other pairs ({int((~is_complementary).sum())})",
    )
    ax.scatter(
        pairs.loc[is_complementary, "overlap_jaccard_rpa"],
        pairs.loc[is_complementary, "exclusive_phi_proximity"],
        s=14, color=PAIR_ACCENT, alpha=0.85, edgecolor="none", zorder=3,
        label=f"Complementary ({int(is_complementary.sum())})",
    )
    ax.format(
        title="Low overlap, still adjacent",
        xlabel=r"Portfolio overlap (Jaccard on $RPA>1$ topics)",
        ylabel=r"Exclusive support proximity in $\phi$",
    )
    ax.legend(loc="lr", frame=False, fontsize=figstyle.FS_LEGEND, ncols=1)

    return {
        "n_pairs": int(len(pairs)),
        "share_complementary": float(len(complementary) / max(len(pairs), 1)),
    }


def build_figure() -> tuple[uplt.Figure, dict]:
    fig, axs = uplt.subplots(
        nrows=1, ncols=2, share=0, refwidth=3.3, refaspect=1, wspace=10.0
    )
    stats_out = {
        "position": _draw_position(axs[0]),
        "complementarity": _draw_complementarity(axs[1]),
    }
    axs.format(
        abc="a", abcloc="ul", abcsize=figstyle.FS_PANEL, grid=False,
        titlesize=figstyle.FS_TITLE, titleweight="bold", titleloc="uc",
    )
    figstyle.apply_typography(axs)
    return fig, stats_out


def main() -> None:
    fig, stats_out = build_figure()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PNG, dpi=300)
    fig.save(OUT_PDF)
    print(f"Wrote {OUT_PNG}")
    print(json.dumps(stats_out, indent=2))


if __name__ == "__main__":
    main()
