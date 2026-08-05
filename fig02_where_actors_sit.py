"""Main-text Figure 2: actors hold distinct, sticky, complementary positions.

The old version opened with a strip plot of actor positions grouped by the mode
each actor had been *assigned from that same position*, which cannot fail to
separate. This replaces it with displays that can fail:

  A  position against tenure and breadth -- the tenure gradient the text claims
     is shown, along with the breadth entanglement that qualifies it
  B  dominant-mode transitions -- positions are sticky, not momentary
  C  pairwise portfolios -- low topic overlap does not mean disconnected work

Colour is reserved: the orange/blue/green triple means "mode" throughout the
paper, so panel B uses a grey ramp and panel C a single off-palette accent
rather than borrowing those hues for unrelated categories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.patches import FancyArrowPatch
from scipy import stats

import fig03_local_portfolio_movement as f3
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


def _draw_transitions(ax) -> dict:
    """Dominant-mode transitions as an ordered chain rather than a 3x3 grid.

    The finding is topological -- movement runs along the gradient and almost
    never skips the middle -- and a grid buries that in two corner cells. Laying
    the three modes out in their concern-axis order turns it into shape: fat
    links to the neighbour, hairlines across the gap.

    The layout is a flattened triangle, not a force-directed graph and not a
    straight line. Raising the middle mode gives every pair a direct, unobscured
    edge -- on a straight line the coordination-to-strategy link had to arc over
    the middle node, crossing its label -- while keeping the left-to-right
    concern-axis order, so the gradient still reads. It also frees the three
    corners for the mode names, which no longer need masking boxes.

    Figure 1 is a network of topics; a free layout here would invite the reader
    to treat these three zones as objects of the same kind. Spacing is even,
    since panel A already carries the real spacing.
    """
    matrix, _ = f3._load_regime_transition_matrix(f3.REGIME_MATRIX_PATHS)
    meta = f3._load_regime_transition_summary(f3.REGIME_SUMMARY_PATHS)
    values = matrix.to_numpy(dtype=float)

    names = ["Coordination", "Compliance", "Strategy"]
    # Flattened triangle: x keeps the concern-axis order, the raised apex gives
    # the 1<->3 pair a clear run along the base.
    pos = [(0.0, 0.0), (1.5, 1.05), (3.0, 0.0)]
    # Where each name sits relative to its node, and how it anchors: the two
    # base corners hang their labels below and outward, the apex above.
    name_offsets = [(-0.10, -0.46, "center", "top"),
                    (0.0, 0.46, "center", "bottom"),
                    (0.10, -0.46, "center", "top")]

    def _width(p: float) -> float:
        # Wide dynamic range: width is the only quantitative channel here, so
        # it has to carry the 1.6% vs 21.1% contrast on its own. The floor keeps
        # the near-zero 1<->3 links visible instead of vanishing.
        return 1.1 + 13.0 * float(p)

    # One rad sign for every edge. Negative rad bows an arc to the left of its
    # direction of travel, so the two directions of a pair land on opposite
    # sides automatically; the obvious `* (1 if forward else -1)` would stack
    # them on top of each other instead.
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            p = float(values[i, j])
            span = abs(j - i)
            rad = -(0.20 if span == 1 else 0.13)
            colour = REGION_COLORS[i + 1]
            p0, p1 = np.array(pos[i], float), np.array(pos[j], float)
            ax.add_patch(
                FancyArrowPatch(
                    tuple(p0), tuple(p1),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>", mutation_scale=11,
                    lw=_width(p), color=colour, alpha=0.92,
                    # Must exceed the node radius (sqrt(s/pi) pt) or the arc and
                    # its arrowhead vanish under the marker.
                    shrinkA=22, shrinkB=22, zorder=2,
                )
            )
            # Put the label on the side the arc actually bulges toward, found
            # from the chord's left normal rather than hard-coded per edge.
            chord = p1 - p0
            normal = np.array([-chord[1], chord[0]])
            normal = normal / np.linalg.norm(normal)
            offset = 0.30 if span == 1 else 0.26
            lx, ly = (p0 + p1) / 2.0 + normal * offset
            ax.text(
                lx, ly, f"{p * 100:.1f}%",
                ha="center", va="center", fontsize=figstyle.FS_ANNOT, color=colour, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.9),
            )

    for i, (name, (x, y)) in enumerate(zip(names, pos)):
        colour = REGION_COLORS[i + 1]
        # The self-edge is the node's own ring: staying put is the diagonal of
        # the matrix, and a separate loop would collide with the arcs. Ring
        # thickness uses the same width scale as the arrows, so persistence and
        # movement are directly comparable.
        ax.scatter(
            [x], [y], s=1500, marker="o", zorder=3,
            c="white", edgecolor=colour, linewidth=_width(values[i, i]),
            absolute_size=True,
        )
        ax.text(x, y, f"{values[i, i] * 100:.0f}%", ha="center", va="center",
                fontsize=figstyle.FS_LABEL, color=colour, fontweight="bold", zorder=4)
        dx, dy, ha, va = name_offsets[i]
        ax.text(x + dx, y + dy, name, ha=ha, va=va, fontsize=figstyle.FS_LEGEND, color=colour,
                zorder=6)

    ax.format(
        xlim=(-1.0, 4.0), ylim=(-1.05, 2.05),
        xticks=[], yticks=[], xlabel="", ylabel="", grid=False,
    )
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(False)
    return {
        "same_mode_rate": float(meta.get("same_region_rate", np.nan)),
        "adjacent_or_same_rate": float(meta.get("adjacent_or_same_rate", np.nan)),
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
        xlabel=r"Portfolio overlap (Jaccard on $RPA>1$ topics)",
        ylabel=r"Exclusive support proximity in $\phi$",
    )
    ax.legend(loc="lr", frame=False, fontsize=figstyle.FS_LEGEND, ncols=1)

    # The upper-left quadrant is the whole point of the panel, and it is the one
    # thing a reader cannot infer from the axis labels: these pairs would look
    # unrelated to anyone counting shared topics. Open a strip of headroom above
    # the cloud rather than dropping the note onto the points.
    # Open a clear band above the cloud and sit the note in the middle of it,
    # centred over the quadrant it describes. Wedged into the corner it competed
    # with the panel letter on one side and the topmost points on the other.
    y_lo, y_hi = ax.get_ylim()
    band = 0.26 * (y_hi - y_lo)
    ax.set_ylim(y_lo, y_hi + band)
    # Sit the note on the floor of the new band, just clear of the topmost
    # points: centred in the band it collided with the panel letter, and pushed
    # to the corner it competed with the y-axis label.
    ax.text(
        0.5 * (ax.get_xlim()[0] + x_threshold),
        y_hi + 0.04 * band,
        "Few shared topics,\n" r"still adjacent in $\phi$",
        ha="center", va="bottom", fontsize=figstyle.FS_ANNOT,
        color=PAIR_ACCENT, zorder=6, linespacing=1.3,
    )
    return {
        "n_pairs": int(len(pairs)),
        "share_complementary": float(len(complementary) / max(len(pairs), 1)),
    }


def build_figure() -> tuple[uplt.Figure, dict]:
    fig, axs = uplt.subplots(
        nrows=1, ncols=3, share=0, refwidth=2.9, refaspect=1, wspace=10.0
    )
    stats_out = {
        "position": _draw_position(axs[0]),
        "transitions": _draw_transitions(axs[1]),
        "complementarity": _draw_complementarity(axs[2]),
    }
    axs.format(abc="[A]", abcloc="ul", abcsize=figstyle.FS_PANEL, grid=False)
    figstyle.apply_typography(axs)

    # Tie B back to A: the mode names on both axes carry the same colours as the
    # zone bands and markers in A, so the two panels read as one view. Applied
    # after format(), which regenerates tick labels and would drop the colours.
    for labels in (axs[1].get_xticklabels(), axs[1].get_yticklabels()):
        for position, label in enumerate(labels, start=1):
            if position in REGION_COLORS:
                label.set_color(REGION_COLORS[position])
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
