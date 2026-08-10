"""Main-text Figure 3: attention moves locally, and that is not an artifact.

Four panels, in the order readable claim -> picture -> proof -> robustness:

  A  dominant-mode transitions between adjacent windows -- the readable version
     of the claim, which inherits the partition caveat
  B  where actors actually stepped, drawn on the concern axis -- the same claim
     without any partition
  C  how strongly does entry fall off with distance?  (observed, nonparametric)
  D  does that hold under every way of building the space?  (one number each)

A and B state one claim at two resolutions: A aggregates actor-windows into
dominant modes and is easy to read but depends on the zone partition, while B
plots each of the 1,895 individual entries from the nearest topic the actor
already held to the topic it moved into, over the same zone bands Figure 2
uses, and needs no partition at all. Keeping the two side by side is the point
-- the reader can see that the partition-free picture says what the matrix
says. C and D are what make it a result rather than an impression.

C and D must not be two views of the same curve. An earlier draft drew D as
fitted decay curves over the panel-C axis, but the primary curve there was just
a smoothed restatement of C, which forced a five-entry legend to carry the only
real content -- the spread across specifications. Collapsing each specification
to a single effect size makes that spread the subject and removes the legend
entirely, since the row labels identify the rows.

The retention curves moved to the appendix; they show that adoptions persist,
which is a weaker and more expected effect than locality itself. The transition
panel moved here from the positioning figure so that locality is shown once,
in one place, instead of being split across two figures.

Colour discipline: the zone bands in B carry the reserved mode palette, because
they are the same three modes Figure 2 shades, and panel A draws its nodes and
arrows in the same colours for the same modes. Nothing else in the figure is
coloured, so no second categorical scheme can be mistaken for a related one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.patches import FancyArrowPatch

import fig03_local_portfolio_movement as f3
import figstyle

SPECIFICATIONS_CSV = Path("output/hazard_locality_specifications.csv")
OR_X_LO = 0.60

# Shared limits for the movement panel: the zone shading, the diagonal and
# the axes all key off these so the band stays flush with the panel edges.
AX_LO, AX_HI = -0.02, 1.02
MOVES_CSV = Path("output/portfolio_displacement_moves.csv")
TOPIC_ORDER_CSV = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
REGION_CSV = Path("output/fig45_portfolio_space_ridgelines_region_summary.csv")

OUT_PNG = Path("figures/fig03_dynamics_are_local.png")
OUT_PDF = Path("figures/fig03_dynamics_are_local.pdf")
OUT_RETENTION_PNG = Path("figures/figS_retention_after_entry.png")
OUT_RETENTION_PDF = Path("figures/figS_retention_after_entry.pdf")

PRIMARY = figstyle.PRIMARY
MUTED = figstyle.MUTED
ENVELOPE = figstyle.ENVELOPE

# Row labels are the only identifier in panel B, so they have to say what each
# specification *does*, not what it is called. The technical names ("cumulative-
# lagged", "leave-one-actor-out", ...) are the crosswalk to Methods and live in
# the caption; here each row states the manipulation in plain terms, under a
# heading that says which of the two questions it answers.
#
# Group 1 varies which data builds the space -- a question about information.
# Group 2 varies how the space is built from the same data -- a question about
# artifacts. Keeping them visually separate is the whole point of the panel.
SPEC_GROUPS = [
    (
        "Which data builds the space",
        [
            ("Cumulative-lagged space", "Data predating each move"),
            ("Previous-window space", "Preceding window only"),
            ("Pooled full-history space", "Whole archive at once"),
        ],
    ),
    (
        "How the space is built",
        [
            ("Leave-one-actor-out space", "Own papers removed"),
            ("Fractional co-sponsorship counting", "Co-sponsorship split $1/n$"),
        ],
    ),
]

GROUP_GAP = 1.7


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
    since panel B already carries the real spacing.
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
            colour = figstyle.MODE_COLORS[i + 1]
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
            # The two 1<->3 labels are placed explicitly below the base instead:
            # above it they crowd the span-1 labels in the middle of the
            # triangle, and the normal-based rule would strand one above anyway.
            if span == 1:
                chord = p1 - p0
                normal = np.array([-chord[1], chord[0]])
                normal = normal / np.linalg.norm(normal)
                lx, ly = (p0 + p1) / 2.0 + normal * 0.30
            else:
                # Both 1<->3 labels are near-identical short strings ("1.6%",
                # "3.4%"), so at +-0.32 their boxes met in the middle and read as
                # one token. The separation has to exceed a label width (~0.55
                # data units at FS_ANNOT on this axis), not merely a gap.
                lx, ly = 1.5 + (-0.62 if i == 0 else 0.62), -0.30
            ax.text(
                lx, ly, f"{p * 100:.1f}%",
                ha="center", va="center", fontsize=figstyle.FS_ANNOT, color=colour, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.9),
            )

    for i, (name, (x, y)) in enumerate(zip(names, pos)):
        colour = figstyle.MODE_COLORS[i + 1]
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


def _draw_movement(ax) -> dict:
    """Where each expansion came from, against where it went.

    One point per (origin, destination) topic pair: the horizontal coordinate
    is the position on the concern axis of the topic the actor already held
    that is nearest in phi, the vertical coordinate is the position of the
    topic it newly specialized in, and the marker area is how many expansions
    in the record took that exact step.

    Read it against the diagonal. A point on the line is a move that did not
    change position; distance from the line is how far across the space the
    move went. Mass on the diagonal means actors expanded into neighbours.

    This replaces an arc diagram in which the height of each arc and its width
    both encoded the same quantity, so the vertical axis carried no information
    a reader could use. Here the two axes are independent measurements and the
    diagonal is the reference; the same zone bands run along both, which makes
    the panel the continuous counterpart of the transition chain in panel A --
    the same claim without the partition that one depends on.
    """
    moves = pd.read_csv(MOVES_CSV)
    topics = pd.read_csv(TOPIC_ORDER_CSV)
    regions = pd.read_csv(REGION_CSV)
    x_of = dict(zip(topics["topic"], topics["x_plot"].astype(float)))

    referenced = set(moves["from_topic"]) | set(moves["to_topic"])
    unresolved = sorted(referenced - set(x_of))
    if unresolved:
        raise ValueError(f"moves reference topics with no axis position: {unresolved}")

    # Only the three diagonal blocks are shaded. Banding both axes in full would
    # tint the off-diagonal cells with a mixture of two mode colours, which is
    # meaningless -- a move between zones belongs to neither. Shading just the
    # diagonal gives the panel a single readable rule: inside a patch is a move
    # that stayed in its zone, outside is a move that crossed.
    # The zone boundaries span exactly 0-1 but the axes carry a small margin, so
    # shading them verbatim leaves the diagonal band floating clear of the two
    # corners. Bleed the outer zones out to the axis limits instead, so the band
    # runs corner to corner and the panel edges read as part of the partition.
    regions = regions.sort_values("region_id").reset_index(drop=True)
    last = len(regions) - 1
    for idx, row in regions.iterrows():
        color = figstyle.MODE_COLORS[int(row["region_id"])]
        left = AX_LO if idx == 0 else float(row["boundary_left"])
        right = AX_HI if idx == last else float(row["boundary_right"])
        ax.fill_between(
            [left, right], left, right,
            facecolor=color, alpha=0.11, lw=0, zorder=0,
        )

    moves["x_from"] = moves["from_topic"].map(x_of)
    moves["x_to"] = moves["to_topic"].map(x_of)
    span = np.abs(moves["x_to"] - moves["x_from"]).to_numpy(float)

    ax.plot([AX_LO, AX_HI], [AX_LO, AX_HI], color="0.45", lw=1.0, zorder=2)

    # Both coordinates come from a fixed set of 45 topics, so the points fall on
    # a lattice and plain scatter would hide every repeat. Collapse to unique
    # steps and let area carry the count instead.
    grouped = (
        moves.groupby(["x_from", "x_to"]).size().reset_index(name="count")
    )
    ax.scatter(
        grouped["x_from"].to_numpy(float),
        grouped["x_to"].to_numpy(float),
        s=4.0 * grouped["count"].to_numpy(float),
        color=PRIMARY,
        alpha=0.45,
        edgecolor="none",
        zorder=3,
    )

    # Marker area is the only encoding a reader cannot name from the axes, so
    # it gets a key rather than a sentence in the caption. Placed low-right,
    # which is empty: a step from the far end of the axis to the near end.
    key_counts = [1, 10, 30]
    for offset, count in enumerate(key_counts):
        key_x = 0.72 + 0.10 * offset
        ax.scatter(
            [key_x], [0.07], s=4.0 * count, color=PRIMARY, alpha=0.45,
            edgecolor="none", zorder=3,
        )
        ax.text(
            key_x, 0.135, str(count), ha="center", va="bottom",
            fontsize=figstyle.FS_ANNOT, color="0.35", zorder=4,
        )
    ax.text(
        0.72, 0.185, "expansions", ha="left", va="bottom",
        fontsize=figstyle.FS_ANNOT, color="0.35", zorder=4,
    )

    ax.format(
        xlabel="Position of nearest topic already held",
        ylabel="Position of topic newly entered",
        xlim=(AX_LO, AX_HI),
        ylim=(AX_LO, AX_HI),
        xlocator=[0.0, 0.25, 0.5, 0.75, 1.0],
        ylocator=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    zone_of = {}
    for _, row in regions.iterrows():
        zone_of[int(row["region_id"])] = (
            float(row["boundary_left"]),
            float(row["boundary_right"]),
        )

    def _zone(position: float) -> int:
        for rid, (left, right) in zone_of.items():
            if left <= position <= right:
                return rid
        return -1

    same_zone = np.mean(
        [
            _zone(a) == _zone(b)
            for a, b in zip(moves["x_from"].to_numpy(float), moves["x_to"].to_numpy(float))
        ]
    )
    return {
        "n_moves": int(len(moves)),
        "n_distinct_steps": int(len(grouped)),
        "median_span": float(np.median(span)),
        "share_within_quarter_axis": float((span <= 0.25).mean()),
        "share_same_zone": float(same_zone),
    }


def _draw_specification_strength(ax) -> pd.DataFrame:
    """Effect size per specification, in the unit the text quotes.

    Panel A already shows the shape of the decay. This panel answers a
    different question -- whether that decay survives every way of building
    the space -- so it plots one number per specification rather than a
    second set of curves. The unit is the odds ratio for a +0.1 step in
    distance, which is what the Results text quotes, and 1.0 is no effect.
    Row labels carry the identity, so no legend is needed.
    """
    table = pd.read_csv(SPECIFICATIONS_CSV).set_index("label")

    # Walk the groups top to bottom, reserving a slot for each group heading so
    # the headings can be rendered as (unmarked) y ticks rather than floating
    # text that has to dodge the data.
    ticks: list[float] = []
    tick_labels: list[str] = []
    is_heading: list[bool] = []
    rows: list[tuple[float, str]] = []

    cursor = 0.0
    for gi, (heading, members) in enumerate(SPEC_GROUPS):
        if gi:
            cursor -= GROUP_GAP - 1.0
        ticks.append(cursor)
        tick_labels.append(heading)
        is_heading.append(True)
        cursor -= 1.0
        for key, plain in members:
            ticks.append(cursor)
            tick_labels.append(plain)
            is_heading.append(False)
            rows.append((cursor, key))
            cursor -= 1.0

    or_mid, or_lo, or_hi = {}, {}, {}
    for _, key in rows:
        row = table.loc[key]
        or_mid[key] = float(np.exp(0.1 * row["distance_coef"]))
        or_lo[key] = float(np.exp(0.1 * row["distance_ci_low_95"]))
        or_hi[key] = float(np.exp(0.1 * row["distance_ci_high_95"]))

    # Anchored at the left spine instead of floating at the envelope's lower
    # bound: a band read as a highlighted interval, but the claim is that every
    # estimate sits left of the null line, so the shading fills that space.
    ax.axvspan(OR_X_LO, max(or_hi.values()),
               color=ENVELOPE, lw=0, zorder=0)
    ax.axvline(1.0, color="0.45", lw=1.0, zorder=2)

    for yi, key in rows:
        primary = table.loc[key, "group"] == "primary"
        ax.plot([or_lo[key], or_hi[key]], [yi, yi], color=PRIMARY, lw=1.6,
                solid_capstyle="round", zorder=3)
        ax.scatter([or_mid[key]], [yi], s=64 if primary else 40, zorder=4,
                   color=PRIMARY if primary else "white",
                   edgecolor=PRIMARY, linewidth=1.3)

    ax.format(
        yticks=ticks, yticklabels=tick_labels,
        xlabel="Odds ratio for a $+0.1$ step in distance",
        ylabel="",
        xlim=(0.60, 1.06),
        ylim=(min(ticks) - 0.7, max(ticks) + 0.7),
        xlocator=[0.6, 0.7, 0.8, 0.9, 1.0],
    )
    for label, heading in zip(ax.get_yticklabels(), is_heading):
        if heading:
            label.set_fontsize(7.4)
            label.set_fontweight("bold")
            label.set_color(PRIMARY)
        else:
            label.set_fontsize(figstyle.FS_ANNOT)
            label.set_color("0.25")
    # Headings label a group, not a value, so suppress their tick marks.
    for tick, heading in zip(ax.yaxis.get_major_ticks(), is_heading):
        if heading:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

    ax.text(
        0.999, max(ticks) + 0.55, "no distance effect",
        fontsize=figstyle.FS_ANNOT, color="0.45", va="top", ha="right", rotation=90, zorder=5,
    )
    return table.reset_index()


def build_figure() -> tuple[uplt.Figure, pd.DataFrame, dict]:
    adoption_df, _ = f3._load_adoption_panel()
    adoption_df = adoption_df.copy()
    adoption_df["plot_distance"] = adoption_df["distance"].to_numpy(dtype=float)
    agg, event_rate = f3._binned_adoption_curve(adoption_df)

    d_lo = float(np.min(agg["x"]))
    d_hi = float(np.max(agg["x"]))

    # Panel D hangs its specification names off the left spine, so its gutter
    # has to be wider than the ones between A/B and B/C. Panel A is a free
    # chain drawing with no tick labels, so it can sit closer to B.
    fig, axs = uplt.subplots(
        ncols=4, share=0, refwidth=2.35, refaspect=1, wspace=(5.0, 9.5, 16.0)
    )
    ax_a, ax_b, ax_c, ax_d = axs

    transitions = _draw_transitions(ax_a)
    movement = _draw_movement(ax_b)

    f3._plot_adoption_panel(
        ax=ax_c, agg=agg,
        plot_distance=adoption_df["plot_distance"].to_numpy(float),
        event_rate=event_rate, show_legend=False, show_title=False,
    )
    # The source panel draws in the Mode 2 blue, which is reserved. Recolour to
    # the neutral primary so the only hues in this figure are the mode colours
    # in panels A and B.
    for line in ax_c.lines:
        line.set_color(PRIMARY)
        line.set_markerfacecolor(PRIMARY)
        line.set_markeredgecolor(PRIMARY)
    for container in ax_c.containers:
        for bar in getattr(container, "lines", [])[1:]:
            for artist in np.atleast_1d(bar):
                artist.set_color(PRIMARY)
    for collection in ax_c.collections:
        collection.set_color(PRIMARY)
    ax_c.format(
        xlabel=r"Distance to prior portfolio, $1-\max(\phi)$",
        ylabel="Entry probability",
        xlim=(d_lo, d_hi),
    )

    table = _draw_specification_strength(ax_d)

    axs.format(abc="a", abcloc="ul", abcsize=figstyle.FS_PANEL, grid=False)
    figstyle.apply_typography(axs)
    return fig, table, {**movement, "transitions": transitions}


def build_retention_figure() -> uplt.Figure:
    """Appendix: how long a specialization stays active after entry."""
    curves, step_label = f3._compute_retention_curves(time_unit="year", window_size=1)
    fig, ax = uplt.subplots(refwidth=3.6, refaspect=1.25)
    f3._plot_retention_panel(ax=ax, retention_curves=curves, step_label=step_label)
    ax.format(
        xlabel=f"Time since entry ({step_label})",
        ylabel="Probability topic remains active",
        title="", grid=False,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Gap tolerance")
        for text in legend.get_texts():
            text.set_fontsize(7.5)
    return fig


def main() -> None:
    fig, table, movement = build_figure()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PNG, dpi=300)
    fig.save(OUT_PDF)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    for key, value in movement.items():
        print(f"  {key}: {value}")

    fig_r = build_retention_figure()
    fig_r.save(OUT_RETENTION_PNG, dpi=300)
    fig_r.save(OUT_RETENTION_PDF)
    print(f"Wrote {OUT_RETENTION_PNG}")
    print(f"Wrote {OUT_RETENTION_PDF}")

    stale = table[table["source"].str.contains("not yet recomputed", na=False)]
    if len(stale):
        print(
            "\nWARNING: these specification rows are transcribed from the "
            "manuscript, not recomputed in-repo:"
        )
        for _, row in stale.iterrows():
            print(f"  - {row['label']} (beta={row['distance_coef']})")


if __name__ == "__main__":
    main()
