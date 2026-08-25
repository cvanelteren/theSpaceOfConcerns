#!/usr/bin/env python3
"""Publication-style figures for the attention--outcome linkage analysis."""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
FIGDIR = ROOT / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle
from analysis.data_loading import load_submitted_with_fallback
from fig01_space_of_concerns_topology import (
    build_graphs,
    draw_base,
    draw_silhouette,
    fitted_layout,
    load_topic_meta,
)
from scripts import explore_lineage_space as lineage
from scripts.link_movement_to_outcomes import (
    build_stable_graph,
    draw_selected_network_labels,
    draw_stable_edges,
    rolling_actor_positions_1d,
)


INSTRUMENT_COLORS = lineage.INSTRUMENT_COLORS
EXAMPLE_ACTORS = ["Australia", "Netherlands", "Ukraine"]
RANDOM_SEED = 20260812


def save(fig, stem: str, *, transparent: bool = False) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.save(FIGDIR / f"{stem}.pdf", transparent=transparent)
    fig.save(FIGDIR / f"{stem}.png", dpi=260, transparent=transparent)
    uplt.close(fig)


def draw_figure1_scaffold(ax, backbone: nx.Graph, stable: nx.Graph, positions) -> None:
    stable_pairs = {frozenset((u, v)) for u, v in stable.edges()}
    for source, target in backbone.edges():
        if frozenset((source, target)) in stable_pairs:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.plot(
            [x0, x1], [y0, y1], color=figstyle.ENVELOPE, lw=0.62,
            ls=(0, (2.0, 2.7)), alpha=0.72, zorder=0,
        )
    draw_stable_edges(ax, stable, positions)


def make_outcome_map() -> None:
    coverage = pd.read_csv(OUTDIR / "annual_output_topic_coverage.csv").set_index("topic")

    backbone, mst, graph, counts, _ = build_graphs()
    topics = list(counts.index)
    _, mode_of, _ = load_topic_meta()
    positions, land, projection_extent = fitted_layout(mst, graph, mode_of)
    edge_table = pd.read_csv(ROOT / "output/movement_outcomes/bootstrap_edge_stability.csv")
    stable = build_stable_graph(topics, edge_table)

    fig, axes = uplt.subplots(
        [[0, 1, 1, 1, 1, 0]], figwidth=12.5, refaspect=1.45, share=False,
    )
    ax = axes[0]
    draw_silhouette(ax, land, projection_extent)
    draw_figure1_scaffold(ax, backbone, stable, positions)

    values = coverage["primary_outcomes"].reindex(topics, fill_value=0).astype(int)
    max_value = max(float(values.max()), 1.0)
    for topic in topics:
        x, y = positions[topic]
        value = int(values[topic])
        radius = 0.095 if value == 0 else 0.115 + 0.50 * math.sqrt(value / max_value)
        if value == 0:
            ax.add_patch(
                Circle(
                    (x, y), radius, facecolor="white", edgecolor=figstyle.MUTED,
                    linewidth=0.9, zorder=3,
                )
            )
        else:
            ax.add_patch(
                Circle(
                    (x, y), radius, facecolor=figstyle.ADOPTION, edgecolor="white",
                    linewidth=0.7, alpha=0.90, zorder=3,
                )
            )

    top_labels = values.sort_values(ascending=False).head(12).index.tolist()
    for topic in ["Drilling", "Mineral resources", "Marine Protected Areas"]:
        if topic not in top_labels:
            top_labels.append(topic)
    draw_selected_network_labels(ax, positions, top_labels)

    ax.format(
        aspect="equal", grid=False, xlocator=[], ylocator=[],
        xspineloc="neither", yspineloc="neither",
    )
    handles = [
        Line2D([0], [0], color=figstyle.TEXT, lw=2.0, label="bootstrap-supported attention tie"),
        Line2D([0], [0], color=figstyle.ENVELOPE, lw=0.9, ls=(0, (2, 2.7)), label="attention-space scaffold"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=figstyle.ADOPTION, markeredgecolor="white",
               label="area = primary annual-output count"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=7,
               markerfacecolor="white", markeredgecolor=figstyle.MUTED,
               label="zero primary annual-output assignments"),
    ]
    fig.legend(handles=handles, loc="b", ncols=4, frame=False)
    fig.format(
        suptitle="Formal outputs are unevenly distributed across the attention space, 1961–2025",
        suptitlesize=14,
    )
    figstyle.apply_typography([ax])
    save(fig, "exploratory_independent_outcomes_map")


def bootstrap_outcome_means(edges: pd.DataFrame, n_bootstrap: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    definitions = {
        "Adoption-linked": ["direct_adoption_or_approval", "documented_contribution"],
        "Discussion-linked": ["direct_proposal_or_discussion"],
    }
    for label, relations in definitions.items():
        subset = edges[edges["relation"].isin(relations)].copy()
        by_outcome = subset.groupby("outcome_id")["within_meeting_percentile"].mean()
        values = by_outcome.to_numpy(dtype=float)
        draws = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
        )
        rows.append(
            {
                "label": label,
                "mean": float(values.mean()),
                "low": float(np.quantile(draws, 0.025)),
                "high": float(np.quantile(draws, 0.975)),
                "n_outcomes": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def errorbar_dot(ax, x, y, low, high, color, marker="o", size=54) -> None:
    ax.errorbar(
        [x], [y], xerr=np.array([[x - low], [high - x]]),
        fmt="none", ecolor=color, elinewidth=2.0, capsize=3.0, zorder=2,
    )
    ax.scatter(
        [x], [y], s=size, marker=marker, color=color,
        edgecolor="white", linewidth=0.65, zorder=3,
    )


def evidence_strip(
    ax,
    *,
    estimate: float,
    low: float,
    high: float,
    neutral: float,
    y: float,
    label: str,
    color: str,
    label_offset: float = 0.22,
) -> None:
    """Draw one directly labelled estimate as movement away from a baseline."""
    ax.errorbar(
        [estimate], [y], xerr=np.array([[estimate - low], [high - estimate]]),
        fmt="none", ecolor=color, elinewidth=2.0, capsize=3.0, zorder=2,
    )
    ax.scatter(
        [estimate], [y], s=66, color=color, edgecolor="white",
        linewidth=0.7, zorder=3,
    )
    axis_y = ax.get_yaxis_transform()
    ax.text(
        0.06, y + label_offset, label, transform=axis_y, ha="left", va="bottom",
        fontsize=figstyle.FS_LEGEND, color=figstyle.TEXT,
    )


def make_evidence_figure() -> None:
    coverage = pd.read_csv(OUTDIR / "annual_output_topic_coverage.csv").set_index("topic")
    accumulation = pd.read_csv(OUTDIR / "attention_accumulation_models.csv")
    trajectories = pd.read_csv(OUTDIR / "attention_accumulation_event_trajectories.csv")

    fig, axes = uplt.subplots(
        ncols=3, refwidth=3.05, refaspect=1.0, share=False, wspace=5.2,
    )
    ax_a, ax_b, ax_c = axes

    backbone, mst, graph, counts, _ = build_graphs()
    _, mode_of, _ = load_topic_meta()
    positions, land, projection_extent = fitted_layout(mst, graph, mode_of)
    draw_silhouette(ax_a, land, projection_extent)
    draw_base(ax_a, backbone, positions, edge_alpha=0.42, edge_lw=0.58, color=figstyle.REFERENCE)
    topics = list(counts.index)
    values = coverage["primary_outcomes"].reindex(topics, fill_value=0).astype(float)
    maximum = max(float(values.max()), 1.0)
    for topic in topics:
        x, y = positions[topic]
        value = float(values[topic])
        if value == 0:
            ax_a.scatter([x], [y], s=18, facecolor="white", edgecolor=figstyle.MUTED,
                         linewidth=0.7, zorder=3, absolute_size=True)
        else:
            ax_a.scatter([x], [y], s=18 + 360 * value / maximum,
                         color=figstyle.ADOPTION, edgecolor="white", linewidth=0.6,
                         alpha=0.90, zorder=3, absolute_size=True)
    xy = np.asarray(list(positions.values()), dtype=float)
    ax_a.format(
        title="Adopted outputs occupy only part of the concern space",
        aspect="equal", xlim=(xy[:, 0].min() - 1.0, xy[:, 0].max() + 1.0),
        ylim=(xy[:, 1].min() - 1.0, xy[:, 1].max() + 1.0),
        xlocator=[], ylocator=[], xspineloc="neither", yspineloc="neither",
        grid=False, titlesize=figstyle.FS_TITLE,
    )

    horizon_data = accumulation[
        accumulation["specification"].eq("accumulated_attention_hard_output")
    ]
    horizon_specs = [
        ("papers_prior", "Same-concern papers", figstyle.FOCAL, "o"),
        ("nearby_prior", "Nearby-concern papers", figstyle.NEARBY, "s"),
        ("outcomes_prior", "Earlier outputs", figstyle.VERMILLION, "^"),
    ]
    for prefix, label, color, marker in horizon_specs:
        subset = horizon_data[
            horizon_data["predictor"].str.startswith(prefix)
        ].sort_values("horizon_meetings")
        meetings = subset["horizon_meetings"].to_numpy(float)
        ratio = subset["ratio_per_doubling_plus_one"].to_numpy(float)
        low = subset["doubling_ci_low"].to_numpy(float)
        high = subset["doubling_ci_high"].to_numpy(float)
        ax_b.plot(
            meetings, ratio, color=color, marker=marker, ms=4.8,
            lw=1.7, label=label, zorder=3,
        )
        ax_b.errorbar(
            meetings, ratio,
            yerr=np.vstack((ratio - low, high - ratio)),
            fmt="none", ecolor=color, elinewidth=1.0, capsize=2.0,
            alpha=0.48, zorder=2,
        )
    ax_b.axhline(1.0, color=figstyle.REFERENCE, lw=0.9, ls="--")
    ax_b.format(
        title="What accumulates before adopted output?",
        xlabel="Preceding ATCM meetings included",
        ylabel="Output rate ratio\nper doubling",
        xlim=(0.65, 10.6), xlocator=[1, 2, 3, 5, 8, 10],
        ylim=(0.70, 1.87), ylocator=[0.75, 1.0, 1.25, 1.5, 1.75],
        grid=False, titlesize=figstyle.FS_TITLE,
    )
    ax_b.text(
        0.92, 1.015, "no association", ha="left", va="bottom",
        fontsize=figstyle.FS_ANNOT, color=figstyle.MUTED,
    )
    ax_b.legend(loc="ur", ncols=1, frame=False)

    for event_type, label, color in (
        ("Continuing output stream", "Continuing stream", figstyle.ADOPTION),
        ("After five meetings without output", "After 5 quiet meetings", figstyle.CYAN),
    ):
        subset = trajectories[trajectories["event_type"].eq(event_type)].sort_values("relative_meeting")
        x = subset["relative_meeting"].to_numpy(float)
        y = subset["mean_residual_attention"].to_numpy(float)
        low = subset["ci_low"].to_numpy(float)
        high = subset["ci_high"].to_numpy(float)
        ax_c.plot(x, y, color=color, lw=1.9, marker="o", ms=4.5, label=label)
        ax_c.fill_between(x, low, high, color=color, alpha=0.14)
    ax_c.axhline(0.0, color=figstyle.REFERENCE, lw=0.9, ls="--")
    ax_c.axvline(0.0, color=figstyle.TEXT, lw=0.9)
    ax_c.format(
        title="Lead-up is clearest in continuing streams",
        xlabel="ATCM meetings before output",
        ylabel="Adjusted paper attention",
        xlim=(-10.3, 0.3), xlocator=[-10, -8, -6, -4, -2, 0],
        ylim=(-0.40, 0.55),
        grid=False, titlesize=figstyle.FS_TITLE,
    )
    ax_c.legend(loc="ul", ncols=1, frame=False)

    for ax in axes:
        ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    fig.format(abc="a", abcloc="ul", abcsize=figstyle.FS_PANEL)
    figstyle.apply_typography(axes)
    save(fig, "fig03_selective_translation", transparent=True)


def make_scope_boundary_figure() -> None:
    fig, ax = uplt.subplots(refwidth=4.5, refaspect=2.8)
    ax.plot([1985.5, 1992.5], [0, 0], color=figstyle.REFERENCE, lw=1.2, zorder=0)
    ax.scatter(
        [1987], [0], s=85, marker="o", color=figstyle.ADOPTION,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.scatter(
        [1991], [0], s=90, marker="D", color=figstyle.CONTRAST,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.plot([1987, 1987], [0.04, 0.42], color=figstyle.ADOPTION, lw=1.0)
    ax.plot([1991, 1991], [-0.04, -0.42], color=figstyle.CONTRAST, lw=1.0)
    ax.text(
        1987, 0.47,
        "Recommendation XIV-3 (1987)\n"
        "Safeguards for scientific drilling\n"
        "annual ATCM output → Drilling",
        ha="center", va="bottom", fontsize=8.5, color=figstyle.ADOPTION,
    )
    ax.text(
        1991, -0.47,
        "Madrid Protocol, SATCM XI-4 (1991)\n"
        "external constitutional instrument\n"
        "prohibition → Mineral resources; research exception → Drilling context",
        ha="center", va="top", fontsize=8.5, color=figstyle.CONTRAST,
    )
    ax.format(
        title="Annual outputs and constitutional instruments are separate outcome layers",
        xlim=(1985.3, 1992.8), ylim=(-0.95, 0.95),
        xlocator=[1987, 1991], ylocator=[],
        xlabel="Year", grid=False, yspineloc="neither",
        titlesize=figstyle.FS_TITLE,
    )
    figstyle.apply_typography([ax])
    save(fig, "exploratory_outcome_scope_boundary")


def nearest_position(trajectory: pd.DataFrame, year: int) -> pd.Series | None:
    if trajectory.empty:
        return None
    exact = trajectory[trajectory["year"] == year]
    if not exact.empty:
        return exact.iloc[0]
    distance = (trajectory["year"] - year).abs()
    if distance.min() <= 2:
        return trajectory.loc[distance.idxmin()]
    return None


def make_actor_trajectory_figure() -> None:
    topic_positions = pd.read_csv(ROOT / "output/fig45_portfolio_space_ridgelines_topic_order.csv")
    submitted = load_submitted_with_fallback()
    lookup = lineage._canonical_topic_lookup(topic_positions["topic"].tolist())
    positions = rolling_actor_positions_1d(submitted, topic_positions, lookup)
    events = pd.read_csv(OUTDIR / "verified_actor_outcome_events_independent.csv")

    fig, axes = uplt.subplots(
        ncols=3, refwidth=2.85, refaspect=1.0, share=False, wspace=3.4,
    )
    for ax, actor_name in zip(axes, EXAMPLE_ACTORS):
        color = figstyle.ACTOR_EXAMPLE_COLORS[actor_name]
        trajectory = positions[
            positions["actor"].eq(actor_name)
            & positions["year"].between(1961, 2025)
        ].copy()
        blocks = trajectory["year"].diff().fillna(1).gt(1).cumsum()
        for _, block in trajectory.groupby(blocks):
            ax.plot(block["year"], block["position"], color=color, lw=2.1, zorder=3)
        ax.scatter(
            trajectory["year"], trajectory["position"], s=13,
            color=color, edgecolor="white", linewidth=0.35, zorder=4,
        )
        actor_events = events[events["actor"].eq(actor_name)].drop_duplicates("outcome_id")
        year_counts = actor_events.groupby("year").size().to_dict()
        year_seen: dict[int, int] = collections.defaultdict(int)
        for event in actor_events.itertuples(index=False):
            state = nearest_position(trajectory, int(event.year))
            if state is None:
                continue
            n = int(year_counts[int(event.year)])
            rank = year_seen[int(event.year)]
            year_seen[int(event.year)] += 1
            event_year = float(event.year) + (rank - (n - 1) / 2) * 0.30
            instrument_color = INSTRUMENT_COLORS.get(event.instrument, "#667085")
            ax.plot(
                [event_year, event_year], [state["position"], event.expected_x],
                color=instrument_color, lw=0.85, alpha=0.50, zorder=2,
            )
            ax.scatter(
                [event_year], [event.expected_x], marker="D", s=30,
                color=instrument_color, edgecolor="white", linewidth=0.5, zorder=6,
            )
        count = int(actor_events["outcome_id"].nunique())
        ax.format(
            title=f"{actor_name}: {count} verified outcome links",
            xlabel="Year", ylabel="Position on the continuous\nconcern axis",
            xlim=(1960, 2026), xlocator=[1961, 1980, 2000, 2020],
            ylim=(-0.03, 1.03), ylocator=[0, 0.25, 0.5, 0.75, 1],
            grid=False, titlesize=figstyle.FS_TITLE,
        )
    for ax in axes[1:]:
        ax.format(ylabel="", yticklabels=[])
    handles = [
        Line2D([0], [0], marker="D", linestyle="none", markersize=6,
               markerfacecolor=INSTRUMENT_COLORS[name], markeredgecolor="white", label=name)
        for name in lineage.INSTRUMENTS
    ]
    axes[1].legend(handles=handles, loc="b", ncols=4, frame=False)
    fig.format(
        abc="A", abcloc="ul", abcsize=figstyle.FS_PANEL,
        suptitle="Verified contributions reach independently classified outcome concerns",
        suptitlesize=14,
    )
    figstyle.apply_typography(axes)
    save(fig, "exploratory_independent_outcome_trajectories")


def main() -> None:
    make_evidence_figure()
    make_scope_boundary_figure()
    print("wrote fig03_selective_translation.[pdf|png]")
    print("wrote exploratory_outcome_scope_boundary.[pdf|png]")


if __name__ == "__main__":
    main()
