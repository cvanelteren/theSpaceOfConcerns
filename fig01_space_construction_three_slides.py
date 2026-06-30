"""Compact 3-slide explainer for constructing the ATS space of concerns."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import ultraplot as uplt

import fig01_space_construction_slides as base

OUT_PREFIX = "fig01_space_construction_compact"
SLIDE1_TOPICS = [
    ("Science issues", "Science"),
    ("Tourism and NG_Activities", "Tourism"),
    ("Environmental Protection General", "Protection"),
    ("Environmental Monitoring and Reporting", "Monitoring"),
    ("Exchange of Information", "Exchange"),
    ("Climate Change", "Climate"),
]
SLIDE1_COUNTRIES = [
    "United Kingdom",
    "Australia",
    "New Zealand",
    "Germany",
    "United States",
    "Norway",
]


def _fig():
    fig = uplt.figure(figsize=base.FIGSIZE)
    fig.patch.set_facecolor("white")
    return fig


def _title(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=base.COLORS["text"],
    )
    if subtitle:
        fig.text(
            0.5,
            0.925,
            subtitle,
            ha="center",
            va="top",
            fontsize=11.5,
            color=base.COLORS["muted"],
        )


def _panel(ax, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.91,
        title,
        ha="center",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=base.COLORS["text"],
        transform=ax.transAxes,
    )


def _save(fig, stem: str) -> None:
    base._save(fig, stem)


def _draw_rpa_panel(ax, counts, rpa) -> None:
    tx, cx = base._bipartite_positions()
    max_count = float(counts.to_numpy(dtype=float).max())
    for topic in base.TOPICS:
        for country in base.COUNTRIES:
            val = float(counts.loc[topic, country])
            if val <= 0:
                continue
            specialised = float(rpa.loc[topic, country]) > base.RPA_THRESHOLD
            base._draw_bipartite_edge(
                ax,
                cx[country],
                tx[topic],
                color=base.COLORS["special_edge"] if specialised else base.COLORS["dim_edge"],
                alpha=0.85 if specialised else 0.35,
                lw=(0.8 + 2.6 * (val / max_count)) if specialised else 0.7,
                zorder=3 if specialised else 1,
            )
    for topic in base.TOPICS:
        base._draw_topic_node(ax, topic, tx[topic])
    for country in base.COUNTRIES:
        base._draw_flag(ax, country, cx[country])


def _slide1_positions():
    tx = {
        topic: (0.12 + i * 0.14, 0.58)
        for i, (topic, _) in enumerate(SLIDE1_TOPICS)
    }
    cx = {
        country: (0.16 + i * 0.13, 0.14)
        for i, country in enumerate(SLIDE1_COUNTRIES)
    }
    return tx, cx


def _draw_slide1_topic(ax, topic: str, label: str, xy: tuple[float, float]) -> None:
    x, y = xy
    patch = mpatches.Circle(
        (x, y),
        0.026,
        facecolor=base.COLORS["topic_fill"],
        edgecolor="#666666",
        linewidth=1.4,
        transform=ax.transAxes,
        zorder=10,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y + base._topic_label_offset(topic),
        label,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=base.COLORS["text"],
        transform=ax.transAxes,
        zorder=11,
    )


def _draw_slide1_flag(ax, country: str, xy: tuple[float, float]) -> None:
    x, y = xy
    img = base._load_flag(country)
    if img is not None:
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.065),
            (x, y),
            frameon=False,
            xycoords=ax.transAxes,
            zorder=9,
        )
        ax.add_artist(ab)
    ax.text(
        x,
        y - base._country_label_offset(country),
        base._country_display(country),
        ha="center",
        va="top",
        fontsize=9.5,
        color=base.COLORS["text"],
        transform=ax.transAxes,
        zorder=10,
    )


def _draw_slide1_bipartite(ax, counts, rpa=None) -> None:
    tx, cx = _slide1_positions()
    topic_names = [t for t, _ in SLIDE1_TOPICS]
    max_count = float(counts.to_numpy(dtype=float).max())
    for topic in topic_names:
        for country in SLIDE1_COUNTRIES:
            val = float(counts.loc[topic, country])
            if val <= 0:
                continue
            if rpa is None:
                color = base.COLORS["raw_edge"]
                alpha = 0.55
                lw = 0.5 + 3.0 * (val / max_count)
            else:
                keep = float(rpa.loc[topic, country]) > base.RPA_THRESHOLD
                color = base.COLORS["special_edge"] if keep else base.COLORS["dim_edge"]
                alpha = 0.85 if keep else 0.35
                lw = (0.8 + 2.6 * (val / max_count)) if keep else 0.7
            base._draw_bipartite_edge(
                ax,
                cx[country],
                tx[topic],
                color=color,
                alpha=alpha,
                lw=lw,
                zorder=3 if rpa is not None else 1,
            )
    for topic, label in SLIDE1_TOPICS:
        _draw_slide1_topic(ax, topic, label, tx[topic])
    for country in SLIDE1_COUNTRIES:
        _draw_slide1_flag(ax, country, cx[country])


def slide1(payload: dict[str, object]) -> None:
    fig, axs = uplt.subplots(
        ncols=2,
        figsize=base.FIGSIZE,
        share=0,
        left=0.04,
        right=0.98,
        bottom=0.10,
        top=0.84,
        wspace=0.06,
    )
    fig.patch.set_facecolor("white")
    _title(
        fig,
        "1. From submissions to specialised actor-topic ties",
        "Raw ATS working papers define the bipartite graph; the proximity construction keeps only above-average ties.",
    )
    ax1, ax2 = axs
    _panel(ax1, "Observed submissions")
    _panel(ax2, r"Specialised ties only ($RPA > 1$)")
    compact_topics = [t for t, _ in SLIDE1_TOPICS]
    compact_counts = payload["counts"].reindex(index=compact_topics, columns=SLIDE1_COUNTRIES)
    compact_rpa = payload["rpa"].reindex(index=compact_topics, columns=SLIDE1_COUNTRIES)
    _draw_slide1_bipartite(ax1, compact_counts)
    _draw_slide1_bipartite(ax2, compact_counts, compact_rpa)
    fig.text(
        0.5,
        0.01,
        r"Blue ties are retained in $M_{ai}=1[\mathrm{RPA}_{ai}>1]$; faint ties are dropped before topic-topic co-specialisation is computed.",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color=base.COLORS["muted"],
    )
    _save(fig, f"{OUT_PREFIX}_s1_bipartite_to_rpa")


def slide2(payload: dict[str, object]) -> None:
    fig, axs = uplt.subplots(
        ncols=2,
        figsize=base.FIGSIZE,
        share=0,
        left=0.06,
        right=0.98,
        bottom=0.14,
        top=0.84,
        wspace=0.18,
    )
    fig.patch.set_facecolor("white")
    spotlight = ", ".join(
        base._country_display(c).replace("\n", " ")
        for c in payload["spotlight_countries"]
    )
    _title(
        fig,
        "2. Co-specialisation becomes symmetric topic proximity",
        f"Highlighted pair: {base.SPOTLIGHT_TOPICS[0].replace('_', ' ')} and {base.SPOTLIGHT_TOPICS[1].replace('_', ' ')}; shared specialised actors = {spotlight}.",
    )
    ax1, ax2 = axs
    pair_idx = (
        base.TOPICS.index(base.SPOTLIGHT_TOPICS[0]),
        base.TOPICS.index(base.SPOTLIGHT_TOPICS[1]),
    )
    im1 = base._draw_heatmap(
        ax1,
        payload["co_counts"].to_numpy(dtype=float),
        base.TOPICS,
        cmap="Blues",
        vmin=0.0,
        vmax=float(payload["co_counts"].to_numpy(dtype=float).max()),
        fmt=".0f",
        title=r"Shared specialised actors, $C_{ij}$",
        highlight_pair=pair_idx,
    )
    im2 = base._draw_heatmap(
        ax2,
        payload["phi"].to_numpy(dtype=float),
        base.TOPICS,
        cmap="YlGn",
        vmin=0.0,
        vmax=float(payload["phi"].to_numpy(dtype=float).max()),
        fmt=".2f",
        title=r"Symmetric proximity, $\phi_{ij}$",
        highlight_pair=pair_idx,
    )
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.78, pad=0.02)
    cbar1.ax.tick_params(labelsize=9)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.78, pad=0.02)
    cbar2.ax.tick_params(labelsize=9)
    fig.text(
        0.25,
        0.05,
        r"$C_{ij}=\sum_a M_{ai}M_{aj}$",
        ha="center",
        va="bottom",
        fontsize=12,
        color=base.COLORS["accent"],
    )
    fig.text(
        0.75,
        0.05,
        r"$\phi_{ij}=\min(C_{ij}/u_i,\;C_{ij}/u_j)$, with $u_i=\sum_a M_{ai}$",
        ha="center",
        va="bottom",
        fontsize=12,
        color=base.COLORS["accent"],
    )
    _save(fig, f"{OUT_PREFIX}_s2_counts_to_phi")


def _draw_subset_network(ax, payload: dict[str, object]) -> None:
    pos = base._subset_network_positions()
    graph = {}
    for i, t0 in enumerate(base.TOPICS):
        for j, t1 in enumerate(base.TOPICS):
            if j <= i:
                continue
            weight = float(payload["phi"].loc[t0, t1])
            if weight >= base.NETWORK_THRESHOLD:
                graph[(t0, t1)] = weight
    max_u = float(payload["ubiquity"].max())
    for (u, v), weight in graph.items():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        highlight = {u, v} == set(base.SPOTLIGHT_TOPICS)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=base.COLORS["spot_edge"] if highlight else base.COLORS["phi_edge_soft"],
            lw=1.0 + 4.2 * weight,
            alpha=0.95 if highlight else 0.8,
            solid_capstyle="round",
            transform=ax.transAxes,
            zorder=2,
        )
    for topic in base.TOPICS:
        x, y = pos[topic]
        radius = 0.018 + 0.020 * (float(payload["ubiquity"].loc[topic]) / max_u)
        patch = mpatches.Circle(
            (x, y),
            radius,
            facecolor=base.COLORS["topic_highlight"] if topic in base.SPOTLIGHT_TOPICS else "#eef5f7",
            edgecolor="#5d6770",
            linewidth=1.2,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.add_patch(patch)
        txt = ax.text(
            x,
            y + radius + 0.02,
            base._topic_display(topic),
            ha="center",
            va="bottom",
            fontsize=10,
            color=base.COLORS["text"],
            transform=ax.transAxes,
            zorder=4,
        )
        txt.set_path_effects(
            [base.path_effects.withStroke(linewidth=3, foreground="white", alpha=0.95)]
        )


def slide3(payload: dict[str, object]) -> None:
    fig, axs = uplt.subplots(
        ncols=2,
        figsize=base.FIGSIZE,
        share=0,
        left=0.04,
        right=0.98,
        bottom=0.10,
        top=0.84,
        wspace=0.10,
    )
    fig.patch.set_facecolor("white")
    _title(
        fig,
        "3. High-$\\phi$ topic pairs become links in the space of concerns",
        "The same proximity rule first yields a topic graph on a subset, then the full ATS network when applied to all topics.",
    )
    fig.text(0.24, 0.885, "Example topic network", ha="center", va="top", fontsize=12.5, fontweight="bold", color=base.COLORS["text"])
    fig.text(0.75, 0.885, "Full ATS topic set", ha="center", va="top", fontsize=12.5, fontweight="bold", color=base.COLORS["text"])
    ax1, ax2 = axs
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    _draw_subset_network(ax1, payload)
    img = np.asarray(Image.open(base.FULL_SPACE_PNG).convert("RGBA"))
    ax2.imshow(img, extent=(0.02, 0.98, 0.02, 0.96), aspect="auto", zorder=1)
    ax2.axis("off")
    handles = [
        Line2D([0], [0], color=base.COLORS["phi_edge_soft"], lw=3, label=r"topic link ($\phi$)"),
        Line2D([0], [0], color=base.COLORS["spot_edge"], lw=3, label="highlighted pair"),
    ]
    fig.legend(
        handles=handles,
        loc="b",
        frameon=False,
        fontsize=11.5,
        ncol=2,
    )
    fig.text(
        0.5,
        0.02,
        r"Using the same $\phi$ construction on the full topic set yields the paper's space-of-concerns network.",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color=base.COLORS["muted"],
    )
    _save(fig, f"{OUT_PREFIX}_s3_network_to_full_space")


def main() -> None:
    payload = base._load_subset()
    slide1(payload)
    slide2(payload)
    slide3(payload)


if __name__ == "__main__":
    main()
