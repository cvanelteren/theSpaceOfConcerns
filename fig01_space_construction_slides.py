"""Slide sequence explaining how the ATS space of concerns is constructed.

Outputs (in slides/):
  fig01_space_construction_s1_raw_bipartite.(png|pdf)
  fig01_space_construction_s2_rpa_filter.(png|pdf)
  fig01_space_construction_s3_pair_example.(png|pdf)
  fig01_space_construction_s4_counts_to_phi.(png|pdf)
  fig01_space_construction_s5_topic_network.(png|pdf)
  fig01_space_construction_s6_full_space.(png|pdf)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from utils import (
    compute_product_space,
    get_rca,
    load_data,
    load_saved_layout_positions,
    standardize_index_labels,
)

Image.MAX_IMAGE_PIXELS = None

DATA_FP = Path("antarctic-database-go/data/processed/document-summary.parquet")
FLAG_DIR = Path("assets/flags")
SLIDES_DIR = Path("slides")
FULL_SPACE_PNG = Path("figures/fig01_space_of_concerns_topology.png")

COUNTRIES = [
    "United Kingdom",
    "Australia",
    "New Zealand",
    "Germany",
    "United States",
    "Argentina",
    "Norway",
    "Chile",
]

TOPICS = [
    "Science issues",
    "Tourism and NG_Activities",
    "Environmental Protection General",
    "Environmental Monitoring and Reporting",
    "Exchange of Information",
    "Climate Change",
    "Safety and Operations in Antarctica",
    "Educational issues",
]

SPOTLIGHT_TOPICS = (
    "Environmental Protection General",
    "Environmental Monitoring and Reporting",
)

DPI = 300
FIGSIZE = (13.333, 7.5)
RPA_THRESHOLD = 1.0
NETWORK_THRESHOLD = 1.0 / 3.0
FLAG_ZOOM = 0.095

COLORS = {
    "raw_edge": "#b9c0c9",
    "dim_edge": "#d9dde2",
    "special_edge": "#2b7bba",
    "spot_edge": "#d96b2b",
    "topic_fill": "#fbfbfb",
    "topic_highlight": "#ffe6c7",
    "country_dim": 0.45,
    "phi_edge": "#2a9d55",
    "phi_edge_soft": "#7bc48a",
    "text": "#1a1a1a",
    "muted": "#56606b",
    "box": "#f4f6f8",
    "accent": "#0b6e99",
}

COUNTRY_LABELS = {
    "United Kingdom": "United\nKingdom",
    "United States": "United\nStates",
    "New Zealand": "New\nZealand",
}

TOPIC_LABEL_STAGGER = {
    topic: offset
    for topic, offset in zip(
        TOPICS,
        (0.000, 0.070, 0.020, 0.090, 0.012, 0.065, 0.022, 0.055),
    )
}

COUNTRY_LABEL_STAGGER = {
    country: offset
    for country, offset in zip(
        COUNTRIES,
        (0.000, 0.018, 0.006, 0.020, 0.004, 0.018, 0.006, 0.020),
    )
}


def _topic_display(name: str) -> str:
    return (
        str(name)
        .replace("_", " ")
        .replace(" and ", "\n& ")
        .replace(" General", "\n(general)")
        .replace(" in Antarctica", "\nin Antarctica")
    )


def _country_display(name: str) -> str:
    return COUNTRY_LABELS.get(name, name)


def _topic_label_offset(topic: str) -> float:
    return 0.055 + TOPIC_LABEL_STAGGER.get(topic, 0.0)


def _country_label_offset(country: str) -> float:
    return 0.055 + COUNTRY_LABEL_STAGGER.get(country, 0.0)


def _load_subset() -> dict[str, object]:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()
    subset = counts.reindex(index=TOPICS, columns=COUNTRIES).fillna(0.0)
    rpa = get_rca(subset)
    binary = (rpa > RPA_THRESHOLD).astype(int)
    co_counts = pd.DataFrame(
        binary.to_numpy(dtype=int) @ binary.to_numpy(dtype=int).T,
        index=TOPICS,
        columns=TOPICS,
    )
    ubiquity = binary.sum(axis=1)
    phi = compute_product_space(rpa, threshold=RPA_THRESHOLD).reindex(
        index=TOPICS, columns=TOPICS
    )
    spotlight_countries = [
        c for c in COUNTRIES if all(binary.loc[t, c] == 1 for t in SPOTLIGHT_TOPICS)
    ]
    return {
        "counts": subset,
        "rpa": rpa,
        "binary": binary,
        "co_counts": co_counts,
        "ubiquity": ubiquity,
        "phi": phi,
        "spotlight_countries": spotlight_countries,
    }


def _load_flag(country: str) -> np.ndarray | None:
    for suffix in ("_flag.png", "_logo.png"):
        path = FLAG_DIR / f"{country}{suffix}"
        if path.exists():
            return np.asarray(Image.open(path).convert("RGBA"))
    return None


def _canvas():
    fig, ax = uplt.subplots(figsize=FIGSIZE, share=0)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, stem: str) -> None:
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        path = SLIDES_DIR / f"{stem}{ext}"
        fig.savefig(
            path,
            dpi=DPI if ext == ".png" else None,
            bbox_inches="tight",
            transparent=True,
        )
        print(f"Wrote {path}")
    uplt.close(fig)


def _note_box(ax, x: float, y: float, text: str, *, width: float = 0.24) -> None:
    box = mpatches.FancyBboxPatch(
        (x, y),
        width,
        0.12,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#cfd6de",
        facecolor=COLORS["box"],
        transform=ax.transAxes,
        zorder=20,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.014,
        y + 0.06,
        text,
        ha="left",
        va="center",
        fontsize=18,
        color=COLORS["text"],
        transform=ax.transAxes,
        zorder=21,
    )


def _bipartite_positions() -> (
    tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]
):
    tx = {topic: (0.08 + i * 0.115, 0.76) for i, topic in enumerate(TOPICS)}
    cx = {country: (0.10 + i * 0.10, 0.14) for i, country in enumerate(COUNTRIES)}
    return tx, cx


def _draw_topic_node(
    ax, topic: str, xy: tuple[float, float], *, highlight: bool = False
) -> None:
    x, y = xy
    face = COLORS["topic_highlight"] if highlight else COLORS["topic_fill"]
    patch = mpatches.Circle(
        (x, y),
        0.022,
        facecolor=face,
        edgecolor="#666666",
        linewidth=1.4,
        transform=ax.transAxes,
        zorder=10,
    )
    ax.add_patch(patch)
    txt = ax.text(
        x,
        y + _topic_label_offset(topic),
        _topic_display(topic),
        ha="center",
        va="bottom",
        fontsize=18,
        color=COLORS["text"],
        transform=ax.transAxes,
        zorder=11,
    )
    txt.set_path_effects(
        [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.95)]
    )


def _draw_flag(ax, country: str, xy: tuple[float, float], *, dim: bool = False) -> None:
    x, y = xy
    img = _load_flag(country)
    label = _country_display(country)
    if img is None:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
            zorder=10,
        )
        return
    if dim:
        arr = img.astype(float)
        arr[..., :3] = arr[..., :3] * COLORS["country_dim"] + 255.0 * (
            1.0 - COLORS["country_dim"]
        )
        img = arr.astype(np.uint8)
    ab = AnnotationBbox(
        OffsetImage(img, zoom=FLAG_ZOOM),
        (x, y),
        frameon=False,
        xycoords=ax.transAxes,
        zorder=9,
    )
    ax.add_artist(ab)
    txt = ax.text(
        x,
        y - _country_label_offset(country),
        label,
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["text"],
        transform=ax.transAxes,
        zorder=10,
    )
    txt.set_path_effects(
        [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.95)]
    )


def _draw_bipartite_edge(
    ax,
    cxy: tuple[float, float],
    txy: tuple[float, float],
    *,
    color: str,
    alpha: float,
    lw: float,
    zorder: int = 1,
) -> None:
    ax.plot(
        [cxy[0], txy[0]],
        [cxy[1] + 0.04, txy[1] - 0.022],
        color=color,
        lw=lw,
        alpha=alpha,
        solid_capstyle="round",
        transform=ax.transAxes,
        zorder=zorder,
    )


def _draw_raw_bipartite(ax, counts: pd.DataFrame) -> None:
    tx, cx = _bipartite_positions()
    max_count = float(counts.to_numpy(dtype=float).max())
    for topic in TOPICS:
        for country in COUNTRIES:
            val = float(counts.loc[topic, country])
            if val <= 0:
                continue
            _draw_bipartite_edge(
                ax,
                cx[country],
                tx[topic],
                color=COLORS["raw_edge"],
                alpha=0.55,
                lw=0.5 + 3.0 * (val / max_count),
            )
    for topic in TOPICS:
        _draw_topic_node(ax, topic, tx[topic])
    for country in COUNTRIES:
        _draw_flag(ax, country, cx[country])


def slide1_raw(counts: pd.DataFrame) -> None:
    fig, ax = _canvas()
    _draw_raw_bipartite(ax, counts)
    _save(fig, "fig01_space_construction_s1_raw_bipartite")


def slide2_rpa(counts: pd.DataFrame, rpa: pd.DataFrame) -> None:
    fig, ax = _canvas()
    tx, cx = _bipartite_positions()
    max_count = float(counts.to_numpy(dtype=float).max())
    for topic in TOPICS:
        for country in COUNTRIES:
            val = float(counts.loc[topic, country])
            if val <= 0:
                continue
            specialised = float(rpa.loc[topic, country]) > RPA_THRESHOLD
            _draw_bipartite_edge(
                ax,
                cx[country],
                tx[topic],
                color=COLORS["special_edge"] if specialised else COLORS["dim_edge"],
                alpha=0.85 if specialised else 0.45,
                lw=(0.8 + 2.6 * (val / max_count)) if specialised else 0.8,
                zorder=3 if specialised else 1,
            )
    for topic in TOPICS:
        _draw_topic_node(ax, topic, tx[topic])
    for country in COUNTRIES:
        _draw_flag(ax, country, cx[country])
    _save(fig, "fig01_space_construction_s2_rpa_filter")


def slide3_pair_example(
    counts: pd.DataFrame,
    rpa: pd.DataFrame,
    spotlight_countries: list[str],
) -> None:
    fig, ax = _canvas()
    tx, cx = _bipartite_positions()
    max_count = float(counts.to_numpy(dtype=float).max())
    for topic in TOPICS:
        for country in COUNTRIES:
            val = float(counts.loc[topic, country])
            if val <= 0:
                continue
            highlight = (
                country in spotlight_countries
                and topic in SPOTLIGHT_TOPICS
                and float(rpa.loc[topic, country]) > RPA_THRESHOLD
            )
            _draw_bipartite_edge(
                ax,
                cx[country],
                tx[topic],
                color=COLORS["spot_edge"] if highlight else COLORS["dim_edge"],
                alpha=0.9 if highlight else 0.35,
                lw=(1.0 + 3.0 * (val / max_count)) if highlight else 0.8,
                zorder=4 if highlight else 1,
            )
    for topic in TOPICS:
        _draw_topic_node(ax, topic, tx[topic], highlight=topic in SPOTLIGHT_TOPICS)
    for country in COUNTRIES:
        _draw_flag(ax, country, cx[country], dim=country not in spotlight_countries)
    _save(fig, "fig01_space_construction_s3_pair_example")


def _draw_heatmap(
    ax,
    mat: np.ndarray,
    labels: list[str],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
    title: str,
    highlight_pair: tuple[int, int] | None = None,
) -> None:
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(
        [_topic_display(x).replace("\n", " ") for x in labels],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    ax.set_yticklabels(
        [_topic_display(x).replace("\n", " ") for x in labels], fontsize=9
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            text = format(val, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if val > 0.6 * vmax else COLORS["text"],
            )
    if highlight_pair is not None:
        i, j = highlight_pair
        for a, b in ((i, j), (j, i)):
            rect = mpatches.Rectangle(
                (b - 0.5, a - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=COLORS["spot_edge"],
                linewidth=2.5,
            )
            ax.add_patch(rect)
    return im


def slide4_counts_to_phi(
    co_counts: pd.DataFrame, phi: pd.DataFrame, ubiquity: pd.Series
) -> None:
    fig, axs = uplt.subplots(ncols=2, figsize=FIGSIZE, share=0)
    fig.patch.set_facecolor("white")
    ax1, ax2 = axs
    pair_idx = (TOPICS.index(SPOTLIGHT_TOPICS[0]), TOPICS.index(SPOTLIGHT_TOPICS[1]))
    im1 = _draw_heatmap(
        ax1,
        co_counts.to_numpy(dtype=float),
        TOPICS,
        cmap="Blues",
        vmin=0.0,
        vmax=float(co_counts.to_numpy(dtype=float).max()),
        fmt=".0f",
        title="",
        highlight_pair=pair_idx,
    )
    im2 = _draw_heatmap(
        ax2,
        phi.to_numpy(dtype=float),
        TOPICS,
        cmap="YlGn",
        vmin=0.0,
        vmax=float(phi.to_numpy(dtype=float).max()),
        fmt=".2f",
        title="",
        highlight_pair=pair_idx,
    )
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.ax.tick_params(labelsize=9)
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.ax.tick_params(labelsize=9)
    _save(fig, "fig01_space_construction_s4_counts_to_phi")

    for stem, mat, cmap, vmax, fmt in [
        (
            "fig01_space_construction_s4_counts_only",
            co_counts.to_numpy(dtype=float),
            "Blues",
            float(co_counts.to_numpy(dtype=float).max()),
            ".0f",
        ),
        (
            "fig01_space_construction_s4_phi_only",
            phi.to_numpy(dtype=float),
            "YlGn",
            float(phi.to_numpy(dtype=float).max()),
            ".2f",
        ),
    ]:
        fig_single, ax_single = uplt.subplots(figsize=(7.5, 7.5), share=0)
        fig_single.patch.set_facecolor("white")
        im = _draw_heatmap(
            ax_single,
            mat,
            TOPICS,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            fmt=fmt,
            title="",
            highlight_pair=pair_idx,
        )
        cbar = fig_single.colorbar(im, ax=ax_single, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        _save(fig_single, stem)


def _subset_network_positions() -> dict[str, np.ndarray]:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()
    rca = get_rca(counts)
    phi = compute_product_space(rca)
    graph = nx.from_pandas_adjacency(phi)
    pos_full = load_saved_layout_positions(graph)
    if not pos_full:
        subset_graph = nx.Graph()
        subset_graph.add_nodes_from(TOPICS)
        pos = nx.spring_layout(subset_graph, seed=1991)
        return {k: np.asarray(v, dtype=float) for k, v in pos.items()}
    subset = {topic: np.asarray(pos_full[topic], dtype=float) for topic in TOPICS}
    arr = np.array(list(subset.values()), dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.where((maxs - mins) > 1e-9, maxs - mins, 1.0)
    arr[:, 0] = 0.12 + 0.72 * (arr[:, 0] - mins[0]) / span[0]
    arr[:, 1] = 0.14 + 0.68 * (arr[:, 1] - mins[1]) / span[1]
    return {topic: arr[i] for i, topic in enumerate(TOPICS)}


def slide5_topic_network(phi: pd.DataFrame, ubiquity: pd.Series) -> None:
    fig, ax = _canvas()
    pos = _subset_network_positions()
    graph = nx.Graph()
    graph.add_nodes_from(TOPICS)
    for i, t0 in enumerate(TOPICS):
        for j, t1 in enumerate(TOPICS):
            if j <= i:
                continue
            weight = float(phi.loc[t0, t1])
            if weight >= NETWORK_THRESHOLD:
                graph.add_edge(t0, t1, weight=weight)
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        highlight = {u, v} == set(SPOTLIGHT_TOPICS)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=COLORS["spot_edge"] if highlight else COLORS["phi_edge_soft"],
            lw=1.0 + 4.2 * float(data["weight"]),
            alpha=0.95 if highlight else 0.8,
            solid_capstyle="round",
            transform=ax.transAxes,
            zorder=2,
        )
    max_u = float(ubiquity.max())
    for topic in TOPICS:
        x, y = pos[topic]
        radius = 0.018 + 0.020 * (float(ubiquity.loc[topic]) / max_u)
        patch = mpatches.Circle(
            (x, y),
            radius,
            facecolor=(
                COLORS["topic_highlight"] if topic in SPOTLIGHT_TOPICS else "#eef5f7"
            ),
            edgecolor="#5d6770",
            linewidth=1.2,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.add_patch(patch)
        txt = ax.text(
            x,
            y + radius + 0.02,
            _topic_display(topic),
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["text"],
            transform=ax.transAxes,
            zorder=4,
        )
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.95)]
        )
    _save(fig, "fig01_space_construction_s5_topic_network")


def slide6_full_space() -> None:
    fig, ax = _canvas()
    img = np.asarray(Image.open(FULL_SPACE_PNG).convert("RGBA"))
    ax.imshow(img, extent=(0.02, 0.98, 0.02, 0.98), aspect="auto", zorder=1)
    _save(fig, "fig01_space_construction_s6_full_space")


def main() -> None:
    payload = _load_subset()
    slide1_raw(payload["counts"])
    slide2_rpa(payload["counts"], payload["rpa"])
    slide3_pair_example(
        payload["counts"],
        payload["rpa"],
        payload["spotlight_countries"],
    )
    slide4_counts_to_phi(
        payload["co_counts"],
        payload["phi"],
        payload["ubiquity"],
    )
    slide5_topic_network(payload["phi"], payload["ubiquity"])
    slide6_full_space()


if __name__ == "__main__":
    main()
