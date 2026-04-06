"""Slide frames explaining the construction of the space of concerns
via a bipartite country–topic graph.

Frames
------
0  Structure only      — flags (bottom) and topic labels (top), no edges
1  Raw submissions     — all edges, width ∝ submission count
2  RPA highlight       — specialised edges (RPA > 1) coloured, rest dimmed
3  Co-specialisation   — one country highlighted; its specialised topics glow
4  Proximity link      — add topic–topic proximity edge for the highlighted pair
5  Full proximity      — all topic–topic proximity edges (space of concerns seed)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import numpy as np
import ultraplot as uplt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from utils import (
    compute_product_space,
    get_rca,
    load_data,
    standardize_index_labels,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
FLAG_DIR = Path("assets/flags")
OUT_DIR  = Path("figures")

COUNTRIES = [
    "United Kingdom", "United States", "Australia", "Chile",
    "New Zealand", "Argentina", "Norway", "Russian Federation",
    "France", "Germany",
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

# Country to spotlight for the co-specialisation frame
SPOTLIGHT = "Germany"
# Topics Germany is known to specialise in
SPOTLIGHT_TOPICS = ["Science issues", "Environmental Monitoring and Reporting"]

RPA_THRESHOLD = 1.0
SLIDE_W, SLIDE_H = 13.0, 6.0   # wider for better topic spacing
DPI = 180

# Colours
COL_EDGE_RAW   = "#b0b0b0"
COL_EDGE_DIM   = "#e0e0e0"
COL_EDGE_RPA   = "#3f88c5"
COL_EDGE_SPOT  = "#e07b39"
COL_TOPIC_NODE = "#f5f5f5"
COL_TOPIC_HIGH = "#ffe0b2"
COL_PROX_EDGE  = "#2a9d55"

FLAG_ZOOM   = 0.048
NODE_RADIUS = 0.022

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
TOPIC_Y    = 0.78
COUNTRY_Y  = 0.16
LABEL_PAD  = 0.052


def _topic_x(i: int, n: int) -> float:
    margin = 0.06
    return margin + i * (1 - 2 * margin) / (n - 1)


def _country_x(i: int, n: int) -> float:
    margin = 0.05
    return margin + i * (1 - 2 * margin) / (n - 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load() -> tuple:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    counts = counts.groupby(level=0).sum() if counts.index.has_duplicates else counts

    sub = counts.loc[
        counts.index.isin(TOPICS),
        [c for c in COUNTRIES if c in counts.columns],
    ].reindex(index=TOPICS, columns=COUNTRIES).fillna(0)

    rpa = get_rca(sub)
    phi = compute_product_space(rpa, threshold=RPA_THRESHOLD)
    return sub, rpa, phi


# ---------------------------------------------------------------------------
# Flag loading
# ---------------------------------------------------------------------------
def _load_flag(country: str) -> np.ndarray | None:
    for suffix in ("_flag.png", "_logo.png"):
        p = FLAG_DIR / f"{country}{suffix}"
        if p.exists():
            img = Image.open(p).convert("RGBA")
            return np.array(img)
    return None


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _draw_topic_node(ax, x: float, label: str, highlight: bool = False) -> None:
    fc = COL_TOPIC_HIGH if highlight else COL_TOPIC_NODE
    circle = mpatches.Circle(
        (x, TOPIC_Y), radius=NODE_RADIUS,
        facecolor=fc, edgecolor="#555555", linewidth=0.8, zorder=4,
    )
    ax.add_patch(circle)
    txt = ax.text(
        x, TOPIC_Y + LABEL_PAD,
        label.replace(" and ", "\n& ").replace(" General", "\n(general)"),
        ha="center", va="bottom", fontsize=8, zorder=5,
        multialignment="center",
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])


_COUNTRY_LABELS = {
    "Russian Federation": "Russia",
    "United Kingdom":     "United\nKingdom",
    "United States":      "United\nStates",
    "New Zealand":        "New\nZealand",
}


def _draw_flag(ax, x: float, country: str, dim: bool = False) -> None:
    img = _load_flag(country)
    label = _COUNTRY_LABELS.get(country, country)
    if img is None:
        ax.text(x, COUNTRY_Y, label, ha="center", va="center",
                fontsize=8, zorder=5)
        return
    if dim:
        arr = img.astype(float)
        arr[..., :3] = arr[..., :3] * 0.3 + 200 * 0.7
        img = arr.astype(np.uint8)
    oi = OffsetImage(img, zoom=FLAG_ZOOM)
    ab = AnnotationBbox(oi, (x, COUNTRY_Y), frameon=False, zorder=4)
    ax.add_artist(ab)
    txt = ax.text(x, COUNTRY_Y - LABEL_PAD * 1.1, label,
                  ha="center", va="top", fontsize=8, zorder=5,
                  multialignment="center")
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])


def _draw_edge(ax, cx: float, tx: float,
               alpha: float = 1.0, color: str = COL_EDGE_RAW,
               lw: float = 0.8, zorder: int = 2) -> None:
    ax.plot([cx, tx], [COUNTRY_Y + NODE_RADIUS * 0.5, TOPIC_Y - NODE_RADIUS],
            color=color, alpha=alpha, lw=lw, zorder=zorder,
            solid_capstyle="round")


def _draw_prox_edge(ax, x0: float, x1: float, weight: float = 1.0) -> None:
    """Curved topic–topic proximity edge, arcing well above topic labels."""
    import matplotlib.patches as mp
    # rad controls arc height: larger distance → flatter arc
    dist = abs(x1 - x0)
    rad = -(0.35 + 0.4 * dist)
    style = mp.FancyArrowPatch(
        (x0, TOPIC_Y + NODE_RADIUS + LABEL_PAD + 0.04),
        (x1, TOPIC_Y + NODE_RADIUS + LABEL_PAD + 0.04),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-",
        linewidth=1.0 + 2.5 * weight,
        color=COL_PROX_EDGE,
        alpha=0.85,
        zorder=3,
    )
    ax.add_patch(style)


def _base_ax() -> tuple:
    fig, ax = uplt.subplots(refwidth=SLIDE_W, refaspect=SLIDE_W / SLIDE_H)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in [(".pdf", {"transparent": True}),
                    (".png", {"dpi": DPI, "transparent": True})]:
        fig.savefig(OUT_DIR / f"{stem}{ext}", bbox_inches="tight", **kw)
    print(f"Wrote {OUT_DIR}/{stem}.png")
    uplt.close(fig)


def _add_title(ax, title: str) -> None:
    ax.text(0.5, 0.97, title, ha="center", va="top",
            fontsize=13, fontweight="bold", color="#222222", zorder=10,
            transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------
def frame0_structure(counts, tx, cx) -> None:
    """Topics and flags only — no edges."""
    fig, ax = _base_ax()
    _add_title(ax, "Countries submit working papers to topics in the ATS")
    for i, t in enumerate(TOPICS):
        _draw_topic_node(ax, tx[i], t)
    for i, c in enumerate(COUNTRIES):
        _draw_flag(ax, cx[i], c)
    _save(fig, "fig_bipartite_f0_structure")


def frame1_raw(counts, tx, cx) -> None:
    """All submission edges, width ∝ count."""
    fig, ax = _base_ax()
    _add_title(ax, "Each edge shows submission volume between country and topic")
    max_count = counts.values.max()
    for i, t in enumerate(TOPICS):
        for j, c in enumerate(COUNTRIES):
            v = counts.loc[t, c]
            if v > 0:
                lw = 0.3 + 2.5 * (v / max_count)
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_RAW, lw=lw, alpha=0.55)
    for i, t in enumerate(TOPICS):
        _draw_topic_node(ax, tx[i], t)
    for i, c in enumerate(COUNTRIES):
        _draw_flag(ax, cx[i], c)
    _save(fig, "fig_bipartite_f1_raw")


def frame2_rpa(counts, rpa, tx, cx) -> None:
    """Highlight RPA > 1 edges; dim the rest."""
    fig, ax = _base_ax()
    _add_title(ax, "Revealed Policy Advantage (RPA > 1) marks above-average specialisation")
    max_count = counts.values.max()
    for i, t in enumerate(TOPICS):
        for j, c in enumerate(COUNTRIES):
            v = counts.loc[t, c]
            if v == 0:
                continue
            specialised = rpa.loc[t, c] > RPA_THRESHOLD
            if specialised:
                lw = 0.8 + 2.0 * (v / max_count)
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_RPA,
                           lw=lw, alpha=0.85, zorder=3)
            else:
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_DIM,
                           lw=0.4, alpha=0.4, zorder=1)
    for i, t in enumerate(TOPICS):
        _draw_topic_node(ax, tx[i], t)
    for i, c in enumerate(COUNTRIES):
        _draw_flag(ax, cx[i], c)

    handles = [
        Line2D([0], [0], color=COL_EDGE_RPA, lw=2.0, label="RPA > 1  (specialised)"),
        Line2D([0], [0], color=COL_EDGE_DIM, lw=1.0, label="RPA ≤ 1"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="none")
    _save(fig, "fig_bipartite_f2_rpa")


def frame3_spotlight(counts, rpa, tx, cx) -> None:
    """Highlight one country and its specialised topics."""
    fig, ax = _base_ax()
    _add_title(ax,
               f"{SPOTLIGHT} specialises in both highlighted topics →\n"
               "those topics are structurally linked")
    max_count = counts.values.max()
    si = COUNTRIES.index(SPOTLIGHT)

    for i, t in enumerate(TOPICS):
        for j, c in enumerate(COUNTRIES):
            v = counts.loc[t, c]
            if v == 0:
                continue
            is_spot = (c == SPOTLIGHT) and (rpa.loc[t, c] > RPA_THRESHOLD)
            if is_spot:
                lw = 1.0 + 2.0 * (v / max_count)
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_SPOT,
                           lw=lw, alpha=0.95, zorder=4)
            else:
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_DIM,
                           lw=0.4, alpha=0.25, zorder=1)

    for i, t in enumerate(TOPICS):
        highlight = t in SPOTLIGHT_TOPICS
        _draw_topic_node(ax, tx[i], t, highlight=highlight)

    for i, c in enumerate(COUNTRIES):
        _draw_flag(ax, cx[i], c, dim=(c != SPOTLIGHT))

    _save(fig, "fig_bipartite_f3_spotlight")


def frame4_one_prox(counts, rpa, phi, tx, cx) -> None:
    """Add a single proximity edge between the two spotlight topics."""
    fig, ax = _base_ax()
    _add_title(ax,
               "Co-specialisation creates proximity between topics —\n"
               "the building block of the space of concerns")

    max_count = counts.values.max()
    si = COUNTRIES.index(SPOTLIGHT)

    for i, t in enumerate(TOPICS):
        for j, c in enumerate(COUNTRIES):
            v = counts.loc[t, c]
            if v == 0:
                continue
            is_spot = (c == SPOTLIGHT) and (rpa.loc[t, c] > RPA_THRESHOLD)
            if is_spot:
                lw = 1.0 + 2.0 * (v / max_count)
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_SPOT,
                           lw=lw, alpha=0.95, zorder=4)
            else:
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_DIM,
                           lw=0.4, alpha=0.25, zorder=1)

    # Draw single proximity edge
    ti0 = TOPICS.index(SPOTLIGHT_TOPICS[0])
    ti1 = TOPICS.index(SPOTLIGHT_TOPICS[1])
    weight = float(phi.loc[SPOTLIGHT_TOPICS[0], SPOTLIGHT_TOPICS[1]])
    _draw_prox_edge(ax, tx[ti0], tx[ti1], weight=weight)

    for i, t in enumerate(TOPICS):
        highlight = t in SPOTLIGHT_TOPICS
        _draw_topic_node(ax, tx[i], t, highlight=highlight)
    for i, c in enumerate(COUNTRIES):
        _draw_flag(ax, cx[i], c, dim=(c != SPOTLIGHT))

    ax.annotate(
        "proximity\nedge",
        xy=((tx[ti0] + tx[ti1]) / 2, TOPIC_Y + NODE_RADIUS + 0.04),
        xytext=((tx[ti0] + tx[ti1]) / 2, TOPIC_Y + 0.22),
        ha="center", va="bottom", fontsize=8, color=COL_PROX_EDGE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COL_PROX_EDGE, lw=1.0),
    )
    _save(fig, "fig_bipartite_f4_one_proximity")


def frame5_full_prox(counts, rpa, phi, tx, cx) -> None:
    """All topic–topic proximity edges above the bipartite graph."""
    fig, ax = _base_ax()
    _add_title(ax,
               "Doing this for all actors and all topic pairs\n"
               "produces the space of concerns")

    max_count = counts.values.max()
    # Dim all bipartite edges
    for i, t in enumerate(TOPICS):
        for j, c in enumerate(COUNTRIES):
            v = counts.loc[t, c]
            if v > 0:
                lw = 0.3 + 1.5 * (v / max_count)
                _draw_edge(ax, cx[j], tx[i], color=COL_EDGE_DIM,
                           lw=lw, alpha=0.25, zorder=1)

    # Draw all proximity edges above threshold
    phi_sub = phi.loc[TOPICS, TOPICS]
    threshold = float(np.percentile(phi_sub.values[phi_sub.values > 0], 40))
    for i, t0 in enumerate(TOPICS):
        for j, t1 in enumerate(TOPICS):
            if j <= i:
                continue
            w = float(phi_sub.loc[t0, t1])
            if w >= threshold:
                _draw_prox_edge(ax, tx[i], tx[j], weight=w)

    for i, t in enumerate(TOPICS):
        _draw_topic_node(ax, tx[i], t)
    for i, c in enumerate(COUNTRIES):
        _draw_flag(ax, cx[i], c)

    handles = [
        Line2D([0], [0], color=COL_PROX_EDGE, lw=2.5, label="Topic proximity (φ)"),
        Line2D([0], [0], color=COL_EDGE_DIM, lw=1.0, label="Submissions"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="none")
    _save(fig, "fig_bipartite_f5_full_proximity")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    counts, rpa, phi = _load()

    n_t = len(TOPICS)
    n_c = len(COUNTRIES)
    tx = [_topic_x(i, n_t) for i in range(n_t)]
    cx = [_country_x(i, n_c) for i in range(n_c)]

    frame0_structure(counts, tx, cx)
    frame1_raw(counts, tx, cx)
    frame2_rpa(counts, rpa, tx, cx)
    frame3_spotlight(counts, rpa, tx, cx)
    frame4_one_prox(counts, rpa, phi, tx, cx)
    frame5_full_prox(counts, rpa, phi, tx, cx)


if __name__ == "__main__":
    main()
