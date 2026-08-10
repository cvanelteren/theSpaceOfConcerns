# %%
"""Main-text Figure 1: the space of concerns, with actors placed in it.

The main map uses the layout this figure originally shipped with -- Kamada-
Kawai snapped to the Antarctic silhouette's interior, point-reflected, rotated
45 degrees, spread past the coast -- because that is the version that reads as
an iconic surface. The silhouette is a display frame, not a data channel: the
caption states that nothing inside it encodes geography. Below, three small
multiples reuse the identical snapped layout to show where individual actors
sit in the space (topics with RPA>1, sized by RPA, coloured by mode), so
co-specialization and positioning are one visual statement.

Annotation stays light: a few halo landmark labels and the grey italic
structural callouts. No margin label columns, no leaders crossing the map,
no inset, no mode pills.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import contains_xy
from shapely.ops import unary_union

import figstyle
from utils import compute_product_space, get_rca, load_data

LABEL_SET = os.environ.get("FIG01_LABELS", "landmark")

OUT_PDF = Path("figures/fig01_space_of_concerns_topology.pdf")
OUT_SVG = Path("figures/fig01_space_of_concerns_topology.svg")
OUT_ALL_PDF = Path("figures/figS23_space_all_labels.pdf")
OUT_ALL_SVG = Path("figures/figS23_space_all_labels.svg")
OUT_POSTER_PDF = Path("figures/fig01_poster.pdf")
OUT_POSTER_SVG = Path("figures/fig01_poster.svg")

SAVE_POSTER = True
POSTER_TEXT_SCALE = 1.3
POSTER_FIG_SCALE = 1.4

MAP_PROJECTION = ccrs.SouthPolarStereo(central_longitude=0)
MAP_RESOLUTION = "50m"
MAP_LAT_MAX = -60.0
MAP_MARGIN = 0.04
MAP_LAND_COLOR = "#e9e9e9"
MAP_COAST_COLOR = "#9a9a9a"
MAP_SILHOUETTE_WIDTH = 7.0
GRAPH_ROTATION_DEG = 45.0
# >1 pushes the graph past the coast so the graph, not the map, sets the scale;
# raised from 4.0 to give the larger node discs room to breathe in the centre.
GRAPH_SPREAD = 4.8

LANDMARK_TOPICS = {
    "opening statements",
    "inspections",
    "environmental monitoring and reporting",
    "climate change",
    "tourism and ng activities",
    "mineral resources",
}

CARD_ACTORS = ["Australia", "Netherlands", "Ukraine"]

STRUCTURE_CALLOUTS = [
    ("procedural core", "Opening statements", -1.1, -1.0),
    ("resource–science corridor", "Drilling", 0.4, 1.5),
    ("tourism & visitation branch", "Tourism and NG Activities", 1.7, -1.1),
]

TEXT = figstyle.TEXT


def normalize_topic_key(name):
    if name is None:
        return ""
    text = str(name).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def load_data_with_fallback():
    paths = [
        Path("antarctic-database-go/data/processed/document-summary.parquet"),
        Path("../antarctic-database-go/data/processed/document-summary.parquet"),
        Path(
            "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"
        ),
    ]
    for path in paths:
        if path.exists():
            try:
                return load_data(str(path))
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load {path}: {exc}")
    raise FileNotFoundError("No usable data file found.")


def filter_space_topics(counts_df):
    excluded = {"all", "other"}
    keep = [
        topic for topic in counts_df.index if str(topic).strip().lower() not in excluded
    ]
    return counts_df.loc[keep]


def build_graphs():
    counts_df, _, _, _ = load_data_with_fallback()
    counts_df = filter_space_topics(counts_df)
    rca = get_rca(counts_df)
    phi = compute_product_space(rca)
    import networkx as nx

    g = nx.from_pandas_adjacency(phi)
    g.remove_edges_from(nx.selfloop_edges(g))
    for i, j, d in g.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["weight"] = w
        d["distance"] = float(-np.log(np.clip(w, 1e-12, 1.0)))
    mst = nx.maximum_spanning_tree(g)
    backbone = mst.copy()
    weights = np.array(
        [d.get("weight", 1.0) for _, _, d in g.edges(data=True)], dtype=float
    )
    if weights.size:
        cutoff = np.percentile(weights, 95)
        for i, j, d in g.edges(data=True):
            if float(d["weight"]) >= cutoff:
                backbone.add_edge(i, j, **d)
    return backbone, mst, g, counts_df, rca


def load_topic_meta():
    order = pd.read_csv("output/fig45_portfolio_space_ridgelines_topic_order.csv")
    display_of, mode_of, xplot_of = {}, {}, {}
    for _, row in order.iterrows():
        key = normalize_topic_key(row["topic"])
        display_of[key] = str(row["topic"]).replace("_", " ")
        mode_of[key] = int(row["region_id"])
        xplot_of[key] = float(row["x_plot"])
    return display_of, mode_of, xplot_of


def antarctic_landmass(projection, lat_max=MAP_LAT_MAX):
    feature = cfeature.NaturalEarthFeature("physical", "land", MAP_RESOLUTION)
    polygons = []
    for geom in feature.geometries():
        if geom.bounds[3] > lat_max:
            continue
        projected = projection.project_geometry(geom, ccrs.PlateCarree())
        if not projected.is_empty:
            polygons.append(projected)
    if not polygons:
        raise RuntimeError("No Antarctic land geometry in Natural Earth data.")
    return unary_union(polygons).buffer(0)


def build_antarctica_basemap(projection=MAP_PROJECTION, n_across=180):
    land = antarctic_landmass(projection)
    x_lo, y_lo, x_hi, y_hi = land.bounds
    margin_x = MAP_MARGIN * (x_hi - x_lo)
    margin_y = MAP_MARGIN * (y_hi - y_lo)
    proj_extent = (x_lo - margin_x, x_hi + margin_x, y_lo - margin_y, y_hi + margin_y)

    width = MAP_SILHOUETTE_WIDTH
    height = width * (proj_extent[3] - proj_extent[2]) / (proj_extent[1] - proj_extent[0])
    data_extent = [-width / 2, width / 2, -height / 2, height / 2]

    n_down = max(8, int(round(n_across * (y_hi - y_lo) / (x_hi - x_lo))))
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_lo, x_hi, n_across), np.linspace(y_lo, y_hi, n_down)
    )
    inside = contains_xy(land, grid_x, grid_y)
    if not inside.any():
        raise RuntimeError("No interior points found for tessellation.")
    tess = np.column_stack(
        [
            np.interp(grid_x[inside], proj_extent[:2], data_extent[:2]),
            np.interp(grid_y[inside], proj_extent[2:], data_extent[2:]),
        ]
    )
    return tess, proj_extent, data_extent, land


def snap_to_tessellation(pos, tess_points):
    nodes = list(pos.keys())
    pos_arr = np.array([pos[n] for n in nodes])
    d = ((pos_arr[:, None, :] - tess_points[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d, axis=1)
    return {n: tess_points[i] for n, i in zip(nodes, idx)}


def fitted_layout(mst, g, mode_of=None):
    """(SGD)^2-style multicriteria layout (Ahmed et al., arXiv:2112.01571),
    implemented directly: stochastic gradient descent on two differentiable
    readability criteria at once --

      * distance preservation (stress): every pair of topics is pulled toward
        its graph distance in the spanning tree, weighted 1/d^2, which gives
        ideal edge lengths and an honest, organic spread;
      * node resolution: pairs closer than a minimum separation are pushed
        apart, so discs never crowd no matter how dense the core is.

    The silhouette is only a frame the result is scaled into; the wheel is
    rotated so the coordination branch runs west, as in the earlier maps;
    non-tree ties are drawn separately as faint chords.
    """
    import networkx as nx

    _, proj_extent, data_extent, land = build_antarctica_basemap()

    nodes = list(mst.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    spl = dict(nx.all_pairs_shortest_path_length(mst))
    D = np.array([[spl[u][v] for v in nodes] for u in nodes], dtype=float)
    np.fill_diagonal(D, 1.0)

    init = nx.kamada_kawai_layout(mst, weight="distance")
    P = np.array([init[v] for v in nodes], dtype=float)
    P -= P.mean(axis=0)
    P *= 3.0 / np.max(np.linalg.norm(P, axis=1))

    pi, pj = np.triu_indices(n, 1)
    min_sep = 0.85
    iters = 3000
    rng = np.random.default_rng(7)
    n_pairs = len(pi)
    for t in range(iters):
        eta = 0.06 * (1.0 - t / iters) + 0.004
        sel = rng.choice(n_pairs, size=160, replace=False)
        i, j = pi[sel], pj[sel]
        d = P[i] - P[j]
        norm = np.hypot(d[:, 0], d[:, 1])
        norm = np.maximum(norm, 1e-9)
        # criterion 1: pull sampled pairs toward their graph distance
        w = 1.0 / (D[i, j] ** 2)
        grad = ((norm - D[i, j]) / norm * w)[:, None] * d
        P[i] -= eta * grad
        P[j] += eta * grad
        # criterion 2: node resolution -- enforce the minimum separation
        close = norm < min_sep
        if close.any():
            gsep = ((norm[close] - min_sep) / norm[close])[:, None] * d[close]
            P[i[close]] -= 0.5 * gsep
            P[j[close]] += 0.5 * gsep

    pos = {v: (float(P[i][0]), float(P[i][1])) for i, v in enumerate(nodes)}

    # Rotate so the coordination branch runs west, as in the earlier maps.
    if mode_of is not None:
        mem = [n for n in nodes if mode_of.get(normalize_topic_key(n), 2) == 1]
        if mem:
            zs = sum(
                np.exp(1j * np.arctan2(pos[n][1], pos[n][0])) for n in mem
            )
            rot = np.pi - np.angle(zs)
            ca, sa = np.cos(rot), np.sin(rot)
            pos = {n: (x * ca - y * sa, x * sa + y * ca)
                   for n, (x, y) in pos.items()}

    # Fit into the silhouette frame, aspect preserved, graph setting the scale.
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    w = data_extent[1] - data_extent[0]
    h = data_extent[3] - data_extent[2]
    s = min(1.05 * w / np.ptp(xs), 1.05 * h / np.ptp(ys))
    pos = {
        n: (float((x - 0.5 * (xs.max() + xs.min())) * s),
            float((y - 0.5 * (ys.max() + ys.min())) * s))
        for n, (x, y) in pos.items()
    }
    return pos, land, proj_extent


def _declutter(pos, min_d=0.8, iters=120):
    """Separate nodes that the tessellation snap stacked on top of each other.

    The snap can place several topics on nearly identical points (the dense
    north-east cluster), hiding the internal structure. A few relaxation
    iterations push overlapping discs apart while leaving the global layout
    untouched, so every node and every edge endpoint stays readable.
    """
    nodes = list(pos)
    p = np.array([pos[n] for n in nodes], dtype=float)
    for _ in range(iters):
        diff = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        close = dist < min_d
        if not close.any():
            break
        push = np.where(close, (min_d - dist) / 2.0, 0.0)
        unit = np.zeros_like(diff)
        mask = np.isfinite(dist) & (dist > 1e-9)
        unit[mask] = diff[mask] / dist[mask][:, None]
        p += (push[:, :, None] * unit).sum(axis=1)
    return {n: (float(p[i][0]), float(p[i][1])) for i, n in enumerate(nodes)}


def draw_silhouette(ax_map, land, proj_extent):
    geoax = ax_map.inset([0, 0, 1, 1], proj=MAP_PROJECTION, zoom=False, zorder=-2)
    geoax.set_extent(proj_extent, crs=MAP_PROJECTION)
    geoax.add_geometries(
        [land], crs=MAP_PROJECTION,
        facecolor=MAP_LAND_COLOR, edgecolor=MAP_COAST_COLOR, linewidth=0.5,
        zorder=0,
    )
    geoax.patch.set_visible(False)
    geoax.format(grid=False, labels=False)
    for spine in geoax.spines.values():
        spine.set_visible(False)


def draw_base(ax, backbone, pos, *, edge_alpha=0.8, edge_lw=1.0, color="0.70"):
    for u, v, d in backbone.edges(data=True):
        p0, p1 = np.array(pos[u]), np.array(pos[v])
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=color, lw=edge_lw * (0.6 + 3.6 * float(d.get("weight", 0.0))),
            alpha=edge_alpha, zorder=1, solid_capstyle="round",
        )


def _monotone_packed(desired, start, end, min_gap):
    desired = np.asarray(desired, dtype=float)
    n = desired.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([float(np.clip(desired[0], start, end))], dtype=float)
    span = float(end - start)
    eff_gap = min(float(min_gap), span / max(n - 1, 1))
    pos = np.clip(desired.copy(), start, end)
    pos[0] = max(pos[0], start)
    for i in range(1, n):
        pos[i] = max(pos[i], pos[i - 1] + eff_gap)
    if pos[-1] > end:
        pos[-1] = end
        for i in range(n - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - eff_gap)
        if pos[0] < start:
            pos[0] = start
            for i in range(1, n):
                pos[i] = max(pos[i], pos[i - 1] + eff_gap)
    return np.clip(pos, start, end)


def _spaced_monotone(desired, start, end, min_gap, blend=0.62):
    desired = np.asarray(desired, dtype=float)
    n = desired.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([float(np.clip(desired[0], start, end))], dtype=float)
    even = np.linspace(start, end, n)
    return _monotone_packed(float(blend) * desired + (1.0 - blend) * even, start, end, min_gap)


def draw_rect_labels(ax, pos, display_of, *, rect_margin=0.65):
    """Side-label scaffold from the original Figure 1: all topic labels sit
    around the four sides of a rectangle framing the map, each connected to
    its node by a light line, instead of inside the topology."""
    nodes = list(pos.keys())
    xy = np.array([pos[n] for n in nodes], dtype=float)
    x_lo, x_hi = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_lo, y_hi = float(xy[:, 1].min()), float(xy[:, 1].max())
    pad = rect_margin
    rect = (x_lo - pad, x_hi + pad, y_lo - pad, y_hi + pad)
    cx, cy = 0.5 * (rect[0] + rect[1]), 0.5 * (rect[2] + rect[3])

    sides: dict[str, list] = {"top": [], "right": [], "bottom": [], "left": []}
    for n in nodes:
        x, y = pos[n]
        ang = float(np.degrees(np.arctan2(y - cy, x - cx))) % 360.0
        if 45 <= ang < 135:
            sides["top"].append(n)
        elif 135 <= ang < 225:
            sides["left"].append(n)
        elif 225 <= ang < 315:
            sides["bottom"].append(n)
        else:
            sides["right"].append(n)
    for s in sides:
        if s in ("top", "bottom"):
            sides[s].sort(key=lambda n: pos[n][0])
        else:
            sides[s].sort(key=lambda n: pos[n][1])

    ax.add_patch(
        __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (rect[0], rect[2]), rect[1] - rect[0], rect[3] - rect[2],
            fill=False, edgecolor="0.6", linewidth=1.0, zorder=0.5,
        )
    )

    fs = 10.0
    span_x = rect[1] - rect[0]
    span_y = rect[3] - rect[2]
    label_style = {
        "top": dict(ha="center", va="bottom", rotation=90),
        "bottom": dict(ha="center", va="top", rotation=90),
        "left": dict(ha="right", va="center", rotation=0),
        "right": dict(ha="left", va="center", rotation=0),
    }

    def _place(side, nodes_, anchor_start, anchor_end, min_gap, blend):
        raw = np.array(
            [pos[n][0] if side in ("top", "bottom") else pos[n][1] for n in nodes_]
        )
        posn = _spaced_monotone(raw, anchor_start, anchor_end, min_gap, blend=blend)
        for n, pv in zip(nodes_, posn):
            node_xy = np.asarray(pos[n], dtype=float)
            if side == "top":
                txy = (float(pv), rect[3])
            elif side == "bottom":
                txy = (float(pv), rect[2])
            elif side == "left":
                txy = (rect[0], float(pv))
            else:
                txy = (rect[1], float(pv))
            name = display_of.get(normalize_topic_key(n), str(n))
            name = "\n".join(textwrap.wrap(name, 22) or [name])
            ax.annotate(
                name,
                xy=(float(node_xy[0]), float(node_xy[1])),
                xytext=txy,
                fontsize=fs,
                color=TEXT,
                ha=label_style[side]["ha"],
                va=label_style[side]["va"],
                rotation=label_style[side]["rotation"],
                zorder=4,
                arrowprops=dict(
                    arrowstyle="-", color="0.55", lw=0.7,
                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.0",
                ),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
            )

    _place("top", sides["top"], rect[0] + 0.10 * span_x, rect[1] - 0.10 * span_x,
           0.09 * span_x, blend=0.62)
    _place("bottom", sides["bottom"], rect[0] + 0.10 * span_x,
           rect[1] - 0.10 * span_x, 0.09 * span_x, blend=0.62)
    _place("left", sides["left"], rect[2] + 0.10 * span_y, rect[3] - 0.10 * span_y,
           0.14 * span_y, blend=0.30)
    _place("right", sides["right"], rect[2] + 0.10 * span_y,
           rect[3] - 0.10 * span_y, 0.14 * span_y, blend=0.30)


def draw_main_map(ax, backbone, mst, g, pos, mode_of, display_of, *, all_labels,
                  rect_labels=True):
    nodes = list(backbone.nodes())
    deg = dict(g.degree(weight="weight"))
    dvals = np.array([deg.get(n, 0.0) for n in nodes], dtype=float)
    dnorm = (dvals - dvals.min()) / max(float(np.ptp(dvals)), 1e-12)
    sizes = 60.0 + 430.0 * dnorm

    for u, v, d in backbone.edges(data=True):
        p0, p1 = np.array(pos[u]), np.array(pos[v])
        w = float(d.get("weight", 0.0))
        primary = mst.has_edge(u, v)
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color="0.70",
            lw=(0.6 + 3.6 * w) if primary else (0.4 + 1.4 * w),
            alpha=0.85 if primary else 0.35,
            zorder=1, solid_capstyle="round",
        )

    xy = np.array([pos[n] for n in nodes], dtype=float)
    ax.scatter(
        xy[:, 0], xy[:, 1], s=sizes,
        c=figstyle.PRIMARY, alpha=0.92, edgecolor="white", linewidth=0.8,
        zorder=3, absolute_size=True,
    )

    # Obstacle boxes (data units) that labels must dodge. Hard obstacles are
    # the structural callouts and already-placed labels; node discs are soft
    # obstacles, dodged in a first pass so that a label never rides on a
    # foreign node when free space exists nearby, but tolerated in a second
    # pass so labels stay close to their own node either way.
    hard: list[tuple[float, float, float, float]] = []

    by_key = {normalize_topic_key(k): v for k, v in pos.items()}
    callouts = []
    for label, anchor, ddx, ddy in STRUCTURE_CALLOUTS:
        p = by_key.get(normalize_topic_key(anchor))
        if p is None:
            continue
        ctr = xy.mean(axis=0)
        d = np.array([float(p[0]) - ctr[0], float(p[1]) - ctr[1]])
        d = d / (np.linalg.norm(d) or 1.0)
        w = 0.13 * len(label)
        # Keep the tag clear of the mode pills *and* of the callouts already
        # placed: these three tags are long, and the resource-science corridor
        # and tourism branch anchor close enough that dodging only the pills let
        # their boxes overlap. `hard` already holds both sets, so testing the
        # whole list is the fix; pushing the radius out on later sweeps gives the
        # rotation somewhere to go when every nearby angle is taken.
        base_ang = np.arctan2(d[1], d[0])
        tx = ty = None
        for k in range(26):
            ang = base_ang + 0.3 * k
            r = 1.35 + 0.35 * (k // 13)
            tx, ty = float(p[0]) + np.cos(ang) * r, float(p[1]) + np.sin(ang) * r
            box = (tx - w / 2, tx + w / 2, ty - 0.25, ty + 0.25)
            if not any(
                box[0] < o[1] and box[1] > o[0] and box[2] < o[3] and box[3] > o[2]
                for o in hard
            ):
                break
        hard.append((tx - w / 2, tx + w / 2, ty - 0.25, ty + 0.25))
        callouts.append((label, p, tx, ty))

    node_obs = [(x - 0.45, x + 0.45, y - 0.45, y + 0.45) for x, y in pos.values()]

    pill_x = [b[k] for b in hard for k in (0, 1)]
    pill_y = [b[k] for b in hard for k in (2, 3)]
    lo = np.array([min(xy[:, 0].min(), min(pill_x)), min(xy[:, 1].min(), min(pill_y))]) - 0.6
    hi = np.array([max(xy[:, 0].max(), max(pill_x)), max(xy[:, 1].max(), max(pill_y))]) + 0.6

    def _hits(box, obs):
        return any(
            box[0] < o[1] and box[1] > o[0] and box[2] < o[3] and box[3] > o[2]
            for o in obs
        )

    keys = set(mode_of) if all_labels else LANDMARK_TOPICS
    fs = 11.0 if all_labels else 13.0
    char_w = 0.11 if all_labels else 0.13
    line_h = 0.26 if all_labels else 0.30
    centroid = xy.mean(axis=0)
    for key in sorted(keys):
        if not all_labels and not rect_labels:
            continue
        node = next((n for n in pos if normalize_topic_key(n) == key), None)
        if node is None:
            continue
        x, y = pos[node]
        name = display_of.get(key, str(node))
        lines = textwrap.wrap(name, 22) or [name]
        w = char_w * max(len(s) for s in lines)
        h = line_h * len(lines)
        away = np.array([x - centroid[0], y - centroid[1]])
        away = away / (np.linalg.norm(away) or 1.0)
        angles = sorted(
            range(0, 360, 20),
            key=lambda a: -np.cos(np.radians(a)) * away[0]
            - np.sin(np.radians(a)) * away[1],
        )
        placed_box = None
        for with_nodes in (True, False):
            obs = hard + node_obs if with_nodes else hard
            for ang in angles:
                d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
                for off in np.arange(0.55, 1.35, 0.1):
                    cxl, cyl = x + d[0] * off, y + d[1] * off
                    box = (cxl - w / 2, cxl + w / 2, cyl - h / 2, cyl + h / 2)
                    if box[0] < lo[0] or box[1] > hi[0] or box[2] < lo[1] or box[3] > hi[1]:
                        continue
                    if not _hits(box, obs):
                        placed_box = box
                        break
                if placed_box:
                    break
            if placed_box:
                break
        if placed_box is None:
            placed_box = (x - w / 2, x + w / 2, y + 0.45, y + 0.45 + h)
        hard.append(placed_box)
        cxl = 0.5 * (placed_box[0] + placed_box[1])
        cyl = 0.5 * (placed_box[2] + placed_box[3])
        if np.hypot(cxl - x, cyl - y) > 0.85:
            ax.plot([x, cxl], [y, cyl], color="0.55", lw=0.7, zorder=2.5)
        ax.text(
            cxl, cyl, "\n".join(lines),
            ha="center", va="center",
            fontsize=fs, color=TEXT, zorder=4, linespacing=1.05,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )

    for label, p, tx, ty in callouts:
        ax.text(
            tx, ty, label,
            fontsize=13.5, fontstyle="italic", color="0.45",
            ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
            zorder=3.3,
        )

    ax.format(
        xlim=(xy[:, 0].min() - 1.2, xy[:, 0].max() + 1.2),
        ylim=(xy[:, 1].min() - 1.2, xy[:, 1].max() + 1.2),
        xticks=[], yticks=[], grid=False,
    )
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(False)


def draw_actor_card(ax, actor, pos, backbone, rca, mode_of):
    draw_base(ax, backbone, pos, edge_alpha=0.18, edge_lw=0.5)
    nodes = list(backbone.nodes())
    xy = np.array([pos[n] for n in nodes], dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], s=16, color="0.82", alpha=0.7,
               edgecolor="none", zorder=2, absolute_size=True)

    held = []
    for n in nodes:
        key = normalize_topic_key(n)
        if key in rca.index and actor in rca.columns:
            val = float(rca.loc[key, actor])
            if val > 1.0:
                held.append((n, val))
    if held:
        hx = np.array([pos[n] for n, _ in held], dtype=float)
        hv = np.array([v for _, v in held], dtype=float)
        ax.scatter(
            hx[:, 0], hx[:, 1], s=32 + 90 * hv,
            c=figstyle.PRIMARY,
            alpha=0.95, edgecolor="white", linewidth=0.8, zorder=3,
            absolute_size=True,
        )

    ax.format(
        xlim=(xy[:, 0].min() - 1.2, xy[:, 0].max() + 1.2),
        ylim=(xy[:, 1].min() - 1.2, xy[:, 1].max() + 1.2),
        xticks=[], yticks=[], grid=False,
        title=actor, titleloc="l", titlesize=figstyle.FS_LABEL,
        titleweight="bold", titlecolor=figstyle.PRIMARY,
    )
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.text(
        0.99, 1.02, f"{len(held)} specialized topics",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11.0, color="0.55",
    )


def build_figure(*, all_labels: bool):
    import ultraplot as uplt

    backbone, mst, g, counts_df, rca = build_graphs()
    display_of, mode_of, xplot_of = load_topic_meta()
    pos, land, proj_extent = fitted_layout(mst, g, mode_of)

    if all_labels:
        # the appendix variant draws no actor cards, so it gets a single axes;
        # keeping the two-row layout left three empty framed panels below the map
        layout = [[0, 1, 1, 1, 1, 0]]
        fig, axs = uplt.subplots(layout, figwidth=11.8, share=False)
    else:
        layout = [
            [0, 1, 1, 1, 1, 0],
            [2, 2, 3, 3, 4, 4],
        ]
        fig, axs = uplt.subplots(
            layout, figwidth=11.8, hratios=[4.1, 1.15], share=False
        )
    ax_map = axs[0]
    draw_silhouette(ax_map, land, proj_extent)
    if all_labels:
        draw_main_map(ax_map, backbone, mst, g, pos, mode_of, display_of,
                      all_labels=all_labels)
    else:
        draw_main_map(ax_map, backbone, mst, g, pos, mode_of, display_of,
                      all_labels=all_labels, rect_labels=False)
        draw_rect_labels(ax_map, pos, display_of)

    if not all_labels:
        rca_keyed = rca.copy()
        rca_keyed.index = [normalize_topic_key(t) for t in rca_keyed.index]
        for ax_c, actor in zip(axs[1:], CARD_ACTORS):
            draw_actor_card(ax_c, actor, pos, backbone, rca_keyed, mode_of)
    return fig


def save_poster(fig):
    figsize = fig.get_size_inches().copy()
    fig.set_size_inches(figsize * POSTER_FIG_SCALE)
    for ax in fig.axes:
        for txt in ax.texts:
            txt.set_fontsize(txt.get_fontsize() * POSTER_TEXT_SCALE)
    fig.savefig(OUT_POSTER_PDF)
    fig.savefig(OUT_POSTER_SVG)
    print("[fig01] Saved poster variant")


def main():
    all_labels = LABEL_SET == "all"
    fig = build_figure(all_labels=all_labels)
    if all_labels:
        OUT_ALL_PDF.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_ALL_PDF)
        fig.savefig(OUT_ALL_SVG)
        print("[fig01] Saved all-label appendix variant")
        return
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    print("[fig01] Saved main figure assets")
    if SAVE_POSTER:
        save_poster(fig)


if __name__ == "__main__":
    main()
