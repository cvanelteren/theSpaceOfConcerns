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
# Node fill for the space itself: a warm ink that reads as a single neutral
# family alongside the white node halo and grey land silhouette.
NODE_COLOR = "#2E3A4A"
# Offset (data units) from a node centre to its disc rim, used to end the
# side-label leaders on the inward-facing edge of the node instead of its
# centre.
NODE_RIM = 0.35
MAP_SILHOUETTE_WIDTH = 7.0
GRAPH_ROTATION_DEG = 45.0
# >1 pushes the graph past the coast so the graph, not the map, sets the scale;
# the label frame is fixed in draw_rect_labels, so this controls how far the
# network spreads to fill the space between the map and the side labels.
GRAPH_SPREAD = 2.4

LANDMARK_TOPICS = {
    "opening statements",
    "inspections",
    "environmental monitoring and reporting",
    "climate change",
    "tourism and ng activities",
    "mineral resources",
}

CARD_ACTORS = ["Australia", "Netherlands", "Ukraine"]

# Distinct accent per actor card so the three bottom panels read as separate
# allocations rather than one neutral field. Picked to stay clear of the
# neutral map nodes and of one another.
CARD_COLORS = {
    "Australia": "#C96A2B",
    "Netherlands": "#2C6E9C",
    "Ukraine": "#5B7F45",
}

STRUCTURE_CALLOUTS = []

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
    # Push the graph past the silhouette so the topology spreads across the
    # frame and uses the white space between the map and the side labels.
    s *= GRAPH_SPREAD
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


def draw_rect_labels(ax, pos, display_of, *, frame_margin=1.9):
    """Side-label scaffold from the original Figure 1: all topic labels sit
    around the four sides of the map, each connected to its node by a light
    line, instead of inside the topology. The frame itself is not drawn; its
    margins are fixed (data units) so the labels stay put while the network
    spreads to fill the space between them."""
    nodes = list(pos.keys())
    xy = np.array([pos[n] for n in nodes], dtype=float)
    x_lo, x_hi = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_lo, y_hi = float(xy[:, 1].min()), float(xy[:, 1].max())
    cx_map, cy_map = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    node_w, node_h = (x_hi - x_lo), (y_hi - y_lo)
    pad_x, pad_y = frame_margin, frame_margin
    pad = frame_margin
    rect = (x_lo - pad_x, x_hi + pad_x, y_lo - pad_y, y_hi + pad_y)
    cx, cy = cx_map, cy_map

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

    fs = 10.0
    wrap_w = 42
    span_x = rect[1] - rect[0]
    span_y = rect[3] - rect[2]
    label_gap = 1.6 * max(pad_x, pad_y)
    corner_in = 0.14
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
            # The leader terminates on the rim of the disc that faces the label
            # (the inward-facing side toward the map), not at the node centre,
            # so it visibly points at the topic without a line running through
            # the disc.
            if side == "top":
                txy = (float(pv), rect[3] + label_gap)
                rim = (node_xy[0], node_xy[1] + NODE_RIM)
            elif side == "bottom":
                txy = (float(pv), rect[2] - label_gap)
                rim = (node_xy[0], node_xy[1] - NODE_RIM)
            elif side == "left":
                txy = (rect[0] - label_gap, float(pv))
                rim = (node_xy[0] - NODE_RIM, node_xy[1])
            else:
                txy = (rect[1] + label_gap, float(pv))
                rim = (node_xy[0] + NODE_RIM, node_xy[1])
            name = display_of.get(normalize_topic_key(n), str(n))
            name = "\n".join(textwrap.wrap(name, wrap_w) or [name])
            ax.annotate(
                name,
                xy=rim,
                xytext=txy,
                fontsize=fs,
                color=TEXT,
                ha=label_style[side]["ha"],
                va=label_style[side]["va"],
                rotation=label_style[side]["rotation"],
                zorder=4,
                arrowprops=dict(
                    arrowstyle="-", color="0.45", lw=0.7,
                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.0",
                ),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
            )

    _place("top", sides["top"], rect[0] + corner_in * span_x,
           rect[1] - corner_in * span_x, 0.25 * span_x, blend=0.30)
    _place("bottom", sides["bottom"], rect[0] + corner_in * span_x,
           rect[1] - corner_in * span_x, 0.25 * span_x, blend=0.30)
    _place("left", sides["left"], rect[2] + corner_in * span_y,
           rect[3] - corner_in * span_y, 0.18 * span_y, blend=0.30)
    _place("right", sides["right"], rect[2] + corner_in * span_y,
           rect[3] - corner_in * span_y, 0.18 * span_y, blend=0.30)

    # Expand the axes so no label is clipped: measure the placed text boxes
    # and grow the data limits to cover them (plus a small breathing margin).
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    for t in ax.texts:
        bb = t.get_window_extent(renderer=renderer)
        (bx0, by0) = ax.transData.inverted().transform((bb.x0, bb.y0))
        (bx1, by1) = ax.transData.inverted().transform((bb.x1, bb.y1))
        x0, y0 = min(x0, bx0), min(y0, by0)
        x1, y1 = max(x1, bx1), max(y1, by1)
    if np.isfinite(x0):
        ax.set_xlim(x0 - 0.1, x1 + 0.1)
        ax.set_ylim(y0 - 0.1, y1 + 0.1)


# --- Structural-region hulls (the original spline outlines) ----------------
# The three branches and the procedural core are the same regions the earlier
# Figure 1 wrapped in smooth, semi-transparent hulls labelled with the region
# name. Ported from the pre-August figure so the map again shows the core /
# corridor / branch structure behind the dots.

REGION_SPECS = [
    dict(
        label="Drilling, Monitoring\n& CEP Oversight",
        color="#1f77b4",
        nodes=[
            "Drilling", "Sub glacial Lakes", "Operation of the CEP",
            "Environmental Domains Analysis", "Marine Acoustics",
            "State of the Antarctic Environment Report SAER",
        ],
    ),
    dict(
        label="Human Impact\n& Marine Stewardship",
        color="#F57C00",
        nodes=[
            "Tourism and NG Activities", "Marine Protected Areas",
            "Marine living resources", "Prevention of marine pollution",
            "Site Guidelines for Visitors", "Mineral resources",
            "Multiyear strategic workplan", "Human Footprint and wilderness values",
            "Search and Rescue",
        ],
    ),
    dict(
        label="Environmental\nProtection",
        color="#2E7D32",
        nodes=[
            "Nonnative Species and Quarantine", "Specially Protected Species",
            "Climate Change", "CEP Strategy Discussions", "Biological Prospecting",
            "Fauna and Flora General", "Repair and remediation of environmental damage",
            "Operation of the Antarctic Treaty system Reports",
            "Cooperation with Other Organisations",
        ],
    ),
    dict(
        label="Procedural core",
        color="#6A1B9A",
        nodes=[
            "Operational issues", "Liability", "Educational issues",
            "Exchange of Information", "Environmental Protection General",
            "Opening statements", "International Polar Year",
            "Waste management and disposal", "Inspections",
            "Environmental Impact Assessment EIA Other EIA Matters",
            "Science issues", "Management Plans", "Environmental Monitoring and Reporting",
            "Safety and Operations in Antarctica", "Comprehensive Environmental Evaluations",
            "Emergency report and contingency planning", "Institutional and legal matters",
            "Area Protection and Management Plans General", "Historic Sites and Monuments",
            "Operation of the Antarctic Treaty system General",
            "Operation of the Antarctic Treaty system The Secretariat",
        ],
    ),
]


def _chaikin_closed(coords, refinements=2):
    ring = np.asarray(coords, dtype=float)
    if ring.shape[0] < 4:
        return ring
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    for _ in range(max(0, int(refinements))):
        out = []
        for i in range(ring.shape[0] - 1):
            p0, p1 = ring[i], ring[i + 1]
            out.extend([0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1])
        ring = np.vstack(out + [out[0]])
    return ring


def _fit_closed_spline(coords, out_points=220, smooth_scale=0.004):
    from scipy.interpolate import splprep, splev

    ring = np.asarray(coords, dtype=float)
    if ring.shape[0] < 4:
        return ring
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 4:
        return np.vstack([ring, ring[0]])
    keep = [0]
    for i in range(1, ring.shape[0]):
        if np.hypot(*(ring[i] - ring[keep[-1]])) > 1e-8:
            keep.append(i)
    ring = ring[keep]
    if ring.shape[0] < 4:
        return np.vstack([ring, ring[0]])
    diffs = np.diff(np.vstack([ring, ring[0]]), axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    total = float(seg.sum())
    if total <= 1e-10:
        return np.vstack([ring, ring[0]])
    u = np.hstack([[0.0], np.cumsum(seg[:-1]) / total])
    try:
        k = int(min(3, ring.shape[0] - 1))
        s = float(smooth_scale) * ring.shape[0]
        tck, _ = splprep([ring[:, 0], ring[:, 1]], u=u, s=s, per=True, k=k)
        uu = np.linspace(0.0, 1.0, int(max(48, out_points)), endpoint=False)
        x_new, y_new = splev(uu, tck)
        smooth = np.column_stack([x_new, y_new])
        return np.vstack([smooth, smooth[0]])
    except Exception:
        return _chaikin_closed(np.vstack([ring, ring[0]]), refinements=2)


def _region_hulls(pos, backbone_nodes, padding=0.55):
    from shapely.geometry import LineString, Point, Polygon
    from scipy.spatial import ConvexHull

    node_lookup = {normalize_topic_key(n): n for n in backbone_nodes}
    regions = []
    for spec in REGION_SPECS:
        nodes = set()
        for name in spec["nodes"]:
            key = normalize_topic_key(name)
            if key in node_lookup:
                nodes.add(node_lookup[key])
        if len(nodes) < 2:
            continue
        regions.append({"label": spec["label"], "color": spec["color"], "nodes": nodes})
    assigned = set().union(*(r["nodes"] for r in regions))
    leftovers = set(backbone_nodes) - assigned
    if leftovers:
        for r in regions:
            if r["label"] == "Procedural core":
                r["nodes"].update(leftovers)
                break
    regions = [r for r in regions if len(r["nodes"]) >= 2]

    hulls = []
    for cid, reg in enumerate(regions, start=1):
        pts = np.array([pos[n] for n in reg["nodes"]], dtype=float)
        if pts.shape[0] < 2:
            continue
        if pts.shape[0] == 2 or (
            pts.shape[0] == 3 and np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2
        ):
            poly = LineString(pts).buffer(float(padding) * 0.9, cap_style=1, join_style=1)
        else:
            try:
                hull = ConvexHull(pts)
                coords = pts[hull.vertices]
                coords = np.vstack([coords, coords[0]])
                coords = _chaikin_closed(coords, refinements=2)
                poly = Polygon(coords)
            except Exception:
                poly = LineString(pts).buffer(
                    float(padding) * 0.9, cap_style=1, join_style=1
                )
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        poly = poly.buffer(float(padding), join_style=1).buffer(0)
        if poly.is_empty:
            continue
        coords = np.asarray(poly.exterior.coords, dtype=float)
        spline = _fit_closed_spline(coords, out_points=240, smooth_scale=0.003)
        if spline.shape[0] >= 4:
            spoly = Polygon(spline)
            if not spoly.is_valid:
                spoly = spoly.buffer(0)
            if (not spoly.is_empty) and float(spoly.area) > 1e-8:
                node_pts = np.array([pos[n] for n in reg["nodes"]], dtype=float)
                max_miss = 0.0
                for px, py in node_pts:
                    p = Point(float(px), float(py))
                    if not spoly.covers(p):
                        max_miss = max(max_miss, float(spoly.distance(p)))
                if max_miss > 0.0:
                    spoly = spoly.buffer(
                        max_miss + 0.22 * float(padding), join_style=1
                    ).buffer(0)
                outline = np.asarray(spoly.exterior.coords, dtype=float)
            else:
                outline = coords
        else:
            outline = coords
        hulls.append(
            {
                "cluster_id": cid,
                "nodes": reg["nodes"],
                "label": reg["label"],
                "color": reg["color"],
                "outline_coords": outline,
            }
        )
    return hulls


def draw_region_hulls(ax, hulls, alpha_fill=0.12, alpha_edge=0.55):
    for item in hulls:
        x, y = item["outline_coords"][:, 0], item["outline_coords"][:, 1]
        ax.fill(
            x, y,
            facecolor=item["color"], edgecolor=item["color"],
            linewidth=1.2, alpha=alpha_fill, zorder=0.5,
        )
        ax.plot(
            x, y, color=item["color"], lw=1.1, alpha=alpha_edge,
            zorder=0.6, solid_capstyle="round",
        )


def draw_region_labels(ax, hulls, pos):
    from matplotlib.path import Path

    # Manual nudges (data units) so each region label clears its own nodes and
    # sits where the hull has room. Applied on top of the automatic in-hull
    # placement.
    NUDGES = {
        "Environmental\nProtection": (0.55, -0.10),
        "Drilling, Monitoring\n& CEP Oversight": (0.15, -0.70),
        "Human Impact\n& Marine Stewardship": (-0.75, -0.65),
        "Procedural core": (0.0, 0.45),
    }

    for item in hulls:
        label = item["label"]
        color = item["color"]
        x, y = item["outline_coords"][:, 0], item["outline_coords"][:, 1]
        cx, cy = float(np.mean(x)), float(np.mean(y))
        # Annotate just inside the top of the hull so the region name reads
        # attached to its spline.
        idx = int(np.argmax(y))
        anchor = np.array([x[idx], y[idx]])
        tangent = np.array([x[idx] - x[idx - 1], y[idx] - y[(idx + 1) % (x.size - 1)]])
        tnorm = float(np.hypot(tangent[0], tangent[1]))
        if tnorm > 1e-9:
            tangent = tangent / tnorm
        angle = float(np.degrees(np.arctan2(tangent[1], tangent[0])))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        radial = anchor - np.array([cx, cy])
        rnorm = float(np.hypot(radial[0], radial[1]))
        if rnorm > 1e-9:
            radial = radial / rnorm
        lx, ly = anchor + radial * 0.10
        # Keep long branch labels roughly inside their hull: pull the text
        # toward the region centroid along the inward direction.
        dx, dy = anchor - np.array([lx, ly])
        lx, ly = lx + 0.45 * (cx - lx), ly + 0.45 * (cy - ly)
        nudge = NUDGES.get(label)
        if nudge is not None:
            lx += float(nudge[0])
            ly += float(nudge[1])
        ax.text(
            lx, ly, label,
            ha="center", va="center",
            rotation=angle, rotation_mode="anchor",
            fontsize=9.5, color="0.15",
            bbox=dict(
                boxstyle="round,pad=0.22", facecolor="white", edgecolor=color,
                linewidth=0.8, alpha=1.0,
            ),
            zorder=2.2,
        )


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
            color="0.30",
            lw=(0.6 + 3.6 * w) if primary else (0.4 + 1.4 * w),
            alpha=0.95 if primary else 0.55,
            zorder=1, solid_capstyle="round",
        )

    xy = np.array([pos[n] for n in nodes], dtype=float)
    ax.scatter(
        xy[:, 0], xy[:, 1], s=sizes,
        c=NODE_COLOR, alpha=0.92, edgecolor="white", linewidth=0.8,
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

    pill_x = [b[k] for b in hard for k in (0, 1)] or [xy[:, 0].min()]
    pill_y = [b[k] for b in hard for k in (2, 3)] or [xy[:, 1].min()]
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
            c=CARD_COLORS.get(actor, NODE_COLOR),
            alpha=0.95, edgecolor="white", linewidth=0.8, zorder=3,
            absolute_size=True,
        )

    ax.format(
        xlim=(xy[:, 0].min() - 1.2, xy[:, 0].max() + 1.2),
        ylim=(xy[:, 1].min() - 1.2, xy[:, 1].max() + 1.2),
        xticks=[], yticks=[], grid=False,
        title=actor, titleloc="l", titlesize=18.0,
        titleweight="bold", titlecolor=CARD_COLORS.get(actor, figstyle.PRIMARY),
    )
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(False)


def build_figure(*, all_labels: bool):
    import ultraplot as uplt

    backbone, mst, g, counts_df, rca = build_graphs()
    display_of, mode_of, xplot_of = load_topic_meta()
    pos, land, proj_extent = fitted_layout(mst, g, mode_of)

    if all_labels:
        # the appendix variant draws no actor cards, so it gets a single axes;
        # keeping the two-row layout left three empty framed panels below the map
        layout = [[0, 1, 1, 1, 1, 0]]
        fig, axs = uplt.subplots(layout, figwidth=16.0, share=False)
    else:
        layout = [
            [0, 1, 1, 1, 1, 0],
            [2, 2, 3, 3, 4, 4],
        ]
        fig, axs = uplt.subplots(
            layout, figwidth=16.0, hratios=[4.1, 1.15], share=False
        )
    ax_map = axs[0]
    draw_silhouette(ax_map, land, proj_extent)
    if all_labels:
        hulls = _region_hulls(pos, list(backbone.nodes()))
        draw_region_hulls(ax_map, hulls)
        draw_region_labels(ax_map, hulls, pos)
        draw_main_map(ax_map, backbone, mst, g, pos, mode_of, display_of,
                      all_labels=all_labels)
    else:
        hulls = _region_hulls(pos, list(backbone.nodes()))
        draw_region_hulls(ax_map, hulls)
        draw_region_labels(ax_map, hulls, pos)
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
