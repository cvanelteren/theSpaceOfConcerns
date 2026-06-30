# %%
"""Graph-only figure: space of concerns fitted to Antarctica silhouette.

The expensive part of this figure is not the node layout itself, which is
already loaded from disk when available, but the side-label assignment and
connector-routing scaffold. Cache that derived layout so routine reruns only
recompute it when the underlying geometry changes.
"""

import json
import time
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import networkx as nx
import numpy as np
import ultraplot as uplt
from PIL import Image
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull

try:
    from shapely.geometry import LineString, Point, Polygon

    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    HAS_SHAPELY = False

from utils import (
    compute_product_space,
    get_rca,
    load_data,
    load_flag,
    load_saved_layout_positions,
)

USE_CACHED_SIDE_LAYOUT = True
REFRESH_SIDE_LAYOUT = False
SIDE_LAYOUT_CACHE_VERSION = 9
SIDE_LAYOUT_CACHE_PATH = Path("assets/cache/fig01_side_layout_cache.json")
USE_SAVED_NODE_LAYOUT = False
SAVE_MAIN_SVG = True
GENERATE_REVEAL_SEQUENCE = False
DEBUG_PROGRESS = True
REVEAL_OUTPUT_DIR = Path("./output/fig01_reveals")
SAVE_MAIN_PNG = True
SAVE_MAIN_PDF = True
MAIN_PNG_DPI = 1200

# Poster-friendly variant: same layout, but all text is enlarged so it reads
# from a distance, while crispness is preserved via a vector PDF plus a
# high-DPI PNG. Exported in addition to the manuscript figure.
SAVE_POSTER = True
POSTER_TEXT_SCALE = 1.3  # multiply every text size
POSTER_LINE_SCALE = 1.4  # thicken connector arrows to balance the larger text
POSTER_FIG_SCALE = 1.4  # enlarge the canvas so dense side columns gain headroom
POSTER_PNG_DPI = 600  # high enough to stay crisp at large print sizes


def debug_print(message):
    if DEBUG_PROGRESS:
        print(f"[fig01] {message}", flush=True)


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
                return load_data(
                    str(path),
                )
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
    g = nx.from_pandas_adjacency(phi)
    g.remove_edges_from(nx.selfloop_edges(g))
    for i, j, d in g.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["weight"] = w
        d["distance"] = float(-np.log(np.clip(w, 1e-12, 1.0)))
    mst = nx.maximum_spanning_tree(g)
    weights = np.array(
        [d.get("weight", 1.0) for _, _, d in g.edges(data=True)], dtype=float
    )
    if weights.size:
        cutoff = np.percentile(weights, 95)
        for i, j, d in g.edges(data=True):
            if float(d["weight"]) >= cutoff:
                mst.add_edge(i, j, **d)
    return mst, g


def _scale_linear(values, out_min, out_max):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(arr, 0.5 * (out_min + out_max), dtype=float)
    return out_min + (arr - lo) * (out_max - out_min) / (hi - lo)


def bridging_centrality(graph, distance="distance"):
    """Bridging centrality = betweenness * bridging coefficient."""
    betweenness = nx.betweenness_centrality(
        graph,
        weight=distance,
        normalized=True,
    )
    degree = dict(graph.degree())
    scores = {}
    for node in graph.nodes():
        k_node = int(degree.get(node, 0))
        if k_node <= 0:
            scores[node] = 0.0
            continue
        denom = 0.0
        for nbr in graph.neighbors(node):
            k_nbr = int(degree.get(nbr, 0))
            if k_nbr > 0:
                denom += 1.0 / float(k_nbr)
        bridge_coef = (1.0 / float(k_node)) / denom if denom > 0.0 else 0.0
        scores[node] = float(betweenness.get(node, 0.0)) * bridge_coef
    return scores


def sparsify_full_graph(graph, top_k=6, min_weight=None):
    """Keep strongest local ties per node, then re-add MST edges for connectivity."""
    sparse = nx.Graph()
    sparse.add_nodes_from(graph.nodes(data=True))
    for node in graph.nodes():
        neighbors = sorted(
            graph[node].items(),
            key=lambda kv: float(kv[1].get("weight", 0.0)),
            reverse=True,
        )
        kept = 0
        for nbr, data in neighbors:
            weight = float(data.get("weight", 0.0))
            if min_weight is not None and weight < float(min_weight):
                continue
            sparse.add_edge(node, nbr, **data)
            kept += 1
            if kept >= int(top_k):
                break

    mst_full = nx.maximum_spanning_tree(graph, weight="weight")
    for u, v, data in mst_full.edges(data=True):
        if not sparse.has_edge(u, v):
            sparse.add_edge(u, v, **data)
    return sparse


def detect_topic_communities(graph):
    try:
        communities = nx.community.louvain_communities(
            graph, weight="weight", resolution=1.0, seed=1991
        )
    except Exception:  # pragma: no cover
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")
    communities = [set(c) for c in communities]
    communities = sorted(
        communities, key=lambda c: (-len(c), sorted(c)[0] if c else "")
    )
    return communities


def _chaikin_closed(coords: np.ndarray, refinements: int = 2):
    ring = np.asarray(coords, dtype=float)
    if ring.shape[0] < 4:
        return ring
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    for _ in range(max(0, int(refinements))):
        out = []
        for i in range(ring.shape[0] - 1):
            p0 = ring[i]
            p1 = ring[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            out.extend([q, r])
        ring = np.vstack(out + [out[0]])
    return ring


def _fit_closed_spline(
    coords: np.ndarray, out_points: int = 220, smooth_scale: float = 0.004
):
    ring = np.asarray(coords, dtype=float)
    if ring.shape[0] < 4:
        return ring
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 4:
        return np.vstack([ring, ring[0]])

    # Remove near-duplicate vertices for stable spline fitting.
    keep = [0]
    for i in range(1, ring.shape[0]):
        if np.hypot(*(ring[i] - ring[keep[-1]])) > 1e-8:
            keep.append(i)
    ring = ring[keep]
    if ring.shape[0] < 4:
        return np.vstack([ring, ring[0]])

    # Arc-length parameterization for a closed periodic spline.
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
        # Robust fallback if periodic spline fit fails.
        ring2 = _chaikin_closed(np.vstack([ring, ring[0]]), refinements=2)
        return ring2


def build_cluster_hulls(communities, positions, padding):
    hulls = []
    for cid, nodes in enumerate(communities, start=1):
        pts = np.array([positions[n] for n in nodes if n in positions], dtype=float)
        if pts.shape[0] < 2:
            continue
        if HAS_SHAPELY:
            if pts.shape[0] == 2:
                poly = LineString(pts).buffer(
                    float(padding) * 0.9, cap_style=1, join_style=1
                )
            elif (
                pts.shape[0] == 3 and np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2
            ):
                poly = LineString(pts).buffer(
                    float(padding) * 0.9, cap_style=1, join_style=1
                )
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
            hulls.append({"cluster_id": cid, "nodes": set(nodes), "poly": poly})
        else:  # pragma: no cover
            if pts.shape[0] < 3:
                continue
            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            coords = pts[hull.vertices]
            coords = np.vstack([coords, coords[0]])
            coords = _chaikin_closed(coords, refinements=2)
            hulls.append({"cluster_id": cid, "nodes": set(nodes), "coords": coords})

    if HAS_SHAPELY:
        # Keep full padded hulls so every assigned node remains enclosed.
        # Fit smooth closed splines on padded hull boundaries for drawing.
        for item in hulls:
            coords = np.asarray(item["poly"].exterior.coords, dtype=float)
            spline = _fit_closed_spline(coords, out_points=240, smooth_scale=0.003)
            if spline.shape[0] >= 4:
                spoly = Polygon(spline)
                if not spoly.is_valid:
                    spoly = spoly.buffer(0)
                if (not spoly.is_empty) and float(spoly.area) > 1e-8:
                    # Enforce coverage of all assigned nodes for visual demarcation.
                    node_pts = np.array(
                        [positions[n] for n in item["nodes"] if n in positions],
                        dtype=float,
                    )
                    max_miss = 0.0
                    for px, py in node_pts:
                        p = Point(float(px), float(py))
                        if not spoly.covers(p):
                            max_miss = max(max_miss, float(spoly.distance(p)))
                    if max_miss > 0.0:
                        spoly = spoly.buffer(
                            max_miss + 0.22 * float(padding),
                            join_style=1,
                        ).buffer(0)
                    item["poly_spline"] = spoly
                    item["outline_coords"] = np.asarray(
                        spoly.exterior.coords, dtype=float
                    )
                    continue
            item["outline_coords"] = coords
    return hulls


def _short_theme_name(name: str) -> str:
    short = {
        "Environmental Protection": "Environmental",
        "Marine & Wildlife": "Marine/Wildlife",
        "Operations & Safety": "Operations",
        "Governance & Legal": "Governance",
        "Science & Research": "Science",
        "Tourism & Human Activity": "Tourism",
        "Infrastructure & Planning": "Planning",
        "Resource Extraction": "Resource",
    }
    return short.get(name, name)


def _theme_semantic_name(name: str) -> str:
    semantic = {
        "Environmental Protection": "Stewardship & impact",
        "Marine & Wildlife": "Marine ecosystems",
        "Operations & Safety": "Operations & safety",
        "Governance & Legal": "Governance process",
        "Science & Research": "Scientific evidence",
        "Tourism & Human Activity": "Tourism management",
        "Infrastructure & Planning": "Infrastructure planning",
        "Resource Extraction": "Resource politics",
    }
    return semantic.get(name, _short_theme_name(name))


def cluster_theme_counts(nodes, topic_to_theme):
    counts = {}
    for node in nodes:
        theme = topic_to_theme.get(node, "Governance & Legal")
        counts[theme] = counts.get(theme, 0) + 1
    return counts


def cluster_dominant_theme(nodes, topic_to_theme):
    counts = cluster_theme_counts(nodes, topic_to_theme)
    if not counts:
        return "Governance & Legal"
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0]


def semantic_cluster_label(nodes, topic_to_theme):
    counts = cluster_theme_counts(nodes, topic_to_theme)
    if not counts:
        return "Mixed cluster"
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_name, top_n = ranked[0]
    if len(ranked) == 1:
        return _theme_semantic_name(top_name)
    second_name, second_n = ranked[1]
    if second_n / max(top_n, 1) < 0.5:
        return _theme_semantic_name(top_name)
    return f"{_theme_semantic_name(top_name)}\n+ {_theme_semantic_name(second_name)}"


def longest_leaf_path(tree, weight="distance"):
    leaves = [n for n, d in tree.degree() if d == 1]
    if len(leaves) < 2:
        return list(tree.nodes())
    best_dist = -np.inf
    best_path = None
    for i, u in enumerate(leaves):
        lengths, paths = nx.single_source_dijkstra(tree, u, weight=weight)
        for v in leaves[i + 1 :]:
            dist = lengths.get(v)
            if dist is None:
                continue
            if dist > best_dist:
                best_dist = float(dist)
                best_path = paths[v]
    return list(best_path) if best_path else list(tree.nodes())


def normalize_topic_key(name):
    if name is None:
        return ""
    text = str(name).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return text


def detect_structural_regions(backbone, full_graph, topic_to_theme):
    node_lookup = {normalize_topic_key(n): n for n in backbone.nodes()}

    def resolve_nodes(topic_names):
        resolved = set()
        for name in topic_names:
            key = normalize_topic_key(name)
            if key in node_lookup:
                resolved.add(node_lookup[key])
        return resolved

    regions = [
        dict(
            kind="branch",
            label="Drilling, Monitoring\n& CEP Oversight",
            color="#1f77b4",
            nodes=resolve_nodes(
                [
                    "Drilling",
                    "Sub glacial Lakes",
                    "Operation of the CEP",
                    "Environmental Domains Analysis",
                    "Marine Acoustics",
                    "State of the Antarctic Environment Report SAER",
                ]
            ),
        ),
        dict(
            kind="branch",
            label="Human Impact\n& Marine Stewardship",
            color="#F57C00",
            nodes=resolve_nodes(
                [
                    "Tourism and NG Activities",
                    "Marine Protected Areas",
                    "Marine living resources",
                    "Prevention of marine pollution",
                    "Site Guidelines for Visitors",
                    "Mineral resources",
                    "Multiyear strategic workplan",
                    "Human Footprint and wilderness values",
                    "Search and Rescue",
                ]
            ),
        ),
        dict(
            kind="branch",
            label="Environmental\nProtection",
            color="#2E7D32",
            nodes=resolve_nodes(
                [
                    "Nonnative Species and Quarantine",
                    "Specially Protected Species",
                    "Climate Change",
                    "CEP Strategy Discussions",
                    "Biological Prospecting",
                    "Fauna and Flora General",
                    "Repair and remediation of environmental damage",
                    "Operation of the Antarctic Treaty system Reports",
                    "Cooperation with Other Organisations",
                ]
            ),
        ),
        dict(
            kind="core",
            label="Procedural core",
            color="#6A1B9A",
            nodes=resolve_nodes(
                [
                    "Operational issues",
                    "Liability",
                    "Educational issues",
                    "Exchange of Information",
                    "Environmental Protection General",
                    "Opening statements",
                    "International Polar Year",
                    "Waste management and disposal",
                    "Inspections",
                    "Environmental Impact Assessment EIA Other EIA Matters",
                    "Science issues",
                    "Management Plans",
                    "Environmental Monitoring and Reporting",
                    "Safety and Operations in Antarctica",
                    "Comprehensive Environmental Evaluations",
                    "Emergency report and contingency planning",
                    "Institutional and legal matters",
                    "Area Protection and Management Plans General",
                    "Historic Sites and Monuments",
                    "Operation of the Antarctic Treaty system General",
                    "Operation of the Antarctic Treaty system The Secretariat",
                ]
            ),
        ),
    ]

    assigned = set().union(*(r["nodes"] for r in regions))
    leftovers = set(backbone.nodes()) - assigned
    if leftovers:
        for region in regions:
            if region["kind"] == "core":
                region["nodes"].update(leftovers)
                break

    regions = [r for r in regions if len(r["nodes"]) >= 2]
    return regions


def draw_cluster_hulls(ax, hulls, color_by_id):
    artists_by_cluster = {}
    for item in hulls:
        cluster_id = int(item["cluster_id"])
        color = color_by_id.get(item["cluster_id"], "#777777")
        if HAS_SHAPELY:
            outline = np.asarray(item.get("outline_coords"), dtype=float)
            if outline.ndim == 2 and outline.shape[1] == 2 and outline.shape[0] >= 3:
                x, y = outline[:, 0], outline[:, 1]
            else:
                x, y = item["poly"].exterior.xy
            patches = ax.fill(
                x,
                y,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.14,
                zorder=0.18,
            )
        else:  # pragma: no cover
            xy = np.asarray(item["coords"], dtype=float)
            patches = ax.fill(
                xy[:, 0],
                xy[:, 1],
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.14,
                zorder=0.18,
            )
        artists_by_cluster.setdefault(cluster_id, []).extend(list(patches))
    return artists_by_cluster


def get_hull_bounds(hulls):
    if not hulls:
        return None
    mins = []
    maxs = []
    for item in hulls:
        if HAS_SHAPELY and "poly" in item:
            if "outline_coords" in item:
                xy = np.asarray(item["outline_coords"], dtype=float)
                if xy.ndim == 2 and xy.shape[1] == 2 and xy.shape[0] >= 2:
                    mins.append(np.min(xy, axis=0))
                    maxs.append(np.max(xy, axis=0))
                    continue
            bounds = item["poly"].bounds
            mins.append(np.array([bounds[0], bounds[1]], dtype=float))
            maxs.append(np.array([bounds[2], bounds[3]], dtype=float))
        else:  # pragma: no cover
            xy = np.asarray(item.get("coords", []), dtype=float)
            if xy.ndim == 2 and xy.shape[1] == 2 and xy.shape[0] >= 2:
                mins.append(np.min(xy, axis=0))
                maxs.append(np.max(xy, axis=0))
    if not mins:
        return None
    mins = np.vstack(mins)
    maxs = np.vstack(maxs)
    return (
        float(np.min(mins[:, 0])),
        float(np.max(maxs[:, 0])),
        float(np.min(mins[:, 1])),
        float(np.max(maxs[:, 1])),
    )


def draw_cluster_semantic_labels(ax, hulls, color_by_id, label_by_id, mask_extent):
    from matplotlib.patches import FancyArrowPatch

    width = float(mask_extent[1] - mask_extent[0])
    height = float(mask_extent[3] - mask_extent[2])
    min_dx = 0.10 * width
    min_dy = 0.08 * height
    label_nudges = {
        "Environmental\nProtection": (0.18 * width, -0.18 * height),
        "Drilling, Monitoring\n& CEP Oversight": (-0.18 * width, 0.0),
        "Human Impact\n& Marine Stewardship": (-0.10 * width, 0.0),
        "Procedural core": (0.18 * width, 0.06 * height),
    }
    artists_by_cluster = {}
    placed = []
    for item in hulls:
        cluster_id = int(item["cluster_id"])
        color = color_by_id.get(item["cluster_id"], "#777777")
        label = label_by_id.get(item["cluster_id"], "Cluster")
        if HAS_SHAPELY:
            poly = item["poly"]
            exterior = np.asarray(poly.exterior.coords, dtype=float)
            if exterior.shape[0] < 3:
                continue
            center = poly.representative_point()
            cx, cy = float(center.x), float(center.y)
            idx = int(np.argmax(exterior[:, 1]))
            anchor = exterior[idx]
            prev_pt = exterior[idx - 1]
            next_pt = exterior[(idx + 1) % (exterior.shape[0] - 1)]
        else:  # pragma: no cover
            xy = np.asarray(item["coords"], dtype=float)
            if xy.shape[0] < 3:
                continue
            cx, cy = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
            idx = int(np.argmax(xy[:, 1]))
            anchor = xy[idx]
            prev_pt = xy[idx - 1]
            next_pt = xy[(idx + 1) % (xy.shape[0] - 1)]

        tangent = np.asarray(next_pt - prev_pt, dtype=float)
        tnorm = float(np.hypot(tangent[0], tangent[1]))
        if tnorm < 1e-9:
            tangent = np.array([1.0, 0.0], dtype=float)
            tnorm = 1.0
        tangent /= tnorm
        angle = float(np.degrees(np.arctan2(tangent[1], tangent[0])))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        radial = np.asarray(anchor - np.array([cx, cy]), dtype=float)
        rnorm = float(np.hypot(radial[0], radial[1]))
        if rnorm < 1e-9:
            radial = np.array([0.0, 1.0], dtype=float)
            rnorm = 1.0
        radial /= rnorm
        lx, ly = anchor + radial * (0.06 * max(width, height))

        dx_dy = label_nudges.get(label)
        if dx_dy is not None:
            dx, dy = dx_dy
            lx += float(dx)
            if label == "Environmental\nProtection":
                # Pull toward shape center in y for better in-hull readability.
                ly = 0.55 * float(cy) + 0.45 * float(ly) + float(dy)
            else:
                ly += float(dy)

        for px, py in placed:
            if abs(lx - px) < min_dx and abs(ly - py) < min_dy:
                ly += 0.75 * min_dy
        lx = float(
            np.clip(lx, mask_extent[0] + 0.02 * width, mask_extent[1] - 0.02 * width)
        )
        ly = float(
            np.clip(ly, mask_extent[2] + 0.03 * height, mask_extent[3] - 0.03 * height)
        )
        placed.append((lx, ly))

        # For moved labels, target the nearest hull point and pull slightly
        # inward so the connector stays short and reads as attached "inside"
        # the outline rather than jumping to a distant top anchor.
        if label in {"Environmental\nProtection", "Procedural core"}:
            boundary = np.asarray(exterior[:-1], dtype=float)
            if boundary.ndim == 2 and boundary.shape[0] > 0:
                d2 = (boundary[:, 0] - lx) ** 2 + (boundary[:, 1] - ly) ** 2
                near_idx = int(np.argmin(d2))
                boundary_anchor = boundary[near_idx]
                center_vec = np.array([cx, cy], dtype=float)
                anchor = 0.86 * boundary_anchor + 0.14 * center_vec

        txt = ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            rotation=angle,
            rotation_mode="anchor",
            fontsize=8.6,
            color="0.1",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.7,
                "alpha": 1,
            },
            zorder=2.1,
        )
        artists_by_cluster.setdefault(cluster_id, []).append(txt)

        # Removed leader connectors for cleaner label-only semantic regions.
    return artists_by_cluster


def tessellate_mask(png_path, stride=6):
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    # Premultiply RGB by alpha to avoid white halo fringes on transparent export.
    premult = arr.copy()
    alpha_frac = premult[:, :, 3:4].astype(np.float32) / 255.0
    premult[:, :, :3] = np.clip(
        premult[:, :, :3].astype(np.float32) * alpha_frac, 0.0, 255.0
    ).astype(np.uint8)
    img = Image.fromarray(premult, mode="RGBA")
    arr = premult
    alpha = arr[:, :, 3]
    mask = alpha > 10

    # Erode one pixel so we are inside the boundary.
    m = mask.astype(np.uint8)
    up = np.pad(m[1:, :], ((0, 1), (0, 0)), mode="constant")
    down = np.pad(m[:-1, :], ((1, 0), (0, 0)), mode="constant")
    left = np.pad(m[:, 1:], ((0, 0), (0, 1)), mode="constant")
    right = np.pad(m[:, :-1], ((0, 0), (1, 0)), mode="constant")
    inner = (m == 1) & (up == 1) & (down == 1) & (left == 1) & (right == 1)

    ys, xs = np.where(inner)
    coords = np.vstack([xs, ys]).T

    # Tessellate by subsampling a grid of interior pixels.
    coords = coords[(coords[:, 0] % stride == 0) & (coords[:, 1] % stride == 0)]
    if len(coords) == 0:
        raise ValueError("No interior points found for tessellation.")

    coords = coords.astype(float)
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (
        coords[:, 0].max() - coords[:, 0].min()
    )
    coords[:, 1] = (coords[:, 1].max() - coords[:, 1]) / (
        coords[:, 1].max() - coords[:, 1].min()
    )

    aspect = arr.shape[0] / arr.shape[1]
    target_width = 7.0
    target_height = target_width * aspect
    coords[:, 0] = (coords[:, 0] - 0.5) * target_width
    coords[:, 1] = (coords[:, 1] - 0.5) * target_height

    extent = [
        -target_width / 2,
        target_width / 2,
        -target_height / 2,
        target_height / 2,
    ]
    return coords, img, extent


def snap_to_tessellation(pos, tess_points):
    nodes = list(pos.keys())
    pos_arr = np.array([pos[n] for n in nodes])

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(tess_points)
        _, idx = tree.query(pos_arr, k=1)
    except Exception:
        d = ((pos_arr[:, None, :] - tess_points[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d, axis=1)

    snapped = {n: tess_points[i] for n, i in zip(nodes, idx)}
    return snapped


def deterministic_layout(g):
    """Deterministic Kamada-Kawai layout with circular initialization."""
    init = nx.circular_layout(g, scale=0.5)
    return nx.kamada_kawai_layout(g, pos=init, weight="weight")


_t0 = time.perf_counter()
debug_print("Building concern-space graphs...")
mst, full_graph = build_graphs()
debug_print(f"Built graphs in {time.perf_counter() - _t0:.2f}s")

_t0 = time.perf_counter()
debug_print("Loading saved node layout...")
saved_pos = load_saved_layout_positions(mst)
debug_print(
    "Loaded saved node layout in " f"{time.perf_counter() - _t0:.2f}s"
    if saved_pos is not None
    else f"No saved node layout found ({time.perf_counter() - _t0:.2f}s)"
)
if not USE_SAVED_NODE_LAYOUT:
    debug_print(
        "Ignoring saved node layout; using deterministic layout to preserve figure geometry."
    )
    saved_pos = None
if saved_pos is None:
    _t0 = time.perf_counter()
    debug_print("Computing deterministic node layout...")
    pos = deterministic_layout(mst)
    debug_print(
        f"Computed deterministic node layout in {time.perf_counter() - _t0:.2f}s"
    )
else:
    pos = saved_pos

fp = Path("1024px-AntarcticaContour.svg.png")
# fp = Path("taklsdfj;asdf.png")
if fp.exists():
    _t0 = time.perf_counter()
    debug_print("Loading Antarctica mask and tessellation...")
    tess_points, mask_img, mask_extent = tessellate_mask(fp)
    debug_print(f"Prepared mask/tessellation in {time.perf_counter() - _t0:.2f}s")

    # Map Kamada-Kawai positions to tessellated points.
    _t0 = time.perf_counter()
    debug_print("Snapping graph layout to Antarctica tessellation...")
    snapped = snap_to_tessellation(pos, tess_points)
    x_min, x_max, y_min, y_max = mask_extent
    snapped = {
        n: np.array([x_min + x_max - p[0], y_min + y_max - p[1]])
        for n, p in snapped.items()
    }

    # Rotate snapped positions by 45 degrees around the center of the extent.
    theta = np.deg2rad(45.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    snapped = {
        n: np.array(
            [
                cx + (p[0] - cx) * cos_t - (p[1] - cy) * sin_t,
                cy + (p[0] - cx) * sin_t + (p[1] - cy) * cos_t,
            ]
        )
        for n, p in snapped.items()
    }

    # Scale snapped positions to make the graph occupy more of the map.
    scale = 4
    snapped = {
        n: np.array([cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale])
        for n, p in snapped.items()
    }
    debug_print(f"Snapped and transformed layout in {time.perf_counter() - _t0:.2f}s")
else:
    print(f"Warning: mask image not found at {fp}. Falling back to graph-only layout.")
    mask_img = None
    snapped = {n: np.array(p, dtype=float) * 4.0 for n, p in pos.items()}
    coords = np.array(list(snapped.values()))
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    mask_extent = [
        x_min - 0.3 * dx,
        x_max + 0.3 * dx,
        y_min - 0.3 * dy,
        y_max + 0.3 * dy,
    ]

theme_colors = {
    "Environmental Protection": "#2E7D32",
    "Marine & Wildlife": "#0277BD",
    "Operations & Safety": "#F57C00",
    "Governance & Legal": "#6A1B9A",
    "Science & Research": "#C62828",
    "Tourism & Human Activity": "#D84315",
    "Infrastructure & Planning": "#5D4037",
    "Resource Extraction": "#00838F",
}

topic_to_theme = {
    "State of the Antarctic Environment Report SAER": "Environmental Protection",
    "Management Plans": "Governance & Legal",
    "Biological Prospecting": "Resource Extraction",
    "Climate Change": "Environmental Protection",
    "Environmental Domains Analysis": "Environmental Protection",
    "Educational issues": "Governance & Legal",
    "Comprehensive Environmental Evaluations": "Environmental Protection",
    "Site Guidelines for Visitors": "Tourism & Human Activity",
    "Repair and remediation of environmental damage": "Environmental Protection",
    "Multiyear strategic workplan": "Governance & Legal",
    "Inspections": "Operations & Safety",
    "Sub glacial Lakes": "Science & Research",
    "Drilling": "Resource Extraction",
    "Opening statements": "Governance & Legal",
    "Operation of the Antarctic Treaty system General": "Governance & Legal",
    "CEP Strategy Discussions": "Governance & Legal",
    "Fauna and Flora General": "Marine & Wildlife",
    "Historic Sites and Monuments": "Infrastructure & Planning",
    "Operational issues": "Operations & Safety",
    "Operation of the Antarctic Treaty system Reports": "Governance & Legal",
    "Science issues": "Science & Research",
    "International Polar Year": "Science & Research",
    "Liability": "Governance & Legal",
    "Prevention of marine pollution": "Environmental Protection",
    "Safety and Operations in Antarctica": "Operations & Safety",
    "Marine living resources": "Resource Extraction",
    "Institutional and legal matters": "Governance & Legal",
    "Nonnative Species and Quarantine": "Environmental Protection",
    "Marine Protected Areas": "Marine & Wildlife",
    "Tourism and NG Activities": "Tourism & Human Activity",
    "Cooperation with Other Organisations": "Governance & Legal",
    "Environmental Impact Assessment EIA Other EIA Matters": "Environmental Protection",
    "Specially Protected Species": "Marine & Wildlife",
    "Marine Acoustics": "Science & Research",
    "Mineral resources": "Resource Extraction",
    "Environmental Monitoring and Reporting": "Environmental Protection",
    "Exchange of Information": "Governance & Legal",
    "Area Protection and Management Plans General": "Infrastructure & Planning",
    "Operation of the CEP": "Governance & Legal",
    "Waste management and disposal": "Operations & Safety",
    "Human Footprint and wilderness values": "Environmental Protection",
    "Search and Rescue": "Operations & Safety",
    "Environmental Protection General": "Environmental Protection",
    "Emergency report and contingency planning": "Operations & Safety",
    "Operation of the Antarctic Treaty system The Secretariat": "Governance & Legal",
}

node_colors = [
    theme_colors[topic_to_theme.get(node, "Governance & Legal")] for node in mst.nodes()
]

all_weights = np.array(
    [float(d.get("weight", 0.0)) for _, _, d in full_graph.edges(data=True)],
    dtype=float,
)
min_sparse_weight = float(np.percentile(all_weights, 75)) if all_weights.size else None
sparse_full_graph = sparsify_full_graph(
    full_graph,
    top_k=6,
    min_weight=min_sparse_weight,
)
weighted_degree_scores = dict(full_graph.degree(weight="weight"))
node_sizes = _scale_linear(
    [weighted_degree_scores.get(n, 0.0) for n in mst.nodes()], 190, 600
)
node_sizes_draw = node_sizes * 8 + 10

structural_regions = detect_structural_regions(mst, full_graph, topic_to_theme)
region_nodes = [r["nodes"] for r in structural_regions]
extent_w = float(mask_extent[1] - mask_extent[0])
extent_h = float(mask_extent[3] - mask_extent[2])
coords_arr = np.array(list(snapped.values()), dtype=float)
graph_span = float(
    max(np.ptp(coords_arr[:, 0]), np.ptp(coords_arr[:, 1])) if coords_arr.size else 1.0
)
max_marker_radius_pts = float(np.sqrt(np.max(node_sizes_draw) / np.pi))
# Approximate conversion from marker radius (points) to data units.
marker_radius_data = max_marker_radius_pts * (graph_span / 780.0)
hull_padding = max(0.022 * max(extent_w, extent_h), 1.25 * marker_radius_data)
cluster_hulls = build_cluster_hulls(region_nodes, snapped, padding=hull_padding)
hull_bounds = get_hull_bounds(cluster_hulls)
cluster_color_by_id = {i + 1: r["color"] for i, r in enumerate(structural_regions)}
cluster_label_by_id = {i + 1: r["label"] for i, r in enumerate(structural_regions)}

fig, ax = uplt.subplots(width="30cm")
# Draw graph without the highlighted edge so only the dashed version shows.
edge_a = "Drilling"
edge_b = "Marine Acoustics"
mst_draw = mst.copy()
if mst_draw.has_edge(edge_a, edge_b):
    mst_draw.remove_edge(edge_a, edge_b)
edge_widths = np.array(
    [float(mst_draw[u][v].get("weight", 1.0)) * 8 for u, v in mst_draw.edges()]
)
hull_artists_by_cluster = draw_cluster_hulls(ax, cluster_hulls, cluster_color_by_id)
ax.graph(
    mst_draw,
    snapped,
    node_kw=dict(node_size=node_sizes, node_color=node_colors),
    edge_kw=dict(width=edge_widths),
    rescale=0,
)

o = ax.collections[0].get_sizes()
ax.collections[0].set_sizes(o * 8 + 10)
# Keep vertices above connector arrows.
ax.collections[0].set_zorder(3.2)


semantic_label_artists_by_cluster = draw_cluster_semantic_labels(
    ax,
    cluster_hulls,
    cluster_color_by_id,
    cluster_label_by_id,
    mask_extent=mask_extent,
)
offset = 0
inax = ax.inset((0 - offset, 0 - offset, 1 + offset, 1 + offset), zoom=0)
if mask_img is not None:
    inax.imshow(
        mask_img,
        extent=mask_extent,
        alpha=0.18,
        zorder=-1,
        interpolation="bilinear",
    )
inax.axis("off")
inax.set_facecolor("none")
for spine in inax.spines.values():
    spine.set_visible(False)


def smartwrap(text, width):
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def build_interest_groups(edge_a, edge_b):
    """Group countries by RCA>1 specialization in drilling/marine acoustics."""
    counts_df, _, _, _ = load_data_with_fallback()
    counts_df = filter_space_topics(counts_df)
    rca_df = get_rca(counts_df)

    groups = {"drilling": [], "both": [], "marine": []}
    for country in counts_df.columns:
        rca_a = float(rca_df.loc[edge_a, country])
        rca_b = float(rca_df.loc[edge_b, country])
        has_a = rca_a > 1.0
        has_b = rca_b > 1.0
        if not (has_a or has_b):
            continue
        if has_a and has_b:
            key = "both"
        elif has_a:
            key = "drilling"
        else:
            key = "marine"
        groups[key].append(
            dict(
                country=country,
                rca_a=rca_a,
                rca_b=rca_b,
                strength=max(rca_a, rca_b),
            )
        )

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x["strength"], reverse=True)
    return groups


def draw_grouped_rca_inset(
    ax, edge_a, edge_b, groups, callout_color="#E0007A", callout_linewidth=1.8
):
    """Inset: countries grouped by single vs joint RCA>1 interest."""
    from matplotlib.colors import to_rgb
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    col_a = theme_colors[topic_to_theme.get(edge_a, "Governance & Legal")]
    col_b = theme_colors[topic_to_theme.get(edge_b, "Governance & Legal")]
    col_mix = tuple((np.array(to_rgb(col_a)) + np.array(to_rgb(col_b))) / 2.0)
    # Keep edge semantics fixed for presentation: drilling=blue, marine=red, both=purple.
    edge_semantic = {"drilling": "#1f77b4", "both": "#7b3294", "marine": "#d62728"}

    # Topic anchors at the top.
    anc_a = (0.28, 0.94)
    anc_b = (0.72, 0.94)
    anc_mix = ((anc_a[0] + anc_b[0]) / 2, 0.94)

    ax.plot(
        [anc_a[0], anc_b[0]],
        [anc_a[1], anc_b[1]],
        ls="--",
        lw=callout_linewidth,
        c=callout_color,
        alpha=0.95,
        zorder=1,
    )
    ax.scatter(
        [anc_a[0], anc_b[0]], [anc_a[1], anc_b[1]], s=70, c=[col_a, col_b], zorder=3
    )
    ax.text(
        anc_a[0],
        1.04,
        smartwrap(edge_a, 14),
        ha="center",
        va="top",
        fontsize=9.0,
        clip_on=False,
    )
    ax.text(
        anc_b[0],
        1.04,
        smartwrap(edge_b, 16),
        ha="center",
        va="top",
        fontsize=9.0,
        clip_on=False,
    )

    x_group = {"drilling": 0.20, "both": 0.50, "marine": 0.80}
    edge_color = edge_semantic
    src_anchor = {"drilling": anc_a, "both": anc_mix, "marine": anc_b}
    title = {
        "drilling": "RCA>1 drilling",
        "both": "RCA>1 both",
        "marine": "RCA>1 marine",
    }

    for key in ("drilling", "both", "marine"):
        members = groups.get(key, [])
        if members:
            ys = (
                np.linspace(0.82, 0.10, len(members))
                if len(members) > 1
                else np.array([0.46])
            )
        else:
            ys = np.array([])

        # ax.text(
        #     x_group[key],
        #     0.89,
        #     title[key],
        #     ha="center",
        #     va="bottom",
        #     fontsize=5.3,
        #     color=edge_color[key],
        # )

        for y, rec in zip(ys, members):
            if key == "both":
                # Mixed-interest countries link back to both highlighted topics.
                for sx, sy in (anc_a, anc_b):
                    ax.plot(
                        [sx, x_group[key]],
                        [sy, y],
                        color=edge_color[key],
                        lw=1.0,
                        alpha=0.92,
                        zorder=1,
                    )
            else:
                sx, sy = src_anchor[key]
                ax.plot(
                    [sx, x_group[key]],
                    [sy, y],
                    color=edge_color[key],
                    lw=1.0,
                    alpha=0.92,
                    zorder=1,
                )
            img = load_flag(rec["country"], save=True, base="./assets/flags")
            if img is not None:
                zoom = 30 / max(img.shape[:2])
                ab = AnnotationBbox(
                    OffsetImage(img, zoom=zoom),
                    (x_group[key], y),
                    frameon=True,
                    bboxprops=dict(edgecolor="black", linewidth=1.0),
                    pad=0,
                    zorder=4,
                )
                ax.add_artist(ab)
            else:
                ax.scatter(
                    [x_group[key]],
                    [y],
                    s=28,
                    c=[edge_color[key]],
                    edgecolors="k",
                    linewidths=0.7,
                    zorder=4,
                )

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor((1, 1, 1, 0.88))
    for spine in ax.spines.values():
        spine.set_edgecolor("0.5")
        spine.set_linewidth(0.6)


# Simple, evenly spaced labels around the map.
rect_x_min, rect_x_max, rect_y_min, rect_y_max = mask_extent
r = snapped
cx, cy = (rect_x_min + rect_x_max) / 2, (rect_y_min + rect_y_max) / 2

top_nodes = []
right_nodes = []
bottom_nodes = []
left_nodes = []
r = {k.replace("_", " "): v for k, v in r.items()}
for node, (x, y) in r.items():
    angle = np.degrees(np.arctan2(y - cy, x - cx)) % 360
    if 45 <= angle < 135:
        top_nodes.append(node)
    elif 135 <= angle < 225:
        left_nodes.append(node)
    elif 225 <= angle < 315:
        bottom_nodes.append(node)
    else:
        right_nodes.append(node)

top_nodes.sort(key=lambda n: r[n][0])
right_nodes.sort(key=lambda n: r[n][1])
bottom_nodes.sort(key=lambda n: r[n][0])
left_nodes.sort(key=lambda n: r[n][1])

pad = 0.22 * (rect_x_max - rect_x_min)
side_fs = 11.6
SPREAD_BLEND = 0.62
LEFT_SIDE_SPREAD_BLEND = 0.18
RIGHT_SIDE_SPREAD_BLEND = 0.18


def monotone_packed_positions(desired, start, end, min_gap):
    desired = np.asarray(desired, dtype=float)
    n = desired.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        center = float(np.clip(desired[0], start, end))
        return np.array([center], dtype=float)

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


def spaced_monotone_positions(desired, start, end, min_gap, blend=SPREAD_BLEND):
    desired = np.asarray(desired, dtype=float)
    n = desired.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        center = float(np.clip(desired[0], start, end))
        return np.array([center], dtype=float)

    even = np.linspace(start, end, n)
    target = float(blend) * desired + (1.0 - float(blend)) * even
    return monotone_packed_positions(target, start, end, min_gap)


def build_side_layouts(
    positions,
    top_nodes,
    right_nodes,
    bottom_nodes,
    left_nodes,
    rect_bounds,
    pad,
):
    rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect_bounds
    width = rect_x_max - rect_x_min
    height = rect_y_max - rect_y_min
    top_label_gap_x = 0.046 * width
    bottom_label_gap_x = 0.046 * width
    label_gap_y = 0.06 * height
    left_label_gap_y = 0.102 * height
    right_label_gap_y = 0.108 * height
    top_label_start = rect_x_min - 2.10 * pad
    top_label_end = rect_x_max + 2.10 * pad
    bottom_label_start = rect_x_min - 2.10 * pad
    bottom_label_end = rect_x_max + 2.10 * pad
    left_label_start = rect_y_min - 5.10 * pad
    left_label_end = rect_y_max + 5.10 * pad
    right_label_start = rect_y_min - 5.30 * pad
    right_label_end = rect_y_max + 5.30 * pad

    top_label = spaced_monotone_positions(
        [positions[node][0] for node in top_nodes],
        top_label_start,
        top_label_end,
        top_label_gap_x,
    )
    bottom_label = spaced_monotone_positions(
        [positions[node][0] for node in bottom_nodes],
        bottom_label_start,
        bottom_label_end,
        bottom_label_gap_x,
    )
    left_label = spaced_monotone_positions(
        [positions[node][1] for node in left_nodes],
        left_label_start,
        left_label_end,
        left_label_gap_y,
        blend=LEFT_SIDE_SPREAD_BLEND,
    )
    right_label = spaced_monotone_positions(
        [positions[node][1] for node in right_nodes],
        right_label_start,
        right_label_end,
        right_label_gap_y,
        blend=RIGHT_SIDE_SPREAD_BLEND,
    )
    left_label = np.clip(
        left_label + 0.34 * pad,
        left_label_start,
        left_label_end,
    )

    left_preentry = (
        spaced_monotone_positions(
            [positions[node][1] for node in left_nodes],
            rect_y_min + 0.02 * height,
            rect_y_max - 0.02 * height,
            0.035 * height,
            blend=0.65,
        )
        if len(left_nodes)
        else np.array([])
    )
    if len(left_preentry):
        left_preentry = np.clip(
            left_preentry + 0.20 * pad,
            rect_y_min + 0.02 * height,
            rect_y_max - 0.02 * height,
        )

    layout = {
        "top": {
            "nodes": top_nodes,
            "label_positions": top_label,
            "route_values": (
                np.linspace(
                    rect_y_max + 0.04 * pad, rect_y_max + 0.98 * pad, len(top_nodes)
                )
                if len(top_nodes)
                else np.array([])
            ),
            "entry_extras": (
                np.linspace(0.0, 0.24 * pad, len(top_nodes))
                if len(top_nodes)
                else np.array([])
            ),
            "pre_entry_values": (
                spaced_monotone_positions(
                    [positions[node][0] for node in top_nodes],
                    rect_x_min + 0.02 * width,
                    rect_x_max - 0.02 * width,
                    0.035 * width,
                    blend=0.65,
                )
                if len(top_nodes)
                else np.array([])
            ),
        },
        "right": {
            "nodes": right_nodes,
            "label_positions": right_label,
            "route_values": (
                np.linspace(
                    rect_x_max + 0.04 * pad, rect_x_max + 0.98 * pad, len(right_nodes)
                )
                if len(right_nodes)
                else np.array([])
            ),
            "entry_extras": (
                np.linspace(0.0, 0.24 * pad, len(right_nodes))
                if len(right_nodes)
                else np.array([])
            ),
            "pre_entry_values": (
                spaced_monotone_positions(
                    [positions[node][1] for node in right_nodes],
                    rect_y_min + 0.02 * height,
                    rect_y_max - 0.02 * height,
                    0.035 * height,
                    blend=0.65,
                )
                if len(right_nodes)
                else np.array([])
            ),
        },
        "bottom": {
            "nodes": bottom_nodes,
            "label_positions": bottom_label,
            "route_values": (
                np.linspace(
                    rect_y_min - 0.04 * pad, rect_y_min - 0.98 * pad, len(bottom_nodes)
                )
                if len(bottom_nodes)
                else np.array([])
            ),
            "entry_extras": (
                np.linspace(0.0, 0.24 * pad, len(bottom_nodes))
                if len(bottom_nodes)
                else np.array([])
            ),
            "pre_entry_values": (
                spaced_monotone_positions(
                    [positions[node][0] for node in bottom_nodes],
                    rect_x_min + 0.02 * width,
                    rect_x_max - 0.02 * width,
                    0.035 * width,
                    blend=0.65,
                )
                if len(bottom_nodes)
                else np.array([])
            ),
        },
        "left": {
            "nodes": left_nodes,
            "label_positions": left_label,
            "route_values": (
                np.linspace(
                    rect_x_min - 0.04 * pad, rect_x_min - 0.98 * pad, len(left_nodes)
                )
                if len(left_nodes)
                else np.array([])
            ),
            "entry_extras": (
                np.linspace(0.0, 0.24 * pad, len(left_nodes))
                if len(left_nodes)
                else np.array([])
            ),
            "pre_entry_values": left_preentry,
        },
    }

    def _reorder_side_payload(payload):
        n = len(payload["nodes"])
        if n <= 1:
            return payload
        order = np.argsort(np.asarray(payload["pre_entry_values"], dtype=float))
        payload["nodes"] = [payload["nodes"][i] for i in order]
        payload["label_positions"] = np.asarray(payload["label_positions"])[order]
        payload["route_values"] = np.asarray(payload["route_values"])[order]
        payload["entry_extras"] = np.asarray(payload["entry_extras"])[order]
        payload["pre_entry_values"] = np.asarray(payload["pre_entry_values"])[order]
        return payload

    for side in ("top", "right", "bottom", "left"):
        layout[side] = _reorder_side_payload(layout[side])

    def _rebuild_monotone_side(side):
        payload = layout[side]
        if len(payload["nodes"]) <= 1:
            return payload

        if side in ("left", "right"):
            axis_vals = np.asarray([positions[node][1] for node in payload["nodes"]])
            if side == "right":
                label_start, label_end, label_gap = (
                    right_label_start,
                    right_label_end,
                    right_label_gap_y,
                )
            else:
                label_start, label_end, label_gap = (
                    rect_y_min - pad,
                    rect_y_max + pad,
                    left_label_gap_y,
                )
            pre_start, pre_end, pre_gap = (
                rect_y_min + 0.02 * height,
                rect_y_max - 0.02 * height,
                0.035 * height,
            )
        elif side in ("top", "bottom"):
            axis_vals = np.asarray([positions[node][0] for node in payload["nodes"]])
            if side == "top":
                label_start, label_end, label_gap = (
                    top_label_start,
                    top_label_end,
                    top_label_gap_x,
                )
            else:
                label_start, label_end, label_gap = (
                    bottom_label_start,
                    bottom_label_end,
                    bottom_label_gap_x,
                )
            pre_start, pre_end, pre_gap = (
                rect_x_min + 0.02 * width,
                rect_x_max - 0.02 * width,
                0.035 * width,
            )
        else:
            return payload

        order = np.argsort(axis_vals, kind="mergesort")
        nodes_ordered = [payload["nodes"][i] for i in order]
        axis_sorted = [axis_vals[i] for i in order]
        n = len(nodes_ordered)

        payload["nodes"] = nodes_ordered
        payload["label_positions"] = spaced_monotone_positions(
            axis_sorted,
            label_start,
            label_end,
            label_gap,
        )
        if side == "right":
            payload["route_values"] = np.linspace(
                rect_x_max + 0.04 * pad, rect_x_max + 0.98 * pad, n
            )
        elif side == "left":
            payload["route_values"] = np.linspace(
                rect_x_min - 0.04 * pad, rect_x_min - 0.98 * pad, n
            )
        elif side == "bottom":
            payload["route_values"] = np.linspace(
                rect_y_min - 0.04 * pad, rect_y_min - 0.98 * pad, n
            )
        elif side == "top":
            payload["route_values"] = np.linspace(
                rect_y_max + 0.04 * pad, rect_y_max + 0.98 * pad, n
            )
        payload["entry_extras"] = np.linspace(0.0, 0.24 * pad, n)
        payload["pre_entry_values"] = spaced_monotone_positions(
            axis_sorted,
            pre_start,
            pre_end,
            pre_gap,
            blend=0.65,
        )
        if side == "left":
            payload["pre_entry_values"] = np.clip(
                payload["pre_entry_values"] + 0.20 * pad,
                pre_start,
                pre_end,
            )
        return payload

    # Apply an explicit monotone mapping on all sides to reduce
    # side-label connector crossings.
    for side in ("top", "left", "bottom", "right"):
        layout[side] = _rebuild_monotone_side(side)

    def _label_anchor_local(side_name, pos_val):
        if side_name == "top":
            return (float(pos_val), rect_y_max + pad)
        if side_name == "bottom":
            return (float(pos_val), rect_y_min - pad)
        if side_name == "left":
            return (rect_x_min - pad, float(pos_val))
        return (rect_x_max + pad, float(pos_val))

    def _connector_polyline_local(
        side_name, text_anchor, node_xy, route_value, entry_extra, pre_entry_coord
    ):
        _ = (side_name, route_value, entry_extra, pre_entry_coord)
        return [
            np.asarray(text_anchor, dtype=float),
            np.asarray(node_xy, dtype=float),
        ]

    def _segments_cross(a, b, c, d, eps=1e-9):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)
        d = np.asarray(d, dtype=float)

        r = b - a
        s = d - c
        rxs = float(r[0] * s[1] - r[1] * s[0])
        qmp = c - a
        if abs(rxs) <= eps:
            return False
        t = float((qmp[0] * s[1] - qmp[1] * s[0]) / rxs)
        u = float((qmp[0] * r[1] - qmp[1] * r[0]) / rxs)
        return (eps < t < 1.0 - eps) and (eps < u < 1.0 - eps)

    def _polyline_crossings(poly_a, poly_b):
        total = 0
        for i in range(len(poly_a) - 1):
            a0, a1 = poly_a[i], poly_a[i + 1]
            for j in range(len(poly_b) - 1):
                b0, b1 = poly_b[j], poly_b[j + 1]
                if _segments_cross(a0, a1, b0, b1):
                    total += 1
        return total

    def _copy_layout(layout_obj):
        copied = {}
        for side_name, payload in layout_obj.items():
            copied[side_name] = {
                "nodes": list(payload["nodes"]),
                "label_positions": np.asarray(
                    payload["label_positions"], dtype=float
                ).copy(),
                "route_values": np.asarray(payload["route_values"], dtype=float).copy(),
                "entry_extras": np.asarray(payload["entry_extras"], dtype=float).copy(),
                "pre_entry_values": np.asarray(
                    payload["pre_entry_values"], dtype=float
                ).copy(),
            }
        return copied

    def _crossing_cost(layout_obj):
        polylines = []
        for side_name in ("top", "right", "bottom", "left"):
            payload = layout_obj[side_name]
            for node, label_pos, route_val, entry_extra, pre_entry in zip(
                payload["nodes"],
                payload["label_positions"],
                payload["route_values"],
                payload["entry_extras"],
                payload["pre_entry_values"],
            ):
                text_anchor = _label_anchor_local(side_name, label_pos)
                node_xy = positions[node]
                polylines.append(
                    _connector_polyline_local(
                        side_name,
                        text_anchor,
                        node_xy,
                        route_val,
                        entry_extra,
                        pre_entry,
                    )
                )
        total = 0
        for i in range(len(polylines) - 1):
            for j in range(i + 1, len(polylines)):
                total += _polyline_crossings(polylines[i], polylines[j])
        return int(total)

    def _geometry_penalty(layout_obj):
        penalty = 0.0
        for side_name in ("top", "right", "bottom", "left"):
            payload = layout_obj[side_name]
            for node, label_pos, route_val, entry_extra, pre_entry in zip(
                payload["nodes"],
                payload["label_positions"],
                payload["route_values"],
                payload["entry_extras"],
                payload["pre_entry_values"],
            ):
                node_xy = np.asarray(positions[node], dtype=float)
                text_anchor = np.asarray(
                    _label_anchor_local(side_name, label_pos), dtype=float
                )
                poly = _connector_polyline_local(
                    side_name,
                    text_anchor,
                    node_xy,
                    route_val,
                    entry_extra,
                    pre_entry,
                )
                # Keep connectors compact and near their natural node axis ordering.
                path_len = 0.0
                for i in range(len(poly) - 1):
                    seg = poly[i + 1] - poly[i]
                    path_len += float(np.hypot(seg[0], seg[1]))
                penalty += path_len

                if side_name in ("left", "right"):
                    axis_node = float(node_xy[1])
                else:
                    axis_node = float(node_xy[0])
                penalty += 3.6 * abs(float(label_pos) - axis_node)
        return float(penalty)

    def _objective(layout_obj):
        crosses = _crossing_cost(layout_obj)
        geom = _geometry_penalty(layout_obj)
        # Crossings dominate; geometry terms break ties and prevent odd routes.
        return float(crosses) * 1e6 + float(geom)

    def _optimize_side_node_assignment(side_name, max_passes=4):
        n = len(layout[side_name]["nodes"])
        if n <= 1:
            return
        best_cost = _objective(layout)
        passes = 0
        while passes < max_passes:
            passes += 1
            best_swap = None
            best_swap_cost = best_cost
            for i in range(n - 1):
                for j in range(i + 1, n):
                    trial = _copy_layout(layout)
                    trial_nodes = trial[side_name]["nodes"]
                    trial_nodes[i], trial_nodes[j] = trial_nodes[j], trial_nodes[i]
                    trial_cost = _objective(trial)
                    if trial_cost < best_swap_cost:
                        best_swap_cost = trial_cost
                        best_swap = (i, j)
            if best_swap is None:
                break
            i, j = best_swap
            live_nodes = layout[side_name]["nodes"]
            live_nodes[i], live_nodes[j] = live_nodes[j], live_nodes[i]
            best_cost = best_swap_cost

    # Global crossing pass on full connector routes (not only local lane order).
    for side in ("top", "left", "bottom", "right"):
        _optimize_side_node_assignment(side, max_passes=4)

    return layout


def _round_float(value, ndigits=6):
    return round(float(value), int(ndigits))


def _side_layout_signature(
    positions,
    top_nodes,
    right_nodes,
    bottom_nodes,
    left_nodes,
    rect_bounds,
    pad,
):
    return {
        "version": SIDE_LAYOUT_CACHE_VERSION,
        "positions": {
            node: [_round_float(x), _round_float(y)]
            for node, (x, y) in sorted(positions.items())
        },
        "sides": {
            "top": list(top_nodes),
            "right": list(right_nodes),
            "bottom": list(bottom_nodes),
            "left": list(left_nodes),
        },
        "rect_bounds": [_round_float(v) for v in rect_bounds],
        "pad": _round_float(pad),
    }


def _serialize_side_layout(layout):
    serial = {}
    for side_name, payload in layout.items():
        serial[side_name] = {
            "nodes": list(payload["nodes"]),
            "label_positions": [
                float(v) for v in np.asarray(payload["label_positions"])
            ],
            "route_values": [float(v) for v in np.asarray(payload["route_values"])],
            "entry_extras": [float(v) for v in np.asarray(payload["entry_extras"])],
            "pre_entry_values": [
                float(v) for v in np.asarray(payload["pre_entry_values"])
            ],
        }
    return serial


def _deserialize_side_layout(payload):
    layout = {}
    for side_name, side_payload in payload.items():
        layout[side_name] = {
            "nodes": list(side_payload["nodes"]),
            "label_positions": np.asarray(side_payload["label_positions"], dtype=float),
            "route_values": np.asarray(side_payload["route_values"], dtype=float),
            "entry_extras": np.asarray(side_payload["entry_extras"], dtype=float),
            "pre_entry_values": np.asarray(
                side_payload["pre_entry_values"], dtype=float
            ),
        }
    return layout


def load_cached_side_layout(
    positions,
    top_nodes,
    right_nodes,
    bottom_nodes,
    left_nodes,
    rect_bounds,
    pad,
):
    if not USE_CACHED_SIDE_LAYOUT or REFRESH_SIDE_LAYOUT:
        return None
    if not SIDE_LAYOUT_CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(SIDE_LAYOUT_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read side-layout cache: {exc}")
        return None

    expected = _side_layout_signature(
        positions, top_nodes, right_nodes, bottom_nodes, left_nodes, rect_bounds, pad
    )
    if payload.get("signature") != expected:
        return None

    try:
        layout = _deserialize_side_layout(payload["layout"])
    except Exception as exc:
        print(f"Warning: failed to decode side-layout cache: {exc}")
        return None
    print(f"Loaded cached side layout from {SIDE_LAYOUT_CACHE_PATH}.")
    return layout


def save_cached_side_layout(
    layout,
    positions,
    top_nodes,
    right_nodes,
    bottom_nodes,
    left_nodes,
    rect_bounds,
    pad,
):
    if not USE_CACHED_SIDE_LAYOUT:
        return
    SIDE_LAYOUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": _side_layout_signature(
            positions,
            top_nodes,
            right_nodes,
            bottom_nodes,
            left_nodes,
            rect_bounds,
            pad,
        ),
        "layout": _serialize_side_layout(layout),
    }
    SIDE_LAYOUT_CACHE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved side layout cache to {SIDE_LAYOUT_CACHE_PATH}.")


def draw_chamfered_trace(ax, points, color, lw, alpha, zorder, chamfer):
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 2:
        return

    chamfer = float(max(chamfer, 0.0))
    out_points = [pts[0]]

    for i in range(1, len(pts) - 1):
        p_prev = pts[i - 1]
        p = pts[i]
        p_next = pts[i + 1]

        vin = p - p_prev
        vout = p_next - p
        len_in = float(np.hypot(vin[0], vin[1]))
        len_out = float(np.hypot(vout[0], vout[1]))
        if len_in < 1e-9 or len_out < 1e-9:
            out_points.append(p)
            continue

        same_dir = (
            abs(vin[0] * vout[1] - vin[1] * vout[0]) < 1e-9
            and vin[0] * vout[0] + vin[1] * vout[1] > 0
        )
        if same_dir:
            out_points.append(p)
            continue

        cut = min(chamfer, 0.35 * len_in, 0.35 * len_out)
        p_before = p - vin / len_in * cut
        p_after = p + vout / len_out * cut
        out_points.extend([p_before, p_after])

    out_points.append(pts[-1])
    xs = [float(p[0]) for p in out_points]
    ys = [float(p[1]) for p in out_points]
    ax.plot(
        xs,
        ys,
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def _oct_dir(vec, tol=1e-9):
    vx = float(vec[0])
    vy = float(vec[1])
    sx = 0 if abs(vx) <= tol else (1 if vx > 0 else -1)
    sy = 0 if abs(vy) <= tol else (1 if vy > 0 else -1)
    return sx, sy


def octilinear_smooth_path(points, step):
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) <= 2:
        return pts

    step = float(max(step, 1e-9))
    pts = _compress_polyline(pts, tol=1e-9)
    out = [pts[0]]

    for i in range(1, len(pts) - 1):
        p_prev = out[-1]
        p = pts[i]
        p_next = pts[i + 1]

        vin = p - p_prev
        vout = p_next - p
        len_in = float(np.hypot(vin[0], vin[1]))
        len_out = float(np.hypot(vout[0], vout[1]))
        if len_in < 1e-9 or len_out < 1e-9:
            out.append(p)
            continue

        d1 = _oct_dir(vin)
        d2 = _oct_dir(vout)
        if d1 == d2:
            out.append(p)
            continue

        # For orthogonal cardinal turns, replace hard 90-degree corner with
        # two segments meeting at 45 degrees.
        card1 = (abs(d1[0]) + abs(d1[1])) == 1
        card2 = (abs(d2[0]) + abs(d2[1])) == 1
        orth = (d1[0] * d2[0] + d1[1] * d2[1]) == 0
        if card1 and card2 and orth:
            cut = min(0.48 * step, 0.45 * len_in, 0.45 * len_out)
            before = p - np.array([d1[0], d1[1]], dtype=float) * cut
            after = p + np.array([d2[0], d2[1]], dtype=float) * cut
            out.extend([before, after])
            continue

        out.append(p)

    out.append(pts[-1])
    return _compress_polyline(out, tol=1e-9)


def _octilinear_bridge(a, b, step):
    """Connect two grid-snapped points with cardinal/diagonal octilinear steps."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    step = float(max(step, 1e-9))
    dx_steps = int(round((b[0] - a[0]) / step))
    dy_steps = int(round((b[1] - a[1]) / step))
    sx = 0 if dx_steps == 0 else (1 if dx_steps > 0 else -1)
    sy = 0 if dy_steps == 0 else (1 if dy_steps > 0 else -1)
    nx = abs(dx_steps)
    ny = abs(dy_steps)
    n_diag = min(nx, ny)
    n_x = nx - n_diag
    n_y = ny - n_diag

    out = [a]
    p = a.copy()
    for _ in range(n_diag):
        p = p + np.array([sx * step, sy * step], dtype=float)
        out.append(p.copy())
    for _ in range(n_x):
        p = p + np.array([sx * step, 0.0], dtype=float)
        out.append(p.copy())
    for _ in range(n_y):
        p = p + np.array([0.0, sy * step], dtype=float)
        out.append(p.copy())

    if not np.allclose(out[-1], b, atol=1e-8):
        out.append(np.asarray(b, dtype=float))
    return out


def spline_snap_to_octilinear(points, router, smooth_scale=0.12):
    """
    Fit a smooth route, then snap to router grid and re-express as octilinear steps.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    if router is None or len(pts) < 4:
        return _compress_polyline(pts, tol=1e-9)

    arr = np.vstack(pts)
    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    if not np.isfinite(seg).all() or float(seg.sum()) <= 1e-9:
        return _compress_polyline(pts, tol=1e-9)

    u = np.hstack([[0.0], np.cumsum(seg)])
    u = u / float(u[-1])
    k = int(min(3, len(pts) - 1))
    sample_n = int(max(36, len(pts) * 9))
    try:
        tck, _ = splprep(
            [arr[:, 0], arr[:, 1]],
            u=u,
            s=float(max(0.0, smooth_scale)) * len(pts),
            k=k,
            per=False,
        )
        uu = np.linspace(0.0, 1.0, sample_n)
        x_new, y_new = splev(uu, tck)
        smooth_pts = [np.array([x, y], dtype=float) for x, y in zip(x_new, y_new)]
    except Exception:
        smooth_pts = pts

    snapped = [
        router.cell_to_world(router.world_to_cell(np.asarray(p, dtype=float)))
        for p in smooth_pts
    ]
    if snapped:
        snapped[0] = np.asarray(pts[0], dtype=float)
        snapped[-1] = np.asarray(pts[-1], dtype=float)
    snapped = _compress_polyline(snapped, tol=1e-9)
    if len(snapped) < 2:
        return _compress_polyline(pts, tol=1e-9)

    out = [snapped[0]]
    for i in range(len(snapped) - 1):
        bridge = _octilinear_bridge(snapped[i], snapped[i + 1], step=router.step)
        out.extend(bridge[1:])
    out = _compress_polyline(out, tol=1e-9)
    if len(out) < 2:
        return _compress_polyline(pts, tol=1e-9)
    return out


def _polyline_segments(points):
    pts = [np.asarray(p, dtype=float) for p in points]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _compress_polyline(points, tol=1e-9):
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for idx in range(1, len(pts) - 1):
        a = out[-1]
        b = pts[idx]
        c = pts[idx + 1]
        ab = b - a
        bc = c - b
        if float(np.hypot(ab[0], ab[1])) < tol:
            continue
        if float(np.hypot(bc[0], bc[1])) < tol:
            continue
        cross = abs(float(ab[0] * bc[1] - ab[1] * bc[0]))
        denom = float(np.hypot(ab[0], ab[1]) + np.hypot(bc[0], bc[1]))
        if denom <= tol:
            continue
        if cross / denom <= tol:
            continue
        out.append(b)
    out.append(pts[-1])
    return out


@dataclass(frozen=True)
class ConnectorRoutingConfig:
    grid_step_scale: float = 0.055
    router_margin_scale: float = 1.55
    lane_offset_scale: float = 0.36
    arrow_diag_scale: float = 0.08
    corner_cut_scale: float = 0.22
    reservation_scale: float = 0.06
    corridor_weight: float = 0.38
    turn_penalty: float = 0.45
    stem_grid_steps: float = 2.0
    spline_smooth_scale: float = 0.14
    start_open_radius_scale: float = 0.14
    goal_open_radius_scale: float = 0.22
    allow_diagonal_steps: bool = False
    apply_spline_snap: bool = False
    apply_corner_softening: bool = False


USE_ASTAR_CONNECTOR_ROUTER = False


class GridAStarRouter:
    """Global obstacle-aware connector routing on a rasterized grid."""

    _DIRS = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (-1, -1, np.sqrt(2.0)),
    )

    def __init__(self, bounds, step, turn_penalty=0.24):
        self.x_min, self.x_max, self.y_min, self.y_max = [float(v) for v in bounds]
        self.step = float(max(step, 1e-4))
        self.turn_penalty = float(max(turn_penalty, 0.0))
        self.nx = int(np.ceil((self.x_max - self.x_min) / self.step)) + 1
        self.ny = int(np.ceil((self.y_max - self.y_min) / self.step)) + 1
        self.static_blocked = np.zeros((self.ny, self.nx), dtype=bool)
        self.reserved_blocked = np.zeros((self.ny, self.nx), dtype=bool)

    def _inside(self, ix, iy):
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def _clamp_cell(self, ix, iy):
        return int(np.clip(ix, 0, self.nx - 1)), int(np.clip(iy, 0, self.ny - 1))

    def world_to_cell(self, point):
        x, y = float(point[0]), float(point[1])
        ix = int(round((x - self.x_min) / self.step))
        iy = int(round((y - self.y_min) / self.step))
        return self._clamp_cell(ix, iy)

    def cell_to_world(self, cell):
        ix, iy = int(cell[0]), int(cell[1])
        return np.array(
            [self.x_min + ix * self.step, self.y_min + iy * self.step], dtype=float
        )

    def _mark_disk(self, arr, center_cell, radius_cells):
        ix0, iy0 = int(center_cell[0]), int(center_cell[1])
        rr = int(max(0, radius_cells))
        r2 = rr * rr
        x_lo = max(0, ix0 - rr)
        x_hi = min(self.nx - 1, ix0 + rr)
        y_lo = max(0, iy0 - rr)
        y_hi = min(self.ny - 1, iy0 + rr)
        for iy in range(y_lo, y_hi + 1):
            dy = iy - iy0
            for ix in range(x_lo, x_hi + 1):
                dx = ix - ix0
                if dx * dx + dy * dy <= r2:
                    arr[iy, ix] = True

    def mark_circle(self, center_xy, radius, target="static"):
        radius = float(max(radius, 0.0))
        if radius <= 0.0:
            return
        arr = self.static_blocked if target == "static" else self.reserved_blocked
        cell = self.world_to_cell(center_xy)
        r_cells = int(np.ceil(radius / self.step))
        self._mark_disk(arr, cell, r_cells)

    def mark_segment(self, a, b, radius, target="static"):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        radius = float(max(radius, 0.0))
        arr = self.static_blocked if target == "static" else self.reserved_blocked
        seg_len = float(np.hypot(*(b - a)))
        samples = max(2, int(np.ceil(seg_len / max(0.45 * self.step, 1e-6))))
        r_cells = int(np.ceil(radius / self.step))
        for t in np.linspace(0.0, 1.0, samples):
            pt = a + t * (b - a)
            self._mark_disk(arr, self.world_to_cell(pt), r_cells)

    def reserve_polyline(self, points, radius):
        for a, b in _polyline_segments(points):
            self.mark_segment(a, b, radius=radius, target="reserved")

    def _is_blocked(self, ix, iy, open_cells):
        if not self._inside(ix, iy):
            return True
        if open_cells:
            for cx, cy, rr2 in open_cells:
                dx = ix - cx
                dy = iy - cy
                if dx * dx + dy * dy <= rr2:
                    return False
        return bool(self.static_blocked[iy, ix] or self.reserved_blocked[iy, ix])

    def _nearest_free(
        self, start_cell, open_cells, fallback_cell=None, search_radius=24
    ):
        sx, sy = int(start_cell[0]), int(start_cell[1])
        if not self._is_blocked(sx, sy, open_cells):
            return sx, sy
        best = None
        best_d = np.inf
        fx, fy = (None, None) if fallback_cell is None else fallback_cell
        for rr in range(1, int(search_radius) + 1):
            x_lo = max(0, sx - rr)
            x_hi = min(self.nx - 1, sx + rr)
            y_lo = max(0, sy - rr)
            y_hi = min(self.ny - 1, sy + rr)
            for iy in range(y_lo, y_hi + 1):
                for ix in (x_lo, x_hi):
                    if self._is_blocked(ix, iy, open_cells):
                        continue
                    d = (
                        float(np.hypot(ix - fx, iy - fy))
                        if fx is not None
                        else float(np.hypot(ix - sx, iy - sy))
                    )
                    if d < best_d:
                        best_d = d
                        best = (ix, iy)
            for ix in range(x_lo + 1, x_hi):
                for iy in (y_lo, y_hi):
                    if self._is_blocked(ix, iy, open_cells):
                        continue
                    d = (
                        float(np.hypot(ix - fx, iy - fy))
                        if fx is not None
                        else float(np.hypot(ix - sx, iy - sy))
                    )
                    if d < best_d:
                        best_d = d
                        best = (ix, iy)
            if best is not None:
                return best
        return None

    def route(
        self,
        start_xy,
        goal_xy,
        prefer_axis=None,
        corridor_value=None,
        corridor_weight=0.0,
        start_open_radius=0.0,
        goal_open_radius=0.0,
        snap_endpoints=False,
        allow_diagonal=True,
    ):
        start_cell_raw = self.world_to_cell(start_xy)
        goal_cell_raw = self.world_to_cell(goal_xy)

        open_cells = []
        if start_open_radius > 0.0:
            rr = int(np.ceil(float(start_open_radius) / self.step))
            open_cells.append((start_cell_raw[0], start_cell_raw[1], rr * rr))
        if goal_open_radius > 0.0:
            rr = int(np.ceil(float(goal_open_radius) / self.step))
            open_cells.append((goal_cell_raw[0], goal_cell_raw[1], rr * rr))

        start_cell = self._nearest_free(
            start_cell_raw,
            open_cells=open_cells,
            fallback_cell=goal_cell_raw,
        )
        goal_cell = self._nearest_free(
            goal_cell_raw,
            open_cells=open_cells,
            fallback_cell=start_cell_raw,
        )
        if start_cell is None or goal_cell is None:
            return None

        sx, sy = start_cell
        gx, gy = goal_cell

        def heuristic(ix, iy):
            return float(np.hypot(gx - ix, gy - iy))

        start_state = (sx, sy, -1)
        g_score = {start_state: 0.0}
        parent = {}
        open_heap = []
        heappush(open_heap, (heuristic(sx, sy), 0.0, sx, sy, -1))
        end_state = None
        dirs = self._DIRS if bool(allow_diagonal) else self._DIRS[:4]

        while open_heap:
            _, g_curr, ix, iy, dir_idx = heappop(open_heap)
            state = (ix, iy, dir_idx)
            if g_curr > g_score.get(state, np.inf) + 1e-12:
                continue
            if ix == gx and iy == gy:
                end_state = state
                break

            for ndi, (dx, dy, move_cost) in enumerate(dirs):
                jx = ix + dx
                jy = iy + dy
                if self._is_blocked(jx, jy, open_cells):
                    continue
                turn_cost = (
                    self.turn_penalty if (dir_idx != -1 and ndi != dir_idx) else 0.0
                )
                corridor_cost = 0.0
                if corridor_value is not None and corridor_weight > 0.0:
                    if prefer_axis == "horizontal":
                        y = self.y_min + jy * self.step
                        corridor_cost = (
                            float(corridor_weight)
                            * abs(y - float(corridor_value))
                            / max(self.step, 1e-9)
                        )
                    elif prefer_axis == "vertical":
                        x = self.x_min + jx * self.step
                        corridor_cost = (
                            float(corridor_weight)
                            * abs(x - float(corridor_value))
                            / max(self.step, 1e-9)
                        )
                g_next = float(g_curr + move_cost + turn_cost + corridor_cost)
                nstate = (jx, jy, ndi)
                if g_next + 1e-12 < g_score.get(nstate, np.inf):
                    g_score[nstate] = g_next
                    parent[nstate] = state
                    heappush(
                        open_heap, (g_next + heuristic(jx, jy), g_next, jx, jy, ndi)
                    )

        if end_state is None:
            return None

        cells = []
        state = end_state
        while True:
            cells.append((state[0], state[1]))
            if state == start_state:
                break
            state = parent[state]
        cells.reverse()
        points = [self.cell_to_world(cell) for cell in cells]
        if points:
            if snap_endpoints:
                points[0] = self.cell_to_world(start_cell)
                points[-1] = self.cell_to_world(goal_cell)
            else:
                points[0] = np.asarray(start_xy, dtype=float)
                points[-1] = np.asarray(goal_xy, dtype=float)
        return _compress_polyline(points, tol=1e-8)


def _sign(val, fallback=1.0):
    if abs(float(val)) < 1e-9:
        return float(fallback)
    return 1.0 if float(val) >= 0 else -1.0


def draw_box_connector(
    ax,
    side,
    text_anchor,
    node_xy,
    rect_bounds,
    pad,
    color="0.35",
    route_value=None,
    approach_extra=0.0,
    pre_entry_coord=None,
    router=None,
    routing_config=None,
):
    rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect_bounds
    x_node, y_node = float(node_xy[0]), float(node_xy[1])
    x0, y0 = float(text_anchor[0]), float(text_anchor[1])

    if routing_config is None:
        routing_config = ConnectorRoutingConfig()

    route_color = color
    route_lw = 0.9
    route_alpha = 0.8
    route_z = 0.7
    edge_offset = float(routing_config.lane_offset_scale) * pad
    arrow_diag = float(routing_config.arrow_diag_scale) * pad + float(approach_extra)
    corner_cut = float(routing_config.corner_cut_scale) * pad
    reserve_radius = float(routing_config.reservation_scale) * pad

    if side == "top":
        lane = (
            float(route_value) if route_value is not None else rect_y_max + edge_offset
        )
        leave_dir = np.array([0.0, -1.0], dtype=float)
        pre_axis = (
            float(pre_entry_coord)
            if pre_entry_coord is not None
            else (x_node + _sign(x0 - x_node) * arrow_diag)
        )
        s = _sign(x0 - x_node, fallback=1.0)
        pre_entry = (x_node + s * arrow_diag, y_node + arrow_diag)
        waypoints = [
            (x0, y0),
            (x0, lane),
            (pre_axis, lane),
            (pre_axis, pre_entry[1]),
            pre_entry,
        ]
        hints = [
            ("vertical", x0),
            ("horizontal", lane),
            ("vertical", pre_axis),
            (None, None),
        ]
    elif side == "bottom":
        lane = (
            float(route_value) if route_value is not None else rect_y_min - edge_offset
        )
        leave_dir = np.array([0.0, 1.0], dtype=float)
        pre_axis = (
            float(pre_entry_coord)
            if pre_entry_coord is not None
            else (x_node + _sign(x0 - x_node) * arrow_diag)
        )
        s = _sign(x0 - x_node, fallback=1.0)
        pre_entry = (x_node + s * arrow_diag, y_node - arrow_diag)
        waypoints = [
            (x0, y0),
            (x0, lane),
            (pre_axis, lane),
            (pre_axis, pre_entry[1]),
            pre_entry,
        ]
        hints = [
            ("vertical", x0),
            ("horizontal", lane),
            ("vertical", pre_axis),
            (None, None),
        ]
    elif side == "left":
        lane = (
            float(route_value) if route_value is not None else rect_x_min - edge_offset
        )
        leave_dir = np.array([1.0, 0.0], dtype=float)
        pre_axis = (
            float(pre_entry_coord)
            if pre_entry_coord is not None
            else (y_node + _sign(y0 - y_node) * arrow_diag)
        )
        s = _sign(y0 - y_node, fallback=1.0)
        pre_entry = (x_node - arrow_diag, y_node + s * arrow_diag)
        waypoints = [
            (x0, y0),
            (lane, y0),
            (lane, pre_axis),
            (pre_entry[0], pre_axis),
            pre_entry,
        ]
        hints = [
            ("horizontal", y0),
            ("vertical", lane),
            ("horizontal", pre_axis),
            (None, None),
        ]
    else:
        lane = (
            float(route_value) if route_value is not None else rect_x_max + edge_offset
        )
        leave_dir = np.array([-1.0, 0.0], dtype=float)
        pre_axis = (
            float(pre_entry_coord)
            if pre_entry_coord is not None
            else (y_node + _sign(y0 - y_node) * arrow_diag)
        )
        s = _sign(y0 - y_node, fallback=1.0)
        pre_entry = (x_node + arrow_diag, y_node + s * arrow_diag)
        waypoints = [
            (x0, y0),
            (lane, y0),
            (lane, pre_axis),
            (pre_entry[0], pre_axis),
            pre_entry,
        ]
        hints = [
            ("horizontal", y0),
            ("vertical", lane),
            ("horizontal", pre_axis),
            (None, None),
        ]

    if router is None:
        routed = [np.asarray(p, dtype=float) for p in waypoints]
    else:
        # Mesh-based routing: snap exits/entries to grid and route globally.
        grid_step = float(max(router.step, 1e-9))
        box_anchor = np.asarray([x0, y0], dtype=float)
        stem_len = float(max(1.0, routing_config.stem_grid_steps)) * grid_step
        start_exit = box_anchor + leave_dir * stem_len
        node_xy = np.asarray([x_node, y_node], dtype=float)
        goal_entry = node_xy - leave_dir * stem_len
        start_exit = router.cell_to_world(router.world_to_cell(start_exit))
        goal_entry = router.cell_to_world(router.world_to_cell(goal_entry))

        side_axis = "horizontal" if side in ("left", "right") else "vertical"
        side_corridor = float(
            start_exit[1] if side in ("left", "right") else start_exit[0]
        )
        mid = router.route(
            start_exit,
            goal_entry,
            prefer_axis=side_axis,
            corridor_value=side_corridor,
            corridor_weight=float(max(0.0, routing_config.corridor_weight)),
            start_open_radius=max(
                float(routing_config.start_open_radius_scale) * pad,
                1.2 * grid_step,
            ),
            goal_open_radius=max(
                float(routing_config.goal_open_radius_scale) * pad,
                1.5 * grid_step,
            ),
            snap_endpoints=True,
            allow_diagonal=bool(routing_config.allow_diagonal_steps),
        )
        if mid is None:
            mid = [
                np.asarray(start_exit, dtype=float),
                np.asarray(goal_entry, dtype=float),
            ]
        mid = [np.asarray(p, dtype=float) for p in mid]
        routed = [box_anchor, np.asarray(start_exit, dtype=float)]
        routed.extend(mid[1:])
        routed = _compress_polyline(routed, tol=1e-8)
        if bool(routing_config.apply_spline_snap):
            routed = spline_snap_to_octilinear(
                routed,
                router=router,
                smooth_scale=float(max(0.0, routing_config.spline_smooth_scale)),
            )
        pre_entry = (float(routed[-1][0]), float(routed[-1][1]))
        router.reserve_polyline(routed, radius=reserve_radius)
        router.mark_segment(
            np.asarray(pre_entry, dtype=float),
            np.asarray([x_node, y_node], dtype=float),
            radius=0.85 * reserve_radius,
            target="reserved",
        )

    routed_plot = routed
    chamfer_plot = corner_cut
    if router is not None and bool(routing_config.apply_corner_softening):
        routed_plot = octilinear_smooth_path(
            routed,
            step=float(max(router.step, 1e-9)),
        )
        chamfer_plot = 0.0
    elif router is not None:
        chamfer_plot = 0.0

    draw_chamfered_trace(
        ax,
        routed_plot,
        route_color,
        route_lw,
        route_alpha,
        route_z,
        chamfer_plot,
    )
    # Keep the final approach as a plain segment (no arrowhead).
    ax.plot(
        [float(pre_entry[0]), x_node],
        [float(pre_entry[1]), y_node],
        color=route_color,
        lw=route_lw,
        alpha=route_alpha,
        zorder=route_z,
        solid_capstyle="round",
    )
    return _polyline_segments(routed)


_rect_bounds = (rect_x_min, rect_x_max, rect_y_min, rect_y_max)
_t0 = time.perf_counter()
side_layouts = load_cached_side_layout(
    r,
    top_nodes,
    right_nodes,
    bottom_nodes,
    left_nodes,
    _rect_bounds,
    pad,
)
if side_layouts is None:
    debug_print("No valid side-layout cache found; computing side layout...")
    _t1 = time.perf_counter()
    side_layouts = build_side_layouts(
        r,
        top_nodes,
        right_nodes,
        bottom_nodes,
        left_nodes,
        _rect_bounds,
        pad,
    )
    debug_print(f"Computed side layout in {time.perf_counter() - _t1:.2f}s")
    save_cached_side_layout(
        side_layouts,
        r,
        top_nodes,
        right_nodes,
        bottom_nodes,
        left_nodes,
        _rect_bounds,
        pad,
    )
    debug_print(
        f"Saved side-layout cache to {SIDE_LAYOUT_CACHE_PATH} "
        f"({time.perf_counter() - _t0:.2f}s total)"
    )
else:
    debug_print(
        f"Loaded side-layout cache from {SIDE_LAYOUT_CACHE_PATH} "
        f"in {time.perf_counter() - _t0:.2f}s"
    )


def _swap_label_slots(layout, node_a, node_b):
    loc = {}
    for side in ("top", "right", "bottom", "left"):
        payload = layout.get(side, {})
        nodes = payload.get("nodes", [])
        for idx, name in enumerate(nodes):
            if name == node_a:
                loc[node_a] = (side, idx)
            elif name == node_b:
                loc[node_b] = (side, idx)
    if node_a not in loc or node_b not in loc:
        return False
    side_a, idx_a = loc[node_a]
    side_b, idx_b = loc[node_b]
    layout[side_a]["nodes"][idx_a], layout[side_b]["nodes"][idx_b] = (
        layout[side_b]["nodes"][idx_b],
        layout[side_a]["nodes"][idx_a],
    )
    return True


_swap_label_slots(
    side_layouts,
    "Exchange of Information",
    "Operation of the Antarctic Treaty system General",
)


def estimate_connector_mesh_step(positions, side_layouts, rect_bounds, pad, fallback):
    rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect_bounds
    pts = [np.asarray(v, dtype=float) for v in positions.values()]

    def _anchor_for_side(side, pos):
        if side == "top":
            return np.array([float(pos), rect_y_max + pad], dtype=float)
        if side == "bottom":
            return np.array([float(pos), rect_y_min - pad], dtype=float)
        if side == "left":
            return np.array([rect_x_min - pad, float(pos)], dtype=float)
        return np.array([rect_x_max + pad, float(pos)], dtype=float)

    for side in ("top", "right", "bottom", "left"):
        payload = side_layouts.get(side, {})
        for pos in np.asarray(payload.get("label_positions", []), dtype=float):
            pts.append(_anchor_for_side(side, pos))

    if len(pts) < 3:
        return float(fallback)

    arr = np.vstack(pts)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(arr)
        dists, _ = tree.query(arr, k=2)
        nn = np.asarray(dists[:, 1], dtype=float)
    except Exception:
        diff = arr[:, None, :] - arr[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        nn = np.min(dist, axis=1)

    nn = nn[np.isfinite(nn) & (nn > 1e-8)]
    if nn.size == 0:
        return float(fallback)
    raw = 0.55 * float(np.median(nn))
    return float(np.clip(raw, 0.030 * pad, 0.160 * pad))


graph_edge_segments = []
for u, v in mst_draw.edges():
    if u in r and v in r:
        graph_edge_segments.append(
            (np.asarray(r[u], dtype=float), np.asarray(r[v], dtype=float))
        )
# Keep dashed highlighted edge as an obstacle to prevent label connector overlap.
if edge_a in r and edge_b in r:
    graph_edge_segments.append(
        (np.asarray(r[edge_a], dtype=float), np.asarray(r[edge_b], dtype=float))
    )
node_obstacle_points = [np.asarray(r[n], dtype=float) for n in r]
connector_node_avoid_radius = max(0.28 * pad, 2.1 * marker_radius_data)
connector_edge_clearance = 0.16 * pad
routing_cfg = ConnectorRoutingConfig()
connector_router = None
if USE_ASTAR_CONNECTOR_ROUTER:
    mesh_step = estimate_connector_mesh_step(
        r,
        side_layouts,
        (rect_x_min, rect_x_max, rect_y_min, rect_y_max),
        pad,
        fallback=routing_cfg.grid_step_scale * pad,
    )
    router_bounds = (
        rect_x_min - routing_cfg.router_margin_scale * pad,
        rect_x_max + routing_cfg.router_margin_scale * pad,
        rect_y_min - routing_cfg.router_margin_scale * pad,
        rect_y_max + routing_cfg.router_margin_scale * pad,
    )
    connector_router = GridAStarRouter(
        router_bounds,
        step=mesh_step,
        turn_penalty=routing_cfg.turn_penalty,
    )
    for pt in node_obstacle_points:
        connector_router.mark_circle(
            pt, radius=connector_node_avoid_radius, target="static"
        )
    for a, b in graph_edge_segments:
        connector_router.mark_segment(
            a, b, radius=connector_edge_clearance, target="static"
        )


def _label_anchor(side, pos):
    if side == "top":
        return (float(pos), rect_y_max + pad)
    if side == "bottom":
        return (float(pos), rect_y_min - pad)
    if side == "left":
        return (rect_x_min - pad, float(pos))
    return (rect_x_max + pad, float(pos))


def _label_style(side):
    if side == "top":
        return dict(ha="center", va="bottom", rotation=90)
    if side == "bottom":
        return dict(ha="center", va="top", rotation=90)
    if side == "left":
        return dict(ha="right", va="center")
    return dict(ha="left", va="center")


def _draw_side_labels_and_connectors(side, nodes, label_positions):
    from matplotlib.patches import FancyArrowPatch

    style = _label_style(side)
    fig = ax.figure
    label_records = []
    for node, label_pos in zip(nodes, label_positions):
        xy = np.asarray(r[node], dtype=float)
        xytext = _label_anchor(side, label_pos)
        ann = ax.annotate(
            smartwrap(node, 25),
            xy=xy,
            xytext=xytext,
            fontsize=side_fs,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=(1, 1, 1, 1),
                edgecolor=theme_colors[topic_to_theme.get(node, "Governance & Legal")],
                alpha=1,
                linewidth=1.2,
            ),
            zorder=2.2,
            **style,
        )
        ann.set_zorder(4.0)
        bbox_patch = ann.get_bbox_patch()
        if bbox_patch is not None:
            bbox_patch.set_zorder(4.1)
        label_records.append((ann, xy, xytext, node))

    if not label_records:
        return []

    artists = []
    # Force a renderer pass so bbox extents are up-to-date before we compute
    # side-facing arrow origins from the text boxes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ann, xy, xytext, node in label_records:
        patch = ann.get_bbox_patch()
        if patch is not None:
            bb = patch.get_window_extent(renderer=renderer)
            if side == "left":
                p_disp = (bb.x1, 0.5 * (bb.y0 + bb.y1))
            elif side == "right":
                p_disp = (bb.x0, 0.5 * (bb.y0 + bb.y1))
            elif side == "top":
                p_disp = (0.5 * (bb.x0 + bb.x1), bb.y0)
            else:  # bottom
                p_disp = (0.5 * (bb.x0 + bb.x1), bb.y1)
            start = np.asarray(ax.transData.inverted().transform(p_disp), dtype=float)
        else:
            start = np.asarray(xytext, dtype=float)

        node_color = theme_colors[topic_to_theme.get(node, "Governance & Legal")]
        arrow = FancyArrowPatch(
            (float(start[0]), float(start[1])),
            (float(xy[0]), float(xy[1])),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=0.9,
            facecolor=node_color,
            edgecolor="0.35",
            alpha=1.0,
            shrinkA=2.0,
            shrinkB=10.0,
            connectionstyle="arc3,rad=0.0",
            zorder=1.2,
            clip_on=False,
        )
        ax.add_patch(arrow)
        artists.append({"node": node, "label": ann, "arrow": arrow})
    return artists


side_label_artists = []
side_label_artists.extend(
    _draw_side_labels_and_connectors(
        "top",
        side_layouts["top"]["nodes"],
        side_layouts["top"]["label_positions"],
    )
)
side_label_artists.extend(
    _draw_side_labels_and_connectors(
        "right",
        side_layouts["right"]["nodes"],
        side_layouts["right"]["label_positions"],
    )
)
side_label_artists.extend(
    _draw_side_labels_and_connectors(
        "bottom",
        side_layouts["bottom"]["nodes"],
        side_layouts["bottom"]["label_positions"],
    )
)
side_label_artists.extend(
    _draw_side_labels_and_connectors(
        "left",
        side_layouts["left"]["nodes"],
        side_layouts["left"]["label_positions"],
    )
)

# Highlight a specific edge and add a grouped RCA inset.
edge_a = "Drilling"
edge_b = "Marine Acoustics"
highlight_line_artists = []
highlight_other_artists = []
highlight_axes = []
if edge_a in snapped and edge_b in snapped:
    callout_color = "#E0007A"
    callout_linewidth = 1.8
    mx = (snapped[edge_a][0] + snapped[edge_b][0]) / 2
    my = (snapped[edge_a][1] + snapped[edge_b][1]) / 2
    dx_edge = snapped[edge_b][0] - snapped[edge_a][0]
    dy_edge = snapped[edge_b][1] - snapped[edge_a][1]
    edge_len = float(np.hypot(dx_edge, dy_edge))
    edge_angle = float(np.degrees(np.arctan2(dy_edge, dx_edge)))
    from matplotlib.patches import Ellipse

    region_fill = Ellipse(
        (mx, my),
        width=max(1.22 * edge_len, 0.62 * pad),
        height=0.42 * pad,
        angle=edge_angle,
        facecolor=callout_color,
        edgecolor="none",
        alpha=0.13,
        zorder=0.42,
    )
    region_brim = Ellipse(
        (mx, my),
        width=max(1.22 * edge_len, 0.62 * pad),
        height=0.42 * pad,
        angle=edge_angle,
        facecolor="none",
        edgecolor=callout_color,
        linewidth=1.35,
        linestyle="-",
        alpha=0.62,
        zorder=0.43,
    )
    ax.add_patch(region_fill)
    ax.add_patch(region_brim)
    highlight_other_artists.extend([region_fill, region_brim])
    highlight_line_artists.extend(
        ax.plot(
            [snapped[edge_a][0], snapped[edge_b][0]],
            [snapped[edge_a][1], snapped[edge_b][1]],
            color=callout_color,
            linewidth=callout_linewidth,
            linestyle=(0, (2, 1)),
            zorder=3,
        )
    )
    # Inset positioned near the highlighted edge (data coords).
    extent_w = rect_x_max - rect_x_min
    extent_h = rect_y_max - rect_y_min
    inset_w = 0.43 * extent_w
    inset_h = 0.48 * extent_h

    # Keep the inset left of the highlighted link while letting the vertical
    # leader meet the inset near its right edge.
    drill_xy = snapped.get("Drilling")
    inset_x_target = mx - 0.84 * inset_w
    if drill_xy is not None:
        inset_y_target = float(drill_xy[1]) - 3.45 * pad
    else:
        inset_y_target = rect_y_min - 3.45 * pad

    inset_x = float(
        np.clip(
            inset_x_target,
            rect_x_min - 3.55 * pad,
            rect_x_max - inset_w - 0.02 * extent_w,
        )
    )
    inset_y = float(
        np.clip(
            inset_y_target,
            rect_y_min - 4.35 * pad,
            rect_y_max - inset_h - 0.03 * extent_h,
        )
    )

    grouped = build_interest_groups(edge_a, edge_b)
    bax = ax.inset_axes(
        [inset_x, inset_y, inset_w, inset_h], transform=ax.transData, zoom=0
    )
    highlight_axes.append(bax)
    bax.set_in_layout(False)
    draw_grouped_rca_inset(
        bax,
        edge_a=edge_a,
        edge_b=edge_b,
        groups=grouped,
        callout_color=callout_color,
        callout_linewidth=callout_linewidth,
    )
    bax.set_facecolor("white")
    bax.patch.set_alpha(0.96)
    bax.patch.set_edgecolor(callout_color)
    bax.patch.set_linewidth(callout_linewidth)
    for spine in bax.spines.values():
        spine.set_edgecolor(callout_color)
        spine.set_linewidth(callout_linewidth)
    ax.text(
        inset_x + inset_w / 2,
        inset_y - 0.035 * extent_h,
        "Actors behind this link",
        ha="center",
        va="top",
        fontsize=12.0,
        color=callout_color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.18),
        zorder=6,
        clip_on=False,
    )
    # bax.text(
    #     0.03,
    #     0.03,
    #     "Edges shown only for RCA > 1",
    #     transform=bax.transAxes,
    #     ha="left",
    #     va="bottom",
    #     fontsize=5.3,
    #     color="0.35",
    # )
    #
    # bax.invert_yaxis()

    # Straight leader from edge midpoint to the inset top edge.
    arrow_x = float(np.clip(mx, inset_x + 0.06 * inset_w, inset_x + 0.94 * inset_w))
    arrow_y = inset_y + inset_h
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        (mx, my),
        (arrow_x, arrow_y),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=callout_linewidth,
        color=callout_color,
        alpha=0.9,
        connectionstyle="arc3,rad=0.0",
        transform=ax.transData,
        zorder=3.2,
        clip_on=False,
    )
    ax.add_patch(arrow)
    highlight_other_artists.append(arrow)

    # Keep the callout clean: do not add a separate proximity text box.

from matplotlib import patches as mpatches

legend_elements = [
    mpatches.Patch(facecolor=color, label=theme)
    for theme, color in theme_colors.items()
]
legend_obj = ax.legend(
    handles=legend_elements,
    loc="b",
    fontsize=14,
    framealpha=0.0,
    title="Themes",
    title_fontsize=20,
)
plot_margin_x = 2.1 * pad
plot_margin_y = 2.5 * pad
plot_x_min = rect_x_min
plot_x_max = rect_x_max
plot_y_min = rect_y_min
plot_y_max = rect_y_max
if hull_bounds is not None:
    plot_x_min = min(plot_x_min, hull_bounds[0])
    plot_x_max = max(plot_x_max, hull_bounds[1])
    plot_y_min = min(plot_y_min, hull_bounds[2])
    plot_y_max = max(plot_y_max, hull_bounds[3])
ax.set_xlim(plot_x_min - plot_margin_x, plot_x_max + plot_margin_x)
ax.set_ylim(plot_y_min - plot_margin_y, plot_y_max + plot_margin_y)
Path("./figures").mkdir(parents=True, exist_ok=True)
debug_print("Saving main figure assets...")
_t0 = time.perf_counter()
if SAVE_MAIN_PNG:
    fig.savefig(
        "./figures/fig01_space_of_concerns_topology.png",
        dpi=MAIN_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
if SAVE_MAIN_PDF:
    fig.savefig(
        "./figures/fig01_space_of_concerns_topology.pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
if SAVE_MAIN_SVG:
    debug_print("Saving SVG export...")
    fig.savefig(
        "./output/fig01_space_of_concerns_topology.svg",
        bbox_inches="tight",
        pad_inches=0.05,
    )
debug_print(f"Saved main figure assets in {time.perf_counter() - _t0:.2f}s")


if SAVE_POSTER:
    from matplotlib.patches import FancyArrowPatch as _PosterArrow
    from matplotlib.text import Text as _PosterText

    debug_print(
        f"Saving poster variant (canvas x{POSTER_FIG_SCALE}, text x{POSTER_TEXT_SCALE}, "
        f"lines x{POSTER_LINE_SCALE})..."
    )
    _t0 = time.perf_counter()

    # Enlarge the physical canvas first. Side labels live in data coordinates,
    # so a bigger canvas stretches the gaps between them; enlarging text by a
    # slightly smaller factor then keeps the dense columns collision-free while
    # still printing larger, more legible labels.
    _poster_figsize = fig.get_size_inches().copy()
    fig.set_size_inches(_poster_figsize * float(POSTER_FIG_SCALE))

    # Scale every text artist up so labels read at a distance. bbox_inches=
    # "tight" reflows the canvas around the enlarged text so nothing clips.
    _poster_text_state = []
    for _txt in fig.findobj(match=_PosterText):
        _fs = _txt.get_fontsize()
        if _fs is None:
            continue
        _poster_text_state.append((_txt, _fs))
        _txt.set_fontsize(float(_fs) * float(POSTER_TEXT_SCALE))

    # Thicken connector arrows/arrowheads so they stay proportionate to the
    # larger labels.
    _poster_arrow_state = []
    for _art in fig.findobj(match=_PosterArrow):
        _lw = _art.get_linewidth()
        _ms = _art.get_mutation_scale()
        _poster_arrow_state.append((_art, _lw, _ms))
        if _lw is not None:
            _art.set_linewidth(float(_lw) * float(POSTER_LINE_SCALE))
        if _ms is not None:
            _art.set_mutation_scale(float(_ms) * float(POSTER_LINE_SCALE))

    # Keep node markers prominent relative to the enlarged canvas (marker area
    # is in points**2, so scale by the square of the canvas factor).
    _poster_size_state = []
    for _coll in ax.collections:
        if not hasattr(_coll, "get_sizes"):
            continue
        _sizes = _coll.get_sizes()
        if _sizes is None or len(_sizes) == 0:
            continue
        _poster_size_state.append((_coll, _sizes.copy()))
        _coll.set_sizes(_sizes * float(POSTER_FIG_SCALE) ** 2)

    # Drop the Themes legend on the poster (colors are self-evident at this
    # size and it frees vertical room).
    _poster_legend_visible = None
    if legend_obj is not None:
        _poster_legend_visible = legend_obj.get_visible()
        legend_obj.set_visible(False)

    fig.savefig(
        "./figures/fig01_space_of_concerns_topology_poster.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        "./figures/fig01_space_of_concerns_topology_poster.png",
        dpi=POSTER_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=True,
    )

    # Restore original styling so any later exports (e.g. reveal frames) match
    # the manuscript figure.
    fig.set_size_inches(_poster_figsize)
    for _txt, _fs in _poster_text_state:
        _txt.set_fontsize(_fs)
    for _art, _lw, _ms in _poster_arrow_state:
        _art.set_linewidth(_lw)
        _art.set_mutation_scale(_ms)
    for _coll, _sizes in _poster_size_state:
        _coll.set_sizes(_sizes)
    if legend_obj is not None and _poster_legend_visible is not None:
        legend_obj.set_visible(_poster_legend_visible)

    debug_print(f"Saved poster variant in {time.perf_counter() - _t0:.2f}s")


def _slugify_label(text):
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _set_visible(artist, visible):
    if artist is not None:
        artist.set_visible(bool(visible))


def _set_many_visible(artists, visible):
    for artist in artists:
        _set_visible(artist, visible)


def _save_reveal_frame(stem):
    fig.savefig(
        str(REVEAL_OUTPUT_DIR / f"{stem}.png"),
        dpi=1200,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    fig.savefig(
        str(REVEAL_OUTPUT_DIR / f"{stem}.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )


node_to_cluster = {}
for item in cluster_hulls:
    cid = int(item.get("cluster_id", -1))
    for node in item.get("nodes", []):
        node_to_cluster[node] = cid


def _apply_slide_text_scale(scale=1.55):
    for rec in side_label_artists:
        lbl = rec.get("label")
        if lbl is not None:
            lbl.set_fontsize(float(side_fs) * float(scale))
    for artists in semantic_label_artists_by_cluster.values():
        for artist in artists:
            if artist is not None:
                artist.set_fontsize(8.6 * float(scale))
    if legend_obj is not None:
        for text in legend_obj.get_texts():
            text.set_fontsize(14.0 * float(scale))
        title = legend_obj.get_title()
        if title is not None:
            title.set_fontsize(20.0 * float(scale))


def _apply_semantic_filter(cluster_id, show_dashed=False):
    show_all = cluster_id is not None and str(cluster_id).lower() == "all"
    # Hide/show semantic areas.
    for cid, artists in hull_artists_by_cluster.items():
        _set_many_visible(
            artists,
            bool(show_all or (cluster_id is not None and int(cid) == int(cluster_id))),
        )
    for cid, artists in semantic_label_artists_by_cluster.items():
        _set_many_visible(
            artists,
            bool(show_all or (cluster_id is not None and int(cid) == int(cluster_id))),
        )

    # Hide/show side topic boxes + connectors while preserving fixed coordinates.
    for rec in side_label_artists:
        node = rec.get("node")
        show = bool(
            show_all
            or (cluster_id is not None and node_to_cluster.get(node) == cluster_id)
        )
        _set_visible(rec.get("label"), show)
        _set_visible(rec.get("arrow"), show)

    # Keep reveal frames clean: no inset overlays.
    _set_many_visible(highlight_line_artists, bool(show_dashed))
    _set_many_visible(highlight_other_artists, False)
    for inset_ax in highlight_axes:
        inset_ax.set_visible(False)


if GENERATE_REVEAL_SEQUENCE:
    debug_print("Generating reveal sequence...")
    _t0 = time.perf_counter()
    REVEAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Slide reveal sequence with fixed geometry:
    #   00: themes only (no boxes / no areas)
    #   01..N: one semantic area at a time, with only corresponding topic boxes shown.
    _apply_slide_text_scale(scale=1.0)
    if legend_obj is not None:
        legend_obj.set_visible(False)
    _apply_semantic_filter(cluster_id=None, show_dashed=True)
    _save_reveal_frame("fig01_space_of_concerns_topology_reveal_00_themes_only")
    for idx, cid in enumerate(sorted(hull_artists_by_cluster.keys()), start=1):
        debug_print(f"Saving reveal frame for cluster {cid} ({idx})...")
        _apply_semantic_filter(cluster_id=int(cid), show_dashed=True)
        label = cluster_label_by_id.get(cid, f"cluster_{cid}")
        slug = _slugify_label(label)
        _save_reveal_frame(f"fig01_space_of_concerns_topology_reveal_{idx:02d}_{slug}")
    _apply_semantic_filter(cluster_id="all", show_dashed=True)
    _save_reveal_frame("fig01_space_of_concerns_topology_reveal_05_all_semantics")

    # Extra slide pair:
    # - graph only (no semantic overlays)
    # - graph + callout inset and proximity label
    _apply_semantic_filter(cluster_id=None, show_dashed=False)
    _save_reveal_frame("fig01_space_of_concerns_topology_reveal_06_graph_only")
    _apply_semantic_filter(cluster_id=None, show_dashed=True)
    _set_many_visible(highlight_other_artists, True)
    for inset_ax in highlight_axes:
        inset_ax.set_visible(True)
    _save_reveal_frame(
        "fig01_space_of_concerns_topology_reveal_07_graph_only_with_callout"
    )
    debug_print(f"Generated reveal sequence in {time.perf_counter() - _t0:.2f}s")
