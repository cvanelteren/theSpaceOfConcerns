# %%
"""Graph-only figure: space of concerns fitted to the Antarctica silhouette.

The silhouette is a real cartopy geoaxis (South Polar Stereographic, Antarctic
landmass only) rather than a bitmap tracing, and the node layout is snapped to
points sampled inside that same projected coastline, so map and graph register
exactly.

The expensive part of the figure is the side-label assignment, not the node
layout. Cache that derived layout so routine reruns only recompute it when the
underlying geometry changes.
"""

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import ultraplot as uplt
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull

import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from shapely import contains_xy
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union

    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    HAS_SHAPELY = False

import figstyle
from utils import compute_product_space, get_rca, load_data, load_flag

USE_CACHED_SIDE_LAYOUT = True
REFRESH_SIDE_LAYOUT = False
SIDE_LAYOUT_CACHE_VERSION = 10
SIDE_LAYOUT_CACHE_PATH = Path("assets/cache/fig01_side_layout_cache.json")
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

# Basemap. central_longitude=0 puts the Antarctic Peninsula in the upper left,
# which is the orientation the node layout was tuned against. MAP_LAT_MAX keeps
# every non-Antarctic landmass out of the frame, so nothing can clip in.
MAP_PROJECTION = ccrs.SouthPolarStereo(central_longitude=0)
MAP_RESOLUTION = "50m"
MAP_LAT_MAX = -60.0
MAP_MARGIN = 0.04  # ocean rim around the coastline bounding box
MAP_SILHOUETTE_WIDTH = 7.0  # width of the silhouette in figure data units
MAP_LAND_COLOR = "#e9e9e9"
MAP_COAST_COLOR = "#9a9a9a"

# Layout orientation applied to the coastline-snapped node positions.
GRAPH_ROTATION_DEG = 45.0
GRAPH_SPREAD = 4.0  # >1 pushes the graph past the coast so it dominates the frame


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
            color=figstyle.BRANCH_COLORS["drilling"],
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
            color=figstyle.BRANCH_COLORS["human_impact"],
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
            color=figstyle.BRANCH_COLORS["environmental"],
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
            color=figstyle.BRANCH_COLORS["core"],
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


def antarctic_landmass(projection, lat_max=MAP_LAT_MAX):
    """Antarctic land polygons only, projected.

    Selecting geometries that lie entirely south of ``lat_max`` is the clip:
    South America never enters the frame because its polygon is never added.
    """
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
    """Silhouette plus a tessellation of its interior, in figure data units.

    Returns the interior sample points, the projected map extent, the data
    extent the map occupies, and the projected coastline. The two extents are
    two views of the same rectangle, which is what lets the geoaxis be laid over
    the graph without drift.
    """
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
debug_print("Computing deterministic node layout...")
pos = deterministic_layout(mst)
debug_print(f"Computed deterministic node layout in {time.perf_counter() - _t0:.2f}s")

_t0 = time.perf_counter()
debug_print("Building Antarctica silhouette and tessellation from cartopy...")
tess_points, map_proj_extent, mask_extent, land_geometry = build_antarctica_basemap()
debug_print(f"Prepared silhouette/tessellation in {time.perf_counter() - _t0:.2f}s")

# Map Kamada-Kawai positions onto the coastline interior, then orient the result:
# point-reflect so the long backbone runs from the peninsula inward, rotate onto
# the diagonal, and spread past the coast so the graph -- not the map -- sets the
# scale of the figure.
_t0 = time.perf_counter()
debug_print("Snapping graph layout to Antarctica tessellation...")
snapped = snap_to_tessellation(pos, tess_points)
x_min, x_max, y_min, y_max = mask_extent
cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
theta = np.deg2rad(GRAPH_ROTATION_DEG)
cos_t, sin_t = np.cos(theta), np.sin(theta)


def _orient(point):
    px = x_min + x_max - float(point[0])
    py = y_min + y_max - float(point[1])
    dx, dy = px - cx, py - cy
    return np.array(
        [
            cx + GRAPH_SPREAD * (dx * cos_t - dy * sin_t),
            cy + GRAPH_SPREAD * (dx * sin_t + dy * cos_t),
        ]
    )


snapped = {node: _orient(point) for node, point in snapped.items()}
debug_print(f"Snapped and transformed layout in {time.perf_counter() - _t0:.2f}s")

# Sourced from the shared palette rather than declared here: the three mode
# hues used in Figures 2 and 5 are reserved, and these eight are kept clear of
# them so a reader does not read a theme in this figure as a mode in the next.
theme_colors = dict(figstyle.THEME_COLORS)

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
# Antarctica background: a real geoaxis filling the plot box, so the silhouette
# spans the whole frame the way the graph does. ax.inset keeps an axes locator on
# the parent, so the map follows the graph through tight layout and the poster
# rescale instead of drifting off it, and the geoaxis keeps its true aspect. Only
# the Antarctic polygon is drawn, so no other continent can clip into the frame.
geoax = ax.inset([0, 0, 1, 1], proj=MAP_PROJECTION, zoom=False, zorder=-2)
geoax.set_extent(map_proj_extent, crs=MAP_PROJECTION)
geoax.add_geometries(
    [land_geometry],
    crs=MAP_PROJECTION,
    facecolor=MAP_LAND_COLOR,
    edgecolor=MAP_COAST_COLOR,
    linewidth=0.4,
    zorder=0,
)
geoax.patch.set_visible(False)
geoax.format(grid=False, labels=False)
for spine in geoax.spines.values():
    spine.set_visible(False)
debug_print("Placed Antarctica geoaxis over the graph extent")


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
    # Take the group colours from the two topics' own themes rather than a fixed
    # blue/red/purple triple. The anchors at the top of the inset are already
    # drawn in col_a and col_b, so the groups below now match them, and the
    # inset inherits the guarantee that no mode hue appears in this figure.
    edge_semantic = {"drilling": col_a, "both": col_mix, "marine": col_b}

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
        inset_y_target = float(drill_xy[1]) - 4.05 * pad
    else:
        inset_y_target = rect_y_min - 4.05 * pad

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
            rect_y_min - 4.95 * pad,
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
