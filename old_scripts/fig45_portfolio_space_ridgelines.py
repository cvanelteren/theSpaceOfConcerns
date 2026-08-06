# %%
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import (
    compute_product_space,
    get_rca,
    load_data,
    standardize_index_labels,
)

# %%
DATA_FP = "./antarctic-database-go/data/processed/document-summary.parquet"
LOG_EPS = 1e-12
RCA_THRESHOLD = 1.0
MIN_ACTIVE_TOPICS = 3
RIDGE_HEIGHT = 0.8
GAUSS_SIGMA = 1.15
GAUSS_RADIUS = 3
LOESS_FRAC = 0.14
LOESS_MIN_POINTS = 7
PEAK_MIN_REL_HEIGHT = 0.25
TOP_LABEL_FONTSIZE = 6.5
TOP_LABEL_BASE_PT = 3.0
TOP_LABEL_ROW_STEP_PT = 14.0
TOP_LABEL_PAD_PX = 18.0
SAVE_PAD_INCHES = 0.4
N_REGIONS = 3
FIVETHIRTYEIGHT_COLORS = [
    "#008fd5",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
    "#810f7c",
]
ALL_TOPIC_COLOR = "#9aa0a6"
ACTIVE_TOPIC_COLOR = "#000000"

OUT_FIG = Path("figures/fig45_portfolio_space_ridgelines.png")
OUT_PDF = Path("figures/fig45_portfolio_space_ridgelines.pdf")
OUT_TOPIC_CSV = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
OUT_ACTOR_CSV = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")
OUT_REGION_CSV = Path("output/fig45_portfolio_space_ridgelines_region_summary.csv")
FIG46_CLUSTER_CSV = Path("output/fig46_member_ridge_similarity_order.csv")


# %%
def _prepare_counts():
    counts, submitted, countries, topics = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()
    return counts


def _build_distance_matrix(phi: pd.DataFrame):
    phi = (phi + phi.T) / 2.0
    g = nx.Graph()
    for topic_i in phi.index:
        g.add_node(topic_i)
    for i, topic_i in enumerate(phi.index):
        for j in range(i + 1, len(phi.index)):
            topic_j = phi.index[j]
            weight = float(phi.iat[i, j])
            if weight <= 0:
                continue
            g.add_edge(
                topic_i,
                topic_j,
                weight=weight,
                distance=float(-np.log(np.clip(weight, LOG_EPS, 1.0))),
            )

    dist = nx.floyd_warshall_numpy(g, weight="distance")
    return phi.index.to_list(), np.asarray(dist, dtype=float)


def _mds_1d(distance_matrix: np.ndarray):
    n = distance_matrix.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (distance_matrix**2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = int(np.argmax(eigvals))
    eigval = float(max(eigvals[idx], 0.0))
    coord = eigvecs[:, idx] * np.sqrt(eigval)
    return np.asarray(coord, dtype=float)


def _gaussian_smooth(values: np.ndarray, sigma: float, radius: int):
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _weighted_median(x_positions: np.ndarray, weights: np.ndarray):
    x = np.asarray(x_positions, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
        return float(np.median(x))
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cum, 0.5 * total, side="left"))
    idx = min(max(idx, 0), len(x_sorted) - 1)
    return float(x_sorted[idx])


def _loess_smooth(
    x_positions: np.ndarray,
    values: np.ndarray,
    frac: float = LOESS_FRAC,
    min_points: int = LOESS_MIN_POINTS,
):
    x = np.asarray(x_positions, dtype=float)
    y = np.asarray(values, dtype=float)
    n = y.size
    if n <= 2:
        return y.copy()

    span = int(max(min_points, np.ceil(frac * n)))
    span = min(span, n)
    smoothed = np.zeros_like(y, dtype=float)

    for idx, x0 in enumerate(x):
        dist = np.abs(x - x0)
        h = float(np.partition(dist, span - 1)[span - 1])
        if h <= 0:
            nonzero = dist[dist > 0]
            h = float(nonzero.min()) if nonzero.size else 1.0

        u = np.clip(dist / h, 0.0, 1.0)
        w = (1.0 - u**3) ** 3
        if not np.any(w > 0):
            smoothed[idx] = float(y[idx])
            continue

        x_centered = x - x0
        sw = float(np.sum(w))
        swx = float(np.sum(w * x_centered))
        swxx = float(np.sum(w * x_centered * x_centered))
        swy = float(np.sum(w * y))
        swxy = float(np.sum(w * x_centered * y))

        det = sw * swxx - swx * swx
        if abs(det) <= 1e-12:
            estimate = swy / sw if sw > 0 else float(y[idx])
        else:
            intercept = (swxx * swy - swx * swxy) / det
            estimate = intercept
        smoothed[idx] = max(float(estimate), 0.0)

    return smoothed


def _find_peak_indices(values: np.ndarray, min_rel_height: float = PEAK_MIN_REL_HEIGHT):
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    max_val = float(np.nanmax(values))
    if max_val <= 0:
        return np.array([], dtype=int)
    min_height = min_rel_height * max_val

    peaks = []
    if values[0] > values[1] and values[0] >= min_height:
        peaks.append(0)
    for idx in range(1, n - 1):
        left = float(values[idx - 1])
        center = float(values[idx])
        right = float(values[idx + 1])
        if center < min_height:
            continue
        if center >= left and center >= right and (center > left or center > right):
            peaks.append(idx)
    if values[-1] > values[-2] and values[-1] >= min_height:
        peaks.append(n - 1)

    if not peaks:
        return np.array([int(np.argmax(values))], dtype=int)
    return np.asarray(peaks, dtype=int)


def _ridge_peak_summary(values: np.ndarray, x_positions: np.ndarray, topics: list[str]):
    peak_idx = _find_peak_indices(values)
    if peak_idx.size == 0:
        return {
            "peak_count": 0,
            "largest_peak_x": np.nan,
            "largest_peak_topic": "",
            "peak_separation": np.nan,
        }

    peak_heights = values[peak_idx]
    order = np.argsort(peak_heights)[::-1]
    ranked_idx = peak_idx[order]

    largest_idx = int(ranked_idx[0])
    peak_separation = np.nan
    if ranked_idx.size >= 2:
        peak_separation = float(
            abs(x_positions[int(ranked_idx[0])] - x_positions[int(ranked_idx[1])])
        )

    return {
        "peak_count": int(peak_idx.size),
        "largest_peak_x": float(x_positions[largest_idx]),
        "largest_peak_topic": str(topics[largest_idx]),
        "peak_separation": peak_separation,
    }


def _add_non_overlapping_top_labels(ax, x_positions: np.ndarray, labels: list[str]):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_display = ax.transData.transform(
        np.column_stack([x_positions, np.zeros_like(x_positions)])
    )[:, 0]

    widths = []
    for label in labels:
        tmp = fig.text(0, 0, label, fontsize=TOP_LABEL_FONTSIZE, visible=False)
        bbox = tmp.get_window_extent(renderer=renderer)
        widths.append(float(bbox.width) + TOP_LABEL_PAD_PX)
        tmp.remove()

    row_right_edges = []
    row_indices = []
    for x_px, width_px in zip(x_display, widths):
        left_px = float(x_px - width_px / 2.0)
        right_px = float(x_px + width_px / 2.0)
        placed = False
        for row_idx, last_right in enumerate(row_right_edges):
            if left_px >= last_right + TOP_LABEL_PAD_PX:
                row_right_edges[row_idx] = right_px
                row_indices.append(row_idx)
                placed = True
                break
        if not placed:
            row_right_edges.append(right_px)
            row_indices.append(len(row_right_edges) - 1)

    for xpos, label, row_idx in zip(x_positions, labels, row_indices):
        y_offset_pt = TOP_LABEL_BASE_PT + row_idx * TOP_LABEL_ROW_STEP_PT
        ax.annotate(
            label,
            xy=(float(xpos), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0.0, y_offset_pt),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=TOP_LABEL_FONTSIZE,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.45,
                "color": "0.35",
                "shrinkA": 0,
                "shrinkB": 0,
            },
            clip_on=False,
            annotation_clip=False,
        )


def _select_region_centers(
    x_positions: np.ndarray,
    aggregate_signal: np.ndarray,
    n_regions: int = N_REGIONS,
    n_iter: int = 25,
):
    smooth = _gaussian_smooth(aggregate_signal, sigma=GAUSS_SIGMA, radius=GAUSS_RADIUS)
    if smooth.size == 0:
        return np.array([], dtype=int), smooth

    weights = np.asarray(smooth, dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)

    lo = float(np.min(x_positions))
    hi = float(np.max(x_positions))
    centers = np.linspace(lo, hi, n_regions)

    x = np.asarray(x_positions, dtype=float)
    for _ in range(n_iter):
        assign = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for k in range(n_regions):
            mask = assign == k
            if not np.any(mask):
                continue
            w = weights[mask]
            new_centers[k] = float(np.average(x[mask], weights=w))
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    anchor_idx = []
    for center in centers:
        idx = int(np.argmin(np.abs(x - center)))
        anchor_idx.append(idx)

    # Deduplicate while preserving order, then sort by x-position.
    anchor_idx = sorted(dict.fromkeys(anchor_idx), key=lambda idx: x[idx])
    return np.asarray(anchor_idx, dtype=int), smooth


# %%
counts = _prepare_counts()
rca = get_rca(counts)
phi = compute_product_space(rca)

topic_names, distance_matrix = _build_distance_matrix(phi)
space_coord = _mds_1d(distance_matrix)
topic_order = np.argsort(space_coord)

ordered_topics = [topic_names[i] for i in topic_order]
ordered_coord = space_coord[topic_order]

coord_min = float(ordered_coord.min())
coord_max = float(ordered_coord.max())
if coord_max > coord_min:
    x_plot = (ordered_coord - coord_min) / (coord_max - coord_min)
else:
    x_plot = np.zeros_like(ordered_coord)


# %%
rca = rca.reindex(index=topic_names)
ridge_signal = rca.to_numpy(dtype=float)
active_mask = ridge_signal > RCA_THRESHOLD
region_signal = np.clip(ridge_signal - RCA_THRESHOLD, 0.0, None)
aggregate_region_signal = region_signal.sum(axis=1)[topic_order]
region_peak_idx, aggregate_region_signal_smooth = _select_region_centers(
    x_plot,
    aggregate_region_signal,
)

region_peak_x = x_plot[region_peak_idx]
region_peak_topics = [ordered_topics[idx] for idx in region_peak_idx]
region_boundaries = (
    0.5 * (region_peak_x[:-1] + region_peak_x[1:])
    if len(region_peak_x) > 1
    else np.array([])
)
topic_region_idx = np.digitize(x_plot, region_boundaries).astype(int)

region_palette = uplt.Colormap("batlow")(np.linspace(0.15, 0.85, len(region_peak_idx)))

actor_records = []
for col_idx, actor in enumerate(rca.columns):
    k_active = int(active_mask[:, col_idx].sum())
    if k_active < MIN_ACTIVE_TOPICS:
        continue

    weights = ridge_signal[:, col_idx]
    if weights.sum() <= 0:
        continue

    ridge = ridge_signal[:, col_idx][topic_order]
    ridge_loess = _loess_smooth(x_plot, ridge)
    peak_summary = _ridge_peak_summary(ridge_loess, x_plot, ordered_topics)

    centroid = float(np.average(space_coord, weights=weights))
    centroid_xplot = float(np.average(x_plot, weights=ridge))
    median_xplot = _weighted_median(x_plot, ridge)
    spread = float(np.sqrt(np.average((space_coord - centroid) ** 2, weights=weights)))
    var_xplot = float(np.average((x_plot - centroid_xplot) ** 2, weights=ridge))
    if var_xplot > 0:
        skew_xplot = float(
            np.average((x_plot - centroid_xplot) ** 3, weights=ridge) / (var_xplot**1.5)
        )
    else:
        skew_xplot = 0.0
    region_weights = np.bincount(
        topic_region_idx,
        weights=region_signal[:, col_idx][topic_order],
        minlength=len(region_peak_idx),
    ).astype(float)
    region_total = float(region_weights.sum())
    if region_total > 0:
        region_shares = region_weights / region_total
    else:
        region_shares = np.zeros_like(region_weights)
    dominant_region_idx = int(np.argmax(region_shares)) if region_shares.size else 0
    actor_records.append(
        {
            "actor": actor,
            "column_index": col_idx,
            "k_active": k_active,
            "centroid_coord": centroid,
            "centroid_xplot": centroid_xplot,
            "median_xplot": median_xplot,
            "spread_coord": spread,
            "skew_xplot": skew_xplot,
            "dominant_region": dominant_region_idx + 1,
            "dominant_region_share": float(region_shares[dominant_region_idx]),
            **peak_summary,
            **{
                f"region_{i + 1}_share": float(region_shares[i])
                for i in range(len(region_peak_idx))
            },
        }
    )

actor_df = pd.DataFrame(actor_records)

if FIG46_CLUSTER_CSV.exists():
    cluster_df = pd.read_csv(FIG46_CLUSTER_CSV, usecols=["actor", "cluster_id"])
    cluster_df = cluster_df.drop_duplicates(subset=["actor"])
    actor_df = actor_df.merge(cluster_df, on="actor", how="left")
else:
    actor_df["cluster_id"] = np.nan
actor_df = actor_df.sort_values(
    [
        "largest_peak_x",
        "median_xplot",
        "centroid_xplot",
        "spread_coord",
        "k_active",
        "actor",
    ],
    ascending=[False, False, True, True, False, True],
)


# %%
figheight = max(6.0, 0.34 * len(actor_df) + 1.8)
fig, ax = uplt.subplots(figsize=(10.5, figheight), share=0)

actor_colors = uplt.Colormap("roma")(np.linspace(0.08, 0.92, max(len(actor_df), 2)))
for row_idx, (_, row) in enumerate(actor_df.reset_index(drop=True).iterrows()):
    col_idx = int(row["column_index"])
    baseline = float(row_idx)

    ridge = ridge_signal[:, col_idx][topic_order]
    ridge = _loess_smooth(x_plot, ridge)
    ridge_max = float(ridge.max())
    if ridge_max > 0:
        ridge = ridge / ridge_max

    color = actor_colors[row_idx]
    dominant_region_idx = int(row["dominant_region"]) - 1
    left_edge = (
        0.0
        if dominant_region_idx == 0
        else float(region_boundaries[dominant_region_idx - 1])
    )
    right_edge = (
        1.0
        if dominant_region_idx == len(region_peak_idx) - 1
        else float(region_boundaries[dominant_region_idx])
    )
    ax.axvspan(
        left_edge,
        right_edge,
        ymin=max(0.0, (baseline + 0.02) / max(1.0, len(actor_df))),
        ymax=min(1.0, (baseline + RIDGE_HEIGHT + 0.02) / max(1.0, len(actor_df))),
        color=region_palette[dominant_region_idx],
        alpha=0.03,
        lw=0,
        zorder=0,
    )
    ax.fill_between(
        x_plot,
        baseline,
        baseline + RIDGE_HEIGHT * ridge,
        color=color,
        alpha=0.8,
        lw=0.0,
    )
    ax.plot(
        x_plot,
        baseline + RIDGE_HEIGHT * ridge,
        color=color,
        lw=1.1,
        alpha=0.95,
    )

    all_positions = x_plot
    ax.scatter(
        all_positions,
        np.full_like(all_positions, baseline),
        s=6,
        color=ALL_TOPIC_COLOR,
        alpha=1,
        linewidth=0,
        zorder=2,
    )

    active_positions = x_plot[active_mask[:, col_idx][topic_order]]
    if active_positions.size:
        ax.scatter(
            active_positions,
            np.full_like(active_positions, baseline),
            s=14,
            color=ACTIVE_TOPIC_COLOR,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.98,
            zorder=4,
        )

ax.format(
    xlabel="MDS-Scaled Position in the Space of Concerns (0-1)",
    ylabel="Actor",
)
ax.set_xlim(0.0, 1.0)
ax.margins(x=0)
ax.set_xticks(np.linspace(0.0, 1.0, 5))

ax.set_yticks(np.arange(len(actor_df), dtype=float))
ax.set_yticklabels(actor_df["actor"].tolist(), fontsize=8)
ax.set_ylim(-0.5, len(actor_df) - 0.2 + RIDGE_HEIGHT)
ax.grid(axis="x", alpha=0.2, linewidth=0.6)
ax.grid(axis="y", visible=False)

if "cluster_lr_order" in actor_df and actor_df["cluster_lr_order"].notna().any():
    cluster_rows = actor_df["cluster_lr_order"].to_numpy(dtype=int)
    cluster_text_colors = {
        cid: FIVETHIRTYEIGHT_COLORS[(cid - 1) % len(FIVETHIRTYEIGHT_COLORS)]
        for cid in sorted(np.unique(cluster_rows))
        if cid > 0
    }
    for tick, cid in zip(ax.get_yticklabels(), cluster_rows):
        tick.set_color(cluster_text_colors.get(int(cid), "black"))

    cluster_breaks = np.where(np.diff(cluster_rows) != 0)[0]
    for boundary in cluster_breaks:
        ax.axhline(float(boundary) + 0.5, color="0.15", lw=0.9, alpha=0.35, zorder=2)

    cluster_ids_present = [cid for cid in sorted(np.unique(cluster_rows)) if cid > 0]
    for cid in cluster_ids_present:
        idx = np.where(cluster_rows == cid)[0]
        if idx.size == 0:
            continue
        y_center = 0.5 * (float(idx[0]) + float(idx[-1]))
        ax.text(
            -0.018,
            y_center,
            f"C{cid}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.5,
            color=cluster_text_colors.get(int(cid), "0.25"),
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

_add_non_overlapping_top_labels(
    ax,
    region_peak_x,
    region_peak_topics,
)

for boundary in region_boundaries:
    ax.axvline(float(boundary), color="0.5", lw=0.8, ls=":", alpha=0.6, zorder=1)


# %%
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_TOPIC_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_ACTOR_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_REGION_CSV.parent.mkdir(parents=True, exist_ok=True)

# Actor-level portfolio metrics on full graph distances (log-distance).
ordered_distance_matrix = distance_matrix[np.ix_(topic_order, topic_order)]
x_plot_arr = np.asarray(x_plot, dtype=float)
metric_rows = []
for _, row in actor_df.iterrows():
    col_idx = int(row["column_index"])
    weights = np.clip(ridge_signal[:, col_idx][topic_order], 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        metric_rows.append(
            {
                "actor": row["actor"],
                "centroid_xplot_raw_rca": np.nan,
                "localization_l1_xplot_raw_rca": np.nan,
                "localization_sd_xplot_raw_rca": np.nan,
                "entropy_raw_rca": np.nan,
                "entropy_norm_k_active_raw_rca": np.nan,
                "support_raw_rca": np.nan,
                "entropy_norm_support_raw_rca": np.nan,
                "concentration_hhi_raw_rca": np.nan,
                "effective_topics_raw_rca": np.nan,
                "compactness_pairwise_logdist_raw_rca": np.nan,
                "medoid_radius_logdist_raw_rca": np.nan,
                "medoid_topic_raw_rca": "",
            }
        )
        continue

    p = weights / total
    centroid_xplot = float(np.sum(p * x_plot_arr))
    localization_l1 = float(np.sum(p * np.abs(x_plot_arr - centroid_xplot)))
    localization_sd = float(np.sqrt(np.sum(p * (x_plot_arr - centroid_xplot) ** 2)))
    p_pos = p[p > 0]
    entropy = float(-np.sum(p_pos * np.log2(p_pos)))
    k_active = int(row["k_active"])
    if k_active > 1:
        entropy_norm = float(entropy / np.log2(float(k_active)))
    else:
        entropy_norm = np.nan
    support_raw = int(np.sum(weights > 0))
    if support_raw > 1:
        entropy_norm_support = float(entropy / np.log2(float(support_raw)))
    else:
        entropy_norm_support = np.nan
    concentration_hhi = float(np.sum(p**2))
    if concentration_hhi > 0:
        effective_topics = float(1.0 / concentration_hhi)
    else:
        effective_topics = np.nan

    # Expected pairwise log-distance under portfolio weights.
    compactness_pairwise = float(p @ ordered_distance_matrix @ p)
    medoid_costs = ordered_distance_matrix @ p
    medoid_idx = int(np.argmin(medoid_costs))
    medoid_radius = float(medoid_costs[medoid_idx])
    medoid_topic = str(ordered_topics[medoid_idx])

    metric_rows.append(
        {
            "actor": row["actor"],
            "centroid_xplot_raw_rca": centroid_xplot,
            "localization_l1_xplot_raw_rca": localization_l1,
            "localization_sd_xplot_raw_rca": localization_sd,
            "entropy_raw_rca": entropy,
            "entropy_norm_k_active_raw_rca": entropy_norm,
            "support_raw_rca": support_raw,
            "entropy_norm_support_raw_rca": entropy_norm_support,
            "concentration_hhi_raw_rca": concentration_hhi,
            "effective_topics_raw_rca": effective_topics,
            "compactness_pairwise_logdist_raw_rca": compactness_pairwise,
            "medoid_radius_logdist_raw_rca": medoid_radius,
            "medoid_topic_raw_rca": medoid_topic,
        }
    )

metrics_df = pd.DataFrame(metric_rows)
actor_df = actor_df.merge(metrics_df, on="actor", how="left")

fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)

pd.DataFrame(
    {
        "topic_order": np.arange(len(ordered_topics), dtype=int),
        "topic": ordered_topics,
        "space_coord": ordered_coord,
        "x_plot": x_plot,
        "aggregate_region_signal": aggregate_region_signal,
        "aggregate_region_signal_smooth": aggregate_region_signal_smooth,
        "region_id": topic_region_idx + 1,
        "is_region_anchor": [
            bool(idx in set(region_peak_idx)) for idx in range(len(ordered_topics))
        ],
    }
).to_csv(OUT_TOPIC_CSV, index=False)

actor_df.to_csv(OUT_ACTOR_CSV, index=False)

pd.DataFrame(
    {
        "region_id": np.arange(1, len(region_peak_idx) + 1, dtype=int),
        "anchor_topic": region_peak_topics,
        "anchor_x": region_peak_x,
        "boundary_left": [0.0] + region_boundaries.tolist(),
        "boundary_right": region_boundaries.tolist() + [1.0],
    }
).to_csv(OUT_REGION_CSV, index=False)

uplt.show(block=1)
