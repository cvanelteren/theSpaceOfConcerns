"""Static 3D MDS view of the ATS space of concerns.

This keeps the same underlying topic-topic shortest-path geometry used by the
regime ordering, but retains the first three classical-MDS coordinates instead
of collapsing to one dimension.
"""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.animation import FFMpegWriter, FuncAnimation
import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import compute_product_space, get_rca, load_data, standardize_index_labels

DATA_CANDIDATES = [
    Path("antarctic-database-go/data/processed/document-summary.parquet"),
    Path("antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"),
]
TOPIC_ORDER_CSV = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
OUT_PNG = Path("figures/fig01_space_of_concerns_mds3d.png")
OUT_PDF = Path("figures/fig01_space_of_concerns_mds3d.pdf")
OUT_MP4 = Path("figures/fig01_space_of_concerns_mds3d_rotate.mp4")
OUT_COORDS = Path("output/fig01_space_of_concerns_mds3d_coords.csv")
OUT_META = Path("output/fig01_space_of_concerns_mds3d_meta.json")

LOG_EPS = 1e-12
EDGE_PERCENTILE = 95
LABEL_N_EXTRA = 6
VIEW_ELEV = 19
VIEW_AZIM = -58
ROTATE_FPS = 24
ROTATE_SECONDS = 8
ROTATE_FRAMES = ROTATE_FPS * ROTATE_SECONDS
REGION_COLORS = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
}
FALLBACK_NODE_COLOR = "#51606d"


def load_data_with_fallback():
    last_error = None
    for path in DATA_CANDIDATES:
        if not path.exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Failed to load {path}: {exc}")
    if last_error is not None:
        raise RuntimeError("No usable data file found.") from last_error
    raise FileNotFoundError("No usable data file found.")


def filter_space_topics(counts_df: pd.DataFrame) -> pd.DataFrame:
    excluded = {"all", "other"}
    keep = [
        topic for topic in counts_df.index if str(topic).strip().lower() not in excluded
    ]
    return counts_df.loc[keep]


def build_graphs():
    counts_df, _, _, _ = load_data_with_fallback()
    counts_df = standardize_index_labels(counts_df)
    if counts_df.index.has_duplicates:
        counts_df = counts_df.groupby(level=0).sum()
    counts_df = filter_space_topics(counts_df)
    rca = get_rca(counts_df)
    phi = compute_product_space(rca)

    graph = nx.from_pandas_adjacency(phi)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    for _, _, data in graph.edges(data=True):
        weight = float(data.get("weight", 0.0))
        data["weight"] = weight
        data["distance"] = float(-np.log(np.clip(weight, LOG_EPS, 1.0)))

    scaffold = nx.maximum_spanning_tree(graph, weight="weight")
    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    if weights.size:
        cutoff = float(np.percentile(weights, EDGE_PERCENTILE))
        for u, v, data in graph.edges(data=True):
            if float(data["weight"]) >= cutoff:
                scaffold.add_edge(u, v, **data)

    dist = nx.floyd_warshall_numpy(graph, weight="distance")
    dist = np.asarray(dist, dtype=float)
    if not np.isfinite(dist).all():
        finite = dist[np.isfinite(dist)]
        if finite.size == 0:
            raise ValueError("Distance matrix is fully non-finite.")
        fill = float(finite.max()) * 1.05
        dist = np.where(np.isfinite(dist), dist, fill)

    return phi, graph, scaffold, dist


def classical_mds(distance_matrix: np.ndarray, n_components: int = 3):
    n = distance_matrix.shape[0]
    j = np.eye(n) - np.ones((n, n), dtype=float) / float(n)
    gram = -0.5 * j @ (distance_matrix**2) @ j
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pos = np.clip(eigvals[:n_components], 0.0, None)
    coords = eigvecs[:, :n_components] * np.sqrt(pos)
    return np.asarray(coords, dtype=float), eigvals


def load_topic_metadata(topics: list[str]) -> pd.DataFrame:
    meta = pd.DataFrame({"topic": topics})
    if TOPIC_ORDER_CSV.exists():
        keep = ["topic", "space_coord", "x_plot", "region_id", "is_region_anchor"]
        topic_order_df = pd.read_csv(TOPIC_ORDER_CSV, usecols=keep)
        meta = meta.merge(topic_order_df, on="topic", how="left")
    else:
        meta["space_coord"] = np.nan
        meta["x_plot"] = np.nan
        meta["region_id"] = np.nan
        meta["is_region_anchor"] = False
    meta["is_region_anchor"] = meta["is_region_anchor"].fillna(False).astype(bool)
    return meta


def align_first_axis(coords: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    out = coords.copy()
    if "space_coord" not in meta or meta["space_coord"].isna().all():
        return out
    target = meta["space_coord"].to_numpy(dtype=float)
    valid = np.isfinite(target) & np.isfinite(out[:, 0])
    if valid.sum() < 3:
        return out
    corr = np.corrcoef(out[valid, 0], target[valid])[0, 1]
    if np.isfinite(corr) and corr < 0:
        out[:, 0] *= -1.0
    return out


def scale_coords(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    for idx in range(out.shape[1]):
        col = out[:, idx]
        span = float(col.max() - col.min())
        if span > 1e-12:
            out[:, idx] = 2.0 * (col - col.min()) / span - 1.0
        else:
            out[:, idx] = 0.0
    return out


def choose_labels(meta: pd.DataFrame, graph: nx.Graph, coords: np.ndarray) -> list[str]:
    labels = set(meta.loc[meta["is_region_anchor"], "topic"].tolist())

    degree_strength = pd.Series(
        {
            node: float(
                sum(d.get("weight", 0.0) for _, _, d in graph.edges(node, data=True))
            )
            for node in graph.nodes()
        },
        name="weighted_degree",
    )
    for topic in degree_strength.sort_values(ascending=False).head(LABEL_N_EXTRA).index:
        labels.add(str(topic))

    for axis_idx in range(coords.shape[1]):
        labels.add(str(meta.iloc[int(np.argmin(coords[:, axis_idx]))]["topic"]))
        labels.add(str(meta.iloc[int(np.argmax(coords[:, axis_idx]))]["topic"]))

    return sorted(labels)


def build_3d_figure(
    meta: pd.DataFrame, graph: nx.Graph, scaffold: nx.Graph, coords: np.ndarray
):
    fig, ax = uplt.subplots(
        projection="3d",
        width="16cm",
        height="11.2cm",
        left="0.2em",
        right="0.2em",
        bottom="0.2em",
        top="0.2em",
    )

    pos = {topic: coords[idx] for idx, topic in enumerate(meta["topic"])}

    all_edges = sorted(
        graph.edges(data=True),
        key=lambda item: float(item[2].get("weight", 0.0)),
    )
    for u, v, data in all_edges:
        p0 = pos[u]
        p1 = pos[v]
        weight = float(data.get("weight", 0.0))
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#7f8b95",
            alpha=0.025 + 0.18 * weight,
            lw=0.08 + 10.4 * weight,
            solid_capstyle="round",
            zorder=1,
        )

    weighted_degree = np.array(
        [
            float(
                sum(d.get("weight", 0.0) for _, _, d in graph.edges(topic, data=True))
            )
            for topic in meta["topic"]
        ],
        dtype=float,
    )
    sizes = 10.0 + 85.0 * weighted_degree / max(float(weighted_degree.max()), 1e-9)
    colors = [
        (
            REGION_COLORS.get(int(region), FALLBACK_NODE_COLOR)
            if np.isfinite(region)
            else FALLBACK_NODE_COLOR
        )
        for region in meta["region_id"].to_numpy(dtype=float)
    ]

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=0.35,
        depthshade=False,
        zorder=3,
    )

    labels = choose_labels(meta, graph, coords)
    for topic in labels:
        idx = int(meta.index[meta["topic"] == topic][0])
        x, y, z = coords[idx]
        ax.text(
            float(x),
            float(y),
            float(z) + 0.03,
            str(topic).replace("_", " "),
            fontsize=6.2,
            ha="center",
            va="bottom",
            zorder=5,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.axis(False)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    try:
        ax.dist = 7.0
    except Exception:
        pass

    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    z_span = float(np.ptp(coords[:, 2]))
    ax.set_box_aspect((max(x_span, 1e-3), max(y_span, 1e-3), max(1.35 * z_span, 1e-3)))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))

    return fig, ax


def save_static_outputs(fig):
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=-0.02)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=-0.02)


def save_rotation_mp4(fig, ax):
    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    azims = np.linspace(VIEW_AZIM, VIEW_AZIM + 360.0, ROTATE_FRAMES, endpoint=False)

    def update(frame_idx):
        ax.view_init(elev=VIEW_ELEV, azim=float(azims[frame_idx]))
        return ()

    anim = FuncAnimation(
        fig,
        update,
        frames=len(azims),
        interval=1000 / ROTATE_FPS,
        blit=False,
    )
    writer = FFMpegWriter(
        fps=ROTATE_FPS,
        codec="libx264",
        bitrate=2400,
        metadata={"title": "ATS space of concerns 3D rotation"},
    )
    anim.save(OUT_MP4, writer=writer, dpi=200)


def main():
    phi, graph, scaffold, dist = build_graphs()
    topics = list(phi.index)
    coords_raw, eigvals = classical_mds(dist, n_components=3)
    meta = load_topic_metadata(topics)
    coords_raw = align_first_axis(coords_raw, meta)
    coords = scale_coords(coords_raw)

    weighted_degree = [
        float(sum(d.get("weight", 0.0) for _, _, d in graph.edges(topic, data=True)))
        for topic in topics
    ]
    meta = meta.assign(
        mds_x=coords[:, 0],
        mds_y=coords[:, 1],
        mds_z=coords[:, 2],
        weighted_degree=weighted_degree,
    )
    OUT_COORDS.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUT_COORDS, index=False)

    positive = eigvals[eigvals > 0]
    explained = (
        (np.clip(eigvals[:3], 0.0, None) / positive.sum()).tolist()
        if positive.size
        else []
    )
    OUT_META.write_text(
        json.dumps(
            {
                "distance_definition": "all-pairs_shortest_path_neg_log_phi",
                "embedding": "classical_mds_3d",
                "rendered_edges": "full_graph_all_edges",
                "edge_width_mapping": "0.08 + 2.4 * phi_weight",
                "static_outputs": [str(OUT_PNG), str(OUT_PDF)],
                "rotation_output": str(OUT_MP4),
                "rotation_fps": int(ROTATE_FPS),
                "rotation_seconds": int(ROTATE_SECONDS),
                "n_topics": int(len(topics)),
                "n_edges_full": int(graph.number_of_edges()),
                "n_edges_scaffold": int(scaffold.number_of_edges()),
                "leading_eigenvalues": [float(x) for x in eigvals[:8]],
                "explained_share_first_3_positive_axes": [float(x) for x in explained],
            },
            indent=2,
        )
    )

    fig, ax = build_3d_figure(meta, graph, scaffold, coords)
    save_static_outputs(fig)
    save_rotation_mp4(fig, ax)
    uplt.close(fig)


if __name__ == "__main__":
    main()
