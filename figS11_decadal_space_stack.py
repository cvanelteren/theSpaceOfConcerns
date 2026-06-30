"""Appendix figure: fixed-layout concern space stacked by decade."""

from pathlib import Path

import networkx as nx
import numpy as np
import ultraplot as uplt
from PIL import Image

from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    load_saved_layout_positions,
    standardize_index_labels,
)

OUT_PDF = Path("./figures/figS11_decadal_space_stack.pdf")
OUT_PNG = Path("./figures/figS11_decadal_space_stack.png")
Z_SPACING = 2
Z_VISUAL_SCALE = 2.0
SAMPAIO_BLOCKS = [
    (1961, 1980, "1961–1980"),
    (1981, 2000, "1981–2000"),
    (2001, 2025, "2001–2025"),
]
LAYER_COLORS = ["#2F6C74", "#8FAF7B", "#D49A6A"]


def load_data_with_fallback():
    paths = [
        Path("antarctic-database-go/data/processed/document-summary.parquet"),
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


def collapse_standardized_topics(df):
    df = standardize_index_labels(df)
    if df.index.has_duplicates:
        df = df.groupby(level=0).sum()
    return df


def build_full_scaffold(counts_df):
    rca = get_rca(counts_df)
    phi = compute_product_space(rca)
    graph = nx.from_pandas_adjacency(phi)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    for _, _, data in graph.edges(data=True):
        data["weight"] = float(data.get("weight", 0.0))
    mst = nx.maximum_spanning_tree(graph, weight="weight")
    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    if weights.size:
        cutoff = float(np.percentile(weights, 95))
        for u, v, data in graph.edges(data=True):
            if float(data["weight"]) >= cutoff:
                mst.add_edge(u, v, **data)
    return mst, graph


def tessellate_mask(png_path, stride=6):
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    premult = arr.copy()
    alpha_frac = premult[:, :, 3:4].astype(np.float32) / 255.0
    premult[:, :, :3] = np.clip(
        premult[:, :, :3].astype(np.float32) * alpha_frac, 0.0, 255.0
    ).astype(np.uint8)
    img = Image.fromarray(premult, mode="RGBA")
    arr = premult
    alpha = arr[:, :, 3]
    mask = alpha > 10

    m = mask.astype(np.uint8)
    up = np.pad(m[1:, :], ((0, 1), (0, 0)), mode="constant")
    down = np.pad(m[:-1, :], ((1, 0), (0, 0)), mode="constant")
    left = np.pad(m[:, 1:], ((0, 0), (0, 1)), mode="constant")
    right = np.pad(m[:, :-1], ((0, 0), (1, 0)), mode="constant")
    inner = (m == 1) & (up == 1) & (down == 1) & (left == 1) & (right == 1)

    ys, xs = np.where(inner)
    coords = np.vstack([xs, ys]).T
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
    return {n: tess_points[i] for n, i in zip(nodes, idx)}


def deterministic_layout(g):
    init = nx.circular_layout(g, scale=0.5)
    return nx.kamada_kawai_layout(g, pos=init, weight="weight")


def build_fig1_layout(scaffold, full_topics):
    saved_pos = load_saved_layout_positions(scaffold)
    saved_pos = None
    pos = saved_pos if saved_pos is not None else deterministic_layout(scaffold)

    fp = Path("1024px-AntarcticaContour.svg.png")
    if fp.exists():
        tess_points, _mask_img, mask_extent = tessellate_mask(fp)
        snapped = snap_to_tessellation(pos, tess_points)
        x_min, x_max, y_min, y_max = mask_extent
        snapped = {
            n: np.array([x_min + x_max - p[0], y_min + y_max - p[1]])
            for n, p in snapped.items()
        }
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
        scale = 4.0
        snapped = {
            n: np.array([cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale])
            for n, p in snapped.items()
        }
        return {t: np.asarray(snapped[t], dtype=float) for t in full_topics}

    fallback = {n: np.array(p, dtype=float) * 4.0 for n, p in pos.items()}
    return {t: np.asarray(fallback[t], dtype=float) for t in full_topics}


def sampaio_windows(min_year, max_year):
    lower = int(min_year)
    upper = int(max_year)
    windows = []
    for start, end, label in SAMPAIO_BLOCKS:
        if end < lower or start > upper:
            continue
        windows.append((max(start, lower), min(end, upper), label))
    return windows


def build_block(interaction, full_topics):
    interaction = collapse_standardized_topics(interaction)
    interaction = interaction.reindex(full_topics, fill_value=0)
    rca = get_rca(interaction)
    phi = compute_product_space(rca).reindex(index=full_topics, columns=full_topics)
    phi = phi.fillna(0.0)
    excess = np.maximum(rca - 1.0, 0.0).sum(axis=1).reindex(full_topics, fill_value=0.0)
    return phi, excess


def main():
    counts_df, submitted, countries, topics = load_data_with_fallback()
    counts_df = filter_space_topics(counts_df)
    counts_df = collapse_standardized_topics(counts_df)
    full_topics = list(counts_df.index)

    scaffold, _ = build_full_scaffold(counts_df)
    pos = build_fig1_layout(scaffold, full_topics)
    coords = np.array([np.asarray(pos[t], dtype=float) for t in full_topics])

    windows = sampaio_windows(
        submitted["meeting year"].min(),
        submitted["meeting year"].max(),
    )
    blocks = []
    for start, end, label in windows:
        subset = submitted[
            (submitted["meeting year"] >= start) & (submitted["meeting year"] <= end)
        ]
        if subset.empty:
            continue
        interaction = generate_interaction_matrix(subset, countries, set(full_topics))
        phi, excess = build_block(interaction, full_topics)
        blocks.append(
            {
                "label": label,
                "phi": phi,
                "excess": excess,
            }
        )
    if not blocks:
        raise ValueError("No decadal blocks available for plotting.")

    colors = np.array([uplt.to_rgba(c) for c in LAYER_COLORS[: len(blocks)]])

    fig, ax = uplt.subplots(
        projection="3d",
        height="10.5cm",
        width="6.8cm",
        left="0.00em",
        right="0.00em",
        bottom="0.00em",
        top="0.00em",
    )

    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    pad_x = 0.01 * max(np.ptp(x_vals), 1.0)
    pad_y = 0.01 * max(np.ptp(y_vals), 1.0)

    for layer_idx, (block, color) in enumerate(zip(blocks, colors)):
        z = float(layer_idx) * Z_SPACING
        phi = block["phi"]
        excess = block["excess"]
        for u, v in scaffold.edges():
            w = float(phi.loc[u, v])
            if w <= 0:
                continue
            p0 = pos[u]
            p1 = pos[v]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [z, z],
                color="k",
                alpha=0.18 + 0.55 * w,
                lw=0.15 + 1.6 * w,
                solid_capstyle="round",
                zorder=1,
            )

        face = np.tile(color, (len(full_topics), 1))
        face[:, 3] = 0.9
        ax.scatter(
            x_vals,
            y_vals,
            np.full(len(full_topics), z),
            s=4.0,
            c=face,
            edgecolors="k",
            linewidths=0.18,
            depthshade=False,
            zorder=3,
        )
    ax.set_xlim(float(x_vals.min() - pad_x), float(x_vals.max() + pad_x))
    ax.set_ylim(float(y_vals.min() - pad_y), float(y_vals.max() + pad_y))
    z_levels = [float(i) * Z_SPACING for i in range(len(blocks))]
    ax.set_zlim(-0.2, z_levels[-1] + 0.8)
    # Numeric z spacing alone is visually compressed by the 3D box aspect.
    # Scale the z display dimension so the stacked layers actually separate.
    x_span = float(np.ptp(x_vals) + 2 * pad_x)
    y_span = float(np.ptp(y_vals) + 2 * pad_y)
    z_span = float((z_levels[-1] + 0.8) * Z_VISUAL_SCALE)
    ax.set_box_aspect((x_span, y_span, z_span))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.view_init(elev=20, azim=-56)
    try:
        ax.dist = 7.5
    except Exception:
        pass
    ax.grid(False)
    ax.axis(False)
    ax.margins(-0.45)

    # Label each layer inside the 3D scene so the stack stays centered and the
    # labels read as annotations of the layers rather than external legend text.
    label_x = float(x_vals.mean())
    label_y = float(y_vals.max() - pad_y * 0.35)
    for z, block, color in zip(z_levels, blocks, colors):
        ax.text(
            label_x,
            label_y,
            float(z) + 0.38,
            block["label"],
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=5,
        )

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    padding = -0.2
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=padding)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=padding)
    uplt.close(fig)


if __name__ == "__main__":
    main()
