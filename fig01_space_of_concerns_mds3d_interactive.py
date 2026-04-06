"""Interactive 3D MDS viewer for the ATS space of concerns.

Run inside the plotting environment to open a rotatable Matplotlib figure:

  /opt/homebrew/Cellar/micromamba/2.4.0/envs/ultraplot-dev/bin/python \
      fig01_space_of_concerns_mds3d_interactive.py

Use the mouse to rotate/zoom the scene. The viewer uses the same 3D MDS
embedding and weighted full-edge network as the static exporter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig01_space_of_concerns_mds3d import (
    FALLBACK_NODE_COLOR,
    OUT_COORDS,
    REGION_COLORS,
    align_first_axis,
    build_graphs,
    classical_mds,
    choose_labels,
    load_topic_metadata,
    scale_coords,
)

DEFAULT_SNAPSHOT = Path("figures/fig01_space_of_concerns_mds3d_interactive_snapshot.png")


def build_embedding():
    phi, graph, scaffold, dist = build_graphs()
    topics = list(phi.index)
    coords_raw, _ = classical_mds(dist, n_components=3)
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
    return meta, graph, scaffold, coords


def make_figure(meta, graph, coords, *, label_mode: str = "selected"):
    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    try:
        fig.canvas.manager.set_window_title("ATS space of concerns: 3D MDS")
    except Exception:
        pass

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
            lw=0.08 + 2.4 * weight,
            solid_capstyle="round",
            zorder=1,
        )

    weighted_degree = meta["weighted_degree"].to_numpy(dtype=float)
    sizes = 35.0 + 240.0 * weighted_degree / max(float(weighted_degree.max()), 1e-9)
    colors = [
        REGION_COLORS.get(int(region), FALLBACK_NODE_COLOR)
        if np.isfinite(region)
        else FALLBACK_NODE_COLOR
        for region in meta["region_id"].to_numpy(dtype=float)
    ]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=0.55,
        depthshade=False,
        zorder=3,
    )

    if label_mode == "all":
        labels = meta["topic"].tolist()
    elif label_mode == "none":
        labels = []
    else:
        labels = choose_labels(meta, graph, coords)

    for topic in labels:
        idx = int(meta.index[meta["topic"] == topic][0])
        x, y, z = coords[idx]
        ax.text(
            float(x),
            float(y),
            float(z) + 0.035,
            str(topic).replace("_", " "),
            fontsize=7.2,
            ha="center",
            va="bottom",
            zorder=5,
        )

    ax.view_init(elev=19, azim=-58)
    try:
        ax.dist = 8.0
    except Exception:
        pass

    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    z_span = float(np.ptp(coords[:, 2]))
    ax.set_box_aspect(
        (max(x_span, 1e-3), max(y_span, 1e-3), max(1.35 * z_span, 1e-3))
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-mode",
        choices=("selected", "all", "none"),
        default="selected",
        help="Which topic labels to show.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional snapshot path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figure and optionally save it, but do not open the GUI window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    meta, graph, _, coords = build_embedding()
    fig, _ = make_figure(meta, graph, coords, label_mode=args.label_mode)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Wrote {args.save}")
    if args.no_show:
        plt.close(fig)
        return
    plt.show()


if __name__ == "__main__":
    main()
