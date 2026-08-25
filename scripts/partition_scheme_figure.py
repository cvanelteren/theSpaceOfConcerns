#!/usr/bin/env python
"""Review figure: the three partitioning schemes drawn on one common layout.

Each panel colours the topics of the space of concerns by the community
assignment one scheme produces on the observed data, using the same
(SGD)^2 backbone layout as Figure 1 so the clusterings are directly
comparable by eye:

  A  mds_intervals    intervals on the 1-D MDS axis (the paper's modes)
  B  modularity_full  greedy modularity on the full phi graph
  C  modularity_bb    greedy modularity on the Figure 1 backbone

Community labels are nominal, so a qualitative palette is used and the
colour order carries no meaning; what matters is which topics move together.
Group counts and bootstrap ARI come from output/partition_scheme_comparison.json.

Usage::

    micromamba run -n ultraplot-dev python scripts/partition_scheme_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import ultraplot as uplt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fig01_space_of_concerns_topology as f1  # noqa: E402
from mode_partition_bootstrap import load_pipeline  # noqa: E402
from partition_scheme_comparison import partitions  # noqa: E402

OUT_PDF = ROOT / "figures" / "figS_partition_schemes.pdf"
OUT_PNG = ROOT / "figures" / "figS_partition_schemes.png"

SCHEME_TITLES = {
    "mds_intervals": "1-D MDS intervals",
    "modularity_full": "Modularity, full graph",
    "modularity_bb": "Modularity, backbone",
    "modularity_full_dist": "Modularity, full, dist-wt",
    "modularity_bb_dist": "Modularity, backbone, dist-wt",
}

PALETTE = [
    "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#F0E442",
    "#56B4E9", "#994F00", "#40B0A6", "#E1BE6A", "#7B3294",
]


def main() -> int:
    summary = json.loads(
        (ROOT / "output" / "partition_scheme_comparison.json").read_text()
    )
    ns = load_pipeline()
    counts = ns["_prepare_counts"]()
    obs = partitions(counts, ns)
    names = obs["names"]

    backbone, mst, g, _, _ = f1.build_graphs()
    _, mode_of, _ = f1.load_topic_meta()
    pos, _, _ = f1.fitted_layout(mst, g, mode_of)

    fig, axs = uplt.subplots(ncols=5, refwidth=1.75, refaspect=1, share=0)
    for ax, scheme in zip(axs, SCHEME_TITLES):
        labels = obs[scheme]
        by_name = dict(zip(names, labels))
        f1.draw_base(ax, backbone, pos, edge_alpha=0.30, edge_lw=0.5)
        nodes = [n for n in backbone.nodes()]
        xy = np.array([pos[n] for n in nodes], dtype=float)
        cols = [PALETTE[int(by_name.get(n, 0)) % len(PALETTE)] for n in nodes]
        ax.scatter(
            xy[:, 0], xy[:, 1], s=70, c=cols, alpha=0.95,
            edgecolor="white", linewidth=0.8, zorder=3, absolute_size=True,
        )
        meta = summary["schemes"][scheme]
        ax.format(
            title=f"{SCHEME_TITLES[scheme]}\nk={meta['n_groups_observed']}, "
                  f"ARI {meta['ari_mean']:.2f}",
            titlesize=9.5, titleweight="bold",
            xticks=[], yticks=[], grid=False,
        )
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_visible(False)
    fig.format(abc="a", abcloc="ul", abcsize=11)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Wrote {OUT_PDF.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
