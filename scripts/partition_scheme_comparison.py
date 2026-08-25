#!/usr/bin/env python
"""Which way of partitioning the space of concerns is most reproducible?

The three engagement modes are currently intervals on a one-dimensional
classical-MDS ordering. That ordering is a device for reading the network, and
a good one, but cutting it into blocks discards the 76% of the eigenvalue mass
the axis does not carry and then imposes boundaries on what is left. Bootstrap
agreement for that scheme is an adjusted Rand index of 0.24
(scripts/mode_partition_bootstrap.py).

An alternative is to partition the network directly, using the same filtered
backbone that Figure 1 draws. A topic deep inside a branch should stay there
even when the 1D projection wobbles, so communities ought to be more stable
than intervals. This tests that rather than assuming it, comparing three
schemes under one bootstrap and one yardstick:

  mds_intervals    weighted k-means zones on the 1D MDS axis (current)
  modularity_full  greedy modularity communities on the full phi graph
  modularity_bb    the same on the Figure 1 backbone (maximum spanning tree
                   plus the strongest additional edges)

For each scheme the observed partition is computed on the full data, then the
actors are resampled with replacement and the whole pipeline rebuilt. Agreement
is the adjusted Rand index, which is invariant to how communities are labelled
and to how many are found -- important here, since modularity chooses its own
number of communities and may not choose three.

A scheme that scores near zero is not reproducible. If no scheme scores well,
the honest reading is that the space does not partition, which would be
consistent with the network diagnostics reported in the main text: 37 of 990
pairs at zero proximity, low clustering, and no single structural anchor.

Usage::

    micromamba run -n ultraplot-dev python scripts/partition_scheme_comparison.py [n_boot]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mode_partition_bootstrap import load_pipeline  # noqa: E402

OUT_JSON = ROOT / "output" / "partition_scheme_comparison.json"

SEED = 20260806
DEFAULT_N_BOOT = 200
BACKBONE_EXTRA_QUANTILE = 0.95  # top 5% of edges, as in Figure 1


def _phi_graph(phi) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(phi.index)
    arr = phi.to_numpy(dtype=float)
    for i, ti in enumerate(phi.index):
        for j in range(i + 1, len(phi.index)):
            w = float(arr[i, j])
            if w > 0:
                g.add_edge(ti, phi.index[j], weight=w)
    return g


def _backbone(g: nx.Graph) -> nx.Graph:
    """Figure 1's display graph: maximum spanning tree plus the strongest edges."""
    bb = nx.maximum_spanning_tree(g, weight="weight")
    weights = [d["weight"] for _, _, d in g.edges(data=True)]
    if weights:
        cut = float(np.quantile(weights, BACKBONE_EXTRA_QUANTILE))
        for u, v, d in g.edges(data=True):
            if d["weight"] >= cut:
                bb.add_edge(u, v, weight=d["weight"])
    return bb


def _communities(g: nx.Graph, nodes: list[str]) -> np.ndarray:
    comms = nx.community.greedy_modularity_communities(g, weight="weight")
    label = {n: k for k, c in enumerate(comms) for n in c}
    return np.array([label.get(n, -1) for n in nodes], dtype=int)


def _dist_weighted(g: nx.Graph) -> nx.Graph:
    """Re-weight edges by 1 / (-log phi) so modularity runs on the same
    distance the Figure 1 layout uses, rather than on phi itself. Strong
    ties become disproportionately heavier than under phi weighting."""
    h = g.copy()
    for _, _, d in h.edges(data=True):
        w = min(max(float(d["weight"]), 1e-12), 1.0)
        d["weight"] = 1.0 / max(-np.log(w), 1e-3)
    return h


def partitions(counts, ns) -> dict[str, np.ndarray] | None:
    """All three schemes on one panel, indexed consistently by topic name."""
    rca = ns["get_rca"](counts)
    phi = ns["compute_product_space"](rca)
    names, distance = ns["_build_distance_matrix"](phi)
    if not np.all(np.isfinite(distance)):
        return None

    coord = ns["_mds_1d"](distance)
    order = np.argsort(coord)
    oc = coord[order]
    lo, hi = float(oc.min()), float(oc.max())
    x = (oc - lo) / (hi - lo) if hi > lo else np.zeros_like(oc)

    rca_r = rca.reindex(index=names)
    signal = np.clip(rca_r.to_numpy(dtype=float) - ns["RCA_THRESHOLD"], 0.0, None)
    peak_idx, _ = ns["_select_region_centers"](x, signal.sum(axis=1)[order])
    peak_x = x[peak_idx]
    bounds = 0.5 * (peak_x[:-1] + peak_x[1:]) if len(peak_x) > 1 else np.array([])
    zone_ordered = np.digitize(x, bounds).astype(int)
    mds = np.empty(len(names), dtype=int)
    mds[order] = zone_ordered

    phi_named = phi.reindex(index=names, columns=names)
    g = _phi_graph(phi_named)
    return {
        "names": names,
        "mds_intervals": mds,
        "modularity_full": _communities(g, names),
        "modularity_bb": _communities(_backbone(g), names),
        "modularity_full_dist": _communities(_dist_weighted(g), names),
        "modularity_bb_dist": _communities(_dist_weighted(_backbone(g)), names),
    }


def main() -> int:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_BOOT
    ns = load_pipeline()
    counts = ns["_prepare_counts"]()
    obs = partitions(counts, ns)
    if obs is None:
        print("observed space is disconnected")
        return 1
    names = obs["names"]
    schemes = [k for k in obs if k != "names"]
    n_actors = counts.shape[1]

    print(f"topics: {len(names)}   actors: {n_actors}   bootstrap: {n_boot}\n")
    print("observed number of groups:")
    for s in schemes:
        print(f"  {s:17s} {len(set(obs[s]))}")

    import collections
    import pandas as pd

    groups = collections.defaultdict(list)
    for t, lab in zip(names, obs["modularity_bb"]):
        groups[int(lab)].append(str(t))
    print(f"\nmodularity_bb topic membership (k={len(groups)}):")
    rows = []
    for k in sorted(groups):
        print(f"  community {k + 1}: {', '.join(sorted(groups[k]))}")
        rows.extend((k + 1, t) for t in sorted(groups[k]))
    csv_path = OUT_JSON.with_name("modularity_bb_k7_topics.csv")
    pd.DataFrame(rows, columns=["community", "topic"]).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path.relative_to(ROOT)}")

    rng = np.random.default_rng(SEED)
    aris: dict[str, list[float]] = {s: [] for s in schemes}
    rejected = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_actors, size=n_actors)
        rs = counts.iloc[:, pick]
        rs.columns = [f"{c}__{k}" for k, c in enumerate(rs.columns)]
        try:
            b = partitions(rs, ns)
        except Exception:
            b = None
        if b is None:
            rejected += 1
            continue
        idx = {t: i for i, t in enumerate(b["names"])}
        if not set(names) <= set(idx):
            rejected += 1
            continue
        sel = [idx[t] for t in names]
        for s in schemes:
            aris[s].append(adjusted_rand_score(obs[s], b[s][sel]))

    summary = {"n_topics": len(names), "n_actors": int(n_actors),
               "n_boot": n_boot, "n_rejected": rejected, "seed": SEED,
               "schemes": {}}
    print(f"\nbootstrap agreement (adjusted Rand index), {n_boot - rejected} replicates:")
    for s in schemes:
        a = np.asarray(aris[s])
        summary["schemes"][s] = {
            "n_groups_observed": int(len(set(obs[s]))),
            "ari_mean": float(a.mean()), "ari_median": float(np.median(a)),
            "ari_q05": float(np.quantile(a, 0.05)),
            "ari_q95": float(np.quantile(a, 0.95)),
        }
        print(f"  {s:17s} mean {a.mean():.3f}   median {np.median(a):.3f}"
              f"   90% [{np.quantile(a, 0.05):.3f}, {np.quantile(a, 0.95):.3f}]")

    best = max(schemes, key=lambda s: summary["schemes"][s]["ari_mean"])
    print(f"\nmost reproducible: {best} "
          f"(ARI {summary['schemes'][best]['ari_mean']:.3f})")
    summary["most_reproducible"] = best
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
