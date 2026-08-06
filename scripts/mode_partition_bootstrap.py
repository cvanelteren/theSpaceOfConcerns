#!/usr/bin/env python
"""How stable is the three-mode partition under resampling the actors?

The manuscript describes the modes as "an interpretive division of a continuum
rather than a discovery of three groups", which is true of how $k=3$ is chosen:
the ordering axis has no eigengap and $k$ comes from an elbow on a smooth
inertia curve. But that statement is about *how many* zones there are. It says
nothing about whether the assignment of topics to zones is reproducible, and
those are separate questions. Conflating them is what left the paper stating
the same caveat five times while three of four figures were built on the
partition.

This answers the second question. Actors are resampled with replacement, the
whole pipeline is rebuilt from the resampled panel -- RPA, proximity, the
shortest-path distance matrix, the 1D classical-MDS ordering, and the weighted
k-means zones -- and the resulting partition is compared with the observed one.

Two statistics are reported:

  * **Adjusted Rand index** between each bootstrap partition and the observed
    one, which is invariant to how the zones happen to be numbered.
  * **Per-topic agreement**, the fraction of resamples in which a topic lands
    in the zone it occupies in the full data, after orienting each bootstrap
    axis to the observed one (classical MDS fixes an axis only up to sign).

Definitions are taken from old_scripts/fig45_portfolio_space_ridgelines.py by
executing its preamble, so the pipeline cannot drift away from the one that
produced the figures.

Usage::

    micromamba run -n ultraplot-dev python scripts/mode_partition_bootstrap.py [n_boot]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "old_scripts" / "fig45_portfolio_space_ridgelines.py"
OUT_JSON = ROOT / "output" / "mode_partition_bootstrap.json"
OUT_CSV = ROOT / "output" / "mode_partition_bootstrap_topics.csv"

SEED = 20260806
DEFAULT_N_BOOT = 200


def load_pipeline() -> dict:
    """Execute only the definitions of the ridgeline script, not its analysis."""
    src = SOURCE.read_text()
    marker = "counts = _prepare_counts()"
    cut = src.index(marker)
    # Trim back to the start of the cell that begins the analysis.
    preamble = src[:cut].rstrip()
    if preamble.endswith("# %%"):
        preamble = preamble[: -len("# %%")]
    ns: dict = {"__name__": "_fig45_defs", "__file__": str(SOURCE)}
    exec(compile(preamble, str(SOURCE), "exec"), ns)  # noqa: S102
    return ns


def partition(counts, ns) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Topic names, their position on the concern axis, and their zone index."""
    rca = ns["get_rca"](counts)
    phi = ns["compute_product_space"](rca)
    topic_names, distance = ns["_build_distance_matrix"](phi)
    # Resampling actors can isolate a topic, and floyd_warshall then returns inf
    # for its distances. Squaring those gives inf and the double-centering
    # matmul in classical MDS returns NaN, so the replicate would contribute a
    # meaningless ordering and inflate apparent instability. Such replicates are
    # rejected rather than repaired: a disconnected space is not a noisier
    # estimate of the same object.
    if not np.all(np.isfinite(distance)):
        raise ValueError("disconnected proximity graph in this replicate")
    coord = ns["_mds_1d"](distance)
    order = np.argsort(coord)

    ordered_coord = coord[order]
    lo, hi = float(ordered_coord.min()), float(ordered_coord.max())
    x_plot = (ordered_coord - lo) / (hi - lo) if hi > lo else np.zeros_like(ordered_coord)

    rca = rca.reindex(index=topic_names)
    signal = np.clip(rca.to_numpy(dtype=float) - ns["RCA_THRESHOLD"], 0.0, None)
    aggregate = signal.sum(axis=1)[order]

    peak_idx, _ = ns["_select_region_centers"](x_plot, aggregate)
    peak_x = x_plot[peak_idx]
    bounds = 0.5 * (peak_x[:-1] + peak_x[1:]) if len(peak_x) > 1 else np.array([])
    zone_ordered = np.digitize(x_plot, bounds).astype(int)

    # Return in topic_names order rather than axis order.
    zones = np.empty(len(topic_names), dtype=int)
    positions = np.empty(len(topic_names), dtype=float)
    zones[order] = zone_ordered
    positions[order] = x_plot
    return topic_names, positions, zones


def main() -> int:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_BOOT
    ns = load_pipeline()
    counts = ns["_prepare_counts"]()
    topics, obs_pos, obs_zone = partition(counts, ns)
    n_topics, n_actors = counts.shape
    print(f"topics: {n_topics}   actors: {n_actors}   bootstrap: {n_boot}")

    rng = np.random.default_rng(SEED)
    aris: list[float] = []
    agree = np.zeros(n_topics, dtype=float)
    used = 0
    rejected = 0

    for b in range(n_boot):
        pick = rng.integers(0, n_actors, size=n_actors)
        resampled = counts.iloc[:, pick]
        resampled.columns = [f"{c}__{k}" for k, c in enumerate(resampled.columns)]
        try:
            names_b, pos_b, zone_b = partition(resampled, ns)
        except Exception:
            rejected += 1
            continue
        if list(names_b) != list(topics):
            index = {t: i for i, t in enumerate(names_b)}
            if not set(topics) <= set(index):
                continue
            sel = [index[t] for t in topics]
            pos_b, zone_b = pos_b[sel], zone_b[sel]

        aris.append(adjusted_rand_score(obs_zone, zone_b))
        # Classical MDS fixes the axis only up to sign; orient before comparing
        # zone identities, which are numbered left to right along it.
        if stats.pearsonr(obs_pos, pos_b).statistic < 0:
            zone_b = zone_b.max() - zone_b
        agree += (zone_b == obs_zone).astype(float)
        used += 1

    if not used:
        print("no successful bootstrap replicates")
        return 1

    agree /= used
    aris_arr = np.asarray(aris)
    summary = {
        "n_topics": int(n_topics), "n_actors": int(n_actors),
        "n_boot_requested": n_boot, "n_boot_used": used,
        "n_boot_rejected_disconnected": rejected, "seed": SEED,
        "ari_mean": float(aris_arr.mean()), "ari_median": float(np.median(aris_arr)),
        "ari_q05": float(np.quantile(aris_arr, 0.05)),
        "ari_q95": float(np.quantile(aris_arr, 0.95)),
        "topic_agreement_mean": float(agree.mean()),
        "topic_agreement_min": float(agree.min()),
        "share_topics_above_90": float((agree >= 0.90).mean()),
        "share_topics_above_75": float((agree >= 0.75).mean()),
    }
    print(f"\nAdjusted Rand index vs observed partition:")
    print(f"  mean {summary['ari_mean']:.3f}  median {summary['ari_median']:.3f}"
          f"  90% interval [{summary['ari_q05']:.3f}, {summary['ari_q95']:.3f}]")
    print(f"\nPer-topic agreement with observed zone:")
    print(f"  mean {summary['topic_agreement_mean']:.3f}   min {summary['topic_agreement_min']:.3f}")
    print(f"  {summary['share_topics_above_90']:.1%} of topics agree in >=90% of resamples")
    print(f"  {summary['share_topics_above_75']:.1%} agree in >=75%")

    order = np.argsort(agree)
    print("\nleast stable topics:")
    for i in order[:8]:
        print(f"  {agree[i]:.2f}  zone {obs_zone[i] + 1}  x={obs_pos[i]:.2f}  {topics[i]}")

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    import pandas as pd
    pd.DataFrame({
        "topic": topics, "observed_zone": obs_zone + 1,
        "position": obs_pos, "bootstrap_agreement": agree,
    }).sort_values("position").to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}, {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
