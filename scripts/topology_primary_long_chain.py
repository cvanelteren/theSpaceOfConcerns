#!/usr/bin/env python3
"""Run the final primary topology null with a 20x Curveball chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import build_graphs  # noqa: E402
from topology_fixed_margin_null import (  # noqa: E402
    OUT_DIR,
    exact_margin_draws,
    incidence_to_proximity,
    load_displayed_regions,
    network_metrics,
    summarize_specification,
)


SEED = 20260820
N_DRAWS = 1000
TRADE_MULTIPLIER = 20
LOUVAIN_RESTARTS = 4


def main() -> None:
    _, _, _, counts, rpa = build_graphs()
    counts = counts.sort_index().sort_index(axis=1)
    rpa = rpa.reindex(index=counts.index, columns=counts.columns)
    active = rpa.ge(1.0)
    incidence = active.T.to_numpy(dtype=np.int8)
    labels = load_displayed_regions(counts.index.tolist())
    observed = network_metrics(incidence, labels, LOUVAIN_RESTARTS)
    draws, edge_draws, sampler = exact_margin_draws(
        incidence,
        labels,
        "all_actors_20x",
        N_DRAWS,
        TRADE_MULTIPLIER,
        LOUVAIN_RESTARTS,
        SEED,
    )
    specification = summarize_specification(
        "all_actors_20x", counts, active, observed, draws, sampler
    )

    observed_phi, _ = incidence_to_proximity(incidence)
    upper = np.triu_indices_from(observed_phi, k=1)
    observed_edges = observed_phi[upper]
    null_mean = edge_draws.mean(axis=0)
    null_sd = edge_draws.std(axis=0, ddof=1)
    null_low = np.quantile(edge_draws, 0.025, axis=0)
    null_high = np.quantile(edge_draws, 0.975, axis=0)
    edge_table = pd.DataFrame(
        {
            "concern_a": counts.index.to_numpy()[upper[0]],
            "concern_b": counts.index.to_numpy()[upper[1]],
            "observed_proximity": observed_edges,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "null_q025": null_low,
            "null_q975": null_high,
            "z_vs_null": np.divide(
                observed_edges - null_mean,
                null_sd,
                out=np.zeros_like(observed_edges),
                where=null_sd > 0,
            ),
            "upper_tail_p": (
                1 + (edge_draws >= observed_edges[None, :]).sum(axis=0)
            )
            / (N_DRAWS + 1),
            "lower_tail_p": (
                1 + (edge_draws <= observed_edges[None, :]).sum(axis=0)
            )
            / (N_DRAWS + 1),
        }
    )
    edge_summary = {
        "spearman_observed_vs_null_mean": float(
            spearmanr(observed_edges, null_mean).statistic
        ),
        "rmse_observed_vs_null_mean": float(
            np.sqrt(np.mean((observed_edges - null_mean) ** 2))
        ),
        "pairs_above_null_95_interval": int((observed_edges > null_high).sum()),
        "pairs_below_null_95_interval": int((observed_edges < null_low).sum()),
        "pairs_inside_null_95_interval": int(
            ((observed_edges >= null_low) & (observed_edges <= null_high)).sum()
        ),
    }
    metrics = specification["metrics"]
    interpretation = (
        "'Connected, weakly modular, and locally structured' is claim-safe. Every "
        "fixed-margin null network is connected, so connectivity is descriptive rather "
        "than evidence against random mixing. Observed modularity is low in absolute terms "
        "but exceeds the optimized modularity of every null draw. Avoid 'non-modular', "
        "which would deny the detected local organization. The displayed-region comparison "
        "is supporting evidence because its labels were selected on the observed network."
    )
    payload = {
        "design": {
            "seed": SEED,
            "draws": N_DRAWS,
            "trade_multiplier": TRADE_MULTIPLIER,
            "louvain_restarts": LOUVAIN_RESTARTS,
            "null": (
                "Curveball randomization preserving every actor specialization breadth "
                "and concern holder count exactly"
            ),
        },
        "specification": specification,
        "primary_edge_level_null": edge_summary,
        "claim_safe_interpretation": interpretation,
        "headline": {
            "positive_edges": metrics["edges_support_ge_1"],
            "louvain_modularity": metrics["louvain_modularity"],
            "displayed_region_modularity": metrics["displayed_region_modularity"],
            "displayed_region_mean_difference": metrics[
                "displayed_region_mean_difference"
            ],
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUT_DIR / "topology_primary_20x_draws.csv", index=False)
    edge_table.to_csv(OUT_DIR / "topology_primary_20x_edges.csv", index=False)
    (OUT_DIR / "topology_primary_20x_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["headline"], indent=2))


if __name__ == "__main__":
    main()
