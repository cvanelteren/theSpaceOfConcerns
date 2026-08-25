#!/usr/bin/env python3
"""Check whether topology-null results depend on Curveball chain length."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import build_graphs  # noqa: E402
from topology_fixed_margin_null import (  # noqa: E402
    OUT_DIR,
    exact_margin_draws,
    load_displayed_regions,
)


SEED = 20260821
N_DRAWS = 200
TRADE_MULTIPLIERS = (5, 20, 50)
METRICS = (
    "edges_support_ge_1",
    "edges_support_ge_3",
    "median_proximity",
    "louvain_modularity",
    "displayed_region_modularity",
    "displayed_region_mean_difference",
)


def main() -> None:
    _, _, _, counts, rpa = build_graphs()
    counts = counts.sort_index().sort_index(axis=1)
    incidence = (
        rpa.reindex(index=counts.index, columns=counts.columns)
        .ge(1.0)
        .T.to_numpy(dtype=np.int8)
    )
    labels = load_displayed_regions(counts.index.tolist())
    frames = []
    samplers = {}
    for multiplier in TRADE_MULTIPLIERS:
        draws, _, sampler = exact_margin_draws(
            incidence,
            labels,
            f"trade_multiplier_{multiplier}",
            N_DRAWS,
            multiplier,
            4,
            SEED + 10_000 * multiplier,
        )
        draws["trade_multiplier"] = multiplier
        frames.append(draws)
        samplers[str(multiplier)] = sampler

    combined = pd.concat(frames, ignore_index=True)
    reference = combined.loc[combined["trade_multiplier"].eq(max(TRADE_MULTIPLIERS))]
    rows = []
    for multiplier in TRADE_MULTIPLIERS:
        candidate = combined.loc[combined["trade_multiplier"].eq(multiplier)]
        for metric in METRICS:
            left = candidate[metric].to_numpy(dtype=float)
            right = reference[metric].to_numpy(dtype=float)
            pooled_sd = float(np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2))
            rows.append(
                {
                    "trade_multiplier": multiplier,
                    "reference_trade_multiplier": max(TRADE_MULTIPLIERS),
                    "metric": metric,
                    "mean": float(left.mean()),
                    "reference_mean": float(right.mean()),
                    "standardized_mean_difference": float(
                        (left.mean() - right.mean()) / pooled_sd
                    )
                    if pooled_sd > 0
                    else 0.0,
                    "ks_statistic": float(ks_2samp(left, right).statistic),
                    "ks_p": float(ks_2samp(left, right).pvalue),
                }
            )
    comparison = pd.DataFrame(rows)
    production = comparison.loc[comparison["trade_multiplier"].eq(20)]
    shortest = comparison.loc[comparison["trade_multiplier"].eq(5)]
    summary = {
        "seed": SEED,
        "draws_per_trade_multiplier": N_DRAWS,
        "trade_multipliers": list(TRADE_MULTIPLIERS),
        "samplers": samplers,
        "production_trade_multiplier": 20,
        "maximum_absolute_standardized_mean_difference_20x_vs_50x": float(
            production["standardized_mean_difference"].abs().max()
        ),
        "minimum_ks_p_20x_vs_50x": float(production["ks_p"].min()),
        "five_x_edge_count_standardized_mean_difference_vs_50x": float(
            shortest.loc[
                shortest["metric"].eq("edges_support_ge_1"),
                "standardized_mean_difference",
            ].iloc[0]
        ),
        "five_x_edge_count_ks_p_vs_50x": float(
            shortest.loc[
                shortest["metric"].eq("edges_support_ge_1"), "ks_p"
            ].iloc[0]
        ),
        "interpretation": (
            "The 20x production chain is adequate when its null metric distributions "
            "remain close to the 50x chain. The 5x comparison identifies whether a shorter "
            "chain changes any target. Exact margins hold in every draw regardless of chain "
            "length."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_DIR / "topology_curveball_mixing_draws.csv", index=False)
    comparison.to_csv(OUT_DIR / "topology_curveball_mixing_comparison.csv", index=False)
    (OUT_DIR / "topology_curveball_mixing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
