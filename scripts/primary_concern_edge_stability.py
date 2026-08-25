#!/usr/bin/env python3
"""Actor-bootstrap stability of edges in the inferred-primary concern map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_product_space, get_rca, load_data  # noqa: E402

DATA = ROOT / "data/document-summary-primary-concern.parquet"
OUT_CSV = ROOT / "output/primary_concern_edge_stability.csv"
OUT_JSON = ROOT / "output/primary_concern_edge_stability.json"
N_BOOTSTRAP = 1000
TOP_K = 5
SEED = 20260814


def main() -> None:
    counts, *_ = load_data(str(DATA))
    topics = list(counts.index)
    values = counts.to_numpy(dtype=float)
    n_topics, n_actors = values.shape
    first, second = np.triu_indices(n_topics, k=1)
    edge_index = {
        (int(i), int(j)): position
        for position, (i, j) in enumerate(zip(first, second))
    }
    weights = np.empty((N_BOOTSTRAP, len(first)), dtype=np.float32)
    selected = np.zeros(len(first), dtype=np.int32)
    rng = np.random.default_rng(SEED)

    for draw in range(N_BOOTSTRAP):
        sample = rng.integers(0, n_actors, size=n_actors)
        sampled = pd.DataFrame(
            values[:, sample],
            index=topics,
            columns=[f"actor_{slot}" for slot in range(n_actors)],
        )
        phi = compute_product_space(get_rca(sampled)).to_numpy(dtype=float)
        weights[draw] = phi[first, second]
        chosen: set[tuple[int, int]] = set()
        for i in range(n_topics):
            order = np.argsort(phi[i])[::-1]
            positive = [int(j) for j in order if j != i and phi[i, j] > 0]
            for j in positive[:TOP_K]:
                chosen.add(tuple(sorted((i, j))))
        for pair in chosen:
            selected[edge_index[pair]] += 1

    observed_phi = compute_product_space(get_rca(counts)).to_numpy(dtype=float)
    result = pd.DataFrame(
        {
            "concern_a": [topics[i] for i in first],
            "concern_b": [topics[j] for j in second],
            "phi_observed": observed_phi[first, second],
            "phi_median": np.median(weights, axis=0),
            "phi_ci_low": np.quantile(weights, 0.025, axis=0),
            "phi_ci_high": np.quantile(weights, 0.975, axis=0),
            "endpoint_top5_frequency": selected / N_BOOTSTRAP,
        }
    ).sort_values(
        ["endpoint_top5_frequency", "phi_observed"], ascending=False
    )
    result.to_csv(OUT_CSV, index=False)

    positive = result["phi_observed"].gt(0)
    summary = {
        "papers": 6573,
        "topics": n_topics,
        "actors": n_actors,
        "n_bootstrap": N_BOOTSTRAP,
        "resampling_unit": "submitting actor",
        "positive_observed_edges": int(positive.sum()),
        "positive_95pct_lower_bound_edges": int(result["phi_ci_low"].gt(0).sum()),
        "endpoint_top5_frequency_at_least_0_50": int(
            result["endpoint_top5_frequency"].ge(0.50).sum()
        ),
        "endpoint_top5_frequency_at_least_0_60_and_positive_lower_bound": int(
            (
                result["endpoint_top5_frequency"].ge(0.60)
                & result["phi_ci_low"].gt(0)
            ).sum()
        ),
        "median_endpoint_top5_frequency_among_observed_top10pct_edges": float(
            result.nlargest(int(np.ceil(len(result) * 0.10)), "phi_observed")[
                "endpoint_top5_frequency"
            ].median()
        ),
        "output": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nMost stable edges")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
