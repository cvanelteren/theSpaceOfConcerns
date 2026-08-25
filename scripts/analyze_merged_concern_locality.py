#!/usr/bin/env python3
"""Rebuild the space with the two area-protection labels merged, and retest.

Almost every citation that reaches its focal concern's closest neighbour crosses
between two Secretariat labels -- Management Plans and Area Protection and
Management Plans General -- that name the same administrative object, an ASPA
management plan. If they are one object they should be one node in the space.

This script merges them at source, recomputes revealed policy advantage and the
proximity matrix from the paper record, remaps the outcome labels to match, and
reruns the closest-concern test on the rebuilt space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_citation_bundle_locality import (
    browser_edges,
    load_payload,
    neighbour_rank,
    outcome_frame,
    proximity_matrix,
)
from utils import compute_product_space, get_rca, load_data

PAPERS = ROOT / "data/document-summary-multilabel.parquet"
DEFAULT_OUT = ROOT / "output/merged_concern_locality"
MERGED = "Area protection and management plans"
SOURCES = ["Management Plans", "Area Protection and Management Plans General"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def merged_space() -> pd.DataFrame:
    """Proximity matrix over 44 concerns, the two area labels summed into one."""
    # The interaction matrix is concerns by actors, so the two labels are rows.
    counts, _, _, _ = load_data(str(PAPERS))
    present = [c for c in SOURCES if c in counts.index]
    if len(present) != 2:
        raise SystemExit(f"expected both area labels in the paper matrix, found {present}")
    merged = counts.drop(index=present)
    merged.loc[MERGED] = counts.loc[present].sum(axis=0)
    return compute_product_space(get_rca(merged))


def closest_concern_test(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    index: dict[str, int],
    pool: pd.DataFrame,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    observed = np.array(
        [
            neighbour_rank(index[r.focal_topic], index[r.cited_topic], matrix)
            for r in frame.itertuples()
        ]
    )
    null: list[int] = []
    for r in frame.itertuples():
        candidates = pool[
            (pool.instrument == r.cited_instrument)
            & (pool.year < r.focal_year)
            & (pool.topic != r.focal_topic)
        ]
        if len(candidates) < 3:
            continue
        head = index[r.focal_topic]
        null.extend(
            neighbour_rank(head, index[t], matrix)
            for t in rng.choice(candidates.topic.to_numpy(), size=draws)
        )
    null = np.array(null)
    share, null_share = float(np.mean(observed == 1)), float(np.mean(null == 1))
    return {
        "citations": len(observed),
        "hits": int(np.sum(observed == 1)),
        "share": round(share, 4),
        "null_share": round(null_share, 4),
        "ratio": round(share / null_share, 3) if null_share else np.nan,
    }


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    payload = load_payload()
    outcomes = outcome_frame(payload)
    topics = pd.read_csv(
        ROOT / "output/outcome_linkage/outcome_topic_predictions.csv"
    ).set_index("outcome_id")["topic_top1"]

    edges = browser_edges(payload, outcomes)
    edges["focal_topic"] = edges.focal_id.map(topics)
    edges["cited_topic"] = edges.cited_id.map(topics)
    edges = edges.dropna(subset=["focal_topic", "cited_topic"])

    results = {}
    for name, phi, relabel in [
        ("original", proximity_matrix(payload), False),
        ("merged", merged_space(), True),
    ]:
        index = {t: i for i, t in enumerate(phi.index)}
        matrix = phi.to_numpy()
        frame = edges.copy()
        labels = topics.copy()
        if relabel:
            frame["focal_topic"] = frame.focal_topic.replace(dict.fromkeys(SOURCES, MERGED))
            frame["cited_topic"] = frame.cited_topic.replace(dict.fromkeys(SOURCES, MERGED))
            labels = labels.replace(dict.fromkeys(SOURCES, MERGED))
        missing = {t for t in frame.focal_topic.unique() if t not in index}
        frame = frame[~frame.focal_topic.isin(missing) & ~frame.cited_topic.isin(missing)]

        pool = outcomes.copy()
        pool["topic"] = pool.outcome_id.map(labels)
        pool = pool.dropna(subset=["topic"])
        pool = pool[pool.topic.isin(index)]

        off_label = frame[frame.focal_topic != frame.cited_topic]
        strata = {
            "all off-label citations": off_label,
            "soft law": off_label[off_label.focal_instrument == "Resolution"],
            "binding": off_label[off_label.focal_instrument == "Measure"],
        }
        results[name] = {
            "concerns": len(phi),
            "strata": {
                label: closest_concern_test(subset, matrix, index, pool, rng, args.draws)
                for label, subset in strata.items()
            },
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "merged_concern_locality.json").write_text(json.dumps(results, indent=2))

    print(f"{'space':<10}{'stratum':<26}{'cites':>7}{'hits':>6}{'share':>8}{'null':>8}{'ratio':>8}")
    for name, payload_ in results.items():
        for label, stats in payload_["strata"].items():
            print(f"{name:<10}{label:<26}{stats['citations']:>7}{stats['hits']:>6}"
                  f"{stats['share']:>8.3f}{stats['null_share']:>8.3f}{stats['ratio']:>8.2f}")


if __name__ == "__main__":
    run(parse_args())
