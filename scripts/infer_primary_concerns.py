#!/usr/bin/env python3
"""Infer one primary concern from each paper's archive-category bundle.

The ATS document archive can return one paper under several category filters.
This script treats a paper's category set as the observation and assigns the
label that contributes the most conditional information given the remaining
labels.

For category bundle C and candidate label l in C, the score is

    I(l | C \\ {l}) = -log P(l | C \\ {l})

where probabilities are empirical supports over papers and a paper supports an
itemset when its archive-category bundle contains that itemset. Thus a broad
label that routinely accompanies the other labels contributes little new
information. If conditional scores tie, marginal inverse document frequency,
log(N / n_l), resolves the tie. A final alphabetical rule makes any exact
remaining tie deterministic.

Outputs
-------
data/document-summary-primary-concern.parquet
    Original document rows with ``category`` replaced by the inferred primary
    concern and the full archive bundle retained in ``archive_categories``.
data/paper-primary-concern.csv
    One row per retained paper with scores and assignment diagnostics.
output/primary_concern_assignment_combinations.csv
    One row per unique archive-category bundle.
output/primary_concern_assignment.json
    Corpus-level diagnostics and validation checks.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "document-summary-multilabel.parquet"
LEGACY = ROOT / "antarctic-database-go" / "data" / "processed" / "document-summary.parquet"
OUT_PARQUET = ROOT / "data" / "document-summary-primary-concern.parquet"
OUT_ASSIGNMENTS = ROOT / "data" / "paper-primary-concern.csv"
OUT_COMBINATIONS = ROOT / "output" / "primary_concern_assignment_combinations.csv"
OUT_DIAGNOSTICS = ROOT / "output" / "primary_concern_assignment.json"

TOLERANCE = 1e-12


def parse_bundle(value: object) -> frozenset[str]:
    if value is None or pd.isna(value):
        return frozenset()
    return frozenset(
        label.strip()
        for label in str(value).split("\t")
        if label.strip() and label.strip().upper() not in {"ALL", "OTHER"}
    )


def support(bundle: frozenset[str], counts: Counter[frozenset[str]]) -> int:
    """Number of papers whose observed category set contains ``bundle``."""
    return int(sum(n for observed, n in counts.items() if bundle.issubset(observed)))


def infer_bundle(
    bundle: frozenset[str],
    counts: Counter[frozenset[str]],
    n_papers: int,
) -> dict:
    labels = sorted(bundle)
    if not labels:
        raise ValueError("Cannot assign an empty category bundle")

    full_support = support(bundle, counts)
    conditional_information: dict[str, float] = {}
    conditional_probability: dict[str, float] = {}
    marginal_idf: dict[str, float] = {}
    remainder_support: dict[str, int] = {}

    for label in labels:
        remainder = bundle - {label}
        denominator = support(remainder, counts)
        if denominator <= 0 or full_support <= 0:
            raise ValueError(f"Invalid support for {sorted(bundle)} and {label}")
        probability = full_support / denominator
        conditional_probability[label] = float(probability)
        conditional_information[label] = float(-math.log(probability))
        remainder_support[label] = int(denominator)
        marginal_idf[label] = float(
            math.log(n_papers / support(frozenset({label}), counts))
        )

    maximum_information = max(conditional_information.values())
    information_winners = [
        label
        for label in labels
        if np.isclose(
            conditional_information[label],
            maximum_information,
            rtol=0,
            atol=TOLERANCE,
        )
    ]
    method = "conditional_information"
    candidates = information_winners
    if len(candidates) > 1:
        maximum_idf = max(marginal_idf[label] for label in candidates)
        candidates = [
            label
            for label in candidates
            if np.isclose(marginal_idf[label], maximum_idf, rtol=0, atol=TOLERANCE)
        ]
        method = "marginal_idf_tiebreak"
    primary = sorted(candidates)[0]
    if len(candidates) > 1:
        method = "alphabetical_tiebreak"

    ordered = sorted(
        labels,
        key=lambda label: (
            conditional_information[label],
            marginal_idf[label],
            label,
        ),
        reverse=True,
    )
    runner_up = ordered[1] if len(ordered) > 1 else None
    runner_score = (
        conditional_information[runner_up] if runner_up is not None else np.nan
    )
    return {
        "archive_categories": "\t".join(labels),
        "category_count": len(labels),
        "primary_concern": primary,
        "assignment_method": method if len(labels) > 1 else "single_archive_category",
        "conditional_information": conditional_information[primary],
        "conditional_probability": conditional_probability[primary],
        "marginal_idf": marginal_idf[primary],
        "runner_up": runner_up,
        "runner_up_information": runner_score,
        "information_margin": (
            conditional_information[primary] - runner_score
            if runner_up is not None
            else np.nan
        ),
        "full_bundle_support": full_support,
        "remainder_support": remainder_support[primary],
        "conditional_tie_count": len(information_winners),
        "score_json": json.dumps(
            {
                label: {
                    "conditional_information": conditional_information[label],
                    "conditional_probability": conditional_probability[label],
                    "marginal_idf": marginal_idf[label],
                    "remainder_support": remainder_support[label],
                }
                for label in labels
            },
            sort_keys=True,
        ),
    }


def main() -> None:
    source = pd.read_parquet(SOURCE)
    papers = source.drop_duplicates("paper_id").copy()
    papers["bundle"] = papers["category"].map(parse_bundle)
    papers = papers[papers["bundle"].map(bool)].copy()
    counts: Counter[frozenset[str]] = Counter(papers["bundle"])
    n_papers = int(len(papers))

    inferred = {
        bundle: infer_bundle(bundle, counts, n_papers)
        for bundle in sorted(counts, key=lambda item: (len(item), sorted(item)))
    }
    combination_rows = []
    for bundle, values in inferred.items():
        combination_rows.append(
            {
                **values,
                "papers_with_exact_bundle": int(counts[bundle]),
            }
        )
    combinations = pd.DataFrame(combination_rows)

    assignments = papers[["paper_id", "bundle"]].copy()
    assignment_fields = list(next(iter(inferred.values())).keys())
    for field in assignment_fields:
        assignments[field] = assignments["bundle"].map(
            lambda bundle, key=field: inferred[bundle][key]
        )
    assignments = assignments.drop(columns="bundle")

    legacy = (
        pd.read_parquet(LEGACY)
        .drop_duplicates("paper_id")
        .set_index("paper_id")["category"]
    )
    assignments["collector_first_match"] = assignments["paper_id"].map(legacy)
    assignments["matches_collector_first_match"] = assignments[
        "primary_concern"
    ].eq(assignments["collector_first_match"])

    rebuilt = source.copy()
    mapping = assignments.set_index("paper_id")
    rebuilt = rebuilt[rebuilt["paper_id"].isin(mapping.index)].copy()
    rebuilt["archive_categories"] = rebuilt["paper_id"].map(
        mapping["archive_categories"]
    )
    rebuilt["category"] = rebuilt["paper_id"].map(mapping["primary_concern"])
    for field in (
        "assignment_method",
        "conditional_information",
        "conditional_probability",
        "marginal_idf",
        "information_margin",
        "full_bundle_support",
        "remainder_support",
    ):
        rebuilt[field] = rebuilt["paper_id"].map(mapping[field])

    multi = assignments[assignments["category_count"].gt(1)]
    diagnostics = {
        "source": str(SOURCE),
        "papers_with_primary_concern": n_papers,
        "archive_categories": int(
            len(set().union(*papers["bundle"].tolist()))
        ),
        "unique_category_bundles": int(len(counts)),
        "single_category_bundles": int(sum(len(bundle) == 1 for bundle in counts)),
        "multi_category_bundles": int(sum(len(bundle) > 1 for bundle in counts)),
        "single_category_papers": int(assignments["category_count"].eq(1).sum()),
        "multi_category_papers": int(assignments["category_count"].gt(1).sum()),
        "multi_category_bundles_with_unique_conditional_winner": int(
            combinations.loc[
                combinations["category_count"].gt(1), "conditional_tie_count"
            ].eq(1).sum()
        ),
        "multi_category_papers_using_idf_tiebreak": int(
            multi["assignment_method"].eq("marginal_idf_tiebreak").sum()
        ),
        "multi_category_papers_using_alphabetical_tiebreak": int(
            multi["assignment_method"].eq("alphabetical_tiebreak").sum()
        ),
        "multi_category_primary_matches_collector_first_match": float(
            multi["matches_collector_first_match"].mean()
        ),
        "formula": "-log P(label | remaining labels); marginal IDF breaks ties",
        "probability_support": "paper-weighted superset counts over archive-category bundles",
        "output_parquet": str(OUT_PARQUET),
        "output_assignments": str(OUT_ASSIGNMENTS),
        "output_combinations": str(OUT_COMBINATIONS),
    }

    expected = {
        "papers_with_primary_concern": 6573,
        "archive_categories": 45,
        "unique_category_bundles": 327,
        "single_category_bundles": 45,
        "multi_category_bundles": 282,
        "single_category_papers": 5345,
        "multi_category_papers": 1228,
    }
    for key, value in expected.items():
        if diagnostics[key] != value:
            raise ValueError(f"Unexpected {key}: {diagnostics[key]} != {value}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIAGNOSTICS.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_parquet(OUT_PARQUET, index=False)
    assignments.drop(columns="score_json").to_csv(OUT_ASSIGNMENTS, index=False)
    combinations.to_csv(OUT_COMBINATIONS, index=False)
    OUT_DIAGNOSTICS.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
