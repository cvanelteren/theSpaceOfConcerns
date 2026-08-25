#!/usr/bin/env python3
"""Unblind and evaluate formal-output title classification after consensus."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import build_graphs, load_topic_meta
from scripts import explore_lineage_space as lineage
from utils import compute_product_space, get_rca


OUTDIR = ROOT / "output" / "outcome_linkage"
FINAL = OUTDIR / "outcome_consensus_final.csv"
SUPPLEMENT = OUTDIR / "outcome_consensus_supplement_final.csv"
SCOPE = OUTDIR / "outcome_consensus_validation_scope_key.csv"
PREDICTIONS = OUTDIR / "outcome_topic_predictions.csv"
PROBABILITIES = OUTDIR / "outcome_topic_probabilities.csv"
OUT_ROWS = OUTDIR / "outcome_consensus_model_comparison.csv"
OUT_METRICS = OUTDIR / "outcome_consensus_validation_metrics.csv"
OUT_WEIGHTED = OUTDIR / "outcome_consensus_design_weighted_metrics.csv"
OUT_CONFUSIONS = OUTDIR / "outcome_consensus_confusions.csv"
OUT_SUMMARY = OUTDIR / "outcome_consensus_validation_summary.json"

SPECIAL = {"INSUFFICIENT_TITLE", "OUTSIDE_TAXONOMY"}
RANDOM_SEED = 20260814
N_BOOTSTRAP = 5000


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def scope_metrics(data: pd.DataFrame, scope_name: str) -> dict:
    codable = data[data["codable"]].copy()
    n = len(data)
    n_codable = len(codable)
    exact = int(codable["exact_top1"].sum())
    top3 = int(codable["primary_in_top3"].sum())
    exact_ci = wilson(exact, n_codable)
    top3_ci = wilson(top3, n_codable)
    return {
        "scope": scope_name,
        "n_titles": int(n),
        "n_codable": int(n_codable),
        "n_abstained": int(n - n_codable),
        "coverage": float(n_codable / n) if n else np.nan,
        "exact_top1": float(exact / n_codable) if n_codable else np.nan,
        "exact_ci_low": exact_ci[0],
        "exact_ci_high": exact_ci[1],
        "primary_in_top3": float(top3 / n_codable) if n_codable else np.nan,
        "top3_ci_low": top3_ci[0],
        "top3_ci_high": top3_ci[1],
        "primary_or_secondary_in_top3": float(
            codable["primary_or_secondary_in_top3"].mean()
        ) if n_codable else np.nan,
        "same_descriptive_region": float(
            codable["same_descriptive_region"].mean()
        ) if n_codable else np.nan,
        # phi is proximity: larger values mean closer concerns.
        "mean_human_model_proximity": float(
            codable["human_model_proximity"].mean()
        ) if n_codable else np.nan,
        "median_human_model_proximity": float(
            codable["human_model_proximity"].median()
        ) if n_codable else np.nan,
    }


def design_weighted_metrics(data: pd.DataFrame, population: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    sample = data[data["in_stratified_120"]].copy()
    sample["confidence_band"] = np.where(sample["high_confidence"], "high", "lower")
    population = population.copy()
    population["confidence_band"] = np.where(population["high_confidence"], "high", "lower")
    sample_counts = sample.groupby(["instrument", "confidence_band"]).size().rename("sample_n")
    population_counts = population.groupby(["instrument", "confidence_band"]).size().rename("population_n")
    strata = pd.concat([sample_counts, population_counts], axis=1).reset_index()
    if strata[["sample_n", "population_n"]].isna().any().any():
        raise ValueError("The stratified audit does not cover every population stratum")
    strata["weight"] = strata["population_n"] / strata["sample_n"]
    sample = sample.merge(strata, on=["instrument", "confidence_band"], validate="many_to_one")

    def weighted_ratio(frame: pd.DataFrame, outcome: str, denominator: str = "codable") -> float:
        den = frame.loc[frame[denominator], "weight"].sum()
        num = frame.loc[frame[denominator], "weight"].mul(
            frame.loc[frame[denominator], outcome].astype(float)
        ).sum()
        return float(num / den) if den else np.nan

    coverage = float(sample.loc[sample["codable"], "weight"].sum() / sample["weight"].sum())
    point = {
        "scope": "stratified_120_design_weighted_to_740",
        "n_sample": int(len(sample)),
        "population_titles": int(len(population)),
        "coverage": coverage,
        "exact_top1": weighted_ratio(sample, "exact_top1"),
        "primary_in_top3": weighted_ratio(sample, "primary_in_top3"),
        "same_descriptive_region": weighted_ratio(sample, "same_descriptive_region"),
        "mean_human_model_proximity": weighted_ratio(sample, "human_model_proximity"),
    }

    rng = np.random.default_rng(RANDOM_SEED)
    groups = {
        key: frame.reset_index(drop=True)
        for key, frame in sample.groupby(["instrument", "confidence_band"], sort=True)
    }
    draws = {key: [] for key in ["coverage", "exact_top1", "primary_in_top3", "same_descriptive_region", "mean_human_model_proximity"]}
    for _ in range(N_BOOTSTRAP):
        resampled = pd.concat(
            [
                frame.iloc[rng.integers(0, len(frame), size=len(frame))]
                for frame in groups.values()
            ],
            ignore_index=True,
        )
        draws["coverage"].append(
            float(resampled.loc[resampled["codable"], "weight"].sum() / resampled["weight"].sum())
        )
        for output_metric, row_metric in [
            ("exact_top1", "exact_top1"),
            ("primary_in_top3", "primary_in_top3"),
            ("same_descriptive_region", "same_descriptive_region"),
            ("mean_human_model_proximity", "human_model_proximity"),
        ]:
            draws[output_metric].append(weighted_ratio(resampled, row_metric))
    for metric, values in draws.items():
        point[f"{metric}_ci_low"] = float(np.nanquantile(values, 0.025))
        point[f"{metric}_ci_high"] = float(np.nanquantile(values, 0.975))
    return point, strata


def main() -> None:
    final = pd.read_csv(FINAL, keep_default_na=False)
    scope = pd.read_csv(SCOPE)
    if SUPPLEMENT.exists():
        supplement = pd.read_csv(SUPPLEMENT, keep_default_na=False)
        supplement_scope = supplement[["validation_id", "outcome_id"]].copy()
        supplement_scope["in_stratified_120"] = False
        supplement_scope["in_headline_lineage_95"] = False
        final = pd.concat([final, supplement], ignore_index=True)
        scope = pd.concat([scope, supplement_scope], ignore_index=True)
    predictions = pd.read_csv(PREDICTIONS)
    probabilities = pd.read_csv(PROBABILITIES)
    if final.empty or not final["validation_id"].is_unique or not final["outcome_id"].is_unique:
        raise ValueError("Expected non-empty final consensus with unique validation and outcome IDs")

    _, _, _, counts, _ = build_graphs()
    topics = list(counts.index)
    phi = compute_product_space(get_rca(counts)).reindex(index=topics, columns=topics)
    np.fill_diagonal(phi.values, 1.0)
    _, region_raw, _ = load_topic_meta()
    region = {topic: int(region_raw[lineage.normalize_topic_key(topic)]) for topic in topics}

    rows = final.merge(scope, on=["validation_id", "outcome_id"], validate="one_to_one")
    rows = rows.merge(
        predictions[[
            "outcome_id", "topic_top1", "topic_top2", "topic_top3",
            "probability_top1", "margin_top1_top2", "high_confidence",
        ]],
        on="outcome_id", validate="one_to_one",
    )
    rows["codable"] = ~rows["consensus_primary"].isin(SPECIAL)
    rows["exact_top1"] = rows["codable"] & rows["consensus_primary"].eq(rows["topic_top1"])
    top3_columns = ["topic_top1", "topic_top2", "topic_top3"]
    rows["primary_in_top3"] = rows.apply(
        lambda row: bool(row["codable"] and row["consensus_primary"] in {row[column] for column in top3_columns}),
        axis=1,
    )
    rows["primary_or_secondary_in_top3"] = rows.apply(
        lambda row: bool(
            row["codable"]
            and ({row["consensus_primary"], row["consensus_secondary"]} - {""})
            & {row[column] for column in top3_columns}
        ),
        axis=1,
    )
    rows["same_descriptive_region"] = rows.apply(
        lambda row: bool(
            row["codable"]
            and region[row["consensus_primary"]] == region[row["topic_top1"]]
        ),
        axis=1,
    )
    rows["human_model_proximity"] = rows.apply(
        lambda row: float(phi.loc[row["consensus_primary"], row["topic_top1"]])
        if row["codable"] else np.nan,
        axis=1,
    )
    edges = pd.read_csv(OUTDIR / "direct_edge_outcome_proximity.csv")
    lineage_ids = set(
        edges.loc[
            edges["graph"].eq("decision_map_verified.json")
            & edges["geometry"].eq("cumulative_prior_meeting_space")
            & edges["relation"].isin(
                {
                    "direct_adoption_or_approval", "documented_contribution",
                    "direct_proposal_or_discussion",
                }
            ),
            "outcome_id",
        ]
    )
    rows["in_lineage_union_157"] = rows["outcome_id"].isin(lineage_ids)
    rows.to_csv(OUT_ROWS, index=False)

    metrics = [
        scope_metrics(rows, f"all_{len(rows)}"),
        scope_metrics(rows[rows["in_stratified_120"]], "stratified_120"),
        scope_metrics(rows[rows["in_headline_lineage_95"]], "headline_lineage_95"),
        scope_metrics(rows[rows["in_lineage_union_157"]], "lineage_union_157"),
    ]
    for instrument, group in rows[rows["in_stratified_120"]].groupby("instrument"):
        metrics.append(scope_metrics(group, f"stratified_120::{instrument}"))
    for high, group in rows[rows["in_stratified_120"]].groupby("high_confidence"):
        metrics.append(scope_metrics(group, f"stratified_120::{'high' if high else 'lower'}_model_confidence"))
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(OUT_METRICS, index=False)

    weighted, strata = design_weighted_metrics(rows, predictions)
    pd.DataFrame([weighted]).to_csv(OUT_WEIGHTED, index=False)
    strata.to_csv(OUTDIR / "outcome_consensus_sampling_strata.csv", index=False)

    confusions = (
        rows[rows["codable"] & ~rows["exact_top1"]]
        .groupby(["consensus_primary", "topic_top1"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    confusions.to_csv(OUT_CONFUSIONS, index=False)

    coder_values = {}
    for coder in ("a", "b", "c"):
        parts = [pd.read_csv(OUTDIR / f"outcome_consensus_coder_{coder}.csv")]
        supplement_path = OUTDIR / f"outcome_consensus_supplement_coder_{coder}.csv"
        if supplement_path.exists():
            parts.append(pd.read_csv(supplement_path))
        coder_values[coder] = pd.concat(parts, ignore_index=True)["primary_concern"]
    from sklearn.metrics import cohen_kappa_score
    agreement_rows = []
    for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
        agreement_rows.append(
            {
                "coder_pair": f"{left}-{right}",
                "exact_agreement": float((coder_values[left] == coder_values[right]).mean()),
                "cohen_kappa": float(cohen_kappa_score(coder_values[left], coder_values[right])),
                "n": int(len(coder_values[left])),
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(OUTDIR / "outcome_consensus_intercoder_agreement_combined.csv", index=False)
    unanimous = pd.concat(
        [coder_values[coder].rename(coder) for coder in ("a", "b", "c")], axis=1
    ).nunique(axis=1).eq(1)
    summary = {
        "design": {
            "independent_blind_coders": 3,
            "titles": int(len(rows)),
            "stratified_classifier_audit": 120,
            "complete_headline_lineage_set": 95,
            "overlap": 15,
            "complete_lineage_union": int(rows["in_lineage_union_157"].sum()),
            "unanimous_initial_labels": int(unanimous.sum()),
            "adjudicated_initial_labels": int((~unanimous).sum()),
        },
        "pairwise_intercoder_agreement": agreement.to_dict(orient="records"),
        "scope_metrics": metrics,
        "design_weighted_metrics": weighted,
        "note": "Concern proximity phi is reported on its native scale: larger values mean closer concerns.",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(metrics_table.to_string(index=False))
    print("\nDesign-weighted stratified audit")
    print(pd.DataFrame([weighted]).to_string(index=False))
    print("\nMost common exact-label disagreements")
    print(confusions.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
