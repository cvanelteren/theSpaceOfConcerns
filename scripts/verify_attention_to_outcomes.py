#!/usr/bin/env python3
"""Publication gate for the concern-space and adopted-output analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
LINK = OUT / "outcome_linkage"
FIG = ROOT / "figures"


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, evidence) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    multilabel = json.loads((OUT / "multilabel_category_reconstruction.json").read_text())
    check(
        "multi-label paper taxonomy reconstructed",
        multilabel["papers_with_substantive_categories"] == 6573
        and multilabel["papers_with_multiple_categories"] == 1228
        and multilabel["paper_category_relations"] == 7895
        and multilabel["maximum_categories_per_paper"] == 5,
        multilabel,
    )

    primary_assignment = json.loads((OUT / "primary_concern_assignment.json").read_text())
    check(
        "one primary concern inferred from each substantive archive bundle",
        primary_assignment["papers_with_primary_concern"] == 6573
        and primary_assignment["unique_category_bundles"] == 327
        and primary_assignment["multi_category_papers"] == 1228
        and primary_assignment["multi_category_bundles_with_unique_conditional_winner"] == 277
        and primary_assignment["multi_category_papers_using_idf_tiebreak"] == 10
        and primary_assignment["multi_category_papers_using_alphabetical_tiebreak"] == 0,
        primary_assignment,
    )

    measurement = pd.read_csv(OUT / "primary_concern_measurement_sensitivity.csv")
    check(
        "prospective locality is stable across archive-category treatments",
        set(measurement["category_treatment"])
        == {
            "inferred_primary",
            "fractional_multilabel",
            "single_category_papers",
            "collector_first_match",
        }
        and measurement["odds_ratio_per_0_1"].between(0.78, 0.82).all()
        and (measurement["odds_ratio_per_0_1_ci_high"] < 1).all(),
        measurement.to_dict("records"),
    )

    availability = pd.read_csv(OUT / "hazard_availability_sensitivity.csv").set_index("risk_set")
    primary = availability.loc["appeared_by_prior_window_end"]
    check(
        "primary locality panel is fully prospective",
        int(primary["n_groups"]) == 1035
        and int(primary["n_rows"]) == 32980
        and 0.78 < float(primary["odds_ratio_per_0_1"]) < 0.82,
        primary.to_dict(),
    )

    paired = json.loads((OUT / "paired_null_robustness.json").read_text())
    check(
        "prospective matched comparison resamples complete actor histories",
        paired["lagged_popularity_prior_n_transitions"] == 1035
        and 0.55 < paired["lagged_popularity_prior_share_actor_bootstrap_ci_low"]
        < paired["lagged_popularity_prior_share_nearer_than_null"]
        < paired["lagged_popularity_prior_share_actor_bootstrap_ci_high"] < 0.66,
        {
            key: value for key, value in paired.items()
            if key.startswith("lagged_popularity_prior")
        },
    )

    predictions = pd.read_csv(LINK / "outcome_topic_predictions.csv")
    probabilities = pd.read_csv(LINK / "outcome_topic_probabilities.csv")
    coverage = pd.read_csv(LINK / "annual_output_topic_coverage.csv")
    panel = pd.read_csv(LINK / "topic_meeting_attention_outcomes.csv")
    models = pd.read_csv(LINK / "attention_accumulation_models.csv")
    metrics = json.loads((LINK / "title_classifier_metrics.json").read_text())

    check(
        "complete regular-ATCM output universe",
        len(predictions) == 584
        and predictions["outcome_id"].nunique() == 584
        and predictions["meeting"].min() == 19
        and predictions["meeting"].max() == 47
        and predictions["instrument"].value_counts().to_dict()
        == {"Measure": 277, "Resolution": 172, "Decision": 135},
        {
            "rows": len(predictions),
            "meetings": [int(predictions["meeting"].min()), int(predictions["meeting"].max())],
            "types": predictions["instrument"].value_counts().to_dict(),
        },
    )
    sums = probabilities.groupby("outcome_id")["probability"].sum()
    check(
        "soft output assignments cover 45 concerns and normalize",
        len(probabilities) == 584 * 45
        and probabilities["topic"].nunique() == 45
        and np.allclose(sums, 1.0, atol=1e-8),
        {"rows": len(probabilities), "maximum_sum_error": float((sums - 1).abs().max())},
    )
    check(
        "hard output map is exhaustive",
        len(coverage) == 45
        and int(coverage["primary_outcomes"].sum()) == 584
        and int(coverage["primary_outcomes"].eq(0).sum()) == 8,
        {
            "concerns": len(coverage),
            "assigned_outputs": int(coverage["primary_outcomes"].sum()),
            "unoccupied_concerns": int(coverage["primary_outcomes"].eq(0).sum()),
        },
    )
    check(
        "held-out-meeting title classification clears descriptive thresholds",
        metrics["n_papers"] == 6572
        and metrics["n_concerns"] == 45
        and metrics["top1_matches_any_official_category"] >= 0.60
        and metrics["top3_contains_any_official_category"] >= 0.78
        and metrics["same_descriptive_region"] >= 0.78,
        metrics,
    )
    check(
        "outcome models use ordered regular meetings",
        len(panel) == 45 * 29
        and panel["meeting"].nunique() == 29
        and panel["meeting"].min() == 19
        and panel["meeting"].max() == 47
        and np.isclose(panel["outcome_mass"].sum(), 584.0),
        {
            "rows": len(panel),
            "meetings": int(panel["meeting"].nunique()),
            "output_mass": float(panel["outcome_mass"].sum()),
        },
    )

    hard5 = models[
        models["specification"].eq("accumulated_attention_hard_output")
        & models["horizon_meetings"].eq(5)
    ].set_index("predictor")
    focal = hard5.loc["papers_prior5"]
    nearby = hard5.loc["nearby_prior5"]
    earlier = hard5.loc["outcomes_prior5"]
    check(
        "five-meeting output result separates focal from nearby attention",
        focal["doubling_ci_low"] > 1
        and nearby["doubling_ci_low"] < 1 < nearby["doubling_ci_high"]
        and earlier["doubling_ci_low"] < 1 < earlier["doubling_ci_high"],
        {"focal": focal.to_dict(), "nearby": nearby.to_dict(), "earlier": earlier.to_dict()},
    )
    onset = models[
        models["specification"].eq("new_output_episode_after_five_quiet_meetings")
    ]
    check(
        "quiet-period onset is reported as sparse and uncertain",
        onset["n_events"].nunique() == 1
        and int(onset["n_events"].iloc[0]) == 38
        and (onset["ci_low"] <= 1).all()
        and (onset["ci_high"] >= 1).all(),
        onset.to_dict("records"),
    )

    manuscript = (ROOT / "theSpaceOfConcerns.tex").read_text()
    forbidden = [
        "verified paper lineage",
        "768 annual",
        "740 titled outputs",
        "89 outputs",
        "dividing the weight of multi-category papers",
        "A paper returned under several official categories is divided equally",
        "each paper contributes $1/k$",
    ]
    check(
        "invalid lineage and obsolete output claims are absent from the manuscript",
        not any(term in manuscript for term in forbidden),
        {"found": [term for term in forbidden if term in manuscript]},
    )

    figure_stems = [
        "fig01_space_of_concerns_topology",
        "fig02_local_specialization",
        "fig03_selective_translation",
    ]
    missing = [
        f"{stem}.pdf" for stem in figure_stems
        if not (FIG / f"{stem}.pdf").is_file() or (FIG / f"{stem}.pdf").stat().st_size == 0
    ]
    check("all main figure artifacts exist", not missing, {"missing": missing})

    passed = all(item["passed"] for item in checks)
    report = {"passed": passed, "checks": checks}
    (LINK / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Concern-space and output verification", "", f"Overall: **{'PASS' if passed else 'FAIL'}**", ""]
    lines.extend(f"- [{'x' if item['passed'] else ' '}] {item['check']}" for item in checks)
    (LINK / "verification_report.md").write_text("\n".join(lines) + "\n")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['check']}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
