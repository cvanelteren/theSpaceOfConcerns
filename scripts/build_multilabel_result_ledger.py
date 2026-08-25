#!/usr/bin/env python3
"""Freeze and verify the corrected results used by the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.official_regular_atcm_outputs import load_official_regular_outputs
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from official_regular_atcm_outputs import load_official_regular_outputs


ROOT = Path(__file__).resolve().parents[1]
OUTROOT = ROOT / "output" / "category_treatment_comparison"
SIGNAL_ROOT = ROOT / "output" / "attention_output_signal"
OUT_JSON = ROOT / "output" / "multilabel_result_ledger.json"
OUT_CSV = ROOT / "output" / "multilabel_result_ledger.csv"


def row(rows: list[dict], section: str, result: str, value, low=None, high=None, unit="") -> None:
    rows.append(
        {
            "section": section,
            "result": result,
            "value": value,
            "ci_low": low,
            "ci_high": high,
            "unit": unit,
        }
    )


def main() -> None:
    summary = json.loads((OUTROOT / "major_results_summary.json").read_text())
    bootstrap = json.loads(
        (
            OUTROOT
            / "fractional_multilabel"
            / "locality_actor_history_bootstrap.json"
        ).read_text()
    )
    family_horizons = pd.read_csv(OUTROOT / "output_family_pooled_horizons.csv")
    family_common_support = pd.read_csv(
        OUTROOT / "output_family_common_support_horizons.csv"
    )
    family_sensitivity = pd.read_csv(
        OUTROOT / "output_family_exclusion_sensitivity.csv"
    )
    dependence = json.loads(
        (ROOT / "output" / "hazard_robustness_dependence.json").read_text()
    )
    type_forecasts = pd.read_csv(SIGNAL_ROOT / "type_summary.csv")
    type_contrasts = pd.read_csv(SIGNAL_ROOT / "type_contrast.csv")
    pretest_selection = pd.read_csv(
        SIGNAL_ROOT / "pretest_selected_specifications.csv"
    )
    geometry_null = pd.read_csv(SIGNAL_ROOT / "resolution_geometry_null.csv")
    measure_content = json.loads(
        (SIGNAL_ROOT / "measure_content_summary.json").read_text()
    )
    formal_inheritance = json.loads(
        (SIGNAL_ROOT / "resolution_measure_lineage_summary.json").read_text()
    )
    resolution_follow_up = json.loads(
        (SIGNAL_ROOT / "resolution_hardening_attention_summary.json").read_text()
    )
    concentration = json.loads(
        (SIGNAL_ROOT / "resolution_category_concentration_sensitivity.json").read_text()
    )

    geometry = summary["geometry"]["fractional_multilabel"]
    movement = summary["portfolio_entry"]["fractional_multilabel"]
    locality = movement["conditional_logit"]
    matched = movement["popularity_matched_displacement"]

    assert locality["papers"] == 6573
    assert locality["actor_periods"] == 968
    assert locality["risk_rows"] == 30957
    assert geometry["positive_observed_edges"] if "positive_observed_edges" in geometry else True
    assert geometry["edge_bootstrap"]["positive_observed_edges"] == 956
    assert bootstrap["requested_draws"] == 500
    assert bootstrap["successful_draws"] >= 475
    assert dependence["nonoverlapping_windows"]["n_groups"] == 251
    assert len(type_forecasts) == 9
    assert len(type_contrasts) == 4
    assert len(pretest_selection) == 2
    assert set(pretest_selection["model"]) == {
        "output history",
        "history + direct attention",
    }
    assert set(pretest_selection["history"]) == {5}
    assert set(pretest_selection["alpha"]) == {0.1}
    assert int(
        pretest_selection.loc[
            pretest_selection["model"].eq("history + direct attention"),
            "attention",
        ].iloc[0]
    ) == 1
    assert measure_content["measures"] == 277
    assert measure_content["official_area_category_count"] == 266
    assert measure_content["management_plan_count"] == 230
    assert measure_content["revised_management_plan_count"] == 211
    assert formal_inheritance["measure_pages"] == 277
    assert (
        formal_inheritance.get(
            "measure_pages_loaded_from_pinned_corpus",
            formal_inheritance.get("measure_pages_fetched"),
        )
        == 277
    )
    assert formal_inheritance["measures_with_any_explicit_prior_formal_predecessor"] == 245
    assert resolution_follow_up["eligible_resolutions"] == 115
    assert resolution_follow_up["linked_within_10_meetings"] == 6
    assert concentration["meetings"] == 19

    inventory = load_official_regular_outputs()
    assert len(inventory) == 584
    assert inventory["output_id"].is_unique
    counts = inventory["instrument"].value_counts().to_dict()
    assert counts == {"Measure": 277, "Resolution": 172, "Decision": 135}
    category_mass = pd.read_csv(OUTROOT / "output_family_category_mass.csv")
    assert len(category_mass) == 15
    assert np.isclose(category_mass["category_weight"].sum(), 584.0)

    for source in [
        ROOT / "theSpaceOfConcerns.tex",
        ROOT / "fig01_space_of_concerns_topology.py",
        ROOT / "fig02_entry_and_cohorts.py",
        ROOT / "fig03_selective_translation.py",
        ROOT / "fig03_attention_to_formal_tracks.py",
        ROOT / "scripts" / "analyze_output_category_families.py",
        ROOT / "scripts" / "analyze_resolution_attention_forecast.py",
        ROOT / "scripts" / "analyze_resolution_concentration_sensitivity.py",
        ROOT / "scripts" / "audit_resolution_measure_ladders.py",
        ROOT / "scripts" / "measure_detail_corpus.py",
    ]:
        text = source.read_text()
        for forbidden in (
            "decision_map_verified",
            "direct_lineage.json",
            "verified contributing-paper lineage",
        ):
            if forbidden in text:
                raise AssertionError(f"{source}: forbidden invalid-lineage input {forbidden}")
    assert "document-summary-multilabel.parquet" in (ROOT / "utils.py").read_text()

    rows: list[dict] = []
    row(rows, "geometry", "positive_concern_pairs", 956, unit="of 990")
    row(
        rows,
        "geometry",
        "maximum_louvain_modularity",
        geometry["louvain_modularity_max_100_seeds"],
    )
    row(
        rows,
        "movement",
        "rpa_crossing_or_per_0_1_farther",
        locality["odds_ratio_per_0_1"],
        locality["odds_ratio_per_0_1_ci_low"],
        locality["odds_ratio_per_0_1_ci_high"],
        "conditional-logit model interval",
    )
    row(
        rows,
        "movement",
        "actor_history_bootstrap_or_median",
        bootstrap["odds_ratio_per_0_1_median"],
        bootstrap["odds_ratio_per_0_1_percentile_ci_low"],
        bootstrap["odds_ratio_per_0_1_percentile_ci_high"],
        "central 95% of 500 full-rebuild actor-history bootstrap estimates",
    )
    row(
        rows,
        "movement",
        "first_paper_or_per_0_1_farther",
        locality["new_document_odds_ratio_per_0_1"],
        locality["new_document_odds_ratio_per_0_1_ci_low"],
        locality["new_document_odds_ratio_per_0_1_ci_high"],
    )
    row(
        rows,
        "movement",
        "share_nearer_than_popularity_weighted_alternatives",
        matched["share_nearer_than_popularity_matched_null"],
        matched["actor_bootstrap_ci_low"],
        matched["actor_bootstrap_ci_high"],
    )
    row(
        rows,
        "movement",
        "disjoint_five_meeting_windows_or_per_0_1_farther",
        dependence["nonoverlapping_windows"]["odds_ratio_per_0_1"],
        dependence["nonoverlapping_windows"]["odds_ratio_per_0_1_ci_low"],
        dependence["nonoverlapping_windows"]["odds_ratio_per_0_1_ci_high"],
    )
    primary_output = family_horizons[
        family_horizons["horizon_meetings"].eq(1)
    ]
    for _, result in primary_output.iterrows():
        row(
            rows,
            "formal_output_one_preceding_meeting",
            str(result["predictor"]),
            result["ratio_per_doubling_plus_one"],
            result["doubling_ci_low"],
            result["doubling_ci_high"],
        )
    for horizon in (3, 5):
        result = family_horizons[
            family_horizons["horizon_meetings"].eq(horizon)
            & family_horizons["predictor"].eq("focal_minus_nearby")
        ].iloc[0]
        row(
            rows,
            "formal_output_window_sensitivity",
            f"focal_minus_nearby__{horizon}_preceding_meetings",
            result["ratio_per_doubling_plus_one"],
            result["doubling_ci_low"],
            result["doubling_ci_high"],
        )
    common_one = family_common_support[
        family_common_support["horizon_meetings"].eq(1)
        & family_common_support["predictor"].eq("focal_minus_nearby")
    ].iloc[0]
    row(
        rows,
        "formal_output_common_support",
        "focal_minus_nearby__one_preceding_meeting__same_19_meetings",
        common_one["ratio_per_doubling_plus_one"],
        common_one["doubling_ci_low"],
        common_one["doubling_ci_high"],
    )
    for sensitivity in ("exclude_area_protection", "exclude_both"):
        result = family_sensitivity[
            family_sensitivity["sensitivity"].eq(sensitivity)
            & family_sensitivity["horizon_meetings"].eq(1)
            & family_sensitivity["predictor"].eq("focal_minus_nearby")
        ].iloc[0]
        row(
            rows,
            "formal_output_composition_sensitivity",
            f"focal_minus_nearby__{sensitivity}",
            result["ratio_per_doubling_plus_one"],
            result["doubling_ci_low"],
            result["doubling_ci_high"],
        )

    for _, result in type_forecasts.iterrows():
        row(
            rows,
            "rolling_origin_output_forecast",
            f"{result['instrument']}__{result['comparison']}",
            result["mean_difference"],
            result["bootstrap_low"],
            result["bootstrap_high"],
            f"allocation log-score change across {int(result['meetings'])} rolling-origin test meetings",
        )
    for _, result in type_contrasts.iterrows():
        row(
            rows,
            "rolling_origin_output_type_contrast",
            f"{result['comparison']}__{result['contrast']}",
            result["mean_difference"],
            result["bootstrap_low"],
            result["bootstrap_high"],
            "paired meeting-bootstrap allocation log-score contrast",
        )

    for excluded, group in geometry_null.groupby("exclude_area"):
        observed = float(group["observed_difference_vs_direct"].iloc[0])
        shuffled = group["mean_difference_vs_direct"].to_numpy(float)
        lower_tail_p = float((1 + np.sum(shuffled <= observed)) / (1 + len(shuffled)))
        expected = 1 / 201 if not bool(excluded) else 2 / 201
        assert np.isclose(lower_tail_p, expected)
        row(
            rows,
            "resolution_geometry_null",
            "area_not_scored" if bool(excluded) else "all_categories",
            observed,
            float(np.quantile(shuffled, 0.025)),
            float(np.quantile(shuffled, 0.975)),
            f"observed network-attention score change; lower-tail permutation p={lower_tail_p:.6f}",
        )

    for half in ("lower_entropy_half", "higher_entropy_half"):
        result = concentration[half]
        row(
            rows,
            "resolution_concentration_sensitivity",
            half,
            result["mean_score_change"],
            result["meeting_bootstrap_95_interval"][0],
            result["meeting_bootstrap_95_interval"][1],
            f"allocation log-score change across {result['meetings']} meetings",
        )
    entropy_difference = concentration["higher_minus_lower"]
    row(
        rows,
        "resolution_concentration_sensitivity",
        "higher_minus_lower_entropy_half",
        entropy_difference["mean_difference"],
        entropy_difference["meeting_bootstrap_95_interval"][0],
        entropy_difference["meeting_bootstrap_95_interval"][1],
        "meeting-bootstrap allocation log-score contrast",
    )
    entropy_correlation = concentration["score_change_entropy_correlation"]
    row(
        rows,
        "resolution_concentration_sensitivity",
        "score_change_entropy_pearson_r",
        entropy_correlation["pearson_r"],
        entropy_correlation["meeting_bootstrap_95_interval"][0],
        entropy_correlation["meeting_bootstrap_95_interval"][1],
        "meeting-bootstrap interval",
    )

    for source_type, result in formal_inheritance["formal_predecessors_by_type"].items():
        row(
            rows,
            "measure_formal_inheritance",
            f"measures_citing_prior_{source_type.lower()}",
            result["citing_measures"],
            unit=(
                f"of 277 Measures; {result['cited_predecessors']} distinct predecessors; "
                f"{result['citation_edges']} exact year-qualified links"
            ),
        )
    row(
        rows,
        "measure_formal_inheritance",
        "measures_citing_any_prior_post_1995_output",
        formal_inheritance["measures_with_any_explicit_prior_formal_predecessor"],
        unit="of 277 Measures",
    )
    row(
        rows,
        "resolution_to_measure_follow_up",
        "resolutions_cited_by_measure_within_10_subsequent_meetings",
        resolution_follow_up["linked_within_10_meetings"],
        unit=f"of {resolution_follow_up['eligible_resolutions']} eligible Resolutions",
    )

    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT_CSV, index=False)
    payload = {
        "status": "PASS",
        "primary_category_treatment": "fractional_multilabel",
        "paper_count": 6573,
        "regular_atcm_outputs": counts,
        "output_allocation": "45 fine paper concerns aggregated to 15 analytic families matching official ATS instrument categories; equal fractional weight across each output's official categories",
        "output_hierarchy": "author-coded analytic hierarchy, not an official Secretariat crosswalk",
        "headline_formal_output_analysis": "expanding rolling-origin category-allocation forecasts by output type, followed by an exact year-qualified citation audit of adopted Measure body texts",
        "results": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PASS: wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
