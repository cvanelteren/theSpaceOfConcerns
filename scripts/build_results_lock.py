#!/usr/bin/env python3
"""Assemble the current, analysis-only results lock.

This file deliberately does not read or edit the manuscript.  It gathers the
machine-readable outputs that define the admissible claims after the archive
category and paper--output lineage audits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "results_lock.json"


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text())


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row(table: pd.DataFrame, **conditions) -> dict:
    keep = pd.Series(True, index=table.index)
    for column, value in conditions.items():
        keep &= table[column].eq(value)
    selected = table[keep]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(selected)}")
    return selected.iloc[0].to_dict()


def main() -> None:
    assignment = load_json("output/primary_concern_assignment.json")
    measurement = pd.read_csv(
        ROOT / "output/primary_concern_measurement_sensitivity.csv"
    )
    edge_stability = load_json("output/primary_concern_edge_stability.json")
    edge_rows = pd.read_csv(ROOT / "output/primary_concern_edge_stability.csv")
    semantic = load_json("output/semantic_baseline.json")
    section_diagnostic = load_json("output/section_label_diagnostic.json")
    availability = pd.read_csv(ROOT / "output/hazard_availability_sensitivity.csv")
    entry = pd.read_csv(ROOT / "output/entry_definition_robustness.csv")
    windows = pd.read_csv(ROOT / "output/entry_window_sensitivity.csv")
    phases = pd.read_csv(ROOT / "output/direct_entry_phase_sensitivity.csv")
    dependence = load_json("output/hazard_robustness_dependence.json")
    paired = load_json("output/paired_null_robustness.json")
    circularity = pd.read_csv(ROOT / "output/hazard_circularity_coefficients.csv")
    output_summary = load_json("output/outcome_linkage/analysis_summary.json")
    classifier = load_json("output/outcome_linkage/title_classifier_metrics.json")
    consensus_validation = load_json(
        "output/outcome_linkage/outcome_consensus_validation_summary.json"
    )
    output_models = pd.read_csv(
        ROOT / "output/outcome_linkage/attention_outcome_ppml.csv"
    )
    accumulation = pd.read_csv(
        ROOT / "output/outcome_linkage/attention_accumulation_models.csv"
    )
    consensus_attention = pd.read_csv(
        ROOT / "output/outcome_linkage/consensus_attention_sensitivity.csv"
    )

    main_measurement = row(measurement, category_treatment="inferred_primary")
    liability = edge_rows[
        edge_rows.apply(
            lambda item: {item["concern_a"], item["concern_b"]}
            == {"Inspections", "Liability"},
            axis=1,
        )
    ]
    if len(liability) != 1:
        raise ValueError("Expected exactly one Inspections--Liability edge")

    prior_available = row(
        availability, risk_set="appeared_by_prior_window_end"
    )
    output_scope = output_summary["scope"]
    if output_scope["regular_atcm_outputs"] != 584:
        raise ValueError("Regular ATCM output inventory changed")
    if assignment["papers_with_primary_concern"] != 6573:
        raise ValueError("Primary-concern paper inventory changed")
    if load_json("output/hazard_conditional_logit_meta.json")["period_col"] != "meeting number":
        raise ValueError("Locality analysis is not indexed by ATCM meeting order")

    ppml_specs = {
        name: {
            predictor: row(
                output_models,
                specification=name,
                predictor=predictor,
            )
            for predictor in predictors
        }
        for name, predictors in {
            "prospective_prior3_meetings_soft_assignment": [
                "papers_prior3", "neighbor_papers_prior3", "outcomes_prior3",
                "papers_prior3_minus_neighbor_papers_prior3",
            ],
            "prospective_prior3_meetings_hard_top1": [
                "papers_prior3", "neighbor_papers_prior3", "outcomes_prior3",
                "papers_prior3_minus_neighbor_papers_prior3",
            ],
            "prospective_prior3_meetings_high_confidence": [
                "papers_prior3", "neighbor_papers_prior3", "outcomes_prior3",
            ],
            "prospective_prior3_meetings_measures": [
                "papers_prior3", "neighbor_papers_prior3", "outcomes_prior3",
            ],
            "prospective_prior3_meetings_decisions": [
                "papers_prior3", "neighbor_papers_prior3", "outcomes_prior3",
            ],
            "prospective_prior3_meetings_resolutions": [
                "papers_prior3", "neighbor_papers_prior3", "outcomes_prior3",
            ],
        }.items()
    }

    onset_specs = {
        specification: {
            predictor: row(
                accumulation,
                specification=specification,
                predictor=predictor,
            )
            for predictor in [
                "papers_prior5",
                "nearby_prior5",
                "papers_prior5_minus_nearby_prior5",
            ]
        }
        for specification in [
            "new_output_episode_after_five_quiet_meetings",
            "new_high_confidence_output_episode_after_five_quiet_meetings",
            "new_output_episode_after_five_quiet_meetings_excluding_site_administration",
        ]
    }

    payload = {
        "lock_version": 1,
        "purpose": "analysis lock before any further manuscript editing",
        "input_hashes": {
            "multilabel_archive": sha256("data/document-summary-multilabel.parquet"),
            "inferred_primary_archive": sha256(
                "data/document-summary-primary-concern.parquet"
            ),
            "regular_output_inventory": sha256("../ats_lineage/decision_map.json"),
        },
        "archive_measurement": assignment,
        "concern_space": {
            "primary_network": {
                key: main_measurement[key]
                for key in [
                    "papers", "positive_pair_share", "mean_positive_proximity",
                    "louvain_modularity_mean_100_seeds",
                    "louvain_modularity_max_100_seeds",
                    "louvain_communities_at_max",
                ]
            },
            "construction_sensitivity": measurement.to_dict(orient="records"),
            "edge_bootstrap": edge_stability,
            "inspections_liability_example": liability.iloc[0].to_dict(),
            "literal_wording_baseline": semantic,
            "manual_section_diagnostic": section_diagnostic,
        },
        "portfolio_entry": {
            "primary_prospective_rpa_crossing": prior_available,
            "observable_entry_definitions": entry.to_dict(orient="records"),
            "window_sensitivity": windows.to_dict(orient="records"),
            "nonoverlapping_meeting_sequences": phases.to_dict(orient="records"),
            "category_treatment_sensitivity": measurement[
                [
                    "category_treatment", "odds_ratio_per_0_1",
                    "odds_ratio_per_0_1_ci_low", "odds_ratio_per_0_1_ci_high",
                    "new_document_odds_ratio_per_0_1",
                    "new_document_odds_ratio_per_0_1_ci_low",
                    "new_document_odds_ratio_per_0_1_ci_high",
                ]
            ].to_dict(orient="records"),
            "actor_and_window_dependence": dependence,
            "matched_popularity_null": {
                key: value
                for key, value in paired.items()
                if key.startswith("lagged_popularity_prior")
            },
            "space_circularity_checks": circularity.to_dict(orient="records"),
        },
        "formal_outputs": {
            "scope": output_scope,
            "cross_fitted_title_classifier": classifier,
            "automated_coder_audit": {
                "warning": (
                    "Automated same-family blind coding, not independent human validation"
                ),
                "design_weighted_metrics": consensus_validation[
                    "design_weighted_metrics"
                ],
            },
            "three_meeting_models": ppml_specs,
            "quiet_period_onset_models": onset_specs,
            "automated_consensus_sample_sensitivity": {
                "warning": (
                    "83 codable stratified titles; sampling-weighted sensitivity only"
                ),
                "rows": consensus_attention.to_dict(orient="records"),
            },
        },
        "claim_status": {
            "robust": [
                "The official archive returns some papers under multiple concern categories.",
                "The full concern network is dense and only weakly modular; it does not contain three natural blocs.",
                "An actor's next documentary entry is more likely near its recent specialized portfolio.",
                "Immediate locality survives category treatment, a direct-paper entry definition, topic age, actor removal, co-sponsor weighting, and states-only analysis.",
                "The locality gradient weakens as the inactivity horizon grows.",
            ],
            "bounded_or_exploratory": [
                "Named edges can illustrate the geometry only when their bootstrap stability is reported.",
                "Formal-output associations vary across output coding, time horizon, and instrument type.",
                "For 41 cross-fitted hard-label output onsets after five quiet meetings, focal attention exceeds nearby attention; the high-confidence subset points the same way but is imprecise.",
                "The automated consensus sample instead places more weight on nearby attention, so the output mechanism is not locked to a single focal-only pathway.",
            ],
            "not_supported": [
                "Three reproducible regimes or natural concern modules.",
                "Durable local path dependence across a full subsequent five-meeting block.",
                "A general rule that nearby concern activity directly produces formal output.",
                "A general rule that formal output follows only repeated focal-concern attention.",
                "Causal influence, negotiating intent, implementation, or environmental effectiveness.",
                "Any paper-to-output lineage generated by the year-stripping Cartesian parser.",
            ],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
