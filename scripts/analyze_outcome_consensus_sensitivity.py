#!/usr/bin/env python3
"""Rerun headline formal-output analyses with consensus-coded concerns.

The main outcome models use a probability distribution from the title
classifier. This sensitivity replaces that distribution with a one-hot vector
at the final blinded consensus label for every codable output in the complete
95-output lineage set. It leaves the paper--output links and prior-meeting
concern spaces unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import build_graphs
from scripts import explore_lineage_space as lineage
import scripts.analyze_attention_to_outcomes as base
import scripts.analyze_exposure_corrected_contribution as conversion
import scripts.analyze_space_discrimination as discrimination
from utils import compute_product_space, get_rca


OUTDIR = ROOT / "output" / "outcome_linkage"
FINAL = OUTDIR / "outcome_consensus_final.csv"
SUPPLEMENT = OUTDIR / "outcome_consensus_supplement_final.csv"
SCOPE = OUTDIR / "outcome_consensus_validation_scope_key.csv"
SPECIAL = {"INSUFFICIENT_TITLE", "OUTSIDE_TAXONOMY"}


def consensus_inputs(topics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(FINAL, keep_default_na=False)
    scope = pd.read_csv(SCOPE)
    final = final.merge(
        scope[["validation_id", "outcome_id", "in_headline_lineage_95"]],
        on=["validation_id", "outcome_id"], validate="one_to_one",
    )
    if SUPPLEMENT.exists():
        supplement = pd.read_csv(SUPPLEMENT, keep_default_na=False)
        supplement["in_headline_lineage_95"] = False
        final = pd.concat([final, supplement], ignore_index=True)
    original = pd.read_csv(OUTDIR / "outcome_topic_predictions.csv")
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
    selected = final[final["outcome_id"].isin(lineage_ids)]
    selected = selected[
        ~selected["consensus_primary"].isin(SPECIAL)
    ].copy()
    selected = selected.merge(
        original.drop(columns=["topic_top1", "topic_top2", "topic_top3"]),
        on=["outcome_id", "year", "instrument", "title"],
        validate="one_to_one",
    )
    selected["topic_top1"] = selected["consensus_primary"]
    selected["topic_top2"] = selected["consensus_secondary"].where(
        selected["consensus_secondary"].ne(""), selected["consensus_primary"]
    )
    selected["topic_top3"] = selected["topic_top2"]
    selected["probability_top1"] = 1.0
    selected["margin_top1_top2"] = np.nan
    selected["high_confidence"] = selected["consensus_confidence"].eq("high")

    probability_rows = []
    for row in selected.itertuples(index=False):
        for topic in topics:
            probability_rows.append(
                {
                    "outcome_id": row.outcome_id,
                    "year": int(row.year),
                    "meeting": int(row.meeting),
                    "instrument": row.instrument,
                    "topic": topic,
                    "probability": float(topic == row.consensus_primary),
                }
            )
    return selected, pd.DataFrame(probability_rows)


def main() -> None:
    _, _, _, counts, _ = build_graphs()
    topics = list(counts.index)
    pooled_phi = compute_product_space(get_rca(counts)).reindex(index=topics, columns=topics)
    np.fill_diagonal(pooled_phi.values, 1.0)
    topic_lookup = lineage._canonical_topic_lookup(topics)
    predictions, probabilities = consensus_inputs(topics)
    predictions.to_csv(OUTDIR / "outcome_consensus_lineage_predictions.csv", index=False)
    probabilities.to_csv(OUTDIR / "outcome_consensus_lineage_probabilities.csv", index=False)

    training, paper_categories = base.load_paper_training(topics)
    paper_titles = dict(zip(training["paper_id"], training["title"]))
    submitted = base.load_submitted_with_fallback()
    meetings = sorted(predictions["meeting"].dropna().astype(int).unique())
    phi_by_meeting = base.cumulative_phi_by_meeting(
        submitted, topics, topic_lookup, meetings
    )

    candidate_panel = discrimination.build_candidate_panel(
        topics,
        paper_categories,
        predictions,
        probabilities,
        phi_by_meeting,
        paper_titles=paper_titles,
    )
    candidate_panel.to_csv(
        OUTDIR / "space_discrimination_panel_consensus.csv", index=False
    )
    auc_tests = pd.DataFrame(
        [
            discrimination.discrimination_auc(
                candidate_panel, "expected_proximity", "adoption_linked",
                "Adoption-linked papers, full proximity",
            ),
            discrimination.discrimination_auc(
                candidate_panel, "expected_proximity", "discussion_linked",
                "Discussion-only papers, full proximity",
            ),
            discrimination.discrimination_auc(
                candidate_panel, "related_concern_proximity", "adoption_linked",
                "Adoption-linked papers, off-label geometry only",
            ),
            discrimination.discrimination_auc(
                candidate_panel, "related_concern_proximity", "adoption_linked",
                "Adoption-linked papers, off-label geometry, exact matches removed",
                restrict_off_label=True,
            ),
            discrimination.discrimination_auc(
                candidate_panel, "related_concern_proximity", "discussion_linked",
                "Discussion-only papers, off-label geometry, exact matches removed",
                restrict_off_label=True,
            ),
            discrimination.discrimination_auc(
                candidate_panel, "same_concern_mass", "adoption_linked",
                "Adoption-linked papers, same-concern match only",
            ),
            discrimination.discrimination_auc(
                candidate_panel, "expected_proximity", "adoption_linked",
                "Adoption-linked papers, excluding site administration",
                exclude_site_administration=True,
            ),
            discrimination.discrimination_auc(
                candidate_panel, "expected_proximity", "adoption_linked",
                "Adoption-linked papers, title overlap below 0.30",
                max_title_overlap=0.30,
            ),
            discrimination.discrimination_auc(
                candidate_panel, "expected_proximity", "adoption_linked",
                "Adoption-linked papers, title overlap below 0.15",
                max_title_overlap=0.15,
            ),
        ]
    )
    auc_tests.to_csv(OUTDIR / "space_discrimination_auc_consensus.csv", index=False)

    race = pd.concat(
        [
            discrimination.race_conditional_logit(
                candidate_panel, "adoption_linked", "adoption_linked_papers",
                bootstrap=False,
            ),
            discrimination.race_conditional_logit(
                candidate_panel, "discussion_linked", "discussion_only_papers",
                bootstrap=False,
            ),
            discrimination.race_conditional_logit(
                candidate_panel,
                "adoption_linked",
                "adoption_linked_papers_title_overlap_controlled",
                bootstrap=False,
                terms=[
                    "same_concern_mass", "related_concern_proximity",
                    "title_overlap",
                ],
            ),
        ],
        ignore_index=True,
    )
    race.to_csv(OUTDIR / "space_discrimination_race_consensus.csv", index=False)

    actor_state = base.actor_outcome_panel(
        submitted,
        predictions,
        probabilities,
        paper_categories,
        topics,
        topic_lookup,
    )
    actor_state.to_csv(
        OUTDIR / "actor_outcome_candidate_panel_consensus.csv", index=False
    )
    conversion_panel = conversion.build_panel(candidate_panel, actor_state)
    conversion_model = conversion.fit_model(conversion_panel)
    conversion_panel.to_csv(
        OUTDIR / "actor_output_conversion_panel_consensus.csv", index=False
    )
    conversion_model.to_csv(
        OUTDIR / "actor_output_conversion_model_consensus.csv", index=False
    )

    # A matched classifier comparison isolates relabelling from the omission
    # of titles that the blind consensus could not assign. The original
    # classifier tables include those ambiguous outputs, so comparing them
    # directly with the consensus tables would mix two changes.
    codable_lineage_ids = set(candidate_panel["outcome_id"])
    classifier_panel = pd.read_csv(OUTDIR / "space_discrimination_panel.csv")
    classifier_panel = classifier_panel[
        classifier_panel["outcome_id"].isin(codable_lineage_ids)
    ].copy()
    classifier_auc = pd.DataFrame(
        [
            discrimination.discrimination_auc(
                classifier_panel, "expected_proximity", "adoption_linked",
                "Adoption-linked papers, full proximity",
            ),
            discrimination.discrimination_auc(
                classifier_panel, "expected_proximity", "discussion_linked",
                "Discussion-only papers, full proximity",
            ),
            discrimination.discrimination_auc(
                classifier_panel, "same_concern_mass", "adoption_linked",
                "Adoption-linked papers, same-concern mass only",
            ),
            discrimination.discrimination_auc(
                classifier_panel, "related_concern_proximity", "adoption_linked",
                "Adoption-linked papers, off-label geometry, exact matches removed",
                restrict_off_label=True,
            ),
        ]
    )
    classifier_auc.to_csv(
        OUTDIR / "space_discrimination_auc_classifier_matched.csv", index=False
    )
    classifier_race = pd.concat(
        [
            discrimination.race_conditional_logit(
                classifier_panel, "adoption_linked", "adoption_linked_papers",
                bootstrap=False,
            ),
            discrimination.race_conditional_logit(
                classifier_panel,
                "adoption_linked",
                "adoption_linked_papers_title_overlap_controlled",
                bootstrap=False,
                terms=[
                    "same_concern_mass", "related_concern_proximity",
                    "title_overlap",
                ],
            ),
        ],
        ignore_index=True,
    )
    classifier_race.to_csv(
        OUTDIR / "space_discrimination_race_classifier_matched.csv", index=False
    )
    classifier_actor_state = pd.read_csv(
        OUTDIR / "actor_outcome_candidate_panel_independent.csv"
    )
    classifier_conversion_panel = conversion.build_panel(
        classifier_panel, classifier_actor_state
    )
    classifier_conversion_model = conversion.fit_model(classifier_conversion_panel)
    classifier_conversion_model.to_csv(
        OUTDIR / "actor_output_conversion_model_classifier_matched.csv", index=False
    )

    direct_edges, direct_tests = base.direct_edge_analysis(
        paper_categories,
        predictions,
        probabilities,
        pooled_phi,
        "decision_map_verified.json",
        phi_by_meeting=phi_by_meeting,
        geometry="cumulative_prior_meeting_space_consensus_labels",
    )
    direct_edges.to_csv(OUTDIR / "direct_edge_outcome_proximity_consensus.csv", index=False)
    (OUTDIR / "direct_edge_proximity_tests_consensus.json").write_text(
        json.dumps(direct_tests, indent=2) + "\n"
    )

    summary = {
        "coded_lineage_outputs": int(len(predictions)),
        "abstained_lineage_outputs": int(157 - len(predictions)),
        "discrimination": auc_tests.to_dict(orient="records"),
        "race": race.to_dict(orient="records"),
        "conversion": conversion_model.to_dict(orient="records"),
        "matched_classifier_discrimination": classifier_auc.to_dict(orient="records"),
        "matched_classifier_race": classifier_race.to_dict(orient="records"),
        "matched_classifier_conversion": classifier_conversion_model.to_dict(orient="records"),
        "direct_edge_tests": direct_tests,
        "proximity_direction": "larger phi means closer concerns",
    }
    (OUTDIR / "outcome_consensus_headline_sensitivity.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(auc_tests.to_string(index=False))
    print("\nConditional race")
    print(race.to_string(index=False))
    print("\nExposure-corrected conversion")
    print(conversion_model.to_string(index=False))


if __name__ == "__main__":
    main()
