#!/usr/bin/env python3
"""Build the blind supplement covering all lineage-comparison outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
FINAL_200 = OUTDIR / "outcome_consensus_final.csv"
PREDICTIONS = OUTDIR / "outcome_topic_predictions.csv"
EDGES = OUTDIR / "direct_edge_outcome_proximity.csv"
OUT_BLIND = OUTDIR / "outcome_consensus_supplement_blind.csv"
OUT_SCOPE = OUTDIR / "outcome_consensus_supplement_scope_key.csv"

RELATIONS = {
    "direct_adoption_or_approval",
    "documented_contribution",
    "direct_proposal_or_discussion",
}


def main() -> None:
    final = pd.read_csv(FINAL_200)
    predictions = pd.read_csv(PREDICTIONS)
    edges = pd.read_csv(EDGES)
    verified = edges[
        edges["graph"].eq("decision_map_verified.json")
        & edges["geometry"].eq("cumulative_prior_meeting_space")
        & edges["relation"].isin(RELATIONS)
    ].copy()
    lineage_ids = set(verified["outcome_id"])
    missing_ids = lineage_ids - set(final["outcome_id"])
    selected = predictions.loc[
        predictions["outcome_id"].isin(missing_ids),
        ["outcome_id", "year", "instrument", "title"],
    ].drop_duplicates("outcome_id")
    if len(selected) != len(missing_ids):
        missing = sorted(missing_ids - set(selected["outcome_id"]))
        raise ValueError(f"Missing lineage output titles: {missing}")
    selected = selected.sort_values(
        ["instrument", "year", "outcome_id"], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "validation_id", [f"OS{i:03d}" for i in range(1, len(selected) + 1)])
    selected.to_csv(OUT_BLIND, index=False)

    adoption_ids = set(
        verified.loc[
            verified["relation"].isin(
                {"direct_adoption_or_approval", "documented_contribution"}
            ),
            "outcome_id",
        ]
    )
    discussion_ids = set(
        verified.loc[
            verified["relation"].eq("direct_proposal_or_discussion"),
            "outcome_id",
        ]
    )
    scope = selected[["validation_id", "outcome_id"]].copy()
    scope["in_adoption_95"] = scope["outcome_id"].isin(adoption_ids)
    scope["in_discussion_85"] = scope["outcome_id"].isin(discussion_ids)
    scope["in_lineage_union_157"] = scope["outcome_id"].isin(lineage_ids)
    scope.to_csv(OUT_SCOPE, index=False)
    print(f"Wrote {len(selected)} supplemental blind titles")
    print(f"Complete verified lineage union: {len(lineage_ids)} outputs")
    print(f"Previously coded: {len(lineage_ids & set(final['outcome_id']))}")


if __name__ == "__main__":
    main()
