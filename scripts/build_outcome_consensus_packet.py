#!/usr/bin/env python3
"""Build the blinded packet for final outcome-title validation.

The packet combines the pre-specified 120-outcome stratified audit with every
formal output used by the headline adoption/contribution lineage analyses.  It
contains no classifier prediction, probability, confidence band, paper link,
or lineage status.  A separate scope key is retained for analysis only.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"

BLIND_120 = OUTDIR / "outcome_topic_validation_blind.csv"
PREDICTIONS = OUTDIR / "outcome_topic_predictions.csv"
DIRECT_EDGES = OUTDIR / "direct_edge_outcome_proximity.csv"
OUT_BLIND = OUTDIR / "outcome_consensus_validation_blind.csv"
OUT_SCOPE = OUTDIR / "outcome_consensus_validation_scope_key.csv"

HEADLINE_RELATIONS = {
    "direct_adoption_or_approval",
    "documented_contribution",
}


def main() -> None:
    blind_120 = pd.read_csv(BLIND_120)
    predictions = pd.read_csv(PREDICTIONS)
    edges = pd.read_csv(DIRECT_EDGES)

    audit_ids = set(blind_120["outcome_id"])
    headline_ids = set(
        edges.loc[edges["relation"].isin(HEADLINE_RELATIONS), "outcome_id"]
    )
    selected_ids = audit_ids | headline_ids

    selected = predictions.loc[
        predictions["outcome_id"].isin(selected_ids),
        ["outcome_id", "year", "instrument", "title"],
    ].drop_duplicates("outcome_id")
    if len(selected) != len(selected_ids):
        missing = sorted(selected_ids - set(selected["outcome_id"]))
        raise ValueError(f"Missing selected outputs: {missing}")

    selected = selected.sort_values(
        ["instrument", "year", "outcome_id"], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "validation_id", [f"OV{i:03d}" for i in range(1, len(selected) + 1)])
    selected.to_csv(OUT_BLIND, index=False)

    scope = selected[["validation_id", "outcome_id"]].copy()
    scope["in_stratified_120"] = scope["outcome_id"].isin(audit_ids)
    scope["in_headline_lineage_95"] = scope["outcome_id"].isin(headline_ids)
    scope.to_csv(OUT_SCOPE, index=False)

    print(f"Wrote {len(selected)} blinded titles to {OUT_BLIND}")
    print(f"  stratified audit: {scope['in_stratified_120'].sum()}")
    print(f"  headline lineage: {scope['in_headline_lineage_95'].sum()}")
    print(f"  overlap: {(scope['in_stratified_120'] & scope['in_headline_lineage_95']).sum()}")


if __name__ == "__main__":
    main()
