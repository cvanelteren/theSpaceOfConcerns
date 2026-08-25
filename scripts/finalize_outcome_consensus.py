#!/usr/bin/env python3
"""Combine unanimous blind coding and adversarial adjudication."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
BLIND = OUTDIR / "outcome_consensus_validation_blind.csv"
UNANIMOUS = OUTDIR / "outcome_consensus_unanimous.csv"
ADJUDICATED = OUTDIR / "outcome_consensus_adjudicated.csv"
DISAGREEMENTS = OUTDIR / "outcome_consensus_disagreements_blind.csv"
OUT = OUTDIR / "outcome_consensus_provisional.csv"

SPECIAL = {"INSUFFICIENT_TITLE", "OUTSIDE_TAXONOMY"}


def allowed_topics() -> set[str]:
    probabilities = pd.read_csv(OUTDIR / "outcome_topic_probabilities.csv")
    return set(probabilities["topic"]) | SPECIAL


def main() -> None:
    blind = pd.read_csv(BLIND, keep_default_na=False)
    unanimous = pd.read_csv(UNANIMOUS, keep_default_na=False)
    disagreements = pd.read_csv(DISAGREEMENTS, keep_default_na=False)
    adjudicated = pd.read_csv(ADJUDICATED, keep_default_na=False)

    required = [
        "validation_id", "outcome_id", "adjudicated_primary",
        "adjudicated_secondary", "adjudicated_confidence", "candidate_cases",
        "adjudication_rationale",
    ]
    if list(adjudicated.columns) != required:
        raise ValueError(f"Adjudication columns must be exactly {required}")
    if not adjudicated["validation_id"].is_unique:
        raise ValueError("Adjudication contains duplicate validation IDs")
    expected = disagreements.set_index("validation_id")["outcome_id"].sort_index()
    observed = adjudicated.set_index("validation_id")["outcome_id"].sort_index()
    if not expected.equals(observed):
        raise ValueError("Adjudication IDs do not exactly match the disagreements")

    allowed = allowed_topics()
    if not set(adjudicated["adjudicated_primary"]).issubset(allowed):
        raise ValueError("Adjudication contains an invalid primary label")
    secondaries = set(adjudicated["adjudicated_secondary"]) - {""}
    if not secondaries.issubset(allowed - SPECIAL):
        raise ValueError("Adjudication contains an invalid secondary label")
    if not set(adjudicated["adjudicated_confidence"]).issubset(
        {"high", "medium", "low"}
    ):
        raise ValueError("Adjudication contains invalid confidence values")
    if adjudicated[
        ["adjudicated_primary", "adjudicated_confidence", "candidate_cases",
         "adjudication_rationale"]
    ].eq("").any().any():
        raise ValueError("Adjudication is missing required values")

    adjudicated_out = adjudicated.rename(
        columns={
            "adjudicated_primary": "consensus_primary",
            "adjudicated_secondary": "consensus_secondary",
            "adjudicated_confidence": "consensus_confidence",
        }
    )[
        [
            "validation_id", "outcome_id", "consensus_primary",
            "consensus_secondary", "consensus_confidence",
        ]
    ]
    adjudicated_out["consensus_source"] = "adversarial_adjudication"

    consensus = pd.concat([unanimous, adjudicated_out], ignore_index=True)
    if len(consensus) != len(blind) or not consensus["validation_id"].is_unique:
        raise ValueError("Provisional consensus must contain every title exactly once")
    expected_all = blind.set_index("validation_id")["outcome_id"].sort_index()
    observed_all = consensus.set_index("validation_id")["outcome_id"].sort_index()
    if not expected_all.equals(observed_all):
        raise ValueError("Provisional consensus IDs do not match the blind packet")

    consensus = blind.merge(
        consensus, on=["validation_id", "outcome_id"], validate="one_to_one"
    )
    consensus.to_csv(OUT, index=False)
    print(
        f"Provisional consensus: {len(consensus)} titles; "
        f"{len(unanimous)} unanimous; {len(adjudicated_out)} adjudicated"
    )


if __name__ == "__main__":
    main()
