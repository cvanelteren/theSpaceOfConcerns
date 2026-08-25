#!/usr/bin/env python3
"""Validate the independent audit and prepare anonymous arbitration cases."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
PROVISIONAL = OUTDIR / "outcome_consensus_provisional.csv"
AUDIT = OUTDIR / "outcome_consensus_audit.csv"
OUT_REVISIONS = OUTDIR / "outcome_consensus_revisions_blind.csv"
OUT_FINAL = OUTDIR / "outcome_consensus_final.csv"
RANDOM_SEED = 20260814
SPECIAL = {"INSUFFICIENT_TITLE", "OUTSIDE_TAXONOMY"}


def allowed_topics() -> set[str]:
    probabilities = pd.read_csv(OUTDIR / "outcome_topic_probabilities.csv")
    return set(probabilities["topic"]) | SPECIAL


def main() -> None:
    provisional = pd.read_csv(PROVISIONAL, keep_default_na=False)
    audit = pd.read_csv(AUDIT, keep_default_na=False)
    required = [
        "validation_id", "outcome_id", "audit_decision", "auditor_primary",
        "auditor_secondary", "auditor_confidence", "audit_rationale",
    ]
    if list(audit.columns) != required:
        raise ValueError(f"Audit columns must be exactly {required}")
    expected = provisional.set_index("validation_id")["outcome_id"].sort_index()
    observed = audit.set_index("validation_id")["outcome_id"].sort_index()
    if len(audit) != len(provisional) or not audit["validation_id"].is_unique or not expected.equals(observed):
        raise ValueError("Audit IDs do not exactly match provisional consensus")
    allowed = allowed_topics()
    if not set(audit["audit_decision"]).issubset({"accept", "revise"}):
        raise ValueError("Invalid audit decision")
    if not set(audit["auditor_primary"]).issubset(allowed):
        raise ValueError("Invalid auditor primary label")
    if not (set(audit["auditor_secondary"]) - {""}).issubset(allowed - SPECIAL):
        raise ValueError("Invalid auditor secondary label")
    if not set(audit["auditor_confidence"]).issubset({"high", "medium", "low"}):
        raise ValueError("Invalid auditor confidence")
    if audit[["auditor_primary", "auditor_confidence", "audit_rationale"]].eq("").any().any():
        raise ValueError("Audit is missing required values")

    comparison = provisional.merge(audit, on=["validation_id", "outcome_id"], validate="one_to_one")
    same_pair = comparison["consensus_primary"].eq(comparison["auditor_primary"]) & comparison[
        "consensus_secondary"
    ].eq(comparison["auditor_secondary"])
    if (comparison["audit_decision"].eq("accept") & ~same_pair).any():
        raise ValueError("An accepted audit row changes the proposed label")
    if (comparison["audit_decision"].eq("revise") & same_pair).any():
        raise ValueError("A revised audit row does not change the proposed label")

    revisions = comparison[comparison["audit_decision"].eq("revise")].copy()
    if revisions.empty:
        final = provisional.copy()
        final["consensus_source"] = final["consensus_source"] + "+independent_audit"
        final.to_csv(OUT_FINAL, index=False)
        pd.DataFrame(
            columns=[
                "validation_id", "outcome_id", "year", "instrument", "title",
                "option_a_primary", "option_a_secondary", "option_b_primary",
                "option_b_secondary",
            ]
        ).to_csv(OUT_REVISIONS, index=False)
        print(f"Audit accepted all {len(final)} titles; final consensus written")
        return

    rng = np.random.default_rng(RANDOM_SEED)
    swap = rng.integers(0, 2, size=len(revisions)).astype(bool)
    packet_rows = []
    for use_swap, row in zip(swap, revisions.itertuples(index=False)):
        original = (row.consensus_primary, row.consensus_secondary)
        challenge = (row.auditor_primary, row.auditor_secondary)
        option_a, option_b = (challenge, original) if use_swap else (original, challenge)
        packet_rows.append(
            {
                "validation_id": row.validation_id,
                "outcome_id": row.outcome_id,
                "year": row.year,
                "instrument": row.instrument,
                "title": row.title,
                "option_a_primary": option_a[0],
                "option_a_secondary": option_a[1],
                "option_b_primary": option_b[0],
                "option_b_secondary": option_b[1],
            }
        )
    pd.DataFrame(packet_rows).to_csv(OUT_REVISIONS, index=False)
    print(
        f"Audit accepted {len(comparison) - len(revisions)}/{len(comparison)}; "
        f"wrote {len(revisions)} anonymous arbitration cases"
    )


if __name__ == "__main__":
    main()
