#!/usr/bin/env python3
"""Apply blinded arbitration to the labels challenged by the audit."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
PROVISIONAL = OUTDIR / "outcome_consensus_provisional.csv"
AUDIT = OUTDIR / "outcome_consensus_audit.csv"
CASES = OUTDIR / "outcome_consensus_revisions_blind.csv"
ARBITRATION = OUTDIR / "outcome_consensus_arbitration.csv"
OUT_FINAL = OUTDIR / "outcome_consensus_final.csv"
SPECIAL = {"INSUFFICIENT_TITLE", "OUTSIDE_TAXONOMY"}


def allowed_topics() -> set[str]:
    probabilities = pd.read_csv(OUTDIR / "outcome_topic_probabilities.csv")
    return set(probabilities["topic"]) | SPECIAL


def main() -> None:
    provisional = pd.read_csv(PROVISIONAL, keep_default_na=False)
    audit = pd.read_csv(AUDIT, keep_default_na=False)
    cases = pd.read_csv(CASES, keep_default_na=False)
    arbitration = pd.read_csv(ARBITRATION, keep_default_na=False)
    required = [
        "validation_id", "outcome_id", "arbitrated_primary",
        "arbitrated_secondary", "arbitrated_confidence", "candidate_cases",
        "arbitration_rationale",
    ]
    if list(arbitration.columns) != required:
        raise ValueError(f"Arbitration columns must be exactly {required}")
    expected = cases.set_index("validation_id")["outcome_id"].sort_index()
    observed = arbitration.set_index("validation_id")["outcome_id"].sort_index()
    if len(arbitration) != len(cases) or not arbitration["validation_id"].is_unique or not expected.equals(observed):
        raise ValueError("Arbitration IDs do not exactly match the challenged cases")
    allowed = allowed_topics()
    if not set(arbitration["arbitrated_primary"]).issubset(allowed):
        raise ValueError("Invalid arbitrated primary label")
    if not (set(arbitration["arbitrated_secondary"]) - {""}).issubset(allowed - SPECIAL):
        raise ValueError("Invalid arbitrated secondary label")
    if not set(arbitration["arbitrated_confidence"]).issubset({"high", "medium", "low"}):
        raise ValueError("Invalid arbitrated confidence")
    if arbitration[["arbitrated_primary", "arbitrated_confidence", "candidate_cases", "arbitration_rationale"]].eq("").any().any():
        raise ValueError("Arbitration is missing required values")

    # Consensus requires the independent arbitrator to endorse one of the two
    # independently proposed label pairs. A third pair would trigger another
    # round rather than silently becoming final.
    checked = cases.merge(arbitration, on=["validation_id", "outcome_id"], validate="one_to_one")
    agrees_a = checked["arbitrated_primary"].eq(checked["option_a_primary"]) & checked[
        "arbitrated_secondary"
    ].eq(checked["option_a_secondary"])
    agrees_b = checked["arbitrated_primary"].eq(checked["option_b_primary"]) & checked[
        "arbitrated_secondary"
    ].eq(checked["option_b_secondary"])
    if (~(agrees_a | agrees_b)).any():
        unresolved = checked.loc[~(agrees_a | agrees_b), ["validation_id", "outcome_id"]]
        unresolved.to_csv(OUTDIR / "outcome_consensus_unresolved_after_arbitration.csv", index=False)
        raise ValueError(
            f"{len(unresolved)} arbitration choices match neither independent proposal; another blind round is required"
        )

    final = provisional.copy().set_index("validation_id")
    audit_index = audit.set_index("validation_id")
    for row in arbitration.itertuples(index=False):
        final.loc[row.validation_id, "consensus_primary"] = row.arbitrated_primary
        final.loc[row.validation_id, "consensus_secondary"] = row.arbitrated_secondary
        final.loc[row.validation_id, "consensus_confidence"] = row.arbitrated_confidence
        final.loc[row.validation_id, "consensus_source"] = "independent_challenge+blind_arbitration"
    accepted_ids = audit_index.index[audit_index["audit_decision"].eq("accept")]
    final.loc[accepted_ids, "consensus_source"] = (
        final.loc[accepted_ids, "consensus_source"] + "+independent_audit"
    )
    final.reset_index().to_csv(OUT_FINAL, index=False)
    print(
        f"Final consensus written for {len(final)} titles: "
        f"{len(accepted_ids)} audit-accepted and {len(arbitration)} arbitrated"
    )


if __name__ == "__main__":
    main()
