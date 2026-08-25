#!/usr/bin/env python3
"""Validate independent outcome codings and prepare blind disagreements."""

from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
BLIND = OUTDIR / "outcome_consensus_validation_blind.csv"
CODERS = {
    "a": OUTDIR / "outcome_consensus_coder_a.csv",
    "b": OUTDIR / "outcome_consensus_coder_b.csv",
    "c": OUTDIR / "outcome_consensus_coder_c.csv",
}
OUT_WIDE = OUTDIR / "outcome_consensus_coder_comparison.csv"
OUT_DISAGREE = OUTDIR / "outcome_consensus_disagreements_blind.csv"
OUT_INITIAL = OUTDIR / "outcome_consensus_unanimous.csv"
OUT_AGREEMENT = OUTDIR / "outcome_consensus_intercoder_agreement.csv"

SPECIAL = {"INSUFFICIENT_TITLE", "OUTSIDE_TAXONOMY"}


def allowed_topics() -> set[str]:
    probabilities = pd.read_csv(OUTDIR / "outcome_topic_probabilities.csv")
    return set(probabilities["topic"]) | SPECIAL


def validate_coder(name: str, coded: pd.DataFrame, blind: pd.DataFrame) -> None:
    required = [
        "validation_id", "outcome_id", "primary_concern",
        "secondary_concern", "confidence", "rationale",
    ]
    if list(coded.columns) != required:
        raise ValueError(f"Coder {name}: columns must be exactly {required}")
    if len(coded) != len(blind) or not coded["validation_id"].is_unique:
        raise ValueError(f"Coder {name}: expected {len(blind)} unique rows")
    expected = blind.set_index("validation_id")["outcome_id"].sort_index()
    observed = coded.set_index("validation_id")["outcome_id"].sort_index()
    if not expected.equals(observed):
        raise ValueError(f"Coder {name}: IDs do not match blind packet")
    allowed = allowed_topics()
    bad_primary = sorted(set(coded["primary_concern"].dropna()) - allowed)
    secondary = set(coded["secondary_concern"].dropna()) - {""}
    bad_secondary = sorted(secondary - (allowed - SPECIAL))
    if bad_primary or bad_secondary:
        raise ValueError(
            f"Coder {name}: bad primary={bad_primary}, secondary={bad_secondary}"
        )
    if not set(coded["confidence"]).issubset({"high", "medium", "low"}):
        raise ValueError(f"Coder {name}: invalid confidence")
    if coded[["primary_concern", "confidence", "rationale"]].isna().any().any():
        raise ValueError(f"Coder {name}: missing required values")


def main() -> None:
    blind = pd.read_csv(BLIND, keep_default_na=False)
    wide = blind.copy()
    coded_by_name = {}
    for name, path in CODERS.items():
        coded = pd.read_csv(path, keep_default_na=False)
        validate_coder(name, coded, blind)
        coded_by_name[name] = coded
        renamed = coded.rename(
            columns={
                "primary_concern": f"primary_{name}",
                "secondary_concern": f"secondary_{name}",
                "confidence": f"confidence_{name}",
                "rationale": f"rationale_{name}",
            }
        ).drop(columns=["outcome_id"])
        wide = wide.merge(renamed, on="validation_id", validate="one_to_one")

    primary_cols = [f"primary_{name}" for name in CODERS]
    wide["n_distinct_primary"] = wide[primary_cols].nunique(axis=1)
    wide["unanimous_primary"] = wide["n_distinct_primary"].eq(1)
    wide["majority_primary"] = wide[primary_cols].mode(axis=1)[0]
    wide["majority_count"] = wide.apply(
        lambda row: max(row[primary_cols].value_counts()), axis=1
    )
    wide.to_csv(OUT_WIDE, index=False)

    unanimous = wide.loc[wide["unanimous_primary"]].copy()
    unanimous_out = unanimous[["validation_id", "outcome_id"]].copy()
    unanimous_out["consensus_primary"] = unanimous["primary_a"]
    unanimous_out["consensus_secondary"] = unanimous.apply(
        lambda row: row["secondary_a"]
        if row["secondary_a"] == row["secondary_b"] == row["secondary_c"]
        else "",
        axis=1,
    )
    unanimous_out["consensus_confidence"] = unanimous.apply(
        lambda row: "low" if "low" in {row["confidence_a"], row["confidence_b"], row["confidence_c"]}
        else ("medium" if "medium" in {row["confidence_a"], row["confidence_b"], row["confidence_c"]} else "high"),
        axis=1,
    )
    unanimous_out["consensus_source"] = "three_coder_unanimous"
    unanimous_out.to_csv(OUT_INITIAL, index=False)

    disagreement_cols = [
        "validation_id", "outcome_id", "year", "instrument", "title",
        *primary_cols,
        *[f"secondary_{name}" for name in CODERS],
        *[f"confidence_{name}" for name in CODERS],
        *[f"rationale_{name}" for name in CODERS],
        "majority_primary", "majority_count", "n_distinct_primary",
    ]
    wide.loc[~wide["unanimous_primary"], disagreement_cols].to_csv(
        OUT_DISAGREE, index=False
    )

    agreement_rows = []
    for left, right in combinations(CODERS, 2):
        lvals = coded_by_name[left]["primary_concern"]
        rvals = coded_by_name[right]["primary_concern"]
        agreement_rows.append(
            {
                "coder_pair": f"{left}-{right}",
                "exact_agreement": float((lvals == rvals).mean()),
                "cohen_kappa": float(cohen_kappa_score(lvals, rvals)),
                "n": int(len(lvals)),
            }
        )
    pd.DataFrame(agreement_rows).to_csv(OUT_AGREEMENT, index=False)

    print(f"Unanimous: {int(wide['unanimous_primary'].sum())}/{len(wide)}")
    print(f"Disagreements: {int((~wide['unanimous_primary']).sum())}")
    print(pd.DataFrame(agreement_rows).to_string(index=False))


if __name__ == "__main__":
    main()
