#!/usr/bin/env python3
"""Merge blind crosswalk coders and prepare or apply adjudication."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.official_regular_atcm_outputs import (  # noqa: E402
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
)


OUTDIR = ROOT / "output" / "scientific_checks"
REFERENCE_PATH = OUTDIR / "crosswalk_blind_validation_packet.csv"
COMPARISON_PATH = OUTDIR / "crosswalk_blind_coder_comparison.csv"
ADJUDICATION_PATH = OUTDIR / "crosswalk_blind_adjudication_packet.csv"
CONSENSUS_PATH = OUTDIR / "crosswalk_blind_consensus.csv"
METRICS_PATH = OUTDIR / "crosswalk_blind_agreement.json"
REQUIRED = {
    "validation_id",
    "paper_concern",
    "family_choices",
    "selected_family",
    "confidence",
    "coder_notes",
}
CONFIDENCE = {"high", "medium", "low"}


def coder_label(path: Path, used: set[str]) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "coder"
    base = label
    suffix = 2
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def load_reference() -> tuple[pd.DataFrame, set[str]]:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {REFERENCE_PATH}; run tools/build_crosswalk_blind_validation.py"
        )
    reference = pd.read_csv(REFERENCE_PATH, dtype=str).fillna("")
    choices = {
        choice
        for values in reference["family_choices"]
        for choice in values.split(" | ")
    }
    expected = set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values())
    if choices != expected:
        raise AssertionError("Reference packet does not contain the 15 official families")
    return reference, choices


def load_coder(
    path: Path,
    label: str,
    reference: pd.DataFrame,
    allowed: set[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame["validation_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate validation_id values")
    expected_ids = set(reference["validation_id"])
    observed_ids = set(frame["validation_id"])
    if observed_ids != expected_ids:
        raise ValueError(
            f"{path}: missing IDs={sorted(expected_ids - observed_ids)}; "
            f"unknown IDs={sorted(observed_ids - expected_ids)}"
        )
    check = reference[["validation_id", "paper_concern", "family_choices"]].merge(
        frame[["validation_id", "paper_concern", "family_choices"]],
        on="validation_id",
        suffixes=("_reference", "_coder"),
    )
    for column in ("paper_concern", "family_choices"):
        if not check[f"{column}_reference"].eq(check[f"{column}_coder"]).all():
            raise ValueError(f"{path}: {column} differs from the blind reference")
    selected = frame["selected_family"].str.strip()
    confidence = frame["confidence"].str.strip().str.lower()
    if selected.eq("").any() or confidence.eq("").any():
        incomplete = frame.loc[
            selected.eq("") | confidence.eq(""), "validation_id"
        ].tolist()
        raise ValueError(f"{path}: incomplete rows {incomplete}")
    invalid = sorted(set(selected) - allowed)
    if invalid:
        raise ValueError(f"{path}: invalid selected families {invalid}")
    invalid_confidence = sorted(set(confidence) - CONFIDENCE)
    if invalid_confidence:
        raise ValueError(f"{path}: invalid confidence values {invalid_confidence}")
    return frame[
        ["validation_id", "selected_family", "confidence", "coder_notes"]
    ].rename(
        columns={
            "selected_family": f"{label}_family",
            "confidence": f"{label}_confidence",
            "coder_notes": f"{label}_notes",
        }
    )


def fleiss_kappa(assignments: np.ndarray, families: list[str]) -> float:
    n_items, n_coders = assignments.shape
    if n_items < 1 or n_coders < 2:
        return float("nan")
    counts = np.zeros((n_items, len(families)), dtype=float)
    family_index = {family: index for index, family in enumerate(families)}
    for row in range(n_items):
        for value in assignments[row]:
            counts[row, family_index[str(value)]] += 1
    observed = np.mean(
        (np.square(counts).sum(axis=1) - n_coders) / (n_coders * (n_coders - 1))
    )
    proportions = counts.sum(axis=0) / (n_items * n_coders)
    expected = float(np.square(proportions).sum())
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def load_adjudication(
    path: Path,
    disagreements: pd.DataFrame,
    allowed: set[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"validation_id", "adjudicated_family", "adjudicator_notes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    expected = set(disagreements["validation_id"])
    observed = set(frame["validation_id"])
    if observed != expected:
        raise ValueError(
            f"{path}: missing disagreement IDs={sorted(expected - observed)}; "
            f"unknown IDs={sorted(observed - expected)}"
        )
    selected = frame["adjudicated_family"].str.strip()
    if selected.eq("").any():
        incomplete = frame.loc[selected.eq(""), "validation_id"].tolist()
        raise ValueError(f"{path}: incomplete adjudication rows {incomplete}")
    invalid = sorted(set(selected) - allowed)
    if invalid:
        raise ValueError(f"{path}: invalid adjudicated families {invalid}")
    return frame[["validation_id", "adjudicated_family", "adjudicator_notes"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coder_files", nargs="+", type=Path)
    parser.add_argument("--adjudication", type=Path)
    args = parser.parse_args()
    if len(args.coder_files) < 2:
        parser.error("Provide at least two independently completed coder files")

    reference, allowed = load_reference()
    labels: list[str] = []
    used: set[str] = set()
    comparison = reference[["validation_id", "paper_concern", "family_choices"]].copy()
    for path in args.coder_files:
        label = coder_label(path, used)
        labels.append(label)
        comparison = comparison.merge(
            load_coder(path, label, reference, allowed),
            on="validation_id",
            validate="one_to_one",
        )
    family_columns = [f"{label}_family" for label in labels]
    comparison["exact_agreement"] = comparison[family_columns].nunique(axis=1).eq(1)
    comparison.to_csv(COMPARISON_PATH, index=False)

    disagreements = comparison[~comparison["exact_agreement"]].copy()
    adjudication_columns = [
        "validation_id",
        "paper_concern",
        "family_choices",
        *family_columns,
    ]
    adjudication = disagreements[adjudication_columns].copy()
    adjudication["adjudicated_family"] = ""
    adjudication["adjudicator_notes"] = ""
    adjudication.to_csv(ADJUDICATION_PATH, index=False)

    assignments = comparison[family_columns].to_numpy(str)
    metrics: dict[str, object] = {
        "coders": labels,
        "n_coders": len(labels),
        "n_items": len(comparison),
        "exact_agreements": int(comparison["exact_agreement"].sum()),
        "exact_agreement_share": float(comparison["exact_agreement"].mean()),
        "disagreements": int(len(disagreements)),
        "fleiss_kappa": fleiss_kappa(assignments, sorted(allowed)),
        "current_assignment_hidden_from_coder_packet": True,
        "consensus_complete": False,
    }

    if args.adjudication is not None:
        adjudicated = load_adjudication(args.adjudication, disagreements, allowed)
        final = comparison[["validation_id", "paper_concern", "exact_agreement"]].copy()
        final["consensus_family"] = comparison[family_columns[0]].where(
            comparison["exact_agreement"], ""
        )
        final = final.merge(adjudicated, on="validation_id", how="left")
        final["consensus_family"] = final["consensus_family"].where(
            final["exact_agreement"], final["adjudicated_family"]
        )
        if final["consensus_family"].eq("").any():
            raise AssertionError("Consensus mapping is incomplete")
        final["matches_current_analytical_mapping"] = final.apply(
            lambda row: row["consensus_family"]
            == PAPER_CONCERN_TO_INSTRUMENT_CATEGORY[row["paper_concern"]],
            axis=1,
        )
        final[
            [
                "validation_id",
                "paper_concern",
                "consensus_family",
                "exact_agreement",
                "adjudicator_notes",
                "matches_current_analytical_mapping",
            ]
        ].to_csv(CONSENSUS_PATH, index=False)
        metrics["consensus_complete"] = True
        metrics["consensus_matches_current_count"] = int(
            final["matches_current_analytical_mapping"].sum()
        )
        metrics["consensus_matches_current_share"] = float(
            final["matches_current_analytical_mapping"].mean()
        )

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if disagreements.empty:
        print("No adjudication is required; rerun with the empty packet to finalize.")
    else:
        print(f"Wrote {len(disagreements)} disagreements to {ADJUDICATION_PATH}")


if __name__ == "__main__":
    main()
