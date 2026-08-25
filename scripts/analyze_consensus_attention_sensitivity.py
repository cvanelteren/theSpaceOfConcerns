#!/usr/bin/env python3
"""Check attention--output results against the blinded consensus audit.

The main outcome panel uses cross-fitted title classifications for all 584
regular ATCM outputs.  This sensitivity uses only the independently selected
stratified audit sample and replaces the classifier label with its final
three-coder consensus label.  It reports both raw sample counts and counts
expanded by the sampling-stratum weights.  The exercise is a label-robustness
check, not a new population estimate: the coders were automated and the audit
contains only a sample of outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_attention_to_outcomes import fit_ppml  # noqa: E402

OUTDIR = ROOT / "output" / "outcome_linkage"
OUT_CSV = OUTDIR / "consensus_attention_sensitivity.csv"
OUT_JSON = OUTDIR / "consensus_attention_sensitivity.json"

SPECIAL = {"INSUFFICIENT_TITLE", "OUTSIDE_TAXONOMY"}


def audited_rows() -> pd.DataFrame:
    final = pd.read_csv(OUTDIR / "outcome_consensus_final.csv", keep_default_na=False)
    supplement_path = OUTDIR / "outcome_consensus_supplement_final.csv"
    if supplement_path.exists():
        final = pd.concat(
            [final, pd.read_csv(supplement_path, keep_default_na=False)],
            ignore_index=True,
        )
    scope = pd.read_csv(OUTDIR / "outcome_consensus_validation_scope_key.csv")
    if len(scope) < len(final):
        extra = final.loc[~final["validation_id"].isin(scope["validation_id"]), [
            "validation_id", "outcome_id"
        ]].copy()
        extra["in_stratified_120"] = False
        extra["in_headline_lineage_95"] = False
        scope = pd.concat([scope, extra], ignore_index=True)
    predictions = pd.read_csv(OUTDIR / "outcome_topic_predictions.csv")
    rows = final.merge(
        scope[["validation_id", "outcome_id", "in_stratified_120"]],
        on=["validation_id", "outcome_id"],
        validate="one_to_one",
    ).merge(
        predictions[[
            "outcome_id", "meeting", "instrument", "topic_top1", "high_confidence"
        ]],
        on="outcome_id",
        validate="one_to_one",
        suffixes=("", "_model"),
    )
    rows = rows[
        rows["in_stratified_120"].astype(bool)
        & ~rows["consensus_primary"].isin(SPECIAL)
    ].copy()
    rows["confidence_band"] = rows["high_confidence"].map(
        {True: "high", False: "lower"}
    )
    strata = pd.read_csv(OUTDIR / "outcome_consensus_sampling_strata.csv")
    rows = rows.merge(
        strata[["instrument", "confidence_band", "weight"]],
        on=["instrument", "confidence_band"],
        validate="many_to_one",
    )
    return rows


def build_variant(
    base: pd.DataFrame,
    rows: pd.DataFrame,
    label_col: str,
    weight_col: str | None,
) -> pd.DataFrame:
    data = rows.copy()
    data["outcome_weight"] = 1.0 if weight_col is None else data[weight_col].astype(float)
    counts = data.groupby([label_col, "meeting"])["outcome_weight"].sum()
    counts.index.names = ["topic", "meeting"]
    panel = base.drop(
        columns=[
            column
            for column in base.columns
            if column.startswith("outcome")
            or column.endswith("_mass")
            or column.startswith("measure_")
            or column.startswith("decision_")
            or column.startswith("resolution_")
        ],
        errors="ignore",
    ).copy()
    panel["audited_output_count"] = [
        float(counts.get((topic, meeting), 0.0))
        for topic, meeting in panel[["topic", "meeting"]].itertuples(index=False)
    ]
    panel = panel.sort_values(["topic", "meeting"])
    panel["audited_outcomes_prior3"] = panel.groupby("topic")[
        "audited_output_count"
    ].transform(lambda values: values.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    return panel


def main() -> None:
    base = pd.read_csv(OUTDIR / "topic_meeting_attention_outcomes.csv")
    rows = audited_rows()
    variants = [
        ("consensus_unweighted", "consensus_primary", None),
        ("consensus_design_weighted", "consensus_primary", "weight"),
        ("classifier_same_sample_unweighted", "topic_top1", None),
        ("classifier_same_sample_design_weighted", "topic_top1", "weight"),
    ]
    tables = []
    for name, label, weight in variants:
        panel = build_variant(base, rows, label, weight)
        fitted = fit_ppml(
            panel,
            "audited_output_count",
            ["papers_prior3", "neighbor_papers_prior3", "audited_outcomes_prior3"],
            name,
            minimum_year=4,
            period_column="meeting",
        )
        tables.append(fitted)
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(OUT_CSV, index=False)
    payload = {
        "audited_stratified_outputs": int(len(rows)),
        "coders": "three automated blind coders plus automated adjudication",
        "warning": "sampling-weighted sensitivity, not independent human validation",
        "rows": summary.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary[[
        "specification", "predictor", "incidence_rate_ratio", "ci_low", "ci_high", "p_value"
    ]].to_string(index=False))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
