#!/usr/bin/env python3
"""Score both title classifiers against the same official multi-label truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compare_major_results_category_treatments import (  # noqa: E402
    paper_training_from_frame,
)
from primary_concern_sensitivity import variants  # noqa: E402
from utils import extract_unique_topics, standardize_index_labels  # noqa: E402

OUTROOT = ROOT / "output" / "category_treatment_comparison"
TREATMENTS = ("fractional_multilabel", "inferred_primary")


def expected_calibration_error(frame: pd.DataFrame, bins: int = 10) -> tuple[float, pd.DataFrame]:
    boundaries = np.linspace(0, 1, bins + 1)
    assigned = pd.cut(
        frame["allocation_weight_top1"],
        boundaries,
        include_lowest=True,
        labels=False,
    )
    rows = []
    for bin_index in range(bins):
        subset = frame[assigned.eq(bin_index)]
        if subset.empty:
            continue
        rows.append(
            {
                "bin": bin_index + 1,
                "lower": float(boundaries[bin_index]),
                "upper": float(boundaries[bin_index + 1]),
                "n": int(len(subset)),
                "mean_confidence": float(subset["allocation_weight_top1"].mean()),
                "observed_top1_hit": float(subset["top1_hit"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    ece = float(
        (
            table["n"]
            / table["n"].sum()
            * (table["mean_confidence"] - table["observed_top1_hit"]).abs()
        ).sum()
    )
    return ece, table


def main() -> None:
    data = variants()
    topics = sorted(
        standardize_index_labels(
            pd.DataFrame(index=sorted(extract_unique_topics(data["fractional_multilabel"])))
        ).index
    )
    official, _ = paper_training_from_frame(data["fractional_multilabel"], topics)
    truth = official.set_index("paper_id")["topics"].map(set)

    loaded: dict[str, pd.DataFrame] = {}
    common_ids: set[str] | None = None
    for treatment in TREATMENTS:
        path = OUTROOT / treatment / "paper_title_classifier_oof.csv"
        frame = pd.read_csv(path).set_index("paper_id")
        loaded[treatment] = frame
        identifiers = set(frame.index) & set(truth.index)
        common_ids = identifiers if common_ids is None else common_ids & identifiers
    ordered_ids = sorted(common_ids or set())

    summaries = []
    calibration_tables = []
    scored_tables = []
    for treatment in TREATMENTS:
        frame = loaded[treatment].loc[ordered_ids].copy()
        frame["official_topics"] = [truth.loc[index] for index in frame.index]
        frame["top1_hit"] = [
            prediction in accepted
            for prediction, accepted in zip(
                frame["predicted_topic_top1"], frame["official_topics"]
            )
        ]
        frame["top3_hit"] = [
            bool({first, second, third} & accepted)
            for first, second, third, accepted in zip(
                frame["predicted_topic_top1"],
                frame["predicted_topic_top2"],
                frame["predicted_topic_top3"],
                frame["official_topics"],
            )
        ]
        ece, calibration = expected_calibration_error(frame)
        calibration.insert(0, "category_treatment", treatment)
        calibration_tables.append(calibration)
        summaries.append(
            {
                "category_treatment": treatment,
                "truth_definition": "any official archive category attached to the paper",
                "n_common_papers": int(len(frame)),
                "top1_hit": float(frame["top1_hit"].mean()),
                "top3_hit": float(frame["top3_hit"].mean()),
                "expected_calibration_error_10_bins": ece,
                "mean_top1_allocation_weight": float(
                    frame["allocation_weight_top1"].mean()
                ),
            }
        )
        export = frame.reset_index()[
            [
                "paper_id",
                "meeting",
                "predicted_topic_top1",
                "predicted_topic_top2",
                "predicted_topic_top3",
                "allocation_weight_top1",
                "top1_hit",
                "top3_hit",
            ]
        ].copy()
        export.insert(0, "category_treatment", treatment)
        scored_tables.append(export)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTROOT / "classifier_common_truth_summary.csv", index=False)
    pd.concat(calibration_tables, ignore_index=True).to_csv(
        OUTROOT / "classifier_common_truth_calibration.csv", index=False
    )
    pd.concat(scored_tables, ignore_index=True).to_csv(
        OUTROOT / "classifier_common_truth_papers.csv", index=False
    )
    payload = {
        "comparison_rule": "Both classifiers are scored on the same papers against the original official multi-label assignments.",
        "results": summaries,
    }
    (OUTROOT / "classifier_common_truth_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
