#!/usr/bin/env python3
"""Describe attention around Resolutions later cited by a Measure.

This is a deliberately small boundary test. A Resolution counts as linked only
when a later Measure's legal text contains an exact, year-qualified citation.
Time is measured in ATCM meetings, not calendar years. Resolutions must have ten
subsequent regular meetings in the archive to enter the comparison. "Hardening"
in the filenames is only shorthand for entry into the Measure track; it does not
mean that the later Measure entered into force.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analyze_resolution_attention_forecast import add_features
from scripts.official_regular_atcm_outputs import load_official_regular_outputs


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "attention_output_signal"
CITATIONS_PATH = OUTDIR / "measure_resolution_citations.csv"
RESOLUTION_PATH = OUTDIR / "resolution_hardening_attention.csv"
MATCHED_PATH = OUTDIR / "resolution_hardening_matched_percentiles.csv"
SUMMARY_PATH = OUTDIR / "resolution_hardening_attention_summary.json"

FOLLOW_UP_MEETINGS = (1, 3, 5, 8, 10)
COMPLETE_FOLLOW_UP = 10


def paper_stock(
    paper_count: pd.Series,
    meeting: int,
    categories: list[str],
    start_offset: int,
    end_offset: int,
) -> float:
    """Fractionally average papers across an output's official categories."""
    return float(
        sum(
            paper_count.get((source_meeting, category), 0.0) / len(categories)
            for source_meeting in range(
                meeting + start_offset, meeting + end_offset + 1
            )
            for category in categories
        )
    )


def main() -> None:
    panel = add_features()
    outputs = load_official_regular_outputs()
    resolutions = outputs[outputs["instrument"].eq("Resolution")].copy()
    citations = pd.read_csv(CITATIONS_PATH)
    citations = citations[
        citations["citation_resolves_to_inventory"]
        & citations["citation_is_prior"]
    ].copy()

    meeting_lookup = outputs.set_index("output_id")["meeting"]
    citations["resolution_meeting"] = citations["resolution_id"].map(meeting_lookup)
    citations["measure_meeting"] = citations["measure_id"].map(meeting_lookup)
    citations["lag_meetings"] = (
        citations["measure_meeting"] - citations["resolution_meeting"]
    )
    linked_within_ten = set(
        citations.loc[
            citations["lag_meetings"].between(1, COMPLETE_FOLLOW_UP),
            "resolution_id",
        ]
    )

    latest_complete_meeting = int(panel["meeting"].max()) - COMPLETE_FOLLOW_UP
    resolutions = resolutions[resolutions["meeting"].le(latest_complete_meeting)].copy()
    resolutions["cited_within_10_meetings"] = resolutions["output_id"].isin(
        linked_within_ten
    )
    paper_count = panel.set_index(["meeting", "topic"])["paper_count"]
    resolutions["current_and_previous_2_meetings"] = resolutions.apply(
        lambda row: paper_stock(
            paper_count,
            int(row["meeting"]),
            row["official_categories"],
            -2,
            0,
        ),
        axis=1,
    )
    for horizon in FOLLOW_UP_MEETINGS:
        resolutions[f"papers_next_{horizon}_meetings"] = resolutions.apply(
            lambda row: paper_stock(
                paper_count,
                int(row["meeting"]),
                row["official_categories"],
                1,
                horizon,
            ),
            axis=1,
        )

    feature_columns = [
        "current_and_previous_2_meetings",
        *[f"papers_next_{horizon}_meetings" for horizon in FOLLOW_UP_MEETINGS],
    ]
    matched_rows = []
    positives = resolutions[resolutions["cited_within_10_meetings"]]
    for resolution in positives.itertuples(index=False):
        categories = set(resolution.official_categories)
        controls = resolutions[
            ~resolutions["cited_within_10_meetings"]
            & resolutions["official_categories"].map(
                lambda values: bool(categories.intersection(values))
            )
        ]
        for feature in feature_columns:
            value = float(getattr(resolution, feature))
            percentile = float(
                (
                    controls[feature].lt(value).sum()
                    + 0.5 * controls[feature].eq(value).sum()
                )
                / len(controls)
            )
            matched_rows.append(
                {
                    "resolution_id": resolution.output_id,
                    "resolution_title": resolution.title,
                    "feature": feature,
                    "value": value,
                    "same_category_controls": int(len(controls)),
                    "matched_percentile": percentile,
                }
            )

    matched = pd.DataFrame(matched_rows)
    group_summary = (
        resolutions.groupby("cited_within_10_meetings")[feature_columns]
        .agg(["count", "mean", "median"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "feature"})
    )
    group_summary.to_csv(RESOLUTION_PATH, index=False)
    matched.to_csv(MATCHED_PATH, index=False)

    summary = {
        "eligible_resolutions": int(len(resolutions)),
        "linked_within_10_meetings": int(
            resolutions["cited_within_10_meetings"].sum()
        ),
        "linked_resolution_ids": sorted(positives["output_id"].tolist()),
        "mean_same_category_percentile": {
            feature: float(value)
            for feature, value in matched.groupby("feature")[
                "matched_percentile"
            ].mean().items()
        },
        "interpretation": (
            "Explicitly cited Resolutions do not show unusually high direct paper "
            "attention before or after adoption; the linked set is small and "
            "concentrated in protected-area administration. Entry into the "
            "Measure track does not imply entry into force."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
