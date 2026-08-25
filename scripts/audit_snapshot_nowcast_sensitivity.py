#!/usr/bin/env python3
"""Rerun the nowcast with the January 2024 single-category snapshot.

The snapshot predates ATCMs 46--47 and stores one category per paper. The audit
therefore compares current multi-label and snapshot-single-label results on the
common ATCM 29--45 evaluation period. It tests whether additional category
memberships present in the February 2026 cache drive the nowcast result. It
does not establish that the 2024 category was assigned before older meetings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_output_category_families import (  # noqa: E402
    build_family_panel,
    family_phi_by_meeting,
    local_weight_matrix,
    paper_family_relations,
)
from scripts.analyze_resolution_attention_forecast import (  # noqa: E402
    ATTENTION_WINDOWS,
    HISTORY_WINDOWS,
    NETWORK_K,
    OUTPUT_COLUMNS,
    add_features,
    forecast_meetings,
    model_features,
    paired_summary,
)
from scripts.audit_paper_metadata_timing import (  # noqa: E402
    DEFAULT_LEGACY,
    category_map,
    document_key,
)
from scripts.official_regular_atcm_outputs import (  # noqa: E402
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
)
from scripts.primary_concern_sensitivity import variants  # noqa: E402


OUTDIR = ROOT / "output" / "scientific_checks"
SUMMARY_PATH = OUTDIR / "snapshot_nowcast_sensitivity.csv"
DETAIL_PATH = OUTDIR / "snapshot_nowcast_sensitivity.json"
TEST_START = 29
TEST_END = 45
PANEL_START = 19


def snapshot_submitted() -> pd.DataFrame:
    current = variants()["fractional_multilabel"].copy()
    snapshot = pd.read_csv(DEFAULT_LEGACY)
    new_labels = sorted(
        {
            category
            for cell in current["category"].dropna().astype(str)
            for category in cell.split("\t")
        }
    )
    old_labels = sorted(snapshot["Category"].dropna().astype(str).unique())
    mapping = category_map(old_labels, new_labels).set_index("snapshot_category")[
        "current_category"
    ]
    current["document_key"] = current["paper url"].map(document_key)
    snapshot["document_key"] = snapshot["ID"].map(document_key)
    snapshot["snapshot_category"] = snapshot["Category"].map(mapping)
    submitted = current.merge(
        snapshot[["document_key", "snapshot_category"]],
        on="document_key",
        how="inner",
    )
    submitted = submitted[submitted["snapshot_category"].notna()].copy()
    submitted["category"] = submitted["snapshot_category"]
    meeting = pd.to_numeric(submitted["meeting number"], errors="coerce")
    return submitted[meeting.between(PANEL_START, TEST_END)].copy()


def snapshot_panel(submitted: pd.DataFrame) -> pd.DataFrame:
    relations = paper_family_relations(submitted)
    families = sorted(set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values()))
    meetings = list(range(PANEL_START, TEST_END + 1))
    panel = build_family_panel(relations, families, meetings)

    current_reach = (
        relations.groupby(["family", "meeting"])["actor"]
        .nunique()
        .rename("current_actor_reach")
        .reset_index()
        .rename(columns={"family": "topic"})
    )
    panel = panel.merge(current_reach, on=["topic", "meeting"], how="left")
    panel["current_actor_reach"] = panel["current_actor_reach"].fillna(0.0)

    for output in OUTPUT_COLUMNS.values():
        for horizon in HISTORY_WINDOWS:
            panel[f"{output}_history_{horizon}"] = (
                panel.groupby("topic")[output]
                .transform(
                    lambda values, h=horizon: values.shift(1)
                    .rolling(h, min_periods=1)
                    .sum()
                )
                .fillna(0.0)
            )
    for horizon in HISTORY_WINDOWS:
        panel[f"outcome_mass_history_{horizon}"] = (
            panel.groupby("topic")["outcome_mass"]
            .transform(
                lambda values, h=horizon: values.shift(1)
                .rolling(h, min_periods=1)
                .sum()
            )
            .fillna(0.0)
        )
    for horizon in ATTENTION_WINDOWS:
        panel[f"paper_history_{horizon}"] = (
            panel.groupby("topic")["paper_count"]
            .transform(
                lambda values, h=horizon: values.shift(1)
                .rolling(h, min_periods=1)
                .sum()
            )
            .fillna(0.0)
        )
        reach = []
        for record in panel[["topic", "meeting"]].itertuples(index=False):
            reach.append(
                relations[
                    relations["family"].eq(record.topic)
                    & relations["meeting"].ge(record.meeting - horizon)
                    & relations["meeting"].lt(record.meeting)
                ]["actor"].nunique()
            )
        panel[f"actor_reach_{horizon}"] = reach

    phi = family_phi_by_meeting(relations, families, meetings)
    for k in NETWORK_K:
        panel[f"neighbor_papers_k{k}"] = 0.0
    topic_index = {topic: index for index, topic in enumerate(families)}
    for meeting, indices_raw in panel.groupby("meeting").groups.items():
        indices = list(indices_raw)
        current = (
            panel.loc[indices]
            .set_index("topic")["paper_count"]
            .reindex(families)
            .to_numpy(float)
        )
        topics = panel.loc[indices, "topic"].tolist()
        for k in NETWORK_K:
            nearby = local_weight_matrix(phi[int(meeting)], k=k) @ current
            panel.loc[indices, f"neighbor_papers_k{k}"] = [
                nearby[topic_index[topic]] for topic in topics
            ]
    return panel


def compare(panel: pd.DataFrame, category_source: str) -> list[dict[str, object]]:
    rows = []
    for instrument, output in OUTPUT_COLUMNS.items():
        baseline = forecast_meetings(
            panel,
            output,
            model_features(output, include_focal=False),
            test_start=TEST_START,
            test_end=TEST_END,
        )
        direct = forecast_meetings(
            panel,
            output,
            model_features(output),
            test_start=TEST_START,
            test_end=TEST_END,
        )
        network = forecast_meetings(
            panel,
            output,
            model_features(output, neighbor="neighbor_papers_k14"),
            test_start=TEST_START,
            test_end=TEST_END,
        )
        for comparison, candidate, reference in (
            ("direct vs output history", direct, baseline),
            ("direct + network vs output history", network, baseline),
            ("network vs direct", network, direct),
        ):
            rows.append(
                {
                    "category_source": category_source,
                    "instrument": instrument,
                    "comparison": comparison,
                    **paired_summary(candidate, reference),
                }
            )
    return rows


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    submitted = snapshot_submitted()
    snapshot = snapshot_panel(submitted)
    current = add_features()
    rows = compare(current, "February 2026 multi-label cache")
    rows.extend(compare(snapshot, "January 2024 single-category snapshot"))
    result = pd.DataFrame(rows)
    result.to_csv(SUMMARY_PATH, index=False)

    resolution = result[result["instrument"].eq("Resolution")].to_dict(
        orient="records"
    )
    payload = {
        "evaluation_meetings": [TEST_START, TEST_END],
        "snapshot_papers_meetings_19_45": int(submitted["paper id"].nunique()),
        "interpretation": (
            "The snapshot tests whether additional category memberships in the "
            "2026 cache drive the result. It does not date the single category "
            "before meetings 29-45 and does not cover meetings 46-47."
        ),
        "resolution_results": resolution,
    }
    DETAIL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
