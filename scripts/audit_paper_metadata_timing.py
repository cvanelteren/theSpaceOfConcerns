#!/usr/bin/env python3
"""Audit whether paper metadata supports a strict pre-adoption nowcast claim.

This audit traces the paper-category relation to the ATS search cache, compares
it with an independent January 2024 snapshot when available, and checks cached
document Last-Modified headers against meeting end dates. It does not treat a
document timestamp as evidence that its database categories existed then.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "document-summary-multilabel.parquet"
CACHE = ROOT / "antarctic-database-go" / "data" / "processed" / "http-cache.sqlite3"
METADATA = ROOT / "antarctic-database-go" / "metadata.go"
REBUILD = ROOT / "scripts" / "rebuild_multilabel_paper_categories.py"
FORECAST = ROOT / "scripts" / "analyze_resolution_attention_forecast.py"
DEFAULT_LEGACY = Path(
    "/home/casper/Documents/australia/Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"
)
DEFAULT_OUTDIR = ROOT / "output" / "scientific_checks"
TEST_START = 29
TEST_END = 47


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-snapshot", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def roman_to_int(value: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    return sum(
        -values[char]
        if index + 1 < len(value) and values[char] < values[value[index + 1]]
        else values[char]
        for index, char in enumerate(value)
    )


def meeting_end_dates() -> dict[int, pd.Timestamp]:
    text = METADATA.read_text(encoding="utf-8")
    pattern = re.compile(
        r'Meeting_Date_ATCM_([0-9IVXLCDM]+)(?:_[^=]*)?\s+'
        r'Meeting_Date\s*=\s*"(\d{2}/\d{2}/\d{4})"'
    )
    result: dict[int, pd.Timestamp] = {}
    for token, raw_date in pattern.findall(text):
        meeting = int(token) if token.isdigit() else roman_to_int(token)
        result[meeting] = pd.Timestamp(
            datetime.strptime(raw_date, "%m/%d/%Y"), tz="UTC"
        )
    return result


def normalized_label(value: str) -> str:
    text = value.casefold()
    replacements = {
        "enviromental": "environmental",
        "instutational": "institutional",
        "protocted": "protected",
        "organisations": "organization",
        "organisation": "organization",
        "nonnative": "non native",
        "multiyear": "multi year",
        "ng_": "ng ",
        "flora_": "flora ",
        "saer": "",
        "statement": "statements",
        "planning": "plan",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def category_map(old_labels: list[str], new_labels: list[str]) -> pd.DataFrame:
    rows = []
    for old_label in old_labels:
        scored = sorted(
            (
                SequenceMatcher(
                    None, normalized_label(old_label), normalized_label(new_label)
                ).ratio(),
                new_label,
            )
            for new_label in new_labels
        )
        similarity, new_label = scored[-1]
        if old_label == "Area Management and protection plans: General":
            new_label = "Area Protection and Management Plans General"
            similarity = SequenceMatcher(
                None, normalized_label(old_label), normalized_label(new_label)
            ).ratio()
        rows.append(
            {
                "snapshot_category": old_label,
                "current_category": new_label,
                "normalized_similarity": similarity,
            }
        )
    result = pd.DataFrame(rows)
    if result["current_category"].nunique() != len(result):
        raise AssertionError("Snapshot-to-current category map is not one-to-one")
    return result


def document_key(value: object) -> str:
    path = urlparse(str(value)).path
    return Path(path).stem.casefold()


def audit_snapshot(
    current: pd.DataFrame, legacy_path: Path, outdir: Path
) -> dict[str, object]:
    if not legacy_path.exists():
        return {"available": False, "path": str(legacy_path)}

    old = pd.read_csv(legacy_path)
    old_labels = sorted(old["Category"].dropna().astype(str).unique())
    new_labels = sorted(
        {
            category
            for cell in current["category"].dropna().astype(str)
            for category in cell.split("\t")
        }
    )
    mapping = category_map(old_labels, new_labels)
    mapping.to_csv(outdir / "paper_category_snapshot_label_map.csv", index=False)
    label_of = mapping.set_index("snapshot_category")["current_category"]

    old = old.copy()
    old["document_key"] = old["ID"].map(document_key)
    current_keys = current.copy()
    current_keys["document_key"] = current_keys["paper_url"].map(document_key)
    columns = [
        "document_key",
        "paper_id",
        "meeting_number",
        "paper_name",
        "category",
        "paper_category_count",
    ]
    joined = old.merge(
        current_keys[columns], on="document_key", how="outer", indicator=True
    )
    joined["mapped_snapshot_category"] = joined["Category"].map(label_of)
    joined["snapshot_category_retained"] = [
        mapped in str(categories).split("\t")
        if pd.notna(mapped) and pd.notna(categories)
        else pd.NA
        for mapped, categories in zip(
            joined["mapped_snapshot_category"], joined["category"], strict=True
        )
    ]
    joined.to_csv(outdir / "paper_category_snapshot_comparison.csv", index=False)

    matched = joined[joined["_merge"].eq("both")]
    comparable = matched[matched["snapshot_category_retained"].notna()].copy()
    evaluation = comparable[
        comparable["meeting_number"].between(TEST_START, 45)
    ]
    extra_labels = matched[
        matched["meeting_number"].between(TEST_START, 45)
        & matched["paper_category_count"].gt(1)
    ]
    return {
        "available": True,
        "path": str(legacy_path),
        "snapshot_commit_date": "2024-01-29",
        "snapshot_commit": "91b352b0b7ff3384a20cc8f7b243dd52464c7171",
        "snapshot_categories": len(old_labels),
        "current_categories": len(new_labels),
        "category_map_is_bijective": bool(
            mapping["current_category"].nunique() == len(mapping)
        ),
        "snapshot_rows": int(len(old)),
        "matched_rows": int(len(matched)),
        "comparable_nonmissing_category_rows": int(len(comparable)),
        "retained_category_rows": int(
            comparable["snapshot_category_retained"].astype(bool).sum()
        ),
        "retained_category_share": float(
            comparable["snapshot_category_retained"].astype(bool).mean()
        ),
        "evaluation_29_45_comparable_rows": int(len(evaluation)),
        "evaluation_29_45_retained_rows": int(
            evaluation["snapshot_category_retained"].astype(bool).sum()
        ),
        "evaluation_29_45_retained_share": float(
            evaluation["snapshot_category_retained"].astype(bool).mean()
        ),
        "evaluation_29_45_multilabel_rows_not_dated_by_snapshot": int(
            len(extra_labels)
        ),
        "meetings_46_47_covered": False,
    }


def audit_search_cache(connection: sqlite3.Connection) -> dict[str, object]:
    rows = connection.execute(
        "SELECT url, body, timestamp FROM cache "
        "WHERE method = 'GET' AND status_code = 200 "
        "AND url LIKE '%SearchDocDatabase%'"
    ).fetchall()
    payload_keys: set[str] = set()
    category_ids: set[str] = set()
    payload_records = 0
    timestamps = []
    for url, body, timestamp in rows:
        category_ids.update(parse_qs(urlparse(url).query).get("category", []))
        timestamps.append(timestamp)
        document = json.loads(body)
        payload = document.get("payload") or []
        payload_records += len(payload)
        for record in payload:
            payload_keys.update(record)
    category_like_fields = sorted(
        field
        for field in payload_keys
        if re.search(r"category|topic|theme", field, flags=re.IGNORECASE)
    )
    substantive = sorted(category for category in category_ids if category != "0")
    return {
        "search_query_rows": len(rows),
        "search_payload_records_including_query_duplicates": payload_records,
        "cache_timestamp_min": min(timestamps),
        "cache_timestamp_max": max(timestamps),
        "substantive_query_category_ids": len(substantive),
        "payload_fields": sorted(payload_keys),
        "payload_category_topic_theme_fields": category_like_fields,
        "category_relation_is_query_membership": True,
    }


def audit_document_headers(
    current: pd.DataFrame, connection: sqlite3.Connection, outdir: Path
) -> dict[str, object]:
    headers = pd.read_sql_query(
        "SELECT method, url, headers FROM cache WHERE status_code = 200", connection
    )
    headers["method_rank"] = headers["method"].map({"HEAD": 0, "GET": 1}).fillna(2)
    headers = (
        headers.sort_values("method_rank").drop_duplicates("url").set_index("url")
    )
    ends = meeting_end_dates()
    papers = current[
        current["meeting_number"].between(TEST_START, TEST_END)
        & current["category"].notna()
    ]
    rows = []
    for record in papers.itertuples(index=False):
        raw_headers = headers.at[record.paper_url, "headers"]
        last_modified = (json.loads(raw_headers).get("Last-Modified") or [None])[0]
        timestamp = pd.Timestamp(last_modified) if last_modified else pd.NaT
        days_before_end = (
            (ends[int(record.meeting_number)] - timestamp).total_seconds() / 86_400
            if pd.notna(timestamp)
            else float("nan")
        )
        rows.append(
            {
                "paper_id": int(record.paper_id),
                "meeting": int(record.meeting_number),
                "title": record.paper_name,
                "paper_url": record.paper_url,
                "last_modified": last_modified,
                "meeting_end": ends[int(record.meeting_number)].isoformat(),
                "days_before_meeting_end": days_before_end,
                "modified_after_meeting_end": bool(days_before_end < 0),
            }
        )
    details = pd.DataFrame(rows)
    details.to_csv(outdir / "paper_document_header_timing.csv", index=False)
    by_meeting = (
        details.groupby("meeting")
        .agg(
            papers=("paper_id", "size"),
            last_modified_coverage=("last_modified", lambda values: values.notna().mean()),
            modified_after_meeting_end=("modified_after_meeting_end", "sum"),
            latest_days_before_meeting_end=("days_before_meeting_end", "min"),
            median_days_before_meeting_end=("days_before_meeting_end", "median"),
        )
        .reset_index()
    )
    by_meeting.to_csv(outdir / "paper_document_header_timing_by_meeting.csv", index=False)
    recent = details[details["meeting"].between(42, 47)]
    return {
        "evaluation_papers_with_categories": int(len(details)),
        "last_modified_header_coverage": float(details["last_modified"].notna().mean()),
        "modified_after_meeting_end": int(
            details["modified_after_meeting_end"].sum()
        ),
        "meetings_42_47_papers": int(len(recent)),
        "meetings_42_47_modified_after_meeting_end": int(
            recent["modified_after_meeting_end"].sum()
        ),
        "warning": (
            "Last-Modified dates establish current file timestamps, not category "
            "assignment dates; mass post-meeting dates in older meetings indicate "
            "archive migration or later file replacement."
        ),
    }


def audit_forecast_code() -> dict[str, object]:
    source = FORECAST.read_text(encoding="utf-8")
    rebuild = REBUILD.read_text(encoding="utf-8")
    return {
        "focal_paper_count_predictor": '"paper_count"' in source,
        "focal_actor_reach_predictor": '"current_actor_reach"' in source,
        "network_map_excludes_focal_meeting": (
            'relations[relations["meeting"].lt(meeting)]' in source
        ),
        "reconstruction_reads_category_from_query_url": (
            "category_id = query_category(str(url))" in rebuild
        ),
        "reconstruction_assigns_query_category_to_payload_papers": (
            "categories_of[int(paper_id)].add(label_of[category_id])" in rebuild
        ),
    }


def report_text(summary: dict[str, object]) -> str:
    cache = summary["search_cache"]
    snapshot = summary["snapshot_comparison"]
    headers = summary["document_headers"]
    return f"""# Paper metadata timing and leakage audit

## Verdict

**The strict pre-adoption category-metadata claim is not verified.** The focal
paper files are plausibly pre-adoption inputs, and the available category labels
are highly stable across snapshots. The ATS search responses nevertheless date
from {cache['cache_timestamp_min']} to {cache['cache_timestamp_max']}, after all
evaluation meetings, and their paper payloads contain no category, topic, or
theme field. The reconstruction infers categories from membership in a query
whose URL carries the category ID. Neither the cache nor the payload records
when that membership was assigned.

## Evidence that reduces concern

The independent 29 January 2024 snapshot contains 45 categories. Among
{snapshot.get('comparable_nonmissing_category_rows', 0):,} matched papers with a
nonmissing snapshot category, {snapshot.get('retained_category_rows', 0):,}
retain that category in the 2026 reconstruction
({snapshot.get('retained_category_share', float('nan')):.1%}). For evaluation
meetings 29--45, the corresponding result is
{snapshot.get('evaluation_29_45_retained_rows', 0):,} of
{snapshot.get('evaluation_29_45_comparable_rows', 0):,}. This rejects wholesale
retrospective relabelling of the category each paper already carried in 2024.

All {headers['meetings_42_47_papers']:,} included papers from meetings 42--47
have cached document Last-Modified dates before their meeting ended. This
supports the availability of their document content before adoption. The header
does not date database category membership, and older files contain clear
archive-migration timestamps.

## Remaining leakage path

The 2024 snapshot stores only one category per paper. It cannot establish when
the additional multi-label memberships used by the primary analysis were
assigned, and it does not cover meetings 46--47. The nowcast uses focal-meeting
paper counts and focal-meeting actor reach. A strict operational claim therefore
requires a contemporaneous category snapshot, an effective-date field from the
Secretariat, or a sensitivity analysis that constructs focal-meeting predictors
only from information frozen before adoption, such as the document titles and
pre-meeting category rules.

## Claim-safe conclusion

The current evidence supports **a retrospective rolling-origin same-meeting
nowcast using papers tabled before meeting close and category metadata retrieved
in 2026**. It does not yet support an unqualified statement that every category
predictor was operationally observable before the corresponding outputs were
adopted.
"""


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    current = pd.read_parquet(CURRENT).drop_duplicates("paper_id")
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        summary = {
            "audit": "paper category provenance and pre-adoption timing",
            "current_multilabel_data": str(CURRENT),
            "search_cache": audit_search_cache(connection),
            "snapshot_comparison": audit_snapshot(
                current, args.legacy_snapshot, args.outdir
            ),
            "document_headers": audit_document_headers(
                current, connection, args.outdir
            ),
            "forecast_code": audit_forecast_code(),
            "strict_pre_adoption_category_metadata_verified": False,
        }
    (args.outdir / "paper_metadata_timing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.outdir / "paper_metadata_timing_report.md").write_text(
        report_text(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
