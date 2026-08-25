#!/usr/bin/env python3
"""Audit which official ATS outputs have documented paper pathways.

The audit distinguishes missing recovered lineage from substantive irrelevance.
It compares official regular-ATCM outputs with the confirmed/corroborated
paper--outcome routes used by the 45-concern pathway forecast. A narrow title
flag identifies clearly administrative records for manual scope review; the
flag never removes an output automatically.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_dual_channel_outcomes import (
    BROWSER_DATA,
    EVIDENCE_MINIMUM,
    linked_outcome_topics,
    load_browser,
)
from scripts.official_regular_atcm_outputs import load_official_regular_outputs

OUTDIR = ROOT / "output/documented_pathway_coverage"

ADMINISTRATIVE_TITLE_PATTERN = re.compile(
    r"\b(?:appointment of the executive secretary|external auditor|"
    r"secretariat report,? programme and budget|rules of procedure|"
    r"financial regulations|staff regulations|working languages)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-data", type=Path, default=BROWSER_DATA)
    parser.add_argument("--out-dir", type=Path, default=OUTDIR)
    return parser.parse_args()


def administrative_title_flag(title: str) -> bool:
    """Flag a narrow set of clearly administrative titles for review."""
    return bool(ADMINISTRATIVE_TITLE_PATTERN.search(str(title)))


def build_output_audit(browser: dict) -> pd.DataFrame:
    official = load_official_regular_outputs().copy()
    browser_lookup = {outcome["id"]: outcome for outcome in browser["outcomes"]}
    pathway_topics = linked_outcome_topics(browser)
    rows = []
    for record in official.itertuples(index=False):
        outcome = browser_lookup.get(record.output_id, {})
        qualifying_edges = [
            edge
            for edge in outcome.get("direct_papers", [])
            if int(edge.get("evidence_rank", 0)) >= EVIDENCE_MINIMUM
        ]
        concerns = sorted(pathway_topics.get(record.output_id, frozenset()))
        rows.append(
            {
                "output_id": record.output_id,
                "output_type": record.instrument,
                "year": int(record.year),
                "meeting": int(record.meeting),
                "title": record.title,
                "official_categories": " | ".join(record.official_categories),
                "documented_pathway": bool(concerns),
                "qualifying_paper_links": len(qualifying_edges),
                "pathway_concern_count": len(concerns),
                "pathway_concerns": " | ".join(concerns),
                "administrative_title_flag": administrative_title_flag(record.title),
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != 584 or audit["output_id"].duplicated().any():
        raise AssertionError("Coverage audit must contain 584 unique official outputs")
    return audit


def coverage_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(columns, dropna=False)["documented_pathway"]
    summary = grouped.agg(total_outputs="size", documented_outputs="sum").reset_index()
    summary["coverage"] = summary["documented_outputs"] / summary["total_outputs"]
    return summary


def category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    expanded = frame.assign(
        official_category=frame["official_categories"].str.split(" | ", regex=False)
    ).explode("official_category")
    return coverage_summary(expanded, ["official_category"]).sort_values(
        ["coverage", "total_outputs"], ascending=[False, False]
    )


def period_label(year: int) -> str:
    if year <= 2004:
        return "1995--2004"
    if year <= 2014:
        return "2005--2014"
    return "2015--2025"


def write_report(
    audit: pd.DataFrame,
    by_type: pd.DataFrame,
    by_period: pd.DataFrame,
    by_category: pd.DataFrame,
    path: Path,
) -> None:
    linked = int(audit["documented_pathway"].sum())
    administrative = audit[audit["administrative_title_flag"]]
    admin_linked = int(administrative["documented_pathway"].sum())
    substantive_candidates = audit[~audit["administrative_title_flag"]]
    substantive_linked = int(substantive_candidates["documented_pathway"].sum())

    def markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        rows = []
        for record in frame.itertuples(index=False, name=None):
            values = [
                f"{value:.3f}" if column == "coverage" else str(value)
                for column, value in zip(columns, record)
            ]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, divider, *rows])

    strongest = by_category.head(5)
    weakest = by_category.tail(5).sort_values("coverage")
    report = f"""# Documented pathway coverage

The official regular-ATCM universe contains 584 outputs from 1995--2025.
Confirmed or corroborated paper routes place {linked} outputs ({linked / 584:.1%})
in the 45-concern pathway analysis. Missing routes remain unknown rather than
negative: the audit does not classify unlinked outputs as unrelated to the
concern space.

## Coverage by output type

{markdown_table(by_type)}

## Coverage by period

{markdown_table(by_period)}

## Highest-coverage official categories

{markdown_table(strongest)}

## Lowest-coverage official categories

{markdown_table(weakest)}

## Administrative scope review

A narrow title rule flags {len(administrative)} outputs as clearly
administrative; {admin_linked} have documented paper pathways. Among the
remaining {len(substantive_candidates)} outputs, {substantive_linked}
({substantive_linked / len(substantive_candidates):.1%}) have documented
pathways. This flag supports manual scope review and never excludes records
automatically.

## Interpretation

The pathway forecast estimates where outputs with documented paper foundations
appear in concern space. It does not estimate the location of every adopted
output, and linkage absence does not establish substantive irrelevance. An
all-output analysis requires concern assignments derived independently from
instrument text or official metadata.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    browser = load_browser(args.browser_data)
    audit = build_output_audit(browser)
    audit["period"] = audit["year"].map(period_label)
    by_type = coverage_summary(audit, ["output_type"])
    by_period = coverage_summary(audit, ["period"])
    by_category = category_summary(audit)
    by_admin = coverage_summary(audit, ["administrative_title_flag"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out_dir / "official_output_pathway_audit.csv", index=False)
    by_type.to_csv(args.out_dir / "coverage_by_type.csv", index=False)
    by_period.to_csv(args.out_dir / "coverage_by_period.csv", index=False)
    by_category.to_csv(args.out_dir / "coverage_by_category.csv", index=False)
    by_admin.to_csv(args.out_dir / "coverage_by_administrative_flag.csv", index=False)
    write_report(audit, by_type, by_period, by_category, args.out_dir / "report.md")
    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "official_outputs": len(audit),
                "documented_pathway_outputs": int(audit["documented_pathway"].sum()),
                "evidence_minimum": EVIDENCE_MINIMUM,
                "administrative_flag_is_exclusion": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote pathway coverage audit to {args.out_dir}")


if __name__ == "__main__":
    main()
