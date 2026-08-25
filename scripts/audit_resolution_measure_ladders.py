#!/usr/bin/env python3
"""Audit explicit Resolution-to-Measure references in official ATS records.

Instrument numbers reset every year, so every reference is keyed by type,
number, and year. The audit reads the pinned official output inventory and a
pinned corpus of official Measure bodies, extracts exact year-qualified
Resolution citations, and joins them back to official output categories. It
does not contact the live ATS website.

Shared detailed topics are retained only as a candidate screen. They are never
treated as proof of a lineage unless the later Measure explicitly cites the
earlier Resolution.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.measure_detail_corpus import (
    DEFAULT_CORPUS_PATH,
    corpus_pages,
    load_pinned_measure_corpus,
)
from scripts.official_regular_atcm_outputs import load_official_regular_outputs


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "attention_output_signal"
CITATIONS_PATH = OUTDIR / "measure_resolution_citations.csv"
FORMAL_INHERITANCE_PATH = OUTDIR / "measure_formal_predecessor_citations.csv"
MEASURE_AUDIT_PATH = OUTDIR / "measure_citation_audit.csv"
CANDIDATES_PATH = OUTDIR / "resolution_measure_candidate_audit.csv"
SUMMARY_PATH = OUTDIR / "resolution_measure_lineage_summary.json"
DETAILED_TOPICS_PATH = ROOT.parent / "apparent_consensus" / "data" / "treaty_instrument_topics_long.csv"

MAX_LAG_YEARS = 10

# The common official form is Resolution 4 (2007). This deliberately requires
# the year because Resolution numbers repeat at every meeting.
RESOLUTION_CITATION = re.compile(
    r"\bResolution\s+(\d{1,2})\s*\(\s*(19\d{2}|20\d{2})\s*\)",
    flags=re.IGNORECASE,
)
FORMAL_CITATION = re.compile(
    r"\b(Measure|Decision|Resolution)\s+(\d{1,2})\s*\(\s*(19\d{2}|20\d{2})\s*\)",
    flags=re.IGNORECASE,
)
def citation_rows(
    measures: pd.DataFrame, resolutions: pd.DataFrame, pages: dict[str, tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolution_lookup = resolutions.set_index(["year", "instrument_number"])
    citations = []
    audits = []
    for measure in measures.itertuples(index=False):
        text, status = pages.get(measure.detail_url, ("", "missing"))
        matches = list(RESOLUTION_CITATION.finditer(text))
        audits.append(
            {
                "measure_id": measure.output_id,
                "measure_year": int(measure.year),
                "measure_number": int(measure.instrument_number),
                "measure_title": measure.title,
                "measure_url": measure.detail_url,
                "fetch_status": status,
                "resolution_citations_found": len(matches),
            }
        )
        seen: set[tuple[int, int]] = set()
        for match in matches:
            number = int(match.group(1))
            year = int(match.group(2))
            key = (year, number)
            if key in seen:
                continue
            seen.add(key)
            exists = key in resolution_lookup.index
            resolution = resolution_lookup.loc[key] if exists else None
            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 220)
            citations.append(
                {
                    "resolution_id": f"Resolution {number} ({year})",
                    "resolution_year": year,
                    "resolution_number": number,
                    "resolution_title": resolution.title if exists else "",
                    "resolution_categories": " | ".join(resolution.official_categories)
                    if exists
                    else "",
                    "measure_id": measure.output_id,
                    "measure_year": int(measure.year),
                    "measure_number": int(measure.instrument_number),
                    "measure_title": measure.title,
                    "measure_categories": " | ".join(measure.official_categories),
                    "lag_years": int(measure.year) - year,
                    "citation_resolves_to_inventory": bool(exists),
                    "citation_is_prior": bool(year < int(measure.year)),
                    "snippet": text[start:end],
                    "measure_url": measure.detail_url,
                }
            )
    return pd.DataFrame(citations), pd.DataFrame(audits)


def formal_predecessor_rows(
    measures: pd.DataFrame,
    outputs: pd.DataFrame,
    pages: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Extract exact, year-qualified predecessors cited by each Measure."""
    output_lookup = outputs.set_index(["instrument", "year", "instrument_number"])
    rows = []
    for measure in measures.itertuples(index=False):
        text, status = pages.get(measure.detail_url, ("", "missing"))
        if status != "ok":
            continue
        seen: set[tuple[str, int, int]] = set()
        for match in FORMAL_CITATION.finditer(text):
            instrument = match.group(1).title()
            number = int(match.group(2))
            year = int(match.group(3))
            key = (instrument, year, number)
            if key in seen:
                continue
            seen.add(key)
            exists = key in output_lookup.index
            predecessor = output_lookup.loc[key] if exists else None
            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 220)
            rows.append(
                {
                    "predecessor_id": f"{instrument} {number} ({year})",
                    "predecessor_type": instrument,
                    "predecessor_year": year,
                    "predecessor_number": number,
                    "predecessor_title": predecessor.title if exists else "",
                    "predecessor_categories": " | ".join(
                        predecessor.official_categories
                    )
                    if exists
                    else "",
                    "measure_id": measure.output_id,
                    "measure_year": int(measure.year),
                    "measure_number": int(measure.instrument_number),
                    "measure_title": measure.title,
                    "measure_categories": " | ".join(measure.official_categories),
                    "lag_years": int(measure.year) - year,
                    "citation_resolves_to_inventory": bool(exists),
                    "citation_is_prior": bool(year < int(measure.year)),
                    "snippet": text[start:end],
                    "measure_url": measure.detail_url,
                }
            )
    return pd.DataFrame(rows)


def detailed_topic_candidates(outputs: pd.DataFrame) -> pd.DataFrame:
    if not DETAILED_TOPICS_PATH.exists():
        return pd.DataFrame()
    topics = pd.read_csv(DETAILED_TOPICS_PATH)
    topics["instrument"] = topics["query_doc_type"].map(
        {2: "Measure", 3: "Decision", 4: "Resolution"}
    )
    topics = topics[
        topics["instrument"].isin(["Measure", "Resolution"])
        & topics["ats_topic_id"].notna()
        & topics["ats_topic_id"].ne(0)
    ].copy()
    topics["instrument_number"] = (
        topics["irec_no"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
    topics["ats_topic_id"] = topics["ats_topic_id"].astype(int)
    resolutions = topics[topics["instrument"].eq("Resolution")]
    measures = topics[topics["instrument"].eq("Measure")]
    candidates = resolutions.merge(
        measures,
        on="ats_topic_id",
        suffixes=("_resolution", "_measure"),
    )
    candidates = candidates[
        candidates["year_meeting_measure"].gt(candidates["year_meeting_resolution"])
        & candidates["year_meeting_measure"].le(
            candidates["year_meeting_resolution"] + MAX_LAG_YEARS
        )
    ].copy()
    candidates["resolution_id"] = (
        "Resolution "
        + candidates["instrument_number_resolution"].astype(str)
        + " ("
        + candidates["year_meeting_resolution"].astype(str)
        + ")"
    )
    candidates["measure_id"] = (
        "Measure "
        + candidates["instrument_number_measure"].astype(str)
        + " ("
        + candidates["year_meeting_measure"].astype(str)
        + ")"
    )
    grouped = (
        candidates.groupby(["resolution_id", "measure_id"], as_index=False)
        .agg(
            resolution_year=("year_meeting_resolution", "first"),
            resolution_number=("instrument_number_resolution", "first"),
            resolution_title=("subject_resolution", "first"),
            measure_year=("year_meeting_measure", "first"),
            measure_number=("instrument_number_measure", "first"),
            measure_title=("subject_measure", "first"),
            shared_detailed_topics=("ats_topic_id", "nunique"),
            topic_match_methods=(
                "match_method_resolution",
                lambda values: " | ".join(sorted(set(map(str, values)))),
            ),
        )
    )
    grouped["lag_years"] = grouped["measure_year"] - grouped["resolution_year"]
    return grouped


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outputs = load_official_regular_outputs()
    measures = outputs[outputs["instrument"].eq("Measure")].copy()
    resolutions = outputs[outputs["instrument"].eq("Resolution")].copy()
    corpus_records, corpus_validation = load_pinned_measure_corpus(
        DEFAULT_CORPUS_PATH, measures=measures
    )
    pages = corpus_pages(corpus_records)
    citations, audits = citation_rows(measures, resolutions, pages)
    formal_predecessors = formal_predecessor_rows(measures, outputs, pages)
    citations.to_csv(CITATIONS_PATH, index=False)
    formal_predecessors.to_csv(FORMAL_INHERITANCE_PATH, index=False)
    audits.to_csv(MEASURE_AUDIT_PATH, index=False)

    candidates = detailed_topic_candidates(outputs)
    if not candidates.empty:
        verified = citations[
            citations["citation_resolves_to_inventory"]
            & citations["citation_is_prior"]
        ][["resolution_id", "measure_id", "snippet", "measure_url"]].drop_duplicates()
        candidates = candidates.merge(
            verified.assign(explicit_citation=True),
            on=["resolution_id", "measure_id"],
            how="left",
        )
        candidates["explicit_citation"] = candidates["explicit_citation"].eq(True)
        candidates.to_csv(CANDIDATES_PATH, index=False)

    valid = citations[
        citations["citation_resolves_to_inventory"]
        & citations["citation_is_prior"]
    ]
    valid_formal = formal_predecessors[
        formal_predecessors["citation_resolves_to_inventory"]
        & formal_predecessors["citation_is_prior"]
    ]
    formal_by_type = {
        instrument: {
            "citation_edges": int(len(group)),
            "citing_measures": int(group["measure_id"].nunique()),
            "cited_predecessors": int(group["predecessor_id"].nunique()),
            "median_lag_years": float(group["lag_years"].median()),
        }
        for instrument, group in valid_formal.groupby("predecessor_type")
    }
    summary = {
        "audit_date": date.today().isoformat(),
        "measure_page_source": "pinned_corpus",
        "pinned_measure_corpus": corpus_validation,
        "measure_pages": int(len(measures)),
        "measure_pages_loaded_from_pinned_corpus": int(
            audits["fetch_status"].eq("ok").sum()
        ),
        "year_qualified_resolution_citations": int(len(citations)),
        "resolved_prior_resolution_citations": int(len(valid)),
        "measures_with_resolved_prior_resolution_citations": int(
            valid["measure_id"].nunique()
        ),
        "resolutions_explicitly_cited_by_later_measures": int(
            valid["resolution_id"].nunique()
        ),
        "measures_with_any_explicit_prior_formal_predecessor": int(
            valid_formal["measure_id"].nunique()
        ),
        "formal_predecessors_by_type": formal_by_type,
        "topic_screen_candidate_pairs": int(len(candidates)),
        "topic_screen_pairs_with_explicit_citation": int(
            candidates["explicit_citation"].sum()
        )
        if not candidates.empty
        else 0,
        "interpretation": "Explicit citations are verified formal links; shared detailed topics are candidates only.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
