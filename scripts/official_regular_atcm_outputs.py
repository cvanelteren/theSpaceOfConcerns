"""Load authoritative subjects for post-1995 regular-ATCM outputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_EXPORT = ROOT / "data" / "ats_treaty_instruments_2026-02-09.csv"
INSTRUMENT_EXPORT_SHA256 = (
    "2e2cfaa6cc2742221670d35a3e1effa2a38c8c1e0b1b6532b2020486aff8e3f3"
)
TYPE_FROM_PREFIX = {"M": "Measure", "D": "Decision", "R": "Resolution"}
EXPECTED_COUNTS = {"Measure": 277, "Decision": 135, "Resolution": 172}
EXPECTED_MEETING_COUNTS = {
    19: 16, 20: 7, 21: 10, 22: 12, 23: 9, 24: 12, 25: 7, 26: 12,
    27: 13, 28: 22, 29: 10, 30: 12, 31: 25, 32: 33, 33: 27,
    34: 21, 35: 26, 36: 34, 37: 26, 38: 31, 39: 21, 40: 21,
    41: 14, 42: 26, 43: 40, 44: 29, 45: 28, 46: 25, 47: 15,
}

# The instrument register uses a broader category vocabulary than the paper
# database.  Each entry below maps one official instrument category to the
# corresponding concern in the 45-category paper vocabulary.  Outputs returned
# under several categories retain every match and are divided equally among
# them downstream.
INSTRUMENT_CATEGORY_TO_CONCERN = {
    "Area protection and management": "Area Protection and Management Plans General",
    "Institutional & legal matters": "Institutional and legal matters",
    "Tourism and Non-Governmental Activities": "Tourism and NG_Activities",
    "Operational matters": "Operational issues",
    "Environmental protection": "Environmental Protection General",
    "Historic Sites and Monuments": "Historic Sites and Monuments",
    "Information exchange": "Exchange of Information",
    "Scientific cooperation": "Science issues",
    "Fauna and flora": "Fauna and Flora_General",
    "Marine pollution": "Prevention of marine pollution",
    "Environmental impact assessment": "Environmental Impact Assessment EIA Other EIA Matters",
    "General matters": "Operation of the Antarctic Treaty system General",
    "Marine living resources": "Marine living resources",
    "Waste disposal and management": "Waste management and disposal",
    "Mineral resources": "Mineral resources",
}

# Hierarchy-aware aggregation used when comparing papers with formal outputs.
# The paper database has 45 fine concerns, whereas the instrument register has
# 15 broad categories.  Mapping every paper concern to one broad category puts
# both sides of the comparison at the same resolution and prevents a direct
# subtopic (for example, Management Plans) from being counted as a neighbour of
# its own output family (Area protection and management).
PAPER_CONCERN_TO_INSTRUMENT_CATEGORY = {
    "Operation of the Antarctic Treaty system Reports": "Institutional & legal matters",
    "Environmental Impact Assessment EIA Other EIA Matters": "Environmental impact assessment",
    "Science issues": "Scientific cooperation",
    "Marine Acoustics": "Scientific cooperation",
    "Climate Change": "Environmental protection",
    "Environmental Protection General": "Environmental protection",
    "Operation of the Antarctic Treaty system General": "Institutional & legal matters",
    "Management Plans": "Area protection and management",
    "Comprehensive Environmental Evaluations": "Environmental impact assessment",
    "Exchange of Information": "Information exchange",
    "Inspections": "Operational matters",
    "Tourism and NG_Activities": "Tourism and Non-Governmental Activities",
    "Environmental Monitoring and Reporting": "Environmental protection",
    "Cooperation with Other Organisations": "Institutional & legal matters",
    "Environmental Domains Analysis": "Area protection and management",
    "CEP Strategy Discussions": "Environmental protection",
    "Marine Protected Areas": "Area protection and management",
    "International Polar Year": "Scientific cooperation",
    "Sub glacial Lakes": "Scientific cooperation",
    "Site Guidelines for Visitors": "Tourism and Non-Governmental Activities",
    "Search and Rescue": "Operational matters",
    "Operational issues": "Operational matters",
    "Fauna and Flora_General": "Fauna and flora",
    "Specially Protected Species": "Fauna and flora",
    "Drilling": "Scientific cooperation",
    "Area Protection and Management Plans General": "Area protection and management",
    "Waste management and disposal": "Waste disposal and management",
    "Educational issues": "General matters",
    "Nonnative Species and Quarantine": "Fauna and flora",
    "Prevention of marine pollution": "Marine pollution",
    "Safety and Operations in Antarctica": "Operational matters",
    "Liability": "Institutional & legal matters",
    "Biological Prospecting": "Scientific cooperation",
    "Historic Sites and Monuments": "Historic Sites and Monuments",
    "Emergency report and contingency planning": "Operational matters",
    "Human Footprint and wilderness values": "Environmental protection",
    "Operation of the Antarctic Treaty system The Secretariat": "Institutional & legal matters",
    "Opening statements": "General matters",
    "Operation of the CEP": "Environmental protection",
    "State of the Antarctic Environment Report SAER": "Environmental protection",
    "Institutional and legal matters": "Institutional & legal matters",
    "Marine living resources": "Marine living resources",
    "Mineral resources": "Mineral resources",
    "Repair and remediation of environmental damage": "Environmental protection",
    "Multiyear strategic workplan": "Institutional & legal matters",
}


def _instrument_categories(raw: object) -> list[str]:
    """Extract all official Category values from an instrument detail record."""
    try:
        characteristics = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    categories: list[str] = []
    for item in characteristics:
        if item.get("Title") != "Category":
            continue
        for value in re.split(r"\s*\n\s*", str(item.get("Text") or "")):
            value = re.sub(r"\s+", " ", value).strip()
            if value and value not in categories:
                categories.append(value)
    return categories


def _roman_to_int(value: str) -> int:
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(value.upper()):
        current = numerals[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total


def _meeting_number(raw: object) -> int:
    match = re.search(r"\bATCM\s+([0-9]+|[IVXLCDM]+)\b", str(raw), flags=re.I)
    if match is None:
        raise ValueError(f"Cannot parse ATCM number from {raw!r}")
    token = match.group(1).upper()
    return int(token) if token.isdigit() else _roman_to_int(token)


def load_official_regular_outputs() -> pd.DataFrame:
    source = INSTRUMENT_EXPORT
    if not source.exists():
        raise FileNotFoundError(f"Pinned ATS instrument export is missing: {source}")
    observed_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed_sha != INSTRUMENT_EXPORT_SHA256:
        raise AssertionError(
            f"Pinned ATS export SHA mismatch: {observed_sha} != {INSTRUMENT_EXPORT_SHA256}"
        )
    frame = pd.read_csv(source)
    code = frame["irec_no"].astype(str)
    frame = frame[
        frame["year_meeting"].between(1995, 2025)
        & ~code.str.startswith("S-")
        & code.str.split("-").str[-1].str[:1].isin(TYPE_FROM_PREFIX)
    ].copy()
    frame["instrument"] = (
        frame["irec_no"].astype(str).str.split("-").str[-1].str[:1].map(TYPE_FROM_PREFIX)
    )
    frame["year"] = frame["year_meeting"].astype(int)
    frame["meeting"] = frame["meeting_no"].map(_meeting_number).astype(int)
    frame["instrument_number"] = frame["instrument_no"].astype(int)
    frame["title"] = frame["subject"].astype(str).str.strip()
    frame["official_categories"] = frame["characteristics_json"].map(
        _instrument_categories
    )
    frame["output_id"] = (
        frame["instrument"]
        + " "
        + frame["instrument_number"].astype(str)
        + " ("
        + frame["year"].astype(str)
        + ")"
    )
    frame = frame.drop_duplicates("output_id")
    counts = frame["instrument"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS or len(frame) != sum(EXPECTED_COUNTS.values()):
        raise AssertionError(f"Unexpected regular-ATCM output universe: {counts}, n={len(frame)}")
    meeting_counts = frame.groupby("meeting").size().to_dict()
    if meeting_counts != EXPECTED_MEETING_COUNTS:
        raise AssertionError(f"Unexpected per-meeting output counts: {meeting_counts}")
    if frame["title"].eq("").any() or frame["title"].str.lower().eq("nan").any():
        raise AssertionError("Every official output must have a usable subject")
    if frame["official_categories"].map(len).eq(0).any():
        raise AssertionError("Every official output must have an instrument category")
    unknown = sorted(
        {
            category
            for categories in frame["official_categories"]
            for category in categories
            if category not in INSTRUMENT_CATEGORY_TO_CONCERN
        }
    )
    if unknown:
        raise AssertionError(f"Unmapped official instrument categories: {unknown}")
    frame["official_concerns"] = frame["official_categories"].map(
        lambda values: [INSTRUMENT_CATEGORY_TO_CONCERN[value] for value in values]
    )
    result = frame[
        [
            "output_id",
            "instrument",
            "instrument_number",
            "year",
            "meeting",
            "title",
            "official_categories",
            "official_concerns",
            "detail_url",
        ]
    ].copy()
    result["title_source"] = "official_ats_inventory_subject"
    result["official_export"] = str(source)
    return result.sort_values(["year", "instrument", "instrument_number"]).reset_index(drop=True)
