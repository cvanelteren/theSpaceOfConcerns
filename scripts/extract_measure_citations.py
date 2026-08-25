"""Extract instrument citations from the frozen Measure detail bodies.

The ATS instrument bodies name their predecessors explicitly in the preamble
("Recalling Recommendation XV-8 ..."). This module turns that prose into a
citation edge list. Instrument numbering restarts every year, so a citation is
only resolved when the text supplies the meeting or the year.
"""

import json
import re
from pathlib import Path

import pandas as pd

CORPUS = Path("data/ats_measure_detail_bodies_2026-08-16.jsonl")

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# Recommendation meeting numbers run I-XVIII; SATCM meetings appear as "XI-4".
ROMAN_REF = re.compile(r"\b([IVXLC]{1,6})\s*[-‐-―]\s*(\d{1,2})\b")
YEAR_REF = re.compile(
    r"\b(Measure|Decision|Resolution|Recommendation)s?\s+(\d{1,2})\s*\((\d{4})\)",
    re.I,
)
# "Recommendations XV-8 and XV-9 / VIII-3" - the instrument word governs a run
# of references, so capture the word then scan the span that follows it.
TYPE_WORD = re.compile(r"\b(Measure|Decision|Resolution|Recommendation)s?\b", re.I)
PREAMBLE_VERB = re.compile(
    r"\b(Recalling|Recognising|Recognizing|Noting|Considering|Desiring|Reaffirming|"
    r"Bearing in mind|Having regard|Confirming|Welcoming|Acknowledging)\b",
    re.I,
)


def roman_to_int(text: str) -> int | None:
    total = 0
    previous = 0
    for char in reversed(text.upper()):
        value = ROMAN_VALUES.get(char)
        if value is None:
            return None
        total += value if value >= previous else -value
        previous = max(previous, value)
    return total or None


def meeting_year(meeting_no: int, inventory: dict[int, int]) -> int | None:
    return inventory.get(meeting_no)


def iter_citations(body: str, span_chars: int = 160):
    """Yield (instrument_type, number, meeting_no|None, year|None, verb|None)."""
    for match in YEAR_REF.finditer(body):
        yield (
            match.group(1).capitalize(),
            int(match.group(2)),
            None,
            int(match.group(3)),
            _verb_before(body, match.start()),
        )

    # Roman references inherit the nearest preceding instrument word.
    type_positions = [(m.start(), m.group(1).capitalize()) for m in TYPE_WORD.finditer(body)]
    for match in ROMAN_REF.finditer(body):
        meeting = roman_to_int(match.group(1))
        if meeting is None or not 1 <= meeting <= 45:
            continue
        governing = [(pos, word) for pos, word in type_positions if pos < match.start()]
        if not governing:
            continue
        pos, word = governing[-1]
        if match.start() - pos > span_chars:
            continue
        yield (word, int(match.group(2)), meeting, None, _verb_before(body, match.start()))


def _verb_before(body: str, index: int) -> str | None:
    verbs = [m for m in PREAMBLE_VERB.finditer(body[:index])]
    if not verbs:
        return None
    if index - verbs[-1].start() > 240:
        return None
    return verbs[-1].group(1).capitalize()


def load_bodies(path: Path = CORPUS) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def build_citation_edges(bodies: pd.DataFrame, meeting_years: dict[int, int]) -> pd.DataFrame:
    records = []
    for row in bodies.itertuples():
        seen = set()
        for kind, number, meeting, year, verb in iter_citations(row.body_text):
            if year is None and meeting is not None:
                year = meeting_years.get(meeting)
            if year is None:
                continue
            # Recommendations are identified by meeting and number; the later
            # instruments restart their numbering each year and are identified
            # by number and year.
            if kind == "Recommendation":
                if meeting is None:
                    continue
                source_id = f"Recommendation {meeting}-{number}"
            else:
                source_id = f"{kind} {number} ({year})"
            key = (source_id, verb)
            if key in seen:
                continue
            seen.add(key)
            if row.year - year <= 0 or source_id == row.output_id:
                # Bodies repeat their own identifier and occasionally name an
                # instrument adopted at the same meeting; neither is ancestry.
                continue
            records.append(
                {
                    "focal_id": row.output_id,
                    "focal_year": row.year,
                    "cited_id": source_id,
                    "cited_instrument": kind,
                    "cited_number": number,
                    "cited_meeting": meeting,
                    "cited_year": year,
                    "preamble_verb": verb,
                    "lag_years": row.year - year,
                }
            )
    return pd.DataFrame(records)
