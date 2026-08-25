#!/usr/bin/env python3
"""Plot the concentration of regular ATCM Measures in recurring area work."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import ultraplot as uplt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ats_treaty_instruments_2026-02-09.csv"
OUTPUT = ROOT / "figures" / "figS_measure_composition.pdf"
TOTAL_EXPECTED = 277


def has_area_category(raw: object) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    entries = json.loads(raw)
    return any(
        entry.get("Title") == "Category"
        and "Area protection and management" in str(entry.get("Text", ""))
        for entry in entries
    )


def main() -> None:
    instruments = pd.read_csv(DATA)
    measures = instruments[
        instruments["query_doc_type"].eq(2)
        & instruments["meeting_no"].astype(str).str.startswith("ATCM ")
        & instruments["year_meeting"].between(1995, 2025)
    ].drop_duplicates(["year_meeting", "instrument_no"])
    if len(measures) != TOTAL_EXPECTED:
        raise ValueError(f"Expected {TOTAL_EXPECTED} Measures, found {len(measures)}")

    subject = measures["subject"].fillna("").str.lower()
    counts = pd.Series(
        {
            "Area protection and\nmanagement category": int(
                measures["characteristics_json"].map(has_area_category).sum()
            ),
            "Management plan\nin title": int(subject.str.contains("management plan").sum()),
            "Revised management plan\nin title": int(
                subject.str.contains("revised management plan").sum()
            ),
            "ASPA explicitly named\nin title": int(
                subject.str.contains(r"antarctic specially protected area|\baspa\b", regex=True).sum()
            ),
        }
    )

    fig, ax = uplt.subplots(refwidth=4.2, refaspect=2.2)
    y = list(range(len(counts)))
    colors = ["#176D78", "#2A788E", "#5B8E8D", "#C9872B"]
    ax.barh(y, counts.values, color=colors, edgecolor="white", linewidth=0.7)
    for position, count in zip(y, counts.values):
        ax.text(
            count + 4,
            position,
            f"{count} ({100 * count / TOTAL_EXPECTED:.1f}%)",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.format(
        xlabel="Measures, 1995 to 2025 (n = 277)",
        yticks=y,
        yticklabels=counts.index.tolist(),
        xlim=(0, 300),
        xlocator=[0, 50, 100, 150, 200, 250],
        ylim=(-0.6, len(counts) - 0.4),
        grid=False,
    )
    ax.invert_yaxis()
    fig.save(OUTPUT, dpi=600)


if __name__ == "__main__":
    main()
