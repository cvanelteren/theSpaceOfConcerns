#!/usr/bin/env python3
"""Five disjoint meeting sequences for the direct-entry locality estimate."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.entry_definition_robustness import build_panel, fit  # noqa: E402

OUT = ROOT / "output/direct_entry_phase_sensitivity.csv"


def main() -> None:
    panel, _ = build_panel()
    first_meeting = int(panel["meeting"].min())
    rows = []
    for phase in range(5):
        subset = panel[
            (panel["meeting"] - first_meeting).mod(5).eq(phase)
        ].copy()
        result = fit(subset, "new_document_after_gap")
        result.update(
            {
                "phase": phase,
                "meetings": ",".join(map(str, sorted(subset["meeting"].unique()))),
                "within_phase_prior_windows_overlap": False,
            }
        )
        rows.append(result)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
