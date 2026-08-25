#!/usr/bin/env python3
"""Compare attention--output associations by formal instrument type.

This is a post-processing step for the 2x2 category-treatment comparison.  It
uses the already generated concern-by-meeting panels and fits the same
two-way-fixed-effect PPML specification separately to probability-weighted
Measures, Decisions, and Resolutions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_attention_accumulation import (  # noqa: E402
    add_attention_stocks,
    fit_ppml,
    rolling_prior,
)

BASE = ROOT / "output" / "category_treatment_comparison" / "formal_outputs"
OUT = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "formal_output_instrument_sensitivity.csv"
)
HORIZONS = (1, 2, 3, 5, 8, 10)
TREATMENTS = ("inferred_primary", "fractional_multilabel")
INSTRUMENTS = {
    "Measure": "measure_mass",
    "Decision": "decision_mass",
    "Resolution": "resolution_mass",
}


def main() -> None:
    tables = []
    for attention in TREATMENTS:
        for coding in TREATMENTS:
            directory = BASE / f"attention_{attention}__coding_{coding}"
            panel = add_attention_stocks(
                pd.read_csv(directory / "topic_meeting_panel.csv")
            ).sort_values(["topic", "meeting"])
            for instrument, outcome in INSTRUMENTS.items():
                for horizon in HORIZONS:
                    prior_output = f"{instrument.lower()}_prior{horizon}"
                    panel[prior_output] = panel.groupby("topic")[outcome].transform(
                        lambda values, h=horizon: rolling_prior(values, h)
                    )
                    fitted = fit_ppml(
                        panel,
                        outcome,
                        [
                            f"papers_prior{horizon}",
                            f"nearby_prior{horizon}",
                            prior_output,
                        ],
                        f"{instrument.lower()}_accumulated_attention",
                        horizon,
                    )
                    fitted.insert(0, "instrument", instrument)
                    fitted.insert(0, "output_coding_treatment", coding)
                    fitted.insert(0, "attention_treatment", attention)
                    tables.append(fitted)
    result = pd.concat(tables, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
