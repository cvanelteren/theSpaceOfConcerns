#!/usr/bin/env python3
"""Build a blind human-coding packet for the 45-to-15 crosswalk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.official_regular_atcm_outputs import (  # noqa: E402
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
)


OUTDIR = ROOT / "output" / "scientific_checks"
PACKET_PATH = OUTDIR / "crosswalk_blind_validation_packet.csv"
METADATA_PATH = OUTDIR / "crosswalk_blind_validation_metadata.json"
PROTOCOL_PATH = OUTDIR / "crosswalk_blind_validation_protocol.md"
SEED = 20260820


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    concerns = sorted(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY)
    families = sorted(set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values()))
    if len(concerns) != 45 or len(families) != 15:
        raise AssertionError(
            f"Expected 45 concerns and 15 families; observed {len(concerns)} and {len(families)}"
        )

    rng = np.random.default_rng(SEED)
    concern_order = rng.permutation(len(concerns))
    rows = []
    for packet_position, concern_index in enumerate(concern_order, start=1):
        choices = [families[index] for index in rng.permutation(len(families))]
        rows.append(
            {
                "validation_id": f"CW{packet_position:03d}",
                "paper_concern": concerns[int(concern_index)],
                "family_choices": " | ".join(choices),
                "selected_family": "",
                "confidence": "",
                "coder_notes": "",
            }
        )
    packet = pd.DataFrame(rows)
    if any(
        value in PACKET_PATH.name.lower()
        for value in ("current", "baseline", "official_assignment")
    ):
        raise AssertionError("Blind packet filename discloses the current assignment")
    packet.to_csv(PACKET_PATH, index=False)

    metadata = {
        "purpose": "independent blind validation of the analytical paper-concern crosswalk",
        "rows": len(packet),
        "official_family_choices": len(families),
        "randomization_seed": SEED,
        "blinding": [
            "The packet contains no current crosswalk assignment.",
            "Concern rows and family-choice order are deterministically randomized.",
            "Coders must not inspect forecast results or the current mapping before submission.",
        ],
        "allowed_confidence": ["high", "medium", "low"],
        "allowed_families": families,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    protocol = """# Blind validation protocol for the 45-to-15 crosswalk

## Aim

Classify each paper concern into the single official output-category family that
best represents its dominant institutional function. This exercise validates an
analytical crosswalk. It does not classify individual papers or formal outputs.

## Blinding

Each coder receives a fresh copy of `crosswalk_blind_validation_packet.csv`.
Coders work independently and do not inspect the current crosswalk, forecast
results, another coder's file, or the uncertainty analysis before submitting
their completed files.

## Coding rule

Select one value exactly as written in `family_choices`. Use the concern label
alone. Choose the family that best captures the concern's principal function in
the Treaty system, not every function it could serve. Record `high`, `medium`, or
`low` confidence. Use `coder_notes` to explain boundary decisions, especially for
medium- and low-confidence assignments. Do not leave a row blank.

## Agreement and adjudication

Run `tools/merge_crosswalk_blind_validation.py` with at least two completed coder
files. The merger checks completeness and allowed values, reports exact agreement
and Fleiss' kappa, and creates a disagreement-only packet. An adjudicator who has
not seen the current crosswalk selects one family for each disagreement. Run the
merger again with `--adjudication` to create the final human consensus mapping.

## Commands

```bash
python tools/build_crosswalk_blind_validation.py
python tools/merge_crosswalk_blind_validation.py coder_a.csv coder_b.csv
python tools/merge_crosswalk_blind_validation.py coder_a.csv coder_b.csv \\
  --adjudication output/scientific_checks/crosswalk_blind_adjudication_packet.csv
```

Only after the final consensus file exists should the analyst compare it with the
current mapping and rerun the Resolution forecast using the human-coded mapping.
"""
    PROTOCOL_PATH.write_text(protocol, encoding="utf-8")
    print(f"Wrote {len(packet)} blind rows to {PACKET_PATH}")


if __name__ == "__main__":
    main()
