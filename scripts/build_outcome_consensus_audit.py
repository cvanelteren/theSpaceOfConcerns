#!/usr/bin/env python3
"""Build a model-blind packet for an independent consensus challenge."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
PROVISIONAL = OUTDIR / "outcome_consensus_provisional.csv"
OUT = OUTDIR / "outcome_consensus_audit_blind.csv"


def main() -> None:
    provisional = pd.read_csv(PROVISIONAL, keep_default_na=False)
    required = [
        "validation_id", "outcome_id", "year", "instrument", "title",
        "consensus_primary", "consensus_secondary", "consensus_confidence",
        "consensus_source",
    ]
    missing = set(required) - set(provisional.columns)
    if missing:
        raise ValueError(f"Missing provisional columns: {sorted(missing)}")
    if provisional.empty or not provisional["validation_id"].is_unique:
        raise ValueError("Expected a non-empty provisional consensus with unique IDs")
    provisional[required].to_csv(OUT, index=False)
    print(f"Wrote {len(provisional)} blind consensus-audit rows to {OUT}")


if __name__ == "__main__":
    main()
