#!/usr/bin/env python3
"""Validate the supplement audit and prepare anonymous arbitration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import process_outcome_consensus_audit as process


process.PROVISIONAL = process.OUTDIR / "outcome_consensus_supplement_provisional.csv"
process.AUDIT = process.OUTDIR / "outcome_consensus_supplement_audit.csv"
process.OUT_REVISIONS = process.OUTDIR / "outcome_consensus_supplement_revisions_blind.csv"
process.OUT_FINAL = process.OUTDIR / "outcome_consensus_supplement_final.csv"


if __name__ == "__main__":
    process.main()
