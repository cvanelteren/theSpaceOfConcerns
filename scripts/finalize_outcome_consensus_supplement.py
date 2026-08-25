#!/usr/bin/env python3
"""Combine unanimous coding and adjudication for the lineage supplement."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_outcome_consensus as finalize


finalize.BLIND = finalize.OUTDIR / "outcome_consensus_supplement_blind.csv"
finalize.UNANIMOUS = finalize.OUTDIR / "outcome_consensus_supplement_unanimous.csv"
finalize.ADJUDICATED = finalize.OUTDIR / "outcome_consensus_supplement_adjudicated.csv"
finalize.DISAGREEMENTS = finalize.OUTDIR / "outcome_consensus_supplement_disagreements_blind.csv"
finalize.OUT = finalize.OUTDIR / "outcome_consensus_supplement_provisional.csv"


if __name__ == "__main__":
    finalize.main()
