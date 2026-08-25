#!/usr/bin/env python3
"""Apply final blinded arbitration to the lineage supplement."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_outcome_consensus_arbitration as finalize


finalize.PROVISIONAL = finalize.OUTDIR / "outcome_consensus_supplement_provisional.csv"
finalize.AUDIT = finalize.OUTDIR / "outcome_consensus_supplement_audit.csv"
finalize.CASES = finalize.OUTDIR / "outcome_consensus_supplement_revisions_blind.csv"
finalize.ARBITRATION = finalize.OUTDIR / "outcome_consensus_supplement_arbitration.csv"
finalize.OUT_FINAL = finalize.OUTDIR / "outcome_consensus_supplement_final.csv"


if __name__ == "__main__":
    finalize.main()
