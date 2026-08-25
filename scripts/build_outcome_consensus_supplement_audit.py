#!/usr/bin/env python3
"""Build the independent-audit packet for the lineage supplement."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_outcome_consensus_audit as audit


audit.PROVISIONAL = audit.OUTDIR / "outcome_consensus_supplement_provisional.csv"
audit.OUT = audit.OUTDIR / "outcome_consensus_supplement_audit_blind.csv"


if __name__ == "__main__":
    audit.main()
