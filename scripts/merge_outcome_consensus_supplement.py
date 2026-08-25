#!/usr/bin/env python3
"""Run the standard three-coder merge for the lineage supplement."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import merge_outcome_consensus_coders as merge


merge.BLIND = merge.OUTDIR / "outcome_consensus_supplement_blind.csv"
merge.CODERS = {
    name: merge.OUTDIR / f"outcome_consensus_supplement_coder_{name}.csv"
    for name in ("a", "b", "c")
}
merge.OUT_WIDE = merge.OUTDIR / "outcome_consensus_supplement_coder_comparison.csv"
merge.OUT_DISAGREE = merge.OUTDIR / "outcome_consensus_supplement_disagreements_blind.csv"
merge.OUT_INITIAL = merge.OUTDIR / "outcome_consensus_supplement_unanimous.csv"
merge.OUT_AGREEMENT = merge.OUTDIR / "outcome_consensus_supplement_intercoder_agreement.csv"


if __name__ == "__main__":
    merge.main()
