#!/usr/bin/env python3

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_SCRIPT = PROJECT_ROOT / "old_scripts" / "fig45_portfolio_space_ridgelines.py"
TOPIC_ORDER_CSV = PROJECT_ROOT / "output" / "fig45_portfolio_space_ridgelines_topic_order.csv"
METACLUSTER_CSV = PROJECT_ROOT / "output" / "fig45_topic_metaclusters.csv"
METACLUSTER_LABELS = {
    1: "Institutional Coordination & Information Exchange",
    2: "Compliance, Environmental Management & Operations",
    3: "Frontier Impacts, Resources & Strategic Planning",
}


def _write_metacluster_csv() -> None:
    topic_df = pd.read_csv(TOPIC_ORDER_CSV)
    required = {"region_id", "topic_order", "topic", "x_plot", "is_region_anchor"}
    missing = required.difference(topic_df.columns)
    if missing:
        raise KeyError(
            f"Missing columns in {TOPIC_ORDER_CSV}: {sorted(missing)}"
        )
    out = topic_df[
        ["region_id", "topic_order", "topic", "x_plot", "is_region_anchor"]
    ].copy()
    out["metacluster_label"] = (
        out["region_id"].astype(int).map(METACLUSTER_LABELS)
    )
    out = out[
        [
            "region_id",
            "metacluster_label",
            "topic_order",
            "topic",
            "x_plot",
            "is_region_anchor",
        ]
    ]
    out.to_csv(METACLUSTER_CSV, index=False)


def main() -> None:
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing restored generator at {LEGACY_SCRIPT}. Restore it from the archive branch first."
        )
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    argv = sys.argv[:]
    try:
        sys.argv = [str(LEGACY_SCRIPT), *argv[1:]]
        runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")
    finally:
        sys.argv = argv
    _write_metacluster_csv()


if __name__ == "__main__":
    main()
