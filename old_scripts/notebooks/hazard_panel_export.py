#!/usr/bin/env python3

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "hazard_panel_export.py"


def main() -> None:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing hazard exporter at {SCRIPT_PATH}")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    argv = sys.argv[:]
    try:
        sys.argv = [str(SCRIPT_PATH), *argv[1:]]
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        sys.argv = argv


if __name__ == "__main__":
    main()
