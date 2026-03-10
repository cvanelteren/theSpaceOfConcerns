#!/usr/bin/env python3

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = PROJECT_ROOT / "old_scripts" / "notebooks" / "hazard_panel_export.py"


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


if __name__ == "__main__":
    main()
