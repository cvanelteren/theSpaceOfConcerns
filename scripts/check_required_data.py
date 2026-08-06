#!/usr/bin/env python
"""Verify the repository still contains every data file the figures need.

``output/`` holds roughly 150 MB of intermediates, of which the paper's figure
scripts read about 6 MB. ``.gitignore`` therefore ignores ``output/*`` wholesale
and the needed files are force-added. That arrangement is only safe if
something checks it: without this script a new figure script can reference a
file that exists locally, pass every test on this machine, and fail for anyone
who clones the repository -- which is exactly how the Zenodo release came to
ship code that could not regenerate its own figures.

Run it after adding or changing a figure script::

    micromamba run -n ultraplot-dev python scripts/check_required_data.py

Exit status is 0 when every referenced path is committed, 1 otherwise, so it
can be wired into CI or a pre-commit hook.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories searched for scripts that consume derived data.
SCRIPT_GLOBS = ("*.py", "old_scripts/*.py", "scripts/*.py", "analysis/*.py")

# String literals that look like a derived-data path.
DATA_REF = re.compile(
    r"""["']((?:output|data|derived_data)/[^"']+\.(?:csv|parquet|json|npz|npy))["']"""
)

# Paths built with str.format or f-strings cannot be resolved statically; a
# brace anywhere in the literal marks it as a template rather than a real file.
TEMPLATE = re.compile(r"[{}]")


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def referenced_files() -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for pattern in SCRIPT_GLOBS:
        for script in sorted(ROOT.glob(pattern)):
            try:
                text = script.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for match in DATA_REF.finditer(text):
                path = match.group(1)
                if TEMPLATE.search(path):
                    continue
                refs.setdefault(path, set()).add(
                    str(script.relative_to(ROOT))
                )
    return refs


def main() -> int:
    tracked = tracked_files()
    refs = referenced_files()

    missing_from_git = {}
    missing_from_disk = {}
    for path, users in sorted(refs.items()):
        if path in tracked:
            continue
        # A referenced path that is absent locally too is almost always an
        # output the script writes rather than an input it reads, so it is
        # reported separately and does not fail the check.
        if (ROOT / path).exists():
            missing_from_git[path] = users
        else:
            missing_from_disk[path] = users

    print(f"referenced data paths: {len(refs)}")
    print(f"  committed:           {len(refs) - len(missing_from_git) - len(missing_from_disk)}")
    print(f"  present but not committed: {len(missing_from_git)}")
    print(f"  not present (likely script outputs): {len(missing_from_disk)}")

    if missing_from_disk:
        print("\nreferenced but absent locally (check these are outputs, not inputs):")
        for path, users in sorted(missing_from_disk.items()):
            print(f"  {path}  <- {', '.join(sorted(users))}")

    if missing_from_git:
        print("\nFAIL: these are read by figure scripts but not in the repository:")
        for path, users in sorted(missing_from_git.items()):
            print(f"  {path}  <- {', '.join(sorted(users))}")
        print("\nAdd them with:  git add -f " + " ".join(sorted(missing_from_git)))
        return 1

    print("\nOK: every statically resolvable data reference is committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
