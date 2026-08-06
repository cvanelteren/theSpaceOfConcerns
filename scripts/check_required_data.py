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

# Any quoted literal ending in a data extension. An earlier version matched only
# paths that began with "output/", which missed every script building its paths
# as ``OUTPUT_DIR / "name.csv"`` -- including fig04, whose input was therefore
# absent from a fresh clone while this script reported success. Bare filenames
# are resolved against output/ as well as the repository root.
DATA_REF = re.compile(
    r"""["']([A-Za-z0-9_./\-]+\.(?:csv|parquet|json|npz|npy))["']"""
)

# Resolved relative to the repository root when a reference has no directory.
SEARCH_DIRS = ("output", "data", "derived_data", "")

# The raw submission table is a 12 MB generated artifact that the upstream
# scraper repository does not track. It is documented in the README and
# archived in the Zenodo record rather than committed here, so it is expected
# to be absent from the repository and must not fail this check.
EXTERNAL = ("antarctic-database-go/data/processed/document-summary.parquet",)

# This file's own docstring and examples contain path-shaped strings.
SKIP_SELF = Path(__file__).name

# Paths built with str.format or f-strings cannot be resolved statically; a
# brace anywhere in the literal marks it as a template rather than a real file.
TEMPLATE = re.compile(r"[{}]")

# Templates are the guard's remaining blind spot, and it is a real one: the
# rolling-window regime files are all referenced as
# f"output/{stem}_year_window{window_size}.csv" and were therefore invisible
# here while fig03 failed on a fresh clone. Rather than try to evaluate the
# f-strings, each template is reduced to its literal prefix and every file in
# output/ sharing that prefix is required. That over-collects slightly, which
# is the right direction to err for files of this size.
TEMPLATE_PREFIX_MIN = 12


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def referenced_files() -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for pattern in SCRIPT_GLOBS:
        for script in sorted(ROOT.glob(pattern)):
            if script.name == SKIP_SELF:
                continue
            try:
                text = script.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            who = str(script.relative_to(ROOT))
            for match in DATA_REF.finditer(text):
                path = match.group(1)
                if TEMPLATE.search(path):
                    for expanded in expand_template(path):
                        refs.setdefault(expanded, set()).add(who)
                    continue
                refs.setdefault(path, set()).add(who)
    return refs


def expand_template(path: str) -> list[str]:
    """Every existing file whose name starts with the template's literal prefix."""
    prefix = path.split("{", 1)[0]
    directory, _, stem = prefix.rpartition("/")
    if not directory or len(stem) < TEMPLATE_PREFIX_MIN:
        return []
    base = ROOT / directory
    if not base.is_dir():
        return []
    return [
        f"{directory}/{p.name}"
        for p in sorted(base.iterdir())
        if p.is_file() and p.name.startswith(stem)
    ]


def candidates(ref: str) -> list[str]:
    """Where a reference could resolve to, most specific first."""
    if "/" in ref:
        # Strip leading ./ and ../ so a symlinked sibling checkout and an
        # in-tree path are recognised as the same reference.
        cleaned = ref.lstrip("./")
        return [ref, cleaned]
    return [f"{d}/{ref}" if d else ref for d in SEARCH_DIRS]


def main() -> int:
    tracked = tracked_files()
    refs = referenced_files()

    missing_from_git = {}
    missing_from_disk = {}
    for ref, users in sorted(refs.items()):
        if any(ref.lstrip("./").endswith(e) for e in EXTERNAL):
            continue
        cands = candidates(ref)
        if any(c in tracked for c in cands):
            continue
        # A reference that is absent locally too is almost always an output the
        # script writes rather than an input it reads, so it is reported
        # separately and does not fail the check.
        present = [c for c in cands if (ROOT / c).exists()]
        if present:
            missing_from_git[present[0]] = users
        else:
            missing_from_disk[ref] = users

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
