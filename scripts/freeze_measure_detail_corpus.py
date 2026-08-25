#!/usr/bin/env python3
"""Fetch and freeze the official detail-page bodies for all 277 Measures.

This is the only script in the audit workflow that contacts the live ATS site.
Ordinary analysis reads the resulting pinned JSONL corpus instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from scripts.measure_detail_corpus import (
    DEFAULT_CORPUS_PATH,
    MAX_WORKERS,
    ROOT,
    build_corpus_records,
    fetch_pages_live,
    validate_corpus_records,
    write_corpus,
)
from scripts.official_regular_atcm_outputs import (
    INSTRUMENT_EXPORT,
    load_official_regular_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--retrieval-date", default=date.today().isoformat())
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing frozen corpus. Required to avoid accidental refreshes.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"Refusing to replace existing corpus without --overwrite: {args.output}")

    outputs = load_official_regular_outputs()
    measures = outputs[outputs["instrument"].eq("Measure")].copy()
    pages = fetch_pages_live(measures, workers=args.workers)
    records = build_corpus_records(measures, pages, args.retrieval_date)
    validation = validate_corpus_records(records, measures=measures)
    corpus_sha256 = write_corpus(args.output, records)

    manifest = {
        **validation,
        "corpus_path": str(args.output.resolve().relative_to(ROOT)),
        "corpus_sha256": corpus_sha256,
        "source_inventory_path": str(INSTRUMENT_EXPORT.relative_to(ROOT)),
        "source_inventory_sha256": hashlib.sha256(
            INSTRUMENT_EXPORT.read_bytes()
        ).hexdigest(),
        "body_extraction": (
            "Visible text inside the official detail page's main-col/text-container "
            "instrument body; scripts, styles, tags, and repeated whitespace removed."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
