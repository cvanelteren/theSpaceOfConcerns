#!/usr/bin/env python3
"""Pinned corpus utilities for official ATS Measure detail pages.

The live website is used only by ``freeze_measure_detail_corpus.py``.  Analyses
load the extracted instrument bodies from the versioned JSONL corpus and verify
every record and body hash before use.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = ROOT / "data" / "ats_measure_detail_bodies_2026-08-16.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_CORPUS_PATH.with_suffix(".manifest.json")
EXPECTED_MEASURES = 277
USER_AGENT = "ATS concern-space research audit/1.0"
TIMEOUT_SECONDS = 30
MAX_WORKERS = 8

INSTRUMENT_BODY = re.compile(
    r'<div class="main-cols[^\"]*">.*?<div class="main-col">\s*'
    r'<div class="text-container">(.*?)</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def visible_text(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def instrument_body_text(raw_html: str) -> str:
    """Extract only the enacted instrument body, excluding page metadata."""
    match = INSTRUMENT_BODY.search(raw_html)
    return visible_text(match.group(1)) if match else ""


def body_sha256(body_text: str) -> str:
    return hashlib.sha256(body_text.encode("utf-8")).hexdigest()


def fetch_page(url: str) -> tuple[str, str, str]:
    """Fetch one live detail page for an explicit corpus-freeze operation."""
    try:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw_html = response.read().decode("utf-8", errors="replace")
        body_text = instrument_body_text(raw_html)
        status = "ok" if body_text else "instrument_body_not_found"
        return url, body_text, status
    except Exception as error:  # pragma: no cover - network diagnostic
        return url, "", f"{type(error).__name__}: {error}"


def fetch_pages_live(
    measures: pd.DataFrame, workers: int = MAX_WORKERS
) -> dict[str, tuple[str, str]]:
    urls = sorted(measures["detail_url"].dropna().astype(str).unique())
    pages: dict[str, tuple[str, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for url, body_text, status in pool.map(fetch_page, urls):
            pages[url] = (body_text, status)
    return pages


def build_corpus_records(
    measures: pd.DataFrame,
    pages: dict[str, tuple[str, str]],
    retrieval_date: str,
) -> list[dict]:
    records = []
    ordered = measures.sort_values(
        ["year", "instrument_number", "output_id"], kind="stable"
    )
    for measure in ordered.itertuples(index=False):
        body_text, status = pages.get(str(measure.detail_url), ("", "missing"))
        records.append(
            {
                "output_id": str(measure.output_id),
                "instrument": str(measure.instrument),
                "instrument_number": int(measure.instrument_number),
                "year": int(measure.year),
                "detail_url": str(measure.detail_url),
                "retrieval_date": retrieval_date,
                "body_text": body_text,
                "extraction_status": status,
                "body_sha256": body_sha256(body_text) if body_text else "",
            }
        )
    return records


def _inventory_lookup(measures: pd.DataFrame) -> dict[str, dict]:
    return {
        str(row.output_id): {
            "instrument": str(row.instrument),
            "instrument_number": int(row.instrument_number),
            "year": int(row.year),
            "detail_url": str(row.detail_url),
        }
        for row in measures.itertuples(index=False)
    }


def validate_corpus_records(
    records: list[dict], measures: pd.DataFrame | None = None
) -> dict:
    """Fail closed unless all 277 Measure bodies and hashes are valid."""
    required = {
        "output_id",
        "instrument",
        "instrument_number",
        "year",
        "detail_url",
        "retrieval_date",
        "body_text",
        "extraction_status",
        "body_sha256",
    }
    if len(records) != EXPECTED_MEASURES:
        raise ValueError(
            f"Pinned Measure corpus has {len(records)} rows; expected {EXPECTED_MEASURES}"
        )
    for index, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Corpus row {index} lacks fields: {sorted(missing)}")

    output_ids = [str(record["output_id"]) for record in records]
    urls = [str(record["detail_url"]) for record in records]
    hashes = [str(record["body_sha256"]) for record in records]
    if len(set(output_ids)) != EXPECTED_MEASURES:
        raise ValueError("Pinned Measure corpus must contain 277 unique output IDs")
    if len(set(urls)) != EXPECTED_MEASURES:
        raise ValueError("Pinned Measure corpus must contain 277 unique detail URLs")
    if len(set(hashes)) != EXPECTED_MEASURES:
        raise ValueError("Pinned Measure corpus must contain 277 unique body hashes")

    for record in records:
        output_id = str(record["output_id"])
        if record["instrument"] != "Measure":
            raise ValueError(f"{output_id}: corpus contains a non-Measure record")
        if record["extraction_status"] != "ok" or not record["body_text"]:
            raise ValueError(
                f"{output_id}: invalid extraction status {record['extraction_status']!r}"
            )
        stored_hash = str(record["body_sha256"])
        if not SHA256_PATTERN.fullmatch(stored_hash):
            raise ValueError(f"{output_id}: invalid SHA-256 syntax")
        calculated_hash = body_sha256(str(record["body_text"]))
        if stored_hash != calculated_hash:
            raise ValueError(f"{output_id}: body SHA-256 does not match extracted text")

    if measures is not None:
        expected = _inventory_lookup(measures)
        if set(expected) != set(output_ids):
            missing = sorted(set(expected) - set(output_ids))
            extra = sorted(set(output_ids) - set(expected))
            raise ValueError(
                "Pinned corpus does not match the official Measure inventory: "
                f"{len(missing)} missing, {len(extra)} extra"
            )
        for record in records:
            output_id = str(record["output_id"])
            observed = {
                "instrument": str(record["instrument"]),
                "instrument_number": int(record["instrument_number"]),
                "year": int(record["year"]),
                "detail_url": str(record["detail_url"]),
            }
            if observed != expected[output_id]:
                raise ValueError(f"{output_id}: pinned metadata differs from inventory")

    retrieval_dates = sorted({str(record["retrieval_date"]) for record in records})
    if len(retrieval_dates) != 1:
        raise ValueError("Pinned corpus must have one common retrieval date")
    return {
        "records": len(records),
        "unique_output_ids": len(set(output_ids)),
        "unique_detail_urls": len(set(urls)),
        "unique_valid_body_sha256": len(set(hashes)),
        "retrieval_date": retrieval_dates[0],
    }


def write_corpus(path: Path, records: list[dict]) -> str:
    """Write canonical JSONL and return its file-level SHA-256."""
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_pinned_measure_corpus(
    path: Path = DEFAULT_CORPUS_PATH,
    measures: pd.DataFrame | None = None,
) -> tuple[list[dict], dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Pinned Measure corpus is missing: {path}. "
            "Run scripts/freeze_measure_detail_corpus.py explicitly to create it."
        )
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    validation = validate_corpus_records(records, measures=measures)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pinned Measure corpus manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    corpus_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("corpus_sha256") != corpus_sha256:
        raise ValueError("Pinned Measure corpus does not match its manifest SHA-256")
    for field in ["records", "unique_output_ids", "unique_detail_urls", "unique_valid_body_sha256", "retrieval_date"]:
        if manifest.get(field) != validation[field]:
            raise ValueError(f"Pinned Measure corpus manifest disagrees on {field}")
    validation["corpus_path"] = str(path.relative_to(ROOT))
    validation["manifest_path"] = str(manifest_path.relative_to(ROOT))
    validation["corpus_sha256"] = corpus_sha256
    return records, validation


def corpus_pages(records: list[dict]) -> dict[str, tuple[str, str]]:
    return {
        str(record["detail_url"]): (
            str(record["body_text"]),
            str(record["extraction_status"]),
        )
        for record in records
    }
