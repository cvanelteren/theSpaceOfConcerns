#!/usr/bin/env python3
"""Recover every Secretariat database category returned for each ATCM paper.

The legacy collector queried one category at a time but discarded a paper once
its document URL had been seen. Its saved ``category`` field is therefore the
first matching query category, not a unique paper label. The HTTP cache retains
the category-specific query responses, so this script reconstructs the full
paper--category relation without making network requests.

Outputs
-------
data/document-summary-multilabel.parquet
    The original document rows with ``category`` replaced by a tab-separated
    list of all substantive database categories returned for the paper.
data/paper-category-multilabel.csv
    One row per paper--category relation.
output/multilabel_category_reconstruction.json
    Coverage and multiplicity diagnostics.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "antarctic-database-go/data/processed/document-summary.parquet"
CACHE = ROOT / "antarctic-database-go/data/processed/http-cache.sqlite3"
OUT_PARQUET = ROOT / "data/document-summary-multilabel.parquet"
OUT_RELATIONS = ROOT / "data/paper-category-multilabel.csv"
OUT_DIAGNOSTICS = ROOT / "output/multilabel_category_reconstruction.json"


def query_category(url: str) -> str | None:
    values = parse_qs(urlparse(url).query).get("category", [])
    return values[0] if values else None


def label_map_from_legacy(frame: pd.DataFrame) -> dict[str, str]:
    pairs: dict[str, set[str]] = defaultdict(set)
    for url, label in frame[["page_url", "category"]].dropna().itertuples(index=False):
        category_id = query_category(str(url))
        label = str(label).strip()
        if category_id and category_id != "0" and label.upper() != "ALL":
            pairs[category_id].add(label)
    ambiguous = {key: sorted(values) for key, values in pairs.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"Category IDs have ambiguous labels: {ambiguous}")
    return {key: next(iter(values)) for key, values in pairs.items()}


def recover_relations(label_of: dict[str, str]) -> tuple[dict[int, set[str]], dict]:
    categories_of: dict[int, set[str]] = defaultdict(set)
    rows_seen = 0
    payloads_seen = 0
    malformed = 0
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        cursor = connection.execute(
            "SELECT url, body FROM cache "
            "WHERE status_code = 200 AND url LIKE '%SearchDocDatabase%'"
        )
        for url, body in cursor:
            category_id = query_category(str(url))
            if category_id not in label_of:
                continue
            rows_seen += 1
            try:
                document = json.loads(body)
            except Exception:
                malformed += 1
                continue
            payload = document.get("payload") or []
            payloads_seen += len(payload)
            for paper in payload:
                paper_id = paper.get("Paper_id")
                if paper_id is not None:
                    categories_of[int(paper_id)].add(label_of[category_id])
    diagnostics = {
        "cache_query_rows_read": rows_seen,
        "cache_payload_records_read": payloads_seen,
        "malformed_cache_rows": malformed,
    }
    return categories_of, diagnostics


def main() -> None:
    legacy = pd.read_parquet(LEGACY)
    label_of = label_map_from_legacy(legacy)
    if len(label_of) != 45:
        raise ValueError(f"Expected 45 substantive categories, recovered {len(label_of)}")
    categories_of, diagnostics = recover_relations(label_of)

    unique_papers = legacy.drop_duplicates("paper_id").copy()
    legacy_label_of = unique_papers.set_index("paper_id")["category"].to_dict()
    all_categories_of: dict[int, tuple[str, ...]] = {}
    fallback_ids = []
    excluded_ids = []
    for paper_id in unique_papers["paper_id"].astype(int):
        labels = set(categories_of.get(paper_id, set()))
        if not labels:
            legacy_label = str(legacy_label_of.get(paper_id, "")).strip()
            if legacy_label and legacy_label.upper() != "ALL":
                labels.add(legacy_label)
                fallback_ids.append(paper_id)
            else:
                excluded_ids.append(paper_id)
        all_categories_of[paper_id] = tuple(sorted(labels))

    rebuilt = legacy.copy()
    rebuilt["category"] = rebuilt["paper_id"].astype(int).map(
        lambda paper_id: "\t".join(all_categories_of[paper_id]) or pd.NA
    )
    rebuilt["paper_category_count"] = rebuilt["paper_id"].astype(int).map(
        lambda paper_id: len(all_categories_of[paper_id])
    )

    relation_rows = [
        {"paper_id": paper_id, "category": category, "fractional_weight": 1.0 / len(labels)}
        for paper_id, labels in all_categories_of.items()
        for category in labels
    ]
    relations = pd.DataFrame(relation_rows).sort_values(["paper_id", "category"])
    multiplicity = pd.Series(
        {paper_id: len(labels) for paper_id, labels in all_categories_of.items()},
        dtype=int,
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIAGNOSTICS.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_parquet(OUT_PARQUET, index=False)
    relations.to_csv(OUT_RELATIONS, index=False)

    diagnostics.update(
        {
            "legacy_rows": int(len(legacy)),
            "unique_paper_ids": int(len(unique_papers)),
            "papers_with_substantive_categories": int((multiplicity > 0).sum()),
            "papers_with_multiple_categories": int((multiplicity > 1).sum()),
            "maximum_categories_per_paper": int(multiplicity.max()),
            "papers_recovered_from_legacy_only": int(len(fallback_ids)),
            "papers_without_substantive_category": int(len(excluded_ids)),
            "substantive_categories": int(len(label_of)),
            "paper_category_relations": int(len(relations)),
            "multiplicity_distribution": {
                str(int(count)): int(frequency)
                for count, frequency in multiplicity.value_counts().sort_index().items()
            },
            "source_cache": str(CACHE),
            "source_legacy_parquet": str(LEGACY),
            "output_parquet": str(OUT_PARQUET),
        }
    )
    OUT_DIAGNOSTICS.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
