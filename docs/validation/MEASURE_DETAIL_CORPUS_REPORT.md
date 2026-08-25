# Pinned Measure detail-page corpus

## Methods note

The Resolution-to-Measure audit uses a local snapshot of the official detail-page bodies for all 277 Measures in the pinned regular-ATCM inventory. The pages were retrieved on 2026-08-16. Extraction retains visible text within the page's `main-col/text-container` instrument body and removes scripts, styles, HTML tags, and repeated whitespace. Each JSONL record stores the output identifier, type, number, year, source URL, retrieval date, extracted body, extraction status, and SHA-256 of the extracted text.

`scripts/audit_resolution_measure_ladders.py` does not contact the live website. Before extracting citations, it requires exactly 277 unique Measure identifiers, URLs, and valid body hashes; recomputes every SHA-256; and checks each record against the pinned official inventory. The citation tables reproduced exactly after switching from live pages to the corpus.

## Data availability

- Corpus: `data/ats_measure_detail_bodies_2026-08-16.jsonl`
- Corpus manifest: `data/ats_measure_detail_bodies_2026-08-16.manifest.json`
- Corpus SHA-256: `028ff6f7100e13d7679e207e9e73514e50a9ca33c4260e12e77a8233edee0211`
- Source inventory: `data/ats_treaty_instruments_2026-02-09.csv`
- Source-inventory SHA-256: `2e2cfaa6cc2742221670d35a3e1effa2a38c8c1e0b1b6532b2020486aff8e3f3`

The frozen corpus can be regenerated deliberately with `python -m scripts.freeze_measure_detail_corpus --overwrite`; ordinary audit runs use the pinned file by default.
