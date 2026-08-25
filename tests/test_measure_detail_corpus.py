from __future__ import annotations

import hashlib
import shutil

import pytest

from scripts.measure_detail_corpus import (
    DEFAULT_CORPUS_PATH,
    load_pinned_measure_corpus,
    validate_corpus_records,
    write_corpus,
)
from scripts.official_regular_atcm_outputs import load_official_regular_outputs


def measures():
    outputs = load_official_regular_outputs()
    return outputs[outputs["instrument"].eq("Measure")].copy()


def test_pinned_measure_corpus_matches_inventory_and_hashes():
    records, validation = load_pinned_measure_corpus(
        DEFAULT_CORPUS_PATH, measures=measures()
    )
    assert len(records) == 277
    assert validation["unique_output_ids"] == 277
    assert validation["unique_detail_urls"] == 277
    assert validation["unique_valid_body_sha256"] == 277


def test_corpus_serialization_is_deterministic(tmp_path):
    records, validation = load_pinned_measure_corpus(
        DEFAULT_CORPUS_PATH, measures=measures()
    )
    rewritten = tmp_path / "measure_bodies.jsonl"
    rewritten_hash = write_corpus(rewritten, records)
    assert rewritten_hash == validation["corpus_sha256"]
    assert rewritten.read_bytes() == DEFAULT_CORPUS_PATH.read_bytes()


def test_tampered_body_hash_is_rejected():
    records, _ = load_pinned_measure_corpus(DEFAULT_CORPUS_PATH)
    tampered = [dict(record) for record in records]
    tampered[0]["body_text"] += " altered"
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        validate_corpus_records(tampered, measures=measures())


def test_corpus_file_matches_published_manifest_hash():
    digest = hashlib.sha256(DEFAULT_CORPUS_PATH.read_bytes()).hexdigest()
    assert digest == "028ff6f7100e13d7679e207e9e73514e50a9ca33c4260e12e77a8233edee0211"


def test_file_level_manifest_hash_is_enforced(tmp_path):
    corpus_copy = tmp_path / "measure_bodies.jsonl"
    manifest_copy = corpus_copy.with_suffix(".manifest.json")
    shutil.copyfile(DEFAULT_CORPUS_PATH, corpus_copy)
    shutil.copyfile(DEFAULT_CORPUS_PATH.with_suffix(".manifest.json"), manifest_copy)
    corpus_copy.write_bytes(corpus_copy.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_pinned_measure_corpus(corpus_copy)
