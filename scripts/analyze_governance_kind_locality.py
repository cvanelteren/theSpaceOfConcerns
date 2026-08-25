#!/usr/bin/env python3
"""Which kind of governance decision does the concern space anticipate?

The bundling result says an instrument's off-label citations land on the single
nearest concern in the space far more often than a matched null allows. That is
an average over very different instruments: an ASPA management plan, a
Secretariat budget Decision and a tourism Resolution are not the same act of
governance.

This script cuts the same statistic by instrument type, by the Secretariat's own
category for the instrument, and by period, so the claim can be stated for the
kinds of decision it actually holds for. Constitutional instruments adopted
outside the annual ATCM -- the Madrid Protocol above all -- are not in the
outcome inventory and are therefore outside this test by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_citation_bundle_locality import (
    browser_edges,
    load_payload,
    neighbour_rank,
    outcome_frame,
    proximity_matrix,
)

INSTRUMENTS = ROOT / "data/ats_treaty_instruments_2026-02-09.csv"
DEFAULT_OUT = ROOT / "output/governance_kind_locality"
DOC_TYPES = {2: "Measure", 3: "Decision", 4: "Resolution"}
AREA_CATEGORY = "Area protection and management"
MIN_CITATIONS = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def official_categories() -> pd.Series:
    """Primary Secretariat category for every 1995-2025 instrument.

    The detail page carries one or more categories separated by newlines; the
    first is taken as primary.
    """
    frame = pd.read_csv(INSTRUMENTS)

    def primary(raw: str) -> str | None:
        try:
            for item in json.loads(raw):
                if item.get("Title") == "Category":
                    return item.get("Text").split("\n")[0].strip()
        except Exception:
            return None
        return None

    frame["category"] = frame.characteristics_json.map(primary)
    frame["outcome_id"] = [
        f"{DOC_TYPES[t]} {int(n)} ({int(y)})"
        for t, n, y in zip(frame.query_doc_type, frame.instrument_no, frame.year_meeting)
    ]
    return frame.set_index("outcome_id")["category"]


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    payload = load_payload()
    phi = proximity_matrix(payload)
    outcomes = outcome_frame(payload)
    topics = pd.read_csv(
        ROOT / "output/outcome_linkage/outcome_topic_predictions.csv"
    ).set_index("outcome_id")["topic_top1"]

    matrix = phi.to_numpy()
    topic_index = {name: i for i, name in enumerate(phi.index)}

    edges = browser_edges(payload, outcomes)
    edges["focal_topic"] = edges.focal_id.map(topics)
    edges["cited_topic"] = edges.cited_id.map(topics)
    edges = edges.dropna(subset=["focal_topic", "cited_topic"])
    off_label = edges[edges.focal_topic != edges.cited_topic].copy()
    off_label["category"] = off_label.focal_id.map(official_categories())
    off_label["period"] = pd.cut(
        off_label.focal_year, [1994, 2004, 2014, 2026],
        labels=["1995-2004", "2005-2014", "2015-2025"],
    )

    pool = outcomes.copy()
    pool["topic"] = pool.outcome_id.map(topics)
    pool = pool.dropna(subset=["topic"])

    def statistics(frame: pd.DataFrame) -> dict | None:
        if len(frame) < MIN_CITATIONS:
            return {"citations": len(frame), "underpowered": True}
        observed = np.array(
            [
                neighbour_rank(topic_index[r.focal_topic], topic_index[r.cited_topic], matrix)
                for r in frame.itertuples()
            ]
        )
        null: list[int] = []
        for r in frame.itertuples():
            candidates = pool[
                (pool.instrument == r.cited_instrument)
                & (pool.year < r.focal_year)
                & (pool.topic != r.focal_topic)
            ]
            if len(candidates) < 3:
                continue
            head = topic_index[r.focal_topic]
            null.extend(
                neighbour_rank(head, topic_index[t], matrix)
                for t in rng.choice(candidates.topic.to_numpy(), size=args.draws)
            )
        null = np.array(null)
        share, null_share = float(np.mean(observed == 1)), float(np.mean(null == 1))
        return {
            "citations": len(observed),
            "rank_one_hits": int(np.sum(observed == 1)),
            "observed": round(share, 4),
            "null": round(null_share, 4),
            "enrichment": round(share / null_share, 3) if null_share else np.nan,
            "underpowered": False,
        }

    strata = [("All off-label citations", "overall", off_label)]
    strata += [
        (kind, "instrument type", group)
        for kind, group in off_label.groupby("focal_instrument")
    ]
    strata += [
        (category, "official category", group)
        for category, group in off_label.groupby("category")
    ]
    strata += [
        (str(period), "period", group)
        for period, group in off_label.groupby("period", observed=True)
    ]
    # The period trend is confounded with the growth of the protected-area
    # system, so it is repeated with that category removed.
    strata += [
        (f"{period} excluding area protection", "period", group)
        for period, group in off_label[off_label.category != AREA_CATEGORY].groupby(
            "period", observed=True
        )
    ]

    rows = []
    for name, cut, frame in strata:
        stats = statistics(frame)
        rows.append({"stratum": name, "cut": cut, **stats})
    table = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "governance_kind_locality.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    run(parse_args())
