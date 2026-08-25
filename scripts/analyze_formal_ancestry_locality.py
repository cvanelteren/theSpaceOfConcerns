#!/usr/bin/env python3
"""Test whether formal ancestry remains local in the concern space.

Formal instruments are connected by supported or verified citations in the
outcome browser. Each instrument is represented by concerns assigned to papers
with confirmed or corroborated documentary links. For every focal--ancestor
pair, this script measures concern overlap and concern-space proximity at the
ancestor's minimum graph depth.

The matched null replaces an observed ancestor's concern set with that of an
earlier instrument of the same type, in the same time-lag band, and with the
same number of linked concerns where possible. It preserves the observed
formal graph and depth distribution but breaks the topical correspondence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "output/outcome_concern_browser/browser_data.json"
DEFAULT_OUT = ROOT / "output/formal_ancestry_locality"

OPERATIVE_RELATIONS = {"amends", "supersedes", "designates_under", "pursuant_to"}
AREA_TOPICS = {
    "Area Protection and Management Plans General",
    "Management Plans",
    "Marine Protected Areas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def proximity_matrix(payload: dict) -> pd.DataFrame:
    topics = [node["id"] for node in payload["nodes"]]
    phi = pd.DataFrame(0.0, index=topics, columns=topics)
    np.fill_diagonal(phi.values, 1.0)
    for edge in payload["edges"]:
        source, target = edge["source"], edge["target"]
        phi.loc[source, target] = float(edge["weight"])
        phi.loc[target, source] = float(edge["weight"])
    return phi


def outcome_topics(
    payload: dict,
    *,
    minimum_rank: int = 3,
    exclude_title_matches: bool = False,
    exclude_area: bool = False,
) -> dict[str, frozenset[str]]:
    papers = payload["papers"]
    result: dict[str, frozenset[str]] = {}
    for outcome in payload["outcomes"]:
        topics: set[str] = set()
        for edge in outcome["direct_papers"]:
            if int(edge["evidence_rank"]) < minimum_rank:
                continue
            if exclude_title_matches and edge["channel"] == "wp_title_window":
                continue
            paper = papers.get(edge["paper"])
            if paper:
                topics.update(paper["topics"])
        if exclude_area:
            topics.difference_update(AREA_TOPICS)
        if topics:
            result[outcome["id"]] = frozenset(topics)
    return result


def incoming_edges(payload: dict, operative_only: bool = False) -> dict[str, list[str]]:
    incoming: dict[str, list[str]] = defaultdict(list)
    for outcome in payload["outcomes"]:
        for edge in outcome["incoming_outcomes"]:
            if operative_only and edge["relation"] not in OPERATIVE_RELATIONS:
                continue
            incoming[outcome["id"]].append(edge["outcome"])
    return incoming


def minimum_depths(
    focal: str, incoming: dict[str, list[str]], max_depth: int
) -> dict[str, int]:
    found: dict[str, int] = {}
    queue = deque((source, 1) for source in incoming.get(focal, []))
    while queue:
        source, depth = queue.popleft()
        if depth > max_depth or (source in found and found[source] <= depth):
            continue
        found[source] = depth
        queue.extend((parent, depth + 1) for parent in incoming.get(source, []))
    found.pop(focal, None)
    return found


def pair_metrics(
    focal_topics: frozenset[str],
    ancestor_topics: frozenset[str],
    phi: pd.DataFrame,
    regions: dict[str, int],
) -> dict[str, float]:
    values = phi.loc[list(ancestor_topics), list(focal_topics)].to_numpy()
    nearest = values.max(axis=1)
    same_region = [
        any(regions[a] == regions[f] for f in focal_topics) for a in ancestor_topics
    ]
    return {
        "exact_overlap": float(bool(focal_topics & ancestor_topics)),
        "mean_nearest_phi": float(nearest.mean()),
        "max_phi": float(values.max()),
        "ancestor_topic_share_same_region": float(np.mean(same_region)),
    }


def lag_band(years: int) -> str:
    if years <= 0:
        return "0"
    if years <= 2:
        return "1-2"
    if years <= 5:
        return "3-5"
    if years <= 10:
        return "6-10"
    if years <= 20:
        return "11-20"
    return "21+"


def period(year: int) -> str:
    if year < 1995:
        return "before 1995"
    if year < 2010:
        return "1995-2009"
    return "2010-2025"


def build_pairs(
    payload: dict,
    topics: dict[str, frozenset[str]],
    phi: pd.DataFrame,
    *,
    operative_only: bool,
    max_depth: int,
) -> pd.DataFrame:
    outcomes = {outcome["id"]: outcome for outcome in payload["outcomes"]}
    regions = {node["id"]: int(node["region"]) for node in payload["nodes"]}
    incoming = incoming_edges(payload, operative_only=operative_only)
    rows = []
    for focal_id, focal_topics in topics.items():
        focal = outcomes[focal_id]
        for ancestor_id, depth in minimum_depths(focal_id, incoming, max_depth).items():
            if ancestor_id not in topics:
                continue
            ancestor = outcomes[ancestor_id]
            lag = max(int(focal["year"]) - int(ancestor["year"]), 0)
            rows.append(
                {
                    "focal_id": focal_id,
                    "ancestor_id": ancestor_id,
                    "focal_type": focal["type"],
                    "ancestor_type": ancestor["type"],
                    "focal_year": int(focal["year"]),
                    "ancestor_year": int(ancestor["year"]),
                    "period": period(int(focal["year"])),
                    "lag_years": lag,
                    "lag_band": lag_band(lag),
                    "depth": depth,
                    "focal_topic_count": len(focal_topics),
                    "ancestor_topic_count": len(topics[ancestor_id]),
                    **pair_metrics(focal_topics, topics[ancestor_id], phi, regions),
                }
            )
    return pd.DataFrame(rows)


def candidate_pool(
    row: pd.Series,
    outcomes: dict[str, dict],
    topics: dict[str, frozenset[str]],
) -> list[str]:
    candidates = []
    for candidate_id, candidate_topics in topics.items():
        candidate = outcomes[candidate_id]
        lag = int(row.focal_year) - int(candidate["year"])
        if candidate_id in {row.focal_id, row.ancestor_id} or lag < 0:
            continue
        if candidate["type"] != row.ancestor_type or lag_band(lag) != row.lag_band:
            continue
        if len(candidate_topics) == int(row.ancestor_topic_count):
            candidates.append(candidate_id)
    if candidates:
        return candidates

    # Sparse strata are widened only on concern-set size, while type and lag stay fixed.
    for tolerance in (1, 2, 4, 100):
        candidates = []
        for candidate_id, candidate_topics in topics.items():
            candidate = outcomes[candidate_id]
            lag = int(row.focal_year) - int(candidate["year"])
            if candidate_id in {row.focal_id, row.ancestor_id} or lag < 0:
                continue
            if candidate["type"] != row.ancestor_type or lag_band(lag) != row.lag_band:
                continue
            if abs(len(candidate_topics) - int(row.ancestor_topic_count)) <= tolerance:
                candidates.append(candidate_id)
        if candidates:
            return candidates
    return []


def add_matched_null(
    pairs: pd.DataFrame,
    payload: dict,
    topics: dict[str, frozenset[str]],
    phi: pd.DataFrame,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = {outcome["id"]: outcome for outcome in payload["outcomes"]}
    regions = {node["id"]: int(node["region"]) for node in payload["nodes"]}
    rng = np.random.default_rng(seed)
    topic_positions = {topic: index for index, topic in enumerate(phi.index)}
    phi_values = phi.to_numpy()
    region_values = np.array([regions[topic] for topic in phi.index])

    def fast_metrics(focal_id: str, candidate_id: str) -> np.ndarray:
        focal = np.array([topic_positions[x] for x in topics[focal_id]], dtype=int)
        candidate = np.array([topic_positions[x] for x in topics[candidate_id]], dtype=int)
        values = phi_values[np.ix_(candidate, focal)]
        same_region = (
            region_values[candidate, None] == region_values[focal][None, :]
        ).any(axis=1)
        return np.array(
            [
                np.intersect1d(candidate, focal).size > 0,
                values.max(axis=1).mean(),
                values.max(),
                same_region.mean(),
            ],
            dtype=np.float32,
        )

    pools = [candidate_pool(row, outcomes, topics) for _, row in pairs.iterrows()]
    keep = np.array([bool(pool) for pool in pools])
    pairs = pairs.loc[keep].reset_index(drop=True)
    pools = [pool for pool in pools if pool]

    metric_names = [
        "exact_overlap",
        "mean_nearest_phi",
        "max_phi",
        "ancestor_topic_share_same_region",
    ]
    metric_cache: dict[tuple[str, str], np.ndarray] = {}
    draws = np.empty((permutations, len(pairs), len(metric_names)), dtype=np.float32)
    for position, row in pairs.iterrows():
        options = []
        for candidate_id in pools[position]:
            key = (row.focal_id, candidate_id)
            if key not in metric_cache:
                metric_cache[key] = fast_metrics(*key)
            options.append(metric_cache[key])
        option_values = np.stack(options)
        selected = rng.integers(0, len(option_values), size=permutations)
        draws[:, position, :] = option_values[selected]

    # Equal-weight focal instruments, rather than treating related ancestor pairs
    # as independent observations.
    groupings = {
        "overall": [],
        "type": ["focal_type"],
        "period": ["period"],
    }
    summaries = []
    for stratum, columns in groupings.items():
        key_columns = columns + ["depth"]
        unique_groups = pairs[key_columns].drop_duplicates()
        for _, labels in unique_groups.iterrows():
            selector = np.ones(len(pairs), dtype=bool)
            for column in key_columns:
                selector &= pairs[column].eq(labels[column]).to_numpy()
            selected_pairs = pairs.loc[selector]
            focal_counts = selected_pairs.groupby("focal_id").size()
            weights = selected_pairs["focal_id"].map(
                1.0 / focal_counts / len(focal_counts)
            ).to_numpy()
            observed_values = selected_pairs[metric_names].to_numpy().T @ weights
            null_values = np.einsum("pnm,n->pm", draws[:, selector, :], weights)
            for metric_index, metric in enumerate(metric_names):
                values = null_values[:, metric_index]
                obs = float(observed_values[metric_index])
                summaries.append(
                    {
                        "stratum": stratum,
                        "group": "all" if not columns else str(labels[columns[0]]),
                        "depth": int(labels["depth"]),
                        "metric": metric,
                        "focal_outcomes": len(focal_counts),
                        "observed": obs,
                        "null_mean": float(values.mean()),
                        "excess": obs - float(values.mean()),
                        "null_025": float(np.quantile(values, 0.025)),
                        "null_975": float(np.quantile(values, 0.975)),
                        "randomization_p_upper": float((1 + np.sum(values >= obs)) / (1 + len(values))),
                    }
                )
    return pairs, pd.DataFrame(summaries)


def scenario_specs() -> list[dict]:
    return [
        {"scenario": "primary", "minimum_rank": 3},
        {"scenario": "confirmed_only", "minimum_rank": 4},
        {"scenario": "non_title_evidence", "minimum_rank": 3, "exclude_title_matches": True},
        {"scenario": "exclude_area", "minimum_rank": 3, "exclude_area": True},
        {"scenario": "operative_links", "minimum_rank": 3, "operative_only": True},
    ]


def write_report(summary: pd.DataFrame, path: Path) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        formatted = frame.copy()
        for column in formatted.select_dtypes(include="number"):
            if column in {"depth", "focal_outcomes"}:
                formatted[column] = formatted[column].astype(int).astype(str)
            else:
                formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
        report_rows = [list(formatted.columns)] + formatted.astype(str).values.tolist()
        widths = [
            max(len(row[index]) for row in report_rows)
            for index in range(len(report_rows[0]))
        ]
        markdown = []
        for row_index, row in enumerate(report_rows):
            markdown.append(
                "| "
                + " | ".join(
                    value.ljust(widths[index]) for index, value in enumerate(row)
                )
                + " |"
            )
            if row_index == 0:
                markdown.append(
                    "| " + " | ".join("-" * width for width in widths) + " |"
                )
        return "\n".join(markdown)

    columns = [
        "depth",
        "focal_outcomes",
        "observed",
        "null_mean",
        "excess",
        "randomization_p_upper",
    ]
    primary = summary.query(
        "scenario == 'primary' and stratum == 'overall' "
        "and metric == 'mean_nearest_phi'"
    )[columns]
    by_type = summary.query(
        "scenario == 'primary' and stratum == 'type' "
        "and metric == 'mean_nearest_phi' and depth <= 3"
    )[["group"] + columns].sort_values(["group", "depth"])
    by_period = summary.query(
        "scenario == 'primary' and stratum == 'period' "
        "and metric == 'mean_nearest_phi' and depth <= 3"
    )[["group"] + columns].sort_values(["group", "depth"])
    sensitivities = summary.query(
        "stratum == 'overall' and metric == 'mean_nearest_phi' and depth <= 3"
    )[["scenario"] + columns].sort_values(["scenario", "depth"])
    lines = [
        "# Formal ancestry locality audit",
        "",
        "Formal ancestry depth is the minimum number of supported or verified formal links from an earlier instrument to a focal instrument. Concern evidence uses confirmed or corroborated paper routes unless a sensitivity says otherwise. The null substitutes a temporally available instrument matched on type, time-lag band, and concern-set size. Focal instruments receive equal weight.",
        "",
        "## Primary depth pattern",
        "",
        markdown_table(primary),
        "",
        "## By focal instrument type",
        "",
        markdown_table(by_type),
        "",
        "## By focal period",
        "",
        markdown_table(by_period),
        "",
        "## Sensitivity analyses",
        "",
        markdown_table(sensitivities),
        "",
        "## Interpretation boundary",
        "",
        "This analysis tests documentary topical concordance. It does not show that concern-space proximity caused a formal citation, that a cited paper caused adoption, or that ancestry was observable before the focal instrument was adopted.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = load_payload(args.data)
    phi = proximity_matrix(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    all_summaries = []
    for index, spec in enumerate(scenario_specs()):
        spec = dict(spec)
        name = spec.pop("scenario")
        operative_only = bool(spec.pop("operative_only", False))
        topics = outcome_topics(payload, **spec)
        pairs = build_pairs(
            payload,
            topics,
            phi,
            operative_only=operative_only,
            max_depth=args.max_depth,
        )
        if pairs.empty:
            continue
        pairs, summary = add_matched_null(
            pairs,
            payload,
            topics,
            phi,
            permutations=args.permutations,
            seed=args.seed + index,
        )
        pairs.insert(0, "scenario", name)
        summary.insert(0, "scenario", name)
        all_pairs.append(pairs)
        all_summaries.append(summary)

    pairs_out = pd.concat(all_pairs, ignore_index=True)
    summary_out = pd.concat(all_summaries, ignore_index=True)
    pairs_out.to_csv(args.out_dir / "ancestry_pairs.csv", index=False)
    summary_out.to_csv(args.out_dir / "ancestry_locality_summary.csv", index=False)
    write_report(summary_out, args.out_dir / "report.md")
    print(
        f"Wrote {len(pairs_out):,} matched ancestry pairs and "
        f"{len(summary_out):,} summary rows to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
