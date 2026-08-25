#!/usr/bin/env python3
"""Test whether the concerns an instrument cites are local in the concern space.

A formal outcome names its predecessors in its own preamble. Reading those
citations gives each instrument a concern footprint that does not depend on
reconstructed paper--outcome lineage: the focal instrument's own classified
concern together with the classified concern of every instrument it cites.

Two statistics are reported for each focal instrument, equally weighted across
focal instruments so that heavily citing instruments cannot dominate:

* head proximity -- mean concern-space proximity between the focal
  instrument's concern and the concern of each instrument it cites;
* footprint compactness -- mean pairwise proximity across the distinct
  concerns in the footprint.

Two deflationary quantities accompany them. The same-concern share is the
exact-label-match explanation in its rawest form, and off-label proximity
repeats the head statistic after deleting every citation that shares the focal
concern.

The matched null replaces each cited instrument with an instrument of the same
type, adopted before the focal instrument, in the same lag band. It preserves
citation count, timing, and instrument mix, and breaks only the topical
correspondence.

Citations come either from the frozen Measure detail bodies (`--source body`,
Measures only) or from the outcome browser's citation graph (`--source
browser`, all instrument types).
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

from scripts.extract_measure_citations import build_citation_edges, load_bodies

BROWSER = ROOT / "output/outcome_concern_browser/browser_data.json"
DEFAULT_OUT = ROOT / "output/citation_bundle_locality"
LAG_EDGES = [4, 8, 16]
LAG_LABELS = ["0-3", "4-7", "8-15", "16+"]
AREA_TOPICS = {
    "Area Protection and Management Plans General",
    "Management Plans",
    "Marine Protected Areas",
}
METRICS = ["head_proximity", "compactness", "off_label_proximity", "same_concern_share"]
RENEWAL_RELATIONS = {"supersedes", "amends", "pursuant_to", "designates_under"}


def effective_concerns(counts: pd.Series) -> float:
    """Exponential of Shannon entropy: how many concerns a corpus effectively uses."""
    share = counts / counts.sum()
    return float(np.exp(-(share * np.log(share)).sum()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["body", "browser"], default="browser")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-title-overlap",
        type=float,
        default=None,
        help=(
            "Drop citations whose focal and cited titles share more than this "
            "Jaccard fraction of content words. Instruments that revise a named "
            "site reuse its title, which would hand the classifier the same "
            "label by construction."
        ),
    )
    parser.add_argument(
        "--relations",
        default=None,
        help=(
            "Comma-separated citation relations to keep. Renewal relations "
            "(supersedes, amends, pursuant_to, designates_under) reuse the "
            "predecessor's title legitimately; reference relations (recalls, "
            "cites) do not."
        ),
    )
    parser.add_argument("--label", default=None, help="Suffix for output files.")
    return parser.parse_args()


STOPWORDS = {
    "the", "of", "and", "for", "to", "in", "a", "on", "no", "an", "at", "by",
    "antarctic", "antarctica", "treaty", "measure", "decision", "resolution",
    "recommendation", "revised", "revision", "management", "plan", "plans",
}


def title_tokens(title: str) -> set[str]:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in str(title)).split()
    return {w for w in words if w not in STOPWORDS and not w.isdigit()}


def title_overlap(a: str, b: str) -> float:
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_payload() -> dict:
    return json.loads(BROWSER.read_text())


def proximity_matrix(payload: dict) -> pd.DataFrame:
    topics = [node["id"] for node in payload["nodes"]]
    phi = pd.DataFrame(0.0, index=topics, columns=topics)
    np.fill_diagonal(phi.values, 1.0)
    for edge in payload["edges"]:
        phi.loc[edge["source"], edge["target"]] = float(edge["weight"])
        phi.loc[edge["target"], edge["source"]] = float(edge["weight"])
    return phi


def outcome_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "outcome_id": o["id"],
                "instrument": o["type"],
                "year": int(o["year"]),
                "meeting": int(float(o["meeting"])),
            }
            for o in payload["outcomes"]
        ]
    )


def lag_band(lag: float) -> str:
    return LAG_LABELS[int(np.digitize(lag, LAG_EDGES, right=False))]


def browser_edges(payload: dict, outcomes: pd.DataFrame) -> pd.DataFrame:
    year = outcomes.set_index("outcome_id")["year"]
    kind = outcomes.set_index("outcome_id")["instrument"]
    rows = []
    for outcome in payload["outcomes"]:
        focal_year = int(outcome["year"])
        for inc in outcome["incoming_outcomes"]:
            cited = inc["outcome"]
            if cited not in year.index:
                continue
            lag = focal_year - int(year[cited])
            if lag <= 0 or cited == outcome["id"]:
                continue
            rows.append(
                {
                    "focal_id": outcome["id"],
                    "focal_year": focal_year,
                    "focal_instrument": outcome["type"],
                    "cited_id": cited,
                    "cited_instrument": kind[cited],
                    "relation": inc.get("relation"),
                    "source_tier": inc.get("source_tier"),
                    "lag_years": lag,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["focal_id", "cited_id"])


def body_edges(payload: dict, outcomes: pd.DataFrame) -> pd.DataFrame:
    mmap = outcomes.groupby("meeting")["year"].min().to_dict()
    edges = build_citation_edges(load_bodies(), mmap)
    edges = edges[edges.cited_id.isin(set(outcomes.outcome_id))].copy()
    edges["focal_instrument"] = "Measure"
    edges["relation"] = edges.preamble_verb
    edges["source_tier"] = "body_text"
    return edges.drop_duplicates(subset=["focal_id", "cited_id"])


def null_statistics(head: int, samples: np.ndarray, matrix: np.ndarray) -> dict:
    """Vectorised counterpart of `focal_statistics` over sampled topic codes."""
    head_row = matrix[head]
    prox = head_row[samples]
    off_mask = samples != head
    off_count = off_mask.sum(axis=1)
    with np.errstate(invalid="ignore"):
        off = np.where(off_count > 0, (prox * off_mask).sum(axis=1) / np.maximum(off_count, 1), np.nan)
    compact = np.empty(len(samples))
    for i, row in enumerate(samples):
        footprint = np.unique(np.concatenate(([head], row)))
        if footprint.size < 2:
            compact[i] = np.nan
            continue
        block = matrix[np.ix_(footprint, footprint)]
        compact[i] = block[np.triu_indices(footprint.size, k=1)].mean()
    return {
        "head_proximity": prox.mean(axis=1),
        "compactness": compact,
        "off_label_proximity": off,
        "same_concern_share": (~off_mask).mean(axis=1),
    }


def focal_statistics(head: str, cited: list[str], phi: pd.DataFrame) -> dict:
    head_prox = float(np.mean([phi.at[head, t] for t in cited]))
    footprint = sorted({head, *cited})
    if len(footprint) > 1:
        block = phi.loc[footprint, footprint].to_numpy()
        compact = float(block[np.triu_indices(len(footprint), k=1)].mean())
    else:
        compact = np.nan
    off = [phi.at[head, t] for t in cited if t != head]
    return {
        "head_proximity": head_prox,
        "compactness": compact,
        "off_label_proximity": float(np.mean(off)) if off else np.nan,
        "same_concern_share": float(np.mean([t == head for t in cited])),
        "n_distinct_concerns": len(footprint),
    }


def neighbour_rank(head: int, cited: int, matrix: np.ndarray) -> int:
    """Rank of `cited` among `head`'s other concerns, 1 = nearest neighbour.

    Mean proximity is a poor summary here. Off-diagonal proximities in the space
    are compressed, so averaging over many distant citations hides a real
    concentration among the nearest few. The rank is scale-free and does not.
    """
    row = matrix[head].copy()
    row[head] = -np.inf
    return int(np.where(np.argsort(-row) == cited)[0][0]) + 1


def rank_test(
    subset: pd.DataFrame,
    topic_index: dict[str, int],
    matrix: np.ndarray,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    draws: int = 20,
) -> dict | None:
    """Where do off-label citations fall in the focal concern's neighbourhood?"""
    if len(subset) < 20:
        return None
    observed = np.array(
        [
            neighbour_rank(topic_index[r.focal_topic], topic_index[r.cited_topic], matrix)
            for r in subset.itertuples()
        ]
    )
    null = []
    for r in subset.itertuples():
        cand = pool[
            (pool.instrument == r.cited_instrument)
            & (pool.year < r.focal_year)
            & (pool.topic != r.focal_topic)
        ]
        if len(cand) < 3:
            continue
        head = topic_index[r.focal_topic]
        null.extend(
            neighbour_rank(head, topic_index[t], matrix)
            for t in rng.choice(cand.topic.to_numpy(), size=draws)
        )
    null = np.array(null)
    return {
        "citations": len(observed),
        "median_rank": float(np.median(observed)),
        "null_median_rank": float(np.median(null)),
        "top3_share": round(float(np.mean(observed <= 3)), 4),
        "null_top3_share": round(float(np.mean(null <= 3)), 4),
        "top5_share": round(float(np.mean(observed <= 5)), 4),
        "null_top5_share": round(float(np.mean(null <= 5)), 4),
        "enrichment_top3": round(float(np.mean(observed <= 3) / max(np.mean(null <= 3), 1e-9)), 3),
    }


def ancestry_depth_table(
    edges: pd.DataFrame,
    topics: pd.Series,
    outcomes: pd.DataFrame,
    topic_index: dict[str, int],
    matrix: np.ndarray,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    max_depth: int = 3,
) -> pd.DataFrame:
    """Does concern locality survive a chain of citations?

    Depth 1 is what an instrument cites directly. Deeper concerns enter only
    through an intermediate instrument, so this asks whether formal ancestry
    propagates topical structure or merely accumulates it.
    """
    adjacency: dict[str, set[str]] = {}
    for row in edges.itertuples():
        adjacency.setdefault(row.focal_id, set()).add(row.cited_id)
    kind = outcomes.set_index("outcome_id")["instrument"]

    records = []
    for focal in edges.focal_id.unique():
        head = topics.get(focal)
        if not isinstance(head, str):
            continue
        focal_year = int(edges.loc[edges.focal_id == focal, "focal_year"].iloc[0])
        seen = {focal}
        frontier = set(adjacency.get(focal, set()))
        shallower = {head}
        depth = 1
        while frontier and depth <= max_depth:
            reached = set()
            for ancestor in frontier:
                topic = topics.get(ancestor)
                if not isinstance(topic, str):
                    continue
                reached.add(topic)
                if topic in shallower:
                    continue
                records.append(
                    {
                        "focal_id": focal,
                        "focal_year": focal_year,
                        "focal_topic": head,
                        "cited_id": ancestor,
                        "cited_topic": topic,
                        "cited_instrument": kind.get(ancestor),
                        "depth": depth,
                    }
                )
            shallower |= reached
            seen |= frontier
            frontier = {n for a in frontier for n in adjacency.get(a, set())} - seen
            depth += 1

    reached = pd.DataFrame(records)
    rows = []
    for depth, group in reached.groupby("depth"):
        stats = rank_test(group, topic_index, matrix, pool, rng, draws=10)
        if stats is None:
            continue
        stats.update(depth=int(depth), focal_instruments=int(group.focal_id.nunique()))
        rows.append(stats)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    payload = load_payload()
    phi = proximity_matrix(payload)
    outcomes = outcome_frame(payload)
    topics = pd.read_csv(
        ROOT / "output/outcome_linkage/outcome_topic_predictions.csv"
    ).set_index("outcome_id")["topic_top1"]
    inventory = pd.read_csv(ROOT / "output/outcome_linkage/measure_pathway_inventory.csv")

    edges = browser_edges(payload, outcomes) if args.source == "browser" else body_edges(payload, outcomes)
    titles = {o["id"]: o["title"] for o in payload["outcomes"]}
    edges["title_overlap"] = [
        title_overlap(titles.get(f, ""), titles.get(c, ""))
        for f, c in zip(edges.focal_id, edges.cited_id)
    ]
    if args.relations:
        keep = {r.strip() for r in args.relations.split(",")}
        edges = edges[edges.relation.isin(keep)]
    if args.max_title_overlap is not None:
        edges = edges[edges.title_overlap <= args.max_title_overlap]
    edges["focal_topic"] = edges.focal_id.map(topics)
    edges["cited_topic"] = edges.cited_id.map(topics)
    coverage = {
        "citation_edges": int(len(edges)),
        "edges_with_both_topics": int(edges[["focal_topic", "cited_topic"]].notna().all(axis=1).sum()),
        "focal_instruments_all": int(edges.focal_id.nunique()),
    }
    edges = edges.dropna(subset=["focal_topic", "cited_topic"])
    edges["lag_band"] = edges.lag_years.map(lag_band)
    coverage["focal_instruments_analysed"] = int(edges.focal_id.nunique())

    paper_counts = pd.Series(
        pd.Series([t for p in payload["papers"].values() for t in p["topics"]]).value_counts()
    )
    outcome_counts = topics.value_counts()

    pool = outcomes.copy()
    pool["topic"] = pool.outcome_id.map(topics)
    pool = pool.dropna(subset=["topic"])
    pool_by_type = {kind: grp.reset_index(drop=True) for kind, grp in pool.groupby("instrument")}

    topic_index = {name: i for i, name in enumerate(phi.index)}
    matrix = phi.to_numpy()

    observed_rows, null_draws = [], {}
    for focal_id, grp in edges.groupby("focal_id"):
        head = grp.focal_topic.iloc[0]
        focal_year = int(grp.focal_year.iloc[0])
        cited = grp.cited_topic.tolist()
        stats = focal_statistics(head, cited, phi)
        stats.update(
            focal_id=focal_id,
            focal_year=focal_year,
            focal_instrument=grp.focal_instrument.iloc[0],
            focal_topic=head,
            n_citations=len(cited),
        )
        observed_rows.append(stats)

        candidate_sets = []
        for row in grp.itertuples():
            cand = pool_by_type.get(row.cited_instrument)
            if cand is None:
                candidate_sets.append(None)
                continue
            lag = focal_year - cand.year.to_numpy()
            band = np.array([lag_band(v) for v in lag])
            mask = (lag > 0) & (band == row.lag_band)
            if mask.sum() < 3:  # relax to any earlier instrument of that type
                mask = lag > 0
            candidate_sets.append(cand.topic.to_numpy()[mask] if mask.sum() else None)
        if any(c is None or not len(c) for c in candidate_sets):
            null_draws[focal_id] = None
            continue
        codes = [np.array([topic_index[t] for t in c]) for c in candidate_sets]
        samples = np.column_stack(
            [c[rng.integers(0, len(c), size=args.permutations)] for c in codes]
        )
        null_draws[focal_id] = null_statistics(topic_index[head], samples, matrix)

    observed = pd.DataFrame(observed_rows)
    observed = observed.merge(
        inventory[["measure_id", "functional_family", "recurring_site_administration"]],
        left_on="focal_id",
        right_on="measure_id",
        how="left",
    ).drop(columns="measure_id")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    observed.to_csv(args.out_dir / f"focal_statistics_{args.label or args.source}.csv", index=False)

    administrative = observed.recurring_site_administration.eq(True)
    subsets = {
        "all": observed.index,
        "Measure": observed.index[observed.focal_instrument == "Measure"],
        "Decision": observed.index[observed.focal_instrument == "Decision"],
        "Resolution": observed.index[observed.focal_instrument == "Resolution"],
        "excluding_area_focal": observed.index[~observed.focal_topic.isin(AREA_TOPICS)],
        "excluding_recurring_site_administration": observed.index[~administrative],
        "multi_concern_footprints": observed.index[observed.n_distinct_concerns > 1],
    }

    rows = []
    for name, idx in subsets.items():
        subset = observed.loc[idx].set_index("focal_id")
        ids = [i for i in subset.index if null_draws.get(i) is not None]
        if not ids:
            continue
        for metric in METRICS:
            vals = subset.loc[ids, metric].to_numpy(dtype=float)
            keep = ~np.isnan(vals)
            kept = [i for i, k in zip(ids, keep) if k]
            if len(kept) < 3:
                continue
            obs = float(vals[keep].mean())
            stack = np.vstack([null_draws[i][metric] for i in kept])
            null_means = np.nanmean(stack, axis=0)
            rows.append(
                {
                    "subset": name,
                    "metric": metric,
                    "focal_instruments": len(kept),
                    "observed": round(obs, 4),
                    "null_mean": round(float(np.nanmean(null_means)), 4),
                    "null_p05": round(float(np.nanpercentile(null_means, 5)), 4),
                    "null_p95": round(float(np.nanpercentile(null_means, 95)), 4),
                    "excess": round(obs - float(np.nanmean(null_means)), 4),
                    "randomization_p_upper": round(
                        float((np.sum(null_means >= obs) + 1) / (len(null_means) + 1)), 4
                    ),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / f"bundle_locality_tests_{args.label or args.source}.csv", index=False)

    # Rank tests. Mean proximity understates neighbourhood concentration, so the
    # off-label citations are also scored by their rank in the focal concern's
    # own neighbour ordering.
    administrative = edges.focal_id.map(
        inventory.set_index("measure_id")["recurring_site_administration"]
    ).fillna(False).astype(bool)
    renewal = edges.relation.isin(RENEWAL_RELATIONS)
    off_label = edges[edges.focal_topic != edges.cited_topic]
    rank_subsets = {
        "all_off_label": off_label,
        "excluding_recurring_site_administration": off_label[~administrative.loc[off_label.index]],
        "recurring_site_administration": off_label[administrative.loc[off_label.index]],
        "reference_relations": off_label[~renewal.loc[off_label.index]],
        "renewal_relations": off_label[renewal.loc[off_label.index]],
        **{
            f"focal_{kind}": off_label[off_label.focal_instrument == kind]
            for kind in ["Measure", "Decision", "Resolution"]
        },
    }
    rank_rows = []
    for name, subset in rank_subsets.items():
        stats = rank_test(subset, topic_index, matrix, pool, rng)
        if stats is None:
            continue
        stats.update(subset=name)
        rank_rows.append(stats)
    rank_results = pd.DataFrame(rank_rows)
    rank_results.to_csv(
        args.out_dir / f"neighbour_rank_tests_{args.label or args.source}.csv", index=False
    )

    depth_results = ancestry_depth_table(
        edges, topics, outcomes, topic_index, matrix, pool, rng
    )
    depth_results.to_csv(
        args.out_dir / f"ancestry_depth_tests_{args.label or args.source}.csv", index=False
    )

    breadth = {
        "paper_concerns_used": int(paper_counts.size),
        "paper_effective_concerns": round(effective_concerns(paper_counts), 2),
        "paper_top5_share": round(float(paper_counts.nlargest(5).sum() / paper_counts.sum()), 4),
        "outcome_concerns_used": int(outcome_counts.size),
        "outcome_effective_concerns": round(effective_concerns(outcome_counts), 2),
        "outcome_top5_share": round(float(outcome_counts.nlargest(5).sum() / outcome_counts.sum()), 4),
    }
    coverage["concern_breadth"] = breadth
    off_diagonal = matrix[np.triu_indices(len(matrix), k=1)]
    coverage["space_off_diagonal"] = {
        "median": round(float(np.median(off_diagonal)), 4),
        "p95": round(float(np.percentile(off_diagonal, 95)), 4),
        "max": round(float(off_diagonal.max()), 4),
    }
    coverage.update(
        permutations=args.permutations,
        seed=args.seed,
        source=args.source,
        max_title_overlap=args.max_title_overlap,
    )
    (args.out_dir / f"summary_{args.label or args.source}.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2))
    print(results.to_string(index=False))
    print("\nNeighbour rank of off-label citations")
    print(rank_results.to_string(index=False))
    print("\nConcern locality by citation depth")
    print(depth_results.to_string(index=False))


if __name__ == "__main__":
    run(parse_args())
