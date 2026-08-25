#!/usr/bin/env python3
"""Test whether prior attention and formal memory locate later ATS output.

The unit is a concern, output type, and ATCM meeting. Outcome destinations come
from confirmed or corroborated paper--outcome routes in the lineage browser.
Predictors use only information from earlier meetings:

* direct attention is the paper mass in the concern at the preceding ATCM;
* neighboring attention weights preceding-ATCM papers by a concern map built
  before that ATCM; and
* formal memory measures proximity to the endpoints of formal lineages that
  existed before the focal meeting, carrying concern evidence through at most
  three ancestry steps.

Rolling-origin models predict the distribution of documented output mass over
45 concerns. The exercise is a retrospective strict-lag forecast. It does not
estimate adoption of individual papers, causal influence, or undocumented
output destinations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.data_loading import load_submitted_with_fallback
from scripts import explore_lineage_space as lineage
from scripts.analyze_attention_to_outcomes import (
    cumulative_phi_by_meeting,
    load_paper_training,
    local_weight_matrix,
    paper_attention_meeting_panel,
)
from scripts.official_regular_atcm_outputs import load_official_regular_outputs

BROWSER_DATA = ROOT / "output/outcome_concern_browser/browser_data.json"
OUTDIR = ROOT / "output/dual_channel_forecast"
TEST_START = 29
TEST_END = 47
TRAIN_START = 19
HISTORY_MEETINGS = 5
NEIGHBORS = 5
ANCESTRY_DEPTH = 3
ANCESTRY_DECAY = 0.5
MODEL_ALPHA = 0.5
BOOTSTRAP_DRAWS = 5000
SEED = 20260821
OUTPUT_TYPES = ("Any output", "Measure", "Decision", "Resolution")
EVIDENCE_MINIMUM = 3

MODEL_FEATURES = {
    "output history": ("output_history",),
    "direct attention": ("output_history", "direct_attention"),
    "neighbor attention": (
        "output_history",
        "direct_attention",
        "neighbor_attention",
    ),
    "formal stock": ("output_history", "formal_stock"),
    "recent same-concern output": ("output_history", "recent_formal_same"),
    "recent nearby output": ("output_history", "recent_formal_nearby"),
    "recent formal activity": (
        "output_history",
        "recent_formal_same",
        "recent_formal_nearby",
    ),
    "formal frontier": ("output_history", "formal_memory_depth0"),
    "inherited frontier": ("output_history", "formal_memory"),
    "attention + formal stock": (
        "output_history",
        "direct_attention",
        "formal_stock",
    ),
    "attention + recent same-concern output": (
        "output_history",
        "direct_attention",
        "recent_formal_same",
    ),
    "attention + recent nearby output": (
        "output_history",
        "direct_attention",
        "recent_formal_nearby",
    ),
    "attention + recent formal activity": (
        "output_history",
        "direct_attention",
        "recent_formal_same",
        "recent_formal_nearby",
    ),
    "attention + formal frontier": (
        "output_history",
        "direct_attention",
        "formal_memory_depth0",
    ),
    "attention + inherited frontier": (
        "output_history",
        "direct_attention",
        "formal_memory",
    ),
}

AREA_MANAGEMENT_TOPICS = {
    "Area Protection and Management Plans General",
    "Management Plans",
    "Marine Protected Areas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-data", type=Path, default=BROWSER_DATA)
    parser.add_argument("--out-dir", type=Path, default=OUTDIR)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=MODEL_ALPHA)
    return parser.parse_args()


def load_browser(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def linked_outcome_topics(
    browser: dict, minimum_rank: int = EVIDENCE_MINIMUM
) -> dict[str, frozenset[str]]:
    papers = browser["papers"]
    result = {}
    for outcome in browser["outcomes"]:
        topics: set[str] = set()
        for edge in outcome["direct_papers"]:
            if int(edge["evidence_rank"]) < minimum_rank:
                continue
            paper = papers.get(edge["paper"])
            if paper:
                topics.update(paper["topics"])
        if topics:
            result[outcome["id"]] = frozenset(topics)
    return result


def official_target_topics(
    outcome_topics: dict[str, frozenset[str]],
    official_outputs: pd.DataFrame,
) -> dict[str, frozenset[str]]:
    """Restrict forecast targets to the pinned regular-ATCM output universe."""
    official_ids = frozenset(official_outputs["output_id"].astype(str))
    return {
        outcome_id: topics
        for outcome_id, topics in outcome_topics.items()
        if outcome_id in official_ids
    }


def outcome_mass_panel(
    browser: dict,
    outcome_topics: dict[str, frozenset[str]],
    topics: list[str],
    meetings: list[int],
) -> pd.DataFrame:
    outcome_lookup = {outcome["id"]: outcome for outcome in browser["outcomes"]}
    rows = []
    for outcome_id, assigned_topics in outcome_topics.items():
        outcome = outcome_lookup[outcome_id]
        meeting = int(outcome["meeting"])
        if meeting not in meetings:
            continue
        weight = 1.0 / len(assigned_topics)
        for topic in assigned_topics:
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "meeting": meeting,
                    "topic": topic,
                    "output_type": outcome["type"],
                    "mass": weight,
                }
            )
    allocations = pd.DataFrame(rows)
    index = pd.MultiIndex.from_product(
        [meetings, topics], names=["meeting", "topic"]
    )
    panel = pd.DataFrame(index=index).reset_index()
    for output_type in OUTPUT_TYPES:
        subset = allocations
        if output_type != "Any output":
            subset = subset[subset["output_type"].eq(output_type)]
        mass = subset.groupby(["meeting", "topic"])["mass"].sum()
        name = output_type.lower().replace(" ", "_") + "_mass"
        panel[name] = panel.set_index(["meeting", "topic"]).index.map(mass).fillna(0.0)
    return panel


def formal_graph(browser: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    for outcome in browser["outcomes"]:
        graph.add_node(
            outcome["id"],
            meeting=int(outcome["meeting"]),
            output_type=outcome["type"],
        )
    for outcome in browser["outcomes"]:
        for edge in outcome["incoming_outcomes"]:
            if edge["outcome"] in graph:
                graph.add_edge(edge["outcome"], outcome["id"])
    return graph


def ancestor_topic_evidence(
    graph: nx.DiGraph,
    node: str,
    outcome_topics: dict[str, frozenset[str]],
    max_depth: int,
) -> dict[str, int]:
    evidence: dict[str, int] = {}
    queue = deque([(node, 0)])
    visited: dict[str, int] = {}
    while queue:
        current, depth = queue.popleft()
        if depth > max_depth or (current in visited and visited[current] <= depth):
            continue
        visited[current] = depth
        for topic in outcome_topics.get(current, frozenset()):
            evidence[topic] = min(evidence.get(topic, depth), depth)
        if depth < max_depth:
            queue.extend((parent, depth + 1) for parent in graph.predecessors(current))
    return evidence


def formal_frontier_nodes(graph: nx.DiGraph) -> list[str]:
    """Return nodes in sink components, including reciprocal endpoint cycles."""
    if not graph:
        return []
    condensed = nx.condensation(graph)
    sink_components = [
        component
        for component in condensed
        if condensed.out_degree(component) == 0
    ]
    return sorted(
        node
        for component in sink_components
        for node in condensed.nodes[component]["members"]
    )


def formal_memory_scores(
    graph: nx.DiGraph,
    outcome_topics: dict[str, frozenset[str]],
    phi: pd.DataFrame,
    cutoff_meeting: int,
    max_depth: int = ANCESTRY_DEPTH,
    decay: float = ANCESTRY_DECAY,
) -> pd.Series:
    prior_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if int(attrs["meeting"]) < cutoff_meeting
    ]
    prior = graph.subgraph(prior_nodes)
    frontiers = formal_frontier_nodes(prior)
    route_scores: list[np.ndarray] = []
    for frontier in frontiers:
        evidence = ancestor_topic_evidence(
            prior, frontier, outcome_topics, max_depth=max_depth
        )
        if not evidence:
            continue
        scores = np.zeros(len(phi), dtype=float)
        for topic, depth in evidence.items():
            if topic not in phi.columns:
                continue
            scores = np.maximum(
                scores,
                (decay**depth) * phi[topic].to_numpy(float),
            )
        if scores.max() > 0:
            route_scores.append(scores)
    if not route_scores:
        return pd.Series(0.0, index=phi.index)
    values = np.vstack(route_scores)
    # Multiple compatible lineage endpoints indicate a denser formal frontier.
    # log1p limits the leverage of recurring administrative series.
    return pd.Series(np.log1p(values.sum(axis=0)), index=phi.index)


def formal_stock_scores(
    graph: nx.DiGraph,
    outcome_topics: dict[str, frozenset[str]],
    phi: pd.DataFrame,
    cutoff_meeting: int,
) -> pd.Series:
    """Measure proximity to all previously documented output concerns."""
    scores = []
    for node, attrs in graph.nodes(data=True):
        if int(attrs["meeting"]) >= cutoff_meeting or node not in outcome_topics:
            continue
        topic_scores = [
            phi[topic].to_numpy(float)
            for topic in outcome_topics[node]
            if topic in phi.columns
        ]
        if topic_scores:
            scores.append(np.max(np.vstack(topic_scores), axis=0))
    if not scores:
        return pd.Series(0.0, index=phi.index)
    return pd.Series(np.log1p(np.vstack(scores).sum(axis=0)), index=phi.index)


def recent_formal_activity_scores(
    graph: nx.DiGraph,
    outcome_topics: dict[str, frozenset[str]],
    phi: pd.DataFrame,
    cutoff_meeting: int,
    horizon: int = HISTORY_MEETINGS,
) -> pd.DataFrame:
    """Separate recent output in the same concern from nearby concerns."""
    same = pd.Series(0.0, index=phi.index)
    nearby = pd.Series(0.0, index=phi.index)
    for node, attrs in graph.nodes(data=True):
        meeting = int(attrs["meeting"])
        if not cutoff_meeting - horizon <= meeting < cutoff_meeting:
            continue
        assigned = frozenset(
            topic
            for topic in outcome_topics.get(node, frozenset())
            if topic in phi.columns
        )
        if not assigned:
            continue
        topic_weight = 1.0 / len(assigned)
        for target in phi.index:
            if target in assigned:
                same.loc[target] += topic_weight
                continue
            nearby.loc[target] += float(
                np.mean([phi.loc[target, source] for source in assigned])
            )
    return pd.DataFrame(
        {
            "same": np.log1p(same),
            "nearby": np.log1p(nearby),
        },
        index=phi.index,
    )


def add_predictors(
    panel: pd.DataFrame,
    browser: dict,
    outcome_topics: dict[str, frozenset[str]],
    topics: list[str],
    meetings: list[int],
) -> pd.DataFrame:
    submitted = load_submitted_with_fallback()
    training, _ = load_paper_training(topics)
    attention = paper_attention_meeting_panel(training, topics, meetings)
    attention_lookup = attention.set_index(["meeting", "topic"])["paper_count"]
    topic_lookup = lineage._canonical_topic_lookup(topics)
    map_cutoffs = sorted({meeting - 1 for meeting in meetings if meeting > min(meetings)})
    phi_by_attention_meeting = cumulative_phi_by_meeting(
        submitted,
        topics,
        topic_lookup,
        map_cutoffs,
    )
    graph = formal_graph(browser)

    result = panel.sort_values(["topic", "meeting"]).copy()
    result["direct_attention"] = 0.0
    result["neighbor_attention"] = 0.0
    result["formal_stock"] = 0.0
    result["recent_formal_same"] = 0.0
    result["recent_formal_nearby"] = 0.0
    result["formal_memory_depth0"] = 0.0
    result["formal_memory"] = 0.0
    for meeting in meetings:
        if meeting - 1 not in phi_by_attention_meeting:
            continue
        phi = phi_by_attention_meeting[meeting - 1].reindex(
            index=topics, columns=topics, fill_value=0.0
        )
        direct = np.array(
            [attention_lookup.get((meeting - 1, topic), 0.0) for topic in topics],
            dtype=float,
        )
        neighbor = local_weight_matrix(phi, k=NEIGHBORS) @ direct
        stock = formal_stock_scores(
            graph,
            outcome_topics,
            phi,
            cutoff_meeting=meeting,
        ).reindex(topics, fill_value=0.0)
        recent_activity = recent_formal_activity_scores(
            graph,
            outcome_topics,
            phi,
            cutoff_meeting=meeting,
        ).reindex(topics, fill_value=0.0)
        frontier = formal_memory_scores(
            graph,
            outcome_topics,
            phi,
            cutoff_meeting=meeting,
            max_depth=0,
        ).reindex(topics, fill_value=0.0)
        inherited = formal_memory_scores(
            graph,
            outcome_topics,
            phi,
            cutoff_meeting=meeting,
        ).reindex(topics, fill_value=0.0)
        row_index = result.index[result["meeting"].eq(meeting)]
        position = {topic: index for index, topic in enumerate(topics)}
        result.loc[row_index, "direct_attention"] = [
            direct[position[topic]] for topic in result.loc[row_index, "topic"]
        ]
        result.loc[row_index, "neighbor_attention"] = [
            neighbor[position[topic]] for topic in result.loc[row_index, "topic"]
        ]
        result.loc[row_index, "formal_stock"] = [
            stock.loc[topic] for topic in result.loc[row_index, "topic"]
        ]
        result.loc[row_index, "recent_formal_same"] = [
            recent_activity.loc[topic, "same"]
            for topic in result.loc[row_index, "topic"]
        ]
        result.loc[row_index, "recent_formal_nearby"] = [
            recent_activity.loc[topic, "nearby"]
            for topic in result.loc[row_index, "topic"]
        ]
        result.loc[row_index, "formal_memory_depth0"] = [
            frontier.loc[topic] for topic in result.loc[row_index, "topic"]
        ]
        result.loc[row_index, "formal_memory"] = [
            inherited.loc[topic] for topic in result.loc[row_index, "topic"]
        ]

    for output_type in OUTPUT_TYPES:
        mass_column = output_type.lower().replace(" ", "_") + "_mass"
        history_column = output_type.lower().replace(" ", "_") + "_history"
        result[history_column] = (
            result.groupby("topic")[mass_column]
            .transform(
                lambda values: values.shift(1).rolling(
                    HISTORY_MEETINGS, min_periods=1
                ).sum()
            )
            .fillna(0.0)
        )
    return result


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    alpha: float,
) -> np.ndarray:
    transformer = ColumnTransformer(
        [
            ("topic", OneHotEncoder(handle_unknown="ignore"), ["topic"]),
            ("numeric", StandardScaler(), list(features)),
        ]
    )
    model = make_pipeline(
        transformer,
        PoissonRegressor(alpha=alpha, max_iter=2000),
    )
    model.fit(train[["topic", *features]], train[target])
    return np.maximum(model.predict(test[["topic", *features]]), 1e-12)


def allocation_log_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed_share = observed / observed.sum()
    predicted_share = predicted / predicted.sum()
    return float(-np.sum(observed_share * np.log(np.clip(predicted_share, 1e-12, 1.0))))


def rolling_forecast(
    panel: pd.DataFrame, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    prediction_rows = []
    for output_type in OUTPUT_TYPES:
        stem = output_type.lower().replace(" ", "_")
        target = f"{stem}_mass"
        history = f"{stem}_history"
        available_meetings = (
            panel.groupby("meeting")[target].sum().loc[lambda values: values > 0].index
        )
        for meeting in range(TEST_START, TEST_END + 1):
            if meeting not in available_meetings:
                continue
            train_meetings = [
                value
                for value in available_meetings
                if TRAIN_START <= value < meeting
            ]
            if len(train_meetings) < 6:
                continue
            train = panel[panel["meeting"].isin(train_meetings)].copy()
            test = panel[panel["meeting"].eq(meeting)].copy()
            observed = test[target].to_numpy(float)
            for model_name, raw_features in MODEL_FEATURES.items():
                features = tuple(
                    history if feature == "output_history" else feature
                    for feature in raw_features
                )
                predicted = fit_predict(train, test, features, target, alpha)
                score_rows.append(
                    {
                        "output_type": output_type,
                        "meeting": meeting,
                        "model": model_name,
                        "allocation_log_score": allocation_log_score(
                            observed, predicted
                        ),
                    }
                )
                for topic, observed_mass, predicted_mass in zip(
                    test["topic"], observed, predicted
                ):
                    prediction_rows.append(
                        {
                            "output_type": output_type,
                            "meeting": meeting,
                            "topic": topic,
                            "model": model_name,
                            "observed_mass": observed_mass,
                            "predicted_mass": predicted_mass,
                        }
                    )
    return pd.DataFrame(score_rows), pd.DataFrame(prediction_rows)


def moving_block_sample(
    meetings: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if len(meetings) <= 2:
        return rng.choice(meetings, size=len(meetings), replace=True)
    starts = rng.integers(0, len(meetings) - 1, size=int(np.ceil(len(meetings) / 2)))
    sampled = np.concatenate([meetings[start : start + 2] for start in starts])
    return sampled[: len(meetings)]


def paired_summary(
    scores: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    comparisons = [
        ("direct attention", "output history"),
        ("neighbor attention", "direct attention"),
        ("formal stock", "output history"),
        ("recent same-concern output", "output history"),
        ("recent nearby output", "output history"),
        ("recent formal activity", "output history"),
        ("formal frontier", "output history"),
        ("formal frontier", "formal stock"),
        ("formal frontier", "recent formal activity"),
        ("inherited frontier", "output history"),
        ("inherited frontier", "formal frontier"),
        ("attention + formal stock", "output history"),
        ("attention + recent same-concern output", "output history"),
        ("attention + recent same-concern output", "direct attention"),
        ("attention + recent nearby output", "output history"),
        ("attention + recent nearby output", "direct attention"),
        ("attention + recent formal activity", "output history"),
        ("attention + recent formal activity", "direct attention"),
        (
            "attention + recent formal activity",
            "attention + recent same-concern output",
        ),
        (
            "attention + recent formal activity",
            "attention + recent nearby output",
        ),
        ("attention + formal frontier", "output history"),
        ("attention + formal frontier", "direct attention"),
        ("attention + formal frontier", "attention + formal stock"),
        (
            "attention + formal frontier",
            "attention + recent nearby output",
        ),
        (
            "attention + formal frontier",
            "attention + recent formal activity",
        ),
        ("attention + inherited frontier", "output history"),
        ("attention + inherited frontier", "direct attention"),
        ("attention + inherited frontier", "attention + formal frontier"),
        ("attention + inherited frontier", "attention + formal stock"),
        (
            "attention + inherited frontier",
            "attention + recent formal activity",
        ),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for output_type, group in scores.groupby("output_type"):
        wide = group.pivot(
            index="meeting", columns="model", values="allocation_log_score"
        ).sort_index()
        for model, baseline in comparisons:
            paired = wide[[model, baseline]].dropna()
            differences = (paired[model] - paired[baseline]).to_numpy(float)
            meetings = paired.index.to_numpy(int)
            boot = np.empty(draws, dtype=float)
            lookup = pd.Series(differences, index=meetings)
            for draw in range(draws):
                sampled = moving_block_sample(meetings, rng)
                boot[draw] = lookup.loc[sampled].mean()
            signs = rng.choice([-1.0, 1.0], size=(draws, len(differences)))
            sign_means = (signs * differences).mean(axis=1)
            observed = float(differences.mean())
            rows.append(
                {
                    "output_type": output_type,
                    "model": model,
                    "baseline": baseline,
                    "meetings": len(differences),
                    "mean_difference": observed,
                    "block_bootstrap_low": float(np.quantile(boot, 0.025)),
                    "block_bootstrap_high": float(np.quantile(boot, 0.975)),
                    "meetings_better": int(np.count_nonzero(differences < 0)),
                    "sign_flip_p_two_sided": float(
                        (1 + np.count_nonzero(np.abs(sign_means) >= abs(observed)))
                        / (draws + 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def rescore_predictions(
    predictions: pd.DataFrame,
    excluded_topics: set[str],
) -> pd.DataFrame:
    retained = predictions[~predictions["topic"].isin(excluded_topics)].copy()
    rows = []
    for keys, group in retained.groupby(["output_type", "meeting", "model"]):
        if group["observed_mass"].sum() <= 0:
            continue
        rows.append(
            {
                "output_type": keys[0],
                "meeting": int(keys[1]),
                "model": keys[2],
                "allocation_log_score": allocation_log_score(
                    group["observed_mass"].to_numpy(float),
                    group["predicted_mass"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_summaries(
    predictions: pd.DataFrame, draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = {
        "all concerns": set(),
        "Management Plans omitted": {"Management Plans"},
        "area-management concerns omitted": AREA_MANAGEMENT_TOPICS,
    }
    all_scores = []
    all_summaries = []
    for index, (label, excluded) in enumerate(specifications.items()):
        scores = rescore_predictions(predictions, excluded)
        scores.insert(0, "subset", label)
        summary = paired_summary(scores, draws=draws, seed=seed + index)
        summary.insert(0, "subset", label)
        all_scores.append(scores)
        all_summaries.append(summary)
    return (
        pd.concat(all_scores, ignore_index=True),
        pd.concat(all_summaries, ignore_index=True),
    )


def diagnostics(
    browser: dict,
    lineage_topics: dict[str, frozenset[str]],
    target_topics: dict[str, frozenset[str]],
    official_outputs: pd.DataFrame,
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    alpha: float,
) -> dict:
    outcome_lookup = {outcome["id"]: outcome for outcome in browser["outcomes"]}
    by_type = defaultdict(int)
    for outcome_id in target_topics:
        by_type[outcome_lookup[outcome_id]["type"]] += 1
    return {
        "concerns": int(panel["topic"].nunique()),
        "meetings": [int(panel["meeting"].min()), int(panel["meeting"].max())],
        "test_meetings": [TEST_START, TEST_END],
        "poisson_alpha": alpha,
        "official_regular_outputs": len(official_outputs),
        "documented_target_outputs": len(target_topics),
        "documented_target_outputs_by_type": dict(sorted(by_type.items())),
        "lineage_nodes_with_concern_evidence": len(lineage_topics),
        "scored_meetings_by_type": {
            key: int(value)
            for key, value in scores.groupby("output_type")["meeting"].nunique().items()
        },
        "timing": (
            "attention from t-1; concern map built before t-1; formal graph "
            "and outcome topics restricted to instruments before t"
        ),
        "target_scope": (
            "official regular-ATCM outputs with confirmed or corroborated "
            "paper--outcome routes"
        ),
        "interpretation": (
            "paired rolling-origin prediction of where documented output mass appears; "
            "not causal adoption or complete coverage of all formal outputs"
        ),
    }


def main() -> None:
    args = parse_args()
    browser = load_browser(args.browser_data)
    topics = sorted(node["id"] for node in browser["nodes"])
    meetings = list(range(TRAIN_START, TEST_END + 1))
    official_outputs = load_official_regular_outputs()
    lineage_topics = linked_outcome_topics(browser)
    target_topics = official_target_topics(lineage_topics, official_outputs)
    target_panel = outcome_mass_panel(browser, target_topics, topics, meetings)
    panel = add_predictors(
        target_panel,
        browser,
        lineage_topics,
        topics,
        meetings,
    )
    scores, predictions = rolling_forecast(panel, alpha=args.alpha)
    summary = paired_summary(scores, draws=args.bootstrap_draws, seed=args.seed)
    sensitivity_scores, sensitivity_summary = sensitivity_summaries(
        predictions,
        draws=args.bootstrap_draws,
        seed=args.seed + 100,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_dir / "analysis_panel.csv", index=False)
    scores.to_csv(args.out_dir / "meeting_scores.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    summary.to_csv(args.out_dir / "paired_summary.csv", index=False)
    sensitivity_scores.to_csv(
        args.out_dir / "sensitivity_meeting_scores.csv", index=False
    )
    sensitivity_summary.to_csv(
        args.out_dir / "sensitivity_summary.csv", index=False
    )
    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(
            diagnostics(
                browser,
                lineage_topics,
                target_topics,
                official_outputs,
                panel,
                scores,
                args.alpha,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote dual-channel forecast outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
