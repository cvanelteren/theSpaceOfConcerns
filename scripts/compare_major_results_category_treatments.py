#!/usr/bin/env python3
"""Run the paper's major analyses under two category treatments.

Treatments
----------
1. ``inferred_primary``: one information-theoretically selected concern per paper.
2. ``fractional_multilabel``: every official concern, with one paper divided
   equally across its labels.

The comparison covers the current paper's three empirical stages (geometry,
portfolio entry, and formal output) and the secondary cohort comparison.  For
formal outputs, a 2x2 design separates the category treatment used to measure
paper attention and geometry from the treatment used to train the title-based
output classifier.  The common-coding comparison uses the multi-label-trained
classifier for both attention treatments.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import scripts.analyze_attention_accumulation as accumulation  # noqa: E402
import scripts.analyze_attention_to_outcomes as outcomes_analysis  # noqa: E402
from fig01_space_of_concerns_topology import load_topic_meta  # noqa: E402
from primary_concern_sensitivity import (  # noqa: E402
    corpus_objects,
    fit_prospective_locality,
    modularity_summary,
    variants,
)
from compare_concern_geometries import (  # noqa: E402
    centralities,
    display_backbone,
    edge_set,
    graph_from_phi,
    strongest_neighbor,
    top_neighbors,
)
from hazard_conditional_logit import (  # noqa: E402
    RCA_THRESHOLD,
    WINDOW_MEETINGS,
    build_periods,
    build_window_interaction,
    choose_period_col,
    phi_from_interaction,
    sanitize_periods,
    topic_first_appearance,
)
from portfolio_displacement import _displacement, get_active  # noqa: E402
from scripts import explore_lineage_space as lineage  # noqa: E402
from utils import (  # noqa: E402
    _split_multi_value,
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
    standardize_index_labels,
)


TREATMENTS = ("inferred_primary", "fractional_multilabel")
OUTROOT = ROOT / "output" / "category_treatment_comparison"
OUT_LONG = OUTROOT / "major_results_overview.csv"
OUT_JSON = OUTROOT / "major_results_summary.json"
N_EDGE_BOOTSTRAP = 1_000
N_NULL_DRAWS = 200
N_ACTOR_BOOTSTRAP = 2_000
N_COHORT_BOOTSTRAP = 400
SEED = 20260814
PHI_STEP = 0.30
COHORT_WINDOW_YEARS = 15
COHORTS = (
    (1961, 1980, "1961-80"),
    (1981, 1990, "1981-90"),
    (1991, 2010, "1991-2010"),
)


def canonical_topics(data: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(
        standardize_index_labels(
            pd.DataFrame(
                index=sorted(extract_unique_topics(data["inferred_primary"]))
            )
        ).index
    )


def paper_training_from_frame(
    submitted: pd.DataFrame,
    topics: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build classifier training rows from an explicit category treatment."""
    topic_lookup = lineage._canonical_topic_lookup(topics)
    rows: list[dict] = []
    paper_categories: dict[str, list[str]] = {}
    for _, record in submitted.drop_duplicates("paper id").iterrows():
        raw_title = record.get("paper name")
        title = "" if pd.isna(raw_title) else str(raw_title).strip()
        meeting = pd.to_numeric(record.get("meeting number"), errors="coerce")
        year = pd.to_numeric(record.get("meeting year"), errors="coerce")
        raw = "" if pd.isna(record.get("category")) else str(record.get("category"))
        categories: list[str] = []
        for raw_category in raw.split("\t"):
            category = topic_lookup.get(lineage._normalize(raw_category))
            if category and category not in categories:
                categories.append(category)
        if not title or pd.isna(meeting) or not categories:
            continue
        paper_id = f"paper:{int(record['paper id'])}"
        paper_categories[paper_id] = categories
        rows.append(
            {
                "paper_id": paper_id,
                "meeting": int(meeting),
                "year": int(year) if not pd.isna(year) else np.nan,
                "title": title,
                "topics": categories,
            }
        )
    return pd.DataFrame(rows), paper_categories


def edge_bootstrap(
    counts: pd.DataFrame,
    treatment: str,
) -> tuple[dict, pd.DataFrame]:
    topics = list(counts.index)
    values = counts.to_numpy(dtype=float)
    n_topics, n_actors = values.shape
    first, second = np.triu_indices(n_topics, k=1)
    edge_index = {
        (int(i), int(j)): position
        for position, (i, j) in enumerate(zip(first, second))
    }
    weights = np.empty((N_EDGE_BOOTSTRAP, len(first)), dtype=np.float32)
    selected = np.zeros(len(first), dtype=np.int32)
    rng = np.random.default_rng(SEED)
    for draw in range(N_EDGE_BOOTSTRAP):
        sample = rng.integers(0, n_actors, size=n_actors)
        sampled = pd.DataFrame(
            values[:, sample],
            index=topics,
            columns=[f"actor_{slot}" for slot in range(n_actors)],
        )
        phi = phi_from_interaction(sampled, topics)
        weights[draw] = phi[first, second]
        # Match the project's edge-stability routine exactly.  The diagonal
        # is not a candidate neighbour; setting it to zero before the unstable
        # argsort also makes tied off-diagonal values break identically.
        np.fill_diagonal(phi, 0.0)
        chosen: set[tuple[int, int]] = set()
        for i in range(n_topics):
            order = np.argsort(phi[i])[::-1]
            positive = [int(j) for j in order if j != i and phi[i, j] > 0]
            for j in positive[:5]:
                chosen.add(tuple(sorted((i, j))))
        for pair in chosen:
            selected[edge_index[pair]] += 1
    observed = phi_from_interaction(counts, topics)
    edges = pd.DataFrame(
        {
            "concern_a": [topics[i] for i in first],
            "concern_b": [topics[j] for j in second],
            "phi_observed": observed[first, second],
            "phi_median": np.median(weights, axis=0),
            "phi_ci_low": np.quantile(weights, 0.025, axis=0),
            "phi_ci_high": np.quantile(weights, 0.975, axis=0),
            "endpoint_top5_frequency": selected / N_EDGE_BOOTSTRAP,
        }
    ).sort_values(["endpoint_top5_frequency", "phi_observed"], ascending=False)
    edges.to_csv(OUTROOT / treatment / "edge_bootstrap.csv", index=False)
    positive = edges["phi_observed"].gt(0)
    top_decile = edges.nlargest(int(np.ceil(len(edges) * 0.10)), "phi_observed")
    summary = {
        "positive_observed_edges": int(positive.sum()),
        "positive_95pct_lower_bound_edges": int(edges["phi_ci_low"].gt(0).sum()),
        "endpoint_top5_frequency_at_least_0_50": int(
            edges["endpoint_top5_frequency"].ge(0.50).sum()
        ),
        "endpoint_top5_frequency_at_least_0_60_and_positive_lower_bound": int(
            (
                edges["endpoint_top5_frequency"].ge(0.60)
                & edges["phi_ci_low"].gt(0)
            ).sum()
        ),
        "median_endpoint_top5_frequency_among_top_decile": float(
            top_decile["endpoint_top5_frequency"].median()
        ),
    }
    return summary, edges


def strict_matched_displacement(
    submitted: pd.DataFrame,
    topics: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Fully prospective, popularity-matched displacement comparison."""
    period_col = choose_period_col(submitted)
    clean = sanitize_periods(submitted, period_col)
    members_raw = extract_unique_countries(clean)
    topics_raw = extract_unique_topics(clean)
    base = standardize_index_labels(
        generate_interaction_matrix(clean, members_raw, topics_raw)
    ).reindex(index=topics, fill_value=0.0)
    members = sorted(base.columns)
    period_min = int(clean[period_col].min())
    period_max = int(clean[period_col].max())
    periods = build_periods(period_min, period_max, WINDOW_MEETINGS)
    first = pd.Series(topic_first_appearance(clean, period_col)).reindex(topics)
    if first.isna().any():
        raise ValueError("All canonical topics must have a first appearance")
    first_values = first.to_numpy(dtype=int)
    active_by_period = []
    for start, end in periods:
        interaction = build_window_interaction(
            clean,
            period_col,
            int(start),
            int(end),
            set(members_raw),
            set(topics_raw),
            topics,
            members,
        )
        active_by_period.append(get_active(interaction))
    rng = np.random.default_rng(1994)
    rows = []
    for t in range(1, len(periods)):
        previous_end = int(periods[t - 1][1])
        cumulative = build_window_interaction(
            clean,
            period_col,
            period_min,
            previous_end,
            set(members_raw),
            set(topics_raw),
            topics,
            members,
        )
        phi = phi_from_interaction(cumulative, topics)
        previous = active_by_period[t - 1]
        current = active_by_period[t]
        popularity = previous.sum(axis=1).to_numpy(dtype=float)
        appeared = np.flatnonzero(first_values <= previous_end)
        for member in members:
            held = np.flatnonzero(previous[member].to_numpy(dtype=bool))
            current_indices = np.flatnonzero(current[member].to_numpy(dtype=bool))
            if held.size == 0 or current_indices.size == 0:
                continue
            added = np.intersect1d(np.setdiff1d(current_indices, held), appeared)
            available = np.setdiff1d(appeared, held)
            if added.size == 0 or available.size < added.size:
                continue
            probabilities = popularity[available] + 1.0
            probabilities /= probabilities.sum()
            observed = _displacement(added, held, phi)
            null = np.asarray(
                [
                    _displacement(
                        rng.choice(
                            available,
                            size=added.size,
                            replace=False,
                            p=probabilities,
                        ),
                        held,
                        phi,
                    )
                    for _ in range(N_NULL_DRAWS)
                ]
            )
            rows.append(
                {
                    "member": member,
                    "period_end": int(periods[t][1]),
                    "n_added": int(added.size),
                    "displacement": float(observed),
                    "null_mean": float(null.mean()),
                }
            )
    moved = pd.DataFrame(rows)
    moved["nearer"] = moved["displacement"] < moved["null_mean"]
    by_actor = moved.groupby("member")["nearer"].agg(["sum", "count"])
    successes = by_actor["sum"].to_numpy(dtype=float)
    totals = by_actor["count"].to_numpy(dtype=float)
    rng_boot = np.random.default_rng(1993)
    draws = np.empty(N_ACTOR_BOOTSTRAP)
    for draw in range(N_ACTOR_BOOTSTRAP):
        sample = rng_boot.integers(0, len(by_actor), size=len(by_actor))
        draws[draw] = successes[sample].sum() / totals[sample].sum()
    delta = moved["displacement"] - moved["null_mean"]
    summary = {
        "n_actor_periods": int(len(moved)),
        "share_nearer_than_popularity_matched_null": float(moved["nearer"].mean()),
        "actor_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
        "actor_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        "observed_median_displacement": float(moved["displacement"].median()),
        "null_median_displacement": float(moved["null_mean"].median()),
        "wilcoxon_p": float(stats.wilcoxon(delta, alternative="less").pvalue),
    }
    return summary, moved


def cohort_comparison(
    submitted: pd.DataFrame,
    topics: list[str],
    phi: np.ndarray,
    pooled_active: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Continuous opening-portfolio comparison used in the supplement."""
    off_diagonal = phi.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    reach = np.nansum(off_diagonal >= PHI_STEP, axis=1).astype(float)
    actor_breadth = pooled_active.sum(axis=0).to_numpy(dtype=float)
    holders = pooled_active.sum(axis=1).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        holder_breadth = (
            pooled_active.to_numpy(dtype=float) * actor_breadth[None, :]
        ).sum(axis=1) / holders
    holder_breadth = np.nan_to_num(holder_breadth, nan=0.0)

    data = submitted.copy()
    if "year" not in data.columns:
        data["year"] = data["meeting year"]
    actor_years = []
    for _, record in data.dropna(subset=["submitted by"]).iterrows():
        for actor in _split_multi_value(record["submitted by"], delimiter=","):
            actor_years.append((actor, int(record["year"])))
    actor_years = pd.DataFrame(actor_years, columns=["actor", "year"])
    first = actor_years.groupby("actor")["year"].min()
    last = actor_years.groupby("actor")["year"].max()
    eligible = [
        actor
        for actor in first.index
        if last[actor] - first[actor] >= COHORT_WINDOW_YEARS - 1
    ]
    windows = sorted(
        {
            (int(first[actor]), int(first[actor]) + COHORT_WINDOW_YEARS - 1)
            for actor in eligible
        }
    )
    countries_raw = extract_unique_countries(data)
    topics_raw = extract_unique_topics(data)
    matrices = {}
    for low, high in windows:
        subset = data[data["year"].between(low, high)]
        matrix = standardize_index_labels(
            generate_interaction_matrix(subset, countries_raw, topics_raw)
        )
        if matrix.index.has_duplicates:
            matrix = matrix.groupby(level=0).sum()
        matrices[(low, high)] = matrix.reindex(index=topics, fill_value=0.0)
    all_actors = sorted(matrices[windows[0]].columns)

    def replicate(columns: list[str], seen: dict[str, set[str]] | None = None):
        result = {
            name: {"reach": [], "holder_breadth": []}
            for _, _, name in COHORTS
        }
        for low, high in windows:
            matrix = matrices[(low, high)][columns].copy()
            matrix.columns = [f"{actor}_{i}" for i, actor in enumerate(columns)]
            active = get_rca(matrix).to_numpy(dtype=float) >= RCA_THRESHOLD
            if active.sum() == 0:
                continue
            available_reach = float((active * reach[:, None]).sum() / active.sum())
            available_breadth = float(
                (active * holder_breadth[:, None]).sum() / active.sum()
            )
            for column_index, actor in enumerate(columns):
                if actor not in first.index:
                    continue
                actor_window = (
                    int(first[actor]),
                    int(first[actor]) + COHORT_WINDOW_YEARS - 1,
                )
                if actor_window != (low, high):
                    continue
                held = active[:, column_index]
                if held.sum() < 3:
                    continue
                actor_reach = float((held * reach).sum() / held.sum())
                actor_holder_breadth = float(
                    (held * holder_breadth).sum() / held.sum()
                )
                for cohort_low, cohort_high, name in COHORTS:
                    if cohort_low <= first[actor] <= cohort_high:
                        result[name]["reach"].append(actor_reach / available_reach)
                        result[name]["holder_breadth"].append(
                            actor_holder_breadth / available_breadth
                        )
                        if seen is not None:
                            seen.setdefault(name, set()).add(actor)
        return {
            name: {
                measure: float(np.mean(values)) if values else np.nan
                for measure, values in measures.items()
            }
            for name, measures in result.items()
        }

    contributors: dict[str, set[str]] = {}
    observed = replicate(all_actors, seen=contributors)
    rng = np.random.default_rng(23)
    bootstrap = [
        replicate(list(rng.choice(all_actors, len(all_actors), replace=True)))
        for _ in range(N_COHORT_BOOTSTRAP)
    ]
    rows = []
    for _, _, name in COHORTS:
        for measure in ("holder_breadth", "reach"):
            values = np.asarray([draw[name][measure] for draw in bootstrap])
            rows.append(
                {
                    "cohort": name,
                    "measure": measure,
                    "ratio": observed[name][measure],
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "n_actors": len(contributors.get(name, set())),
                }
            )
    result = pd.DataFrame(rows)
    summary = {}
    for cohort, group in result.groupby("cohort"):
        summary[cohort] = {
            record["measure"]: {
                "ratio": record["ratio"],
                "ci_low": record["ci_low"],
                "ci_high": record["ci_high"],
                "n_actors": int(record["n_actors"]),
            }
            for record in group.to_dict(orient="records")
        }
    return summary, result


def classify_outputs(
    treatment: str,
    submitted: pd.DataFrame,
    topics: list[str],
    phi: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    treatment_dir = OUTROOT / treatment
    treatment_dir.mkdir(parents=True, exist_ok=True)
    outcomes_analysis.OUTDIR = treatment_dir
    training, _ = paper_training_from_frame(submitted, topics)
    outcomes = outcomes_analysis.load_outcomes()
    pooled_phi = pd.DataFrame(phi, index=topics, columns=topics)
    np.fill_diagonal(pooled_phi.values, 1.0)
    _, region_raw, x_raw = load_topic_meta()
    region_of = {
        topic: int(region_raw[lineage.normalize_topic_key(topic)]) for topic in topics
    }
    x_position = {
        topic: float(x_raw[lineage.normalize_topic_key(topic)]) for topic in topics
    }
    title_predictions, title_probabilities, metrics = outcomes_analysis.classify_outcomes(
        training, outcomes, pooled_phi, x_position, region_of
    )
    title_predictions.to_csv(
        treatment_dir / "outcome_topic_predictions_title_model.csv", index=False
    )
    title_probabilities.to_csv(
        treatment_dir / "outcome_topic_probabilities_title_model.csv", index=False
    )
    predictions, probabilities, official_metrics = (
        outcomes_analysis.official_output_allocations(outcomes, topics)
    )
    predictions.to_csv(treatment_dir / "outcome_topic_predictions.csv", index=False)
    probabilities.to_csv(treatment_dir / "outcome_topic_probabilities.csv", index=False)
    (treatment_dir / "title_classifier_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (treatment_dir / "official_output_allocation_metrics.json").write_text(
        json.dumps(official_metrics, indent=2) + "\n", encoding="utf-8"
    )
    return predictions, probabilities, metrics, training


def formal_output_models(
    attention_treatment: str,
    output_coding_treatment: str,
    submitted: pd.DataFrame,
    topics: list[str],
    training: pd.DataFrame,
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Fit accumulation models for one attention/coding combination."""
    topic_lookup = lineage._canonical_topic_lookup(topics)
    meetings = list(
        range(outcomes_analysis.START_MEETING, outcomes_analysis.END_MEETING + 1)
    )
    phi_by_meeting = outcomes_analysis.cumulative_phi_by_meeting(
        submitted, topics, topic_lookup, meetings
    )
    attention = outcomes_analysis.paper_attention_meeting_panel(
        training, topics, meetings
    )
    output_mass = outcomes_analysis.outcome_mass_meeting_panel(
        predictions, probabilities, topics, meetings
    )
    panel = outcomes_analysis.build_topic_meeting_panel(
        attention, output_mass, phi_by_meeting, topics
    )
    stocks = accumulation.add_attention_stocks(panel)
    models = accumulation.fit_all_models(stocks, {})
    combination = f"attention_{attention_treatment}__coding_{output_coding_treatment}"
    combination_dir = OUTROOT / "formal_outputs" / combination
    combination_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(combination_dir / "topic_meeting_panel.csv", index=False)
    models.to_csv(combination_dir / "attention_accumulation_models.csv", index=False)
    summary = {
        "attention_treatment": attention_treatment,
        "output_coding_treatment": output_coding_treatment,
        "n_topic_meetings": int(len(panel)),
        "models_path": str(combination_dir / "attention_accumulation_models.csv"),
    }
    return models, summary


def add_estimate(
    rows: list[dict],
    domain: str,
    treatment: str,
    result: str,
    estimate: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
    scale: str = "",
    output_coding: str = "",
    n: int | None = None,
) -> None:
    rows.append(
        {
            "domain": domain,
            "category_treatment": treatment,
            "output_coding_treatment": output_coding,
            "result": result,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "scale": scale,
            "n": n,
        }
    )


def main() -> None:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    data = {name: frame for name, frame in variants().items() if name in TREATMENTS}
    topics = canonical_topics(data)
    objects = {
        name: corpus_objects(frame, topics) for name, frame in data.items()
    }
    overview_rows: list[dict] = []
    payload: dict = {
        "purpose": "Category-treatment sensitivity of all current major results",
        "treatments": {
            "inferred_primary": "one information-theoretically selected concern per paper",
            "fractional_multilabel": "equal fractional credit across every official archive category",
        },
        "geometry": {},
        "portfolio_entry": {},
        "cohort": {},
        "formal_outputs": {"classifiers": {}, "model_combinations": {}},
    }

    print("[1/4] Geometry and edge stability")
    for treatment in TREATMENTS:
        treatment_dir = OUTROOT / treatment
        treatment_dir.mkdir(parents=True, exist_ok=True)
        counts, phi, active, _, _ = objects[treatment]
        graph = graph_from_phi(phi, topics)
        mst, backbone = display_backbone(graph)
        network = modularity_summary(phi)
        centre = centralities(graph, topics)
        edge_summary, edges = edge_bootstrap(counts, treatment)
        liability = edges[
            (
                edges["concern_a"].eq("Inspections")
                & edges["concern_b"].eq("Liability")
            )
            | (
                edges["concern_a"].eq("Liability")
                & edges["concern_b"].eq("Inspections")
            )
        ].iloc[0]
        geometry = {
            **network,
            "mst_edges": len(edge_set(mst)),
            "display_backbone_edges": len(edge_set(backbone)),
            "highest_strength_concern": str(centre["strength"].idxmax()),
            "highest_closeness_concern": str(centre["closeness"].idxmax()),
            "edge_bootstrap": edge_summary,
            "inspections_liability": {
                "phi": float(liability["phi_observed"]),
                "ci_low": float(liability["phi_ci_low"]),
                "ci_high": float(liability["phi_ci_high"]),
                "endpoint_top5_frequency": float(
                    liability["endpoint_top5_frequency"]
                ),
            },
        }
        payload["geometry"][treatment] = geometry
        add_estimate(
            overview_rows,
            "geometry",
            treatment,
            "maximum_louvain_modularity",
            network["louvain_modularity_max_100_seeds"],
        )
        add_estimate(
            overview_rows,
            "geometry",
            treatment,
            "positive_pair_share",
            network["positive_pair_share"],
            n=990,
        )
        add_estimate(
            overview_rows,
            "geometry",
            treatment,
            "inspections_liability_phi",
            float(liability["phi_observed"]),
            float(liability["phi_ci_low"]),
            float(liability["phi_ci_high"]),
            "actor-bootstrap 95% interval",
            n=N_EDGE_BOOTSTRAP,
        )

    primary_phi = objects["inferred_primary"][1]
    fractional_phi = objects["fractional_multilabel"][1]
    primary_graph = graph_from_phi(primary_phi, topics)
    fractional_graph = graph_from_phi(fractional_phi, topics)
    primary_mst, primary_backbone = display_backbone(primary_graph)
    fractional_mst, fractional_backbone = display_backbone(fractional_graph)
    primary_mst_edges = edge_set(primary_mst)
    fractional_mst_edges = edge_set(fractional_mst)
    primary_backbone_edges = edge_set(primary_backbone)
    fractional_backbone_edges = edge_set(fractional_backbone)
    upper = np.triu_indices(len(topics), k=1)
    primary_tree_steps = np.asarray(
        nx.floyd_warshall_numpy(primary_mst, nodelist=topics, weight=None)
    )[upper]
    fractional_tree_steps = np.asarray(
        nx.floyd_warshall_numpy(fractional_mst, nodelist=topics, weight=None)
    )[upper]
    primary_neighbours = top_neighbors(primary_phi, topics)
    fractional_neighbours = top_neighbors(fractional_phi, topics)
    neighbour_jaccards = []
    for topic in topics:
        left = primary_neighbours[topic]
        right = fractional_neighbours[topic]
        neighbour_jaccards.append(len(left & right) / len(left | right))
    primary_strongest = strongest_neighbor(primary_phi, topics)
    fractional_strongest = strongest_neighbor(fractional_phi, topics)
    primary_centrality = centralities(primary_graph, topics)
    fractional_centrality = centralities(fractional_graph, topics)
    payload["geometry"]["fractional_vs_inferred_primary"] = {
        "mst_edge_jaccard": float(
            len(primary_mst_edges & fractional_mst_edges)
            / len(primary_mst_edges | fractional_mst_edges)
        ),
        "primary_mst_edges_retained": float(
            len(primary_mst_edges & fractional_mst_edges) / len(primary_mst_edges)
        ),
        "display_backbone_edge_jaccard": float(
            len(primary_backbone_edges & fractional_backbone_edges)
            / len(primary_backbone_edges | fractional_backbone_edges)
        ),
        "primary_display_backbone_edges_retained": float(
            len(primary_backbone_edges & fractional_backbone_edges)
            / len(primary_backbone_edges)
        ),
        "mst_pairwise_step_spearman": float(
            stats.spearmanr(primary_tree_steps, fractional_tree_steps).statistic
        ),
        "mean_top5_neighbor_jaccard": float(np.mean(neighbour_jaccards)),
        "strongest_neighbor_agreement": float(
            np.mean(
                [
                    primary_strongest[topic] == fractional_strongest[topic]
                    for topic in topics
                ]
            )
        ),
        "strength_rank_spearman": float(
            stats.spearmanr(
                primary_centrality["strength"],
                fractional_centrality["strength"],
            ).statistic
        ),
        "closeness_rank_spearman": float(
            stats.spearmanr(
                primary_centrality["closeness"],
                fractional_centrality["closeness"],
            ).statistic
        ),
        "betweenness_rank_spearman": float(
            stats.spearmanr(
                primary_centrality["betweenness"],
                fractional_centrality["betweenness"],
            ).statistic
        ),
    }

    print("[2/4] Portfolio entry and matched comparison")
    for treatment in TREATMENTS:
        locality = fit_prospective_locality(data[treatment], topics)
        matched, moved = strict_matched_displacement(data[treatment], topics)
        moved.to_csv(OUTROOT / treatment / "matched_displacement_events.csv", index=False)
        payload["portfolio_entry"][treatment] = {
            "conditional_logit": locality,
            "popularity_matched_displacement": matched,
        }
        add_estimate(
            overview_rows,
            "portfolio_entry",
            treatment,
            "rpa_crossing_or_per_0_1_distance",
            locality["odds_ratio_per_0_1"],
            locality["odds_ratio_per_0_1_ci_low"],
            locality["odds_ratio_per_0_1_ci_high"],
            "conditional odds ratio",
            n=locality["risk_rows"],
        )
        add_estimate(
            overview_rows,
            "portfolio_entry",
            treatment,
            "new_document_or_per_0_1_distance",
            locality["new_document_odds_ratio_per_0_1"],
            locality["new_document_odds_ratio_per_0_1_ci_low"],
            locality["new_document_odds_ratio_per_0_1_ci_high"],
            "conditional odds ratio",
            n=locality["new_document_risk_rows"],
        )
        add_estimate(
            overview_rows,
            "portfolio_entry",
            treatment,
            "share_nearer_than_popularity_matched_null",
            matched["share_nearer_than_popularity_matched_null"],
            matched["actor_bootstrap_ci_low"],
            matched["actor_bootstrap_ci_high"],
            "share of actor-period expansion events",
            n=matched["n_actor_periods"],
        )

    print("[3/4] Cohort comparison")
    for treatment in TREATMENTS:
        _, phi, active, _, _ = objects[treatment]
        cohort_summary, cohort_table = cohort_comparison(
            data[treatment], topics, phi, active
        )
        cohort_table.to_csv(OUTROOT / treatment / "cohort_comparison.csv", index=False)
        payload["cohort"][treatment] = cohort_summary
        for record in cohort_table.to_dict(orient="records"):
            add_estimate(
                overview_rows,
                "cohort",
                treatment,
                f"{record['cohort']}__{record['measure']}",
                record["ratio"],
                record["ci_low"],
                record["ci_high"],
                "opening-portfolio value divided by contemporaneous availability",
                n=int(record["n_actors"]),
            )

    print("[4/4] Output classifiers and formal-output models")
    classifier_results = {}
    for treatment in TREATMENTS:
        _, phi, _, _, _ = objects[treatment]
        predictions, probabilities, metrics, training = classify_outputs(
            treatment, data[treatment], topics, phi
        )
        classifier_results[treatment] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": metrics,
            "training": training,
        }
        payload["formal_outputs"]["classifiers"][treatment] = metrics
        add_estimate(
            overview_rows,
            "formal_output_classifier",
            treatment,
            "top1_matches_any_paper_category",
            metrics["top1_matches_any_official_category"],
            n=metrics["n_papers"],
        )
        add_estimate(
            overview_rows,
            "formal_output_classifier",
            treatment,
            "top3_contains_any_paper_category",
            metrics["top3_contains_any_official_category"],
            n=metrics["n_papers"],
        )

    primary_predictions = classifier_results["inferred_primary"]["predictions"].set_index(
        "outcome_id"
    )
    multilabel_predictions = classifier_results["fractional_multilabel"]["predictions"].set_index(
        "outcome_id"
    )
    common_outcomes = primary_predictions.index.intersection(
        multilabel_predictions.index
    )
    top1_agreement = float(
        (
            primary_predictions.loc[common_outcomes, "topic_top1"]
            == multilabel_predictions.loc[common_outcomes, "topic_top1"]
        ).mean()
    )
    payload["formal_outputs"]["official_output_allocation_treatment_agreement"] = {
        "n_outputs": int(len(common_outcomes)),
        "top1_agreement": top1_agreement,
    }
    add_estimate(
        overview_rows,
        "formal_output_allocation",
        "fractional_multilabel_vs_inferred_primary",
        "official_output_top1_assignment_agreement",
        top1_agreement,
        n=len(common_outcomes),
    )
    for attention_treatment in TREATMENTS:
        for coding_treatment in TREATMENTS:
            coder = classifier_results[coding_treatment]
            models, model_summary = formal_output_models(
                attention_treatment,
                coding_treatment,
                data[attention_treatment],
                topics,
                paper_training_from_frame(data[attention_treatment], topics)[0],
                coder["predictions"],
                coder["probabilities"],
            )
            combination = (
                f"attention_{attention_treatment}__coding_{coding_treatment}"
            )
            payload["formal_outputs"]["model_combinations"][combination] = model_summary
            selected = models[
                models["specification"].eq("accumulated_attention_hard_output")
                | (
                    models["specification"].eq(
                        "accumulated_attention_soft_output"
                    )
                    & models["horizon_meetings"].eq(5)
                )
                | models["specification"].eq(
                    "new_output_episode_after_five_quiet_meetings"
                )
            ]
            for record in selected.to_dict(orient="records"):
                if record["specification"].startswith("accumulated_attention"):
                    estimate = record["ratio_per_doubling_plus_one"]
                    low = record["doubling_ci_low"]
                    high = record["doubling_ci_high"]
                    scale = "rate ratio per doubling of one plus accumulated count"
                else:
                    estimate = record["incidence_rate_ratio"]
                    low = record["ci_low"]
                    high = record["ci_high"]
                    scale = record["scale"]
                add_estimate(
                    overview_rows,
                    "formal_outputs",
                    attention_treatment,
                    f"{record['specification']}__{record['predictor']}",
                    estimate,
                    low,
                    high,
                    scale,
                    output_coding=coding_treatment,
                    n=int(record["n_topic_meetings"]),
                )

    overview = pd.DataFrame(overview_rows)
    overview.to_csv(OUT_LONG, index=False)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_LONG}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
