#!/usr/bin/env python3
"""Test whether historically central paper concerns predict formal contribution.

The unit is a candidate paper--outcome pair.  For every focal ATCM, all graph
quantities are rebuilt from submissions made at earlier meetings.  The models
compare papers around the same output, so centrality belongs to the paper's
concern; the centrality of the output concern itself is constant within an
output and cannot be estimated in an output-stratified model.

The primary question is deliberately incremental.  Does a standard graph
quantity add information after exact concern alignment, proximity through
other concerns, title overlap, paper type, and the paper concern's prior
document volume have already been accounted for?
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.discrete.conditional_models import ConditionalLogit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import build_graphs
from scripts import explore_lineage_space as lineage
import scripts.analyze_attention_to_outcomes as base
from utils import (
    compute_product_space,
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
)


OUTDIR = ROOT / "output" / "outcome_linkage"
PANEL_PATH = OUTDIR / "space_discrimination_panel_consensus.csv"
METRICS_PATH = OUTDIR / "prior_meeting_topic_centrality.csv"
MODELS_PATH = OUTDIR / "outcome_topic_centrality_models.csv"
CORRELATIONS_PATH = OUTDIR / "outcome_topic_centrality_correlations.csv"
CV_PATH = OUTDIR / "outcome_topic_centrality_cv.csv"
TOPIC_MODELS_PATH = OUTDIR / "topic_output_centrality_models.csv"
TOPIC_CV_PATH = OUTDIR / "topic_output_centrality_cv.csv"
REPORT_PATH = OUTDIR / "outcome_topic_centrality_report.md"
SUMMARY_PATH = OUTDIR / "outcome_topic_centrality_summary.json"

RPA_THRESHOLD = 1.0
COMPLEXITY_MIN_PRIOR_ACTIVITY = 20
N_FOLDS = 5
N_BOOTSTRAP = 5000
SEED = 20260814

CORE_TERMS = [
    "same_concern_mass",
    "related_concern_proximity",
    "title_overlap",
    "working_paper",
]
POPULARITY_TERM = "prior_document_volume_percentile"
CENTRALITY_TERMS = [
    "strength_percentile",
    "degree_percentile",
    "closeness_percentile",
    "betweenness_percentile",
    "eigenvector_percentile",
    "pagerank_percentile",
    "clustering_percentile",
    "ubiquity_percentile",
    "holder_breadth_percentile",
    "topic_complexity_percentile",
]


def zscore_safe(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standard_deviation = float(np.nanstd(values))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.nanmean(values))) / standard_deviation


def topic_complexity(active: pd.DataFrame) -> pd.Series:
    """Economic-complexity second-eigenvector score for topics.

    This follows the symmetric topic-side operator used by the project's
    archived complexity analysis.  As in that implementation, the otherwise
    arbitrary eigenvector sign is oriented to correlate positively with topic
    ubiquity.
    """
    topics = active.index
    matrix = active.to_numpy(dtype=float).T  # actors x topics
    actor_diversity = matrix.sum(axis=1)
    topic_ubiquity = matrix.sum(axis=0)
    active_actor = actor_diversity > 0
    active_topic = topic_ubiquity > 0
    result = pd.Series(np.nan, index=topics, name="topic_complexity")
    if active_actor.sum() < 2 or active_topic.sum() < 3:
        return result

    reduced = matrix[np.ix_(active_actor, active_topic)]
    actor_diversity = reduced.sum(axis=1)
    topic_ubiquity = reduced.sum(axis=0)
    inv_actor = 1.0 / actor_diversity
    inv_sqrt_topic = 1.0 / np.sqrt(topic_ubiquity)
    operator = (
        np.diag(inv_sqrt_topic)
        @ reduced.T
        @ np.diag(inv_actor)
        @ reduced
        @ np.diag(inv_sqrt_topic)
    )
    _, eigenvectors = np.linalg.eigh(operator)
    vector = inv_sqrt_topic * eigenvectors[:, -2]

    if np.std(vector) > 0 and np.std(topic_ubiquity) > 0:
        if np.corrcoef(vector, topic_ubiquity)[0, 1] < 0:
            vector = -vector
    vector = zscore_safe(vector)
    result.loc[np.asarray(topics)[active_topic]] = vector
    return result


def graph_metrics(
    phi: pd.DataFrame,
    counts: pd.DataFrame,
    prior_document_volume: pd.Series,
) -> pd.DataFrame:
    topics = list(phi.index)
    values = phi.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, 0.0)

    graph = nx.Graph()
    graph.add_nodes_from(topics)
    for i, source in enumerate(topics):
        for j in range(i + 1, len(topics)):
            weight = float(values[i, j])
            if weight <= 0:
                continue
            graph.add_edge(
                source,
                topics[j],
                weight=weight,
                distance=max(float(-np.log(np.clip(weight, 1e-12, 1.0))), 1e-9),
            )

    strength = pd.Series(values.sum(axis=1), index=topics)
    degree = pd.Series((values > 0).sum(axis=1), index=topics, dtype=float)
    closeness = pd.Series(nx.closeness_centrality(graph, distance="distance"))
    betweenness = pd.Series(
        nx.betweenness_centrality(graph, weight="distance", normalized=True)
    )
    if graph.number_of_edges() == 0:
        eigenvector = pd.Series(0.0, index=topics)
        pagerank = pd.Series(1.0 / len(topics), index=topics)
        clustering = pd.Series(0.0, index=topics)
    else:
        try:
            eigenvector = pd.Series(
                nx.eigenvector_centrality(graph, weight="weight", max_iter=5000)
            )
        except nx.PowerIterationFailedConvergence:
            eigenvector = pd.Series(0.0, index=topics)
        pagerank = pd.Series(nx.pagerank(graph, weight="weight"))
        clustering = pd.Series(nx.clustering(graph, weight="weight"))

    rpa = get_rca(counts)
    # The product-space code uses a strict threshold; portfolio support and
    # the archived complexity construction use RPA >= 1.  Retain that existing
    # distinction here rather than silently redefining either quantity.
    active = (rpa >= RPA_THRESHOLD).astype(int)
    actor_diversity = active.sum(axis=0)
    topic_ubiquity = active.sum(axis=1)
    holder_breadth = active.mul(actor_diversity, axis=1).sum(axis=1).div(
        topic_ubiquity.replace(0, np.nan)
    )

    # Match the archived complexity construction's activity filter, but apply
    # it using only information available before the focal meeting.
    eligible_actors = counts.sum(axis=0) >= COMPLEXITY_MIN_PRIOR_ACTIVITY
    complexity = topic_complexity(active.loc[:, eligible_actors])

    frame = pd.DataFrame(
        {
            "topic": topics,
            "strength": strength.reindex(topics).to_numpy(float),
            "degree": degree.reindex(topics).to_numpy(float),
            "closeness": closeness.reindex(topics).fillna(0).to_numpy(float),
            "betweenness": betweenness.reindex(topics).fillna(0).to_numpy(float),
            "eigenvector": eigenvector.reindex(topics).fillna(0).to_numpy(float),
            "pagerank": pagerank.reindex(topics).fillna(0).to_numpy(float),
            "clustering": clustering.reindex(topics).fillna(0).to_numpy(float),
            "ubiquity": topic_ubiquity.reindex(topics).fillna(0).to_numpy(float),
            "holder_breadth": holder_breadth.reindex(topics).to_numpy(float),
            "topic_complexity": complexity.reindex(topics).to_numpy(float),
            "prior_document_volume": prior_document_volume.reindex(topics)
            .fillna(0)
            .to_numpy(float),
        }
    )
    for term in [
        "strength",
        "degree",
        "closeness",
        "betweenness",
        "eigenvector",
        "pagerank",
        "clustering",
        "ubiquity",
        "holder_breadth",
        "topic_complexity",
        "prior_document_volume",
    ]:
        frame[f"{term}_percentile"] = frame[term].rank(
            pct=True, method="average", na_option="keep"
        )
    return frame


def historical_metrics() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    _, _, _, pooled_counts, _ = build_graphs()
    topics = list(pooled_counts.index)
    topic_lookup = lineage._canonical_topic_lookup(topics)
    training, paper_categories = base.load_paper_training(topics)
    submitted = base.load_submitted_with_fallback()
    candidate_panel = pd.read_csv(PANEL_PATH)
    meetings = set(candidate_panel["meeting"].astype(int).unique())
    topic_meeting_path = OUTDIR / "topic_meeting_attention_outcomes.csv"
    if topic_meeting_path.exists():
        topic_meetings = pd.read_csv(topic_meeting_path, usecols=["meeting"])
        meetings.update(topic_meetings["meeting"].astype(int).unique())
    meetings = sorted(meetings)

    actors_raw = extract_unique_countries(submitted)
    topics_raw = extract_unique_topics(submitted)
    actors = sorted(str(actor) for actor in actors_raw)
    meeting_values = pd.to_numeric(submitted["meeting number"], errors="coerce")
    rows = []
    for meeting in meetings:
        history = submitted[meeting_values < meeting]
        interaction = generate_interaction_matrix(history, actors_raw, topics_raw)
        counts = base.canonicalize_topic_matrix(
            interaction, topics, topic_lookup, actors
        )
        phi = (
            compute_product_space(get_rca(counts), threshold=RPA_THRESHOLD)
            .reindex(index=topics, columns=topics, fill_value=0.0)
            .fillna(0.0)
        )
        document_volume = (
            training.loc[training["meeting"].lt(meeting)]
            .groupby("topic")
            .size()
            .reindex(topics, fill_value=0)
        )
        meeting_metrics = graph_metrics(phi, counts, document_volume)
        meeting_metrics.insert(0, "meeting", int(meeting))
        rows.append(meeting_metrics)
    return pd.concat(rows, ignore_index=True), paper_categories


def attach_paper_topic(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    paper_categories: dict[str, list[str]],
) -> pd.DataFrame:
    single_topic = {
        paper_id: categories[0]
        for paper_id, categories in paper_categories.items()
        if len(categories) == 1
    }
    data = panel.copy()
    data["paper_topic"] = data["paper_id"].map(single_topic)
    if data["paper_topic"].isna().any():
        missing = int(data["paper_topic"].isna().sum())
        raise RuntimeError(f"{missing} candidate rows lack a unique paper concern")
    data["working_paper"] = data["paper_id"].str.contains(r":WP").astype(int)
    return data.merge(
        metrics,
        left_on=["meeting", "paper_topic"],
        right_on=["meeting", "topic"],
        how="left",
        validate="many_to_one",
    )


def usable_groups(data: pd.DataFrame, positive: str = "adoption_linked") -> pd.DataFrame:
    usable = data.groupby("outcome_id")[positive].transform(
        lambda values: 0 < values.sum() < len(values)
    )
    return data.loc[usable].copy()


def standardized_design(data: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    design = pd.DataFrame(index=data.index)
    for term in terms:
        values = data[term].astype(float)
        standard_deviation = max(float(values.std(ddof=0)), 1e-12)
        design[term] = (values - float(values.mean())) / standard_deviation
    return design


def fit_model(
    data: pd.DataFrame,
    terms: list[str],
    specification: str,
    focal_term: str,
) -> dict:
    frame = usable_groups(data.dropna(subset=terms))
    design = standardized_design(frame, terms)
    fitted = ConditionalLogit(
        frame["adoption_linked"].to_numpy(int),
        design,
        groups=frame["outcome_id"].to_numpy(),
    ).fit(method="bfgs", disp=False, maxiter=1000)
    coefficient = float(fitted.params[focal_term])
    standard_error = float(fitted.bse[focal_term])
    return {
        "specification": specification,
        "focal_term": focal_term,
        "coefficient": coefficient,
        "standard_error": standard_error,
        "odds_ratio": math.exp(coefficient),
        "ci_low": math.exp(coefficient - 1.96 * standard_error),
        "ci_high": math.exp(coefficient + 1.96 * standard_error),
        "p_value": float(2 * norm.sf(abs(coefficient / standard_error))),
        "log_likelihood": float(fitted.llf),
        "n_terms": len(terms),
        "n_outputs": int(frame["outcome_id"].nunique()),
        "n_rows": int(len(frame)),
        "scale": "odds ratio per one-SD increase in within-meeting percentile",
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return pd.Series(result, index=p_values.index)


def within_output_auc(group: pd.DataFrame, score: str) -> float:
    positive = group.loc[group["adoption_linked"].eq(1), score].to_numpy(float)
    negative = group.loc[group["adoption_linked"].eq(0), score].to_numpy(float)
    if not len(positive) or not len(negative):
        return np.nan
    return float(
        (
            (positive[:, None] > negative[None, :]).astype(float)
            + 0.5 * (positive[:, None] == negative[None, :])
        ).mean()
    )


def cross_validated_scores(
    data: pd.DataFrame,
    baseline_terms: list[str],
    added_term: str,
    comparison: str,
) -> tuple[pd.DataFrame, dict]:
    """Rank candidate papers in held-out meetings.

    Estimation uses within-output positive--negative paper differences, with
    every output receiving equal total weight.  This is much faster than
    repeatedly evaluating the exact conditional-logit likelihood for the large
    paper choice sets, while targeting the held-out within-output ranking that
    the AUC evaluates.
    """
    terms = list(dict.fromkeys(baseline_terms + [added_term]))
    frame = usable_groups(data.dropna(subset=terms)).copy()
    frame["baseline_score"] = np.nan
    frame["augmented_score"] = np.nan
    splitter = GroupKFold(n_splits=N_FOLDS)
    for train_indices, test_indices in splitter.split(
        frame, groups=frame["meeting"]
    ):
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        for model_terms, score_name in (
            (baseline_terms, "baseline_score"),
            (terms, "augmented_score"),
        ):
            train_design = pd.DataFrame(index=train.index)
            test_design = pd.DataFrame(index=test.index)
            for term in model_terms:
                mean = float(train[term].mean())
                sd = max(float(train[term].std(ddof=0)), 1e-12)
                train_design[term] = (train[term] - mean) / sd
                test_design[term] = (test[term] - mean) / sd
            differences = []
            labels = []
            weights = []
            train_values = train_design.to_numpy(float)
            for _, positions in train.groupby("outcome_id").indices.items():
                positions = np.asarray(positions, dtype=int)
                outcomes = train.iloc[positions]["adoption_linked"].to_numpy(int)
                positive = train_values[positions[outcomes == 1]]
                negative = train_values[positions[outcomes == 0]]
                if not len(positive) or not len(negative):
                    continue
                difference = (
                    positive[:, None, :] - negative[None, :, :]
                ).reshape(-1, train_values.shape[1])
                pair_weight = 0.5 / len(difference)
                differences.extend([difference, -difference])
                labels.extend(
                    [np.ones(len(difference), dtype=int), np.zeros(len(difference), dtype=int)]
                )
                weights.extend(
                    [
                        np.full(len(difference), pair_weight),
                        np.full(len(difference), pair_weight),
                    ]
                )
            pair_design = np.vstack(differences)
            pair_labels = np.concatenate(labels)
            pair_weights = np.concatenate(weights)
            fitted = LogisticRegression(
                C=1e6,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=2000,
            ).fit(pair_design, pair_labels, sample_weight=pair_weights)
            frame.loc[test.index, score_name] = (
                test_design.to_numpy(float) @ fitted.coef_.ravel()
            )

    rows = []
    for outcome_id, group in frame.groupby("outcome_id"):
        rows.append(
            {
                "comparison": comparison,
                "added_term": added_term,
                "outcome_id": outcome_id,
                "meeting": int(group["meeting"].iloc[0]),
                "baseline_auc": within_output_auc(group, "baseline_score"),
                "augmented_auc": within_output_auc(group, "augmented_score"),
            }
        )
    output_scores = pd.DataFrame(rows).dropna()
    output_scores["delta_auc"] = (
        output_scores["augmented_auc"] - output_scores["baseline_auc"]
    )
    rng = np.random.default_rng(SEED)
    deltas = output_scores["delta_auc"].to_numpy(float)
    bootstrap = np.asarray(
        [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    summary = {
        "comparison": comparison,
        "added_term": added_term,
        "baseline_auc": float(output_scores["baseline_auc"].mean()),
        "augmented_auc": float(output_scores["augmented_auc"].mean()),
        "delta_auc": float(deltas.mean()),
        "delta_ci_low": float(np.quantile(bootstrap, 0.025)),
        "delta_ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_outputs": int(len(output_scores)),
        "folds_grouped_by": "meeting",
    }
    return output_scores, summary


def fit_topic_output_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add each historical structural quantity to the main lagged PPML."""
    panel = pd.read_csv(OUTDIR / "topic_meeting_attention_outcomes.csv")
    data = panel.merge(metrics, on=["topic", "meeting"], how="inner")
    controls = [
        "papers_prior3",
        "neighbor_papers_prior3",
        "outcomes_prior3",
        POPULARITY_TERM,
    ]
    rows = []
    for term in CENTRALITY_TERMS:
        usable = data.dropna(subset=[term]).copy()
        fitted = base.fit_ppml(
            usable,
            "outcome_mass",
            controls + [term],
            f"prior_meeting_add_{term}",
            minimum_year=4,
            period_column="meeting",
        )
        rows.append(fitted.loc[fitted["predictor"].eq(term)])
    # Topic complexity is derived from the same actor--concern incidence
    # matrix as ubiquity and holder breadth.  These joint benchmarks show
    # whether its signal is distinct from those simpler summaries.
    for benchmark in ("ubiquity_percentile", "holder_breadth_percentile"):
        usable = data.dropna(
            subset=["topic_complexity_percentile", benchmark]
        ).copy()
        fitted = base.fit_ppml(
            usable,
            "outcome_mass",
            controls + [benchmark, "topic_complexity_percentile"],
            f"topic_complexity_given_{benchmark.replace('_percentile', '')}",
            minimum_year=4,
            period_column="meeting",
        )
        rows.append(
            fitted.loc[fitted["predictor"].eq("topic_complexity_percentile")]
        )
    output = pd.concat(rows, ignore_index=True)
    primary = output["specification"].str.startswith("prior_meeting_add_")
    output.loc[primary, "p_value_bh"] = benjamini_hochberg(
        output.loc[primary, "p_value"]
    )
    return output


def expanding_topic_output_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    """Test whether centrality improves later-meeting concern rankings.

    Models are trained on all available earlier meetings and evaluated on five
    consecutive future blocks.  Topic indicators absorb stable differences
    among concerns.  The score is the Spearman rank correlation across the 45
    concerns within each held-out meeting.
    """
    panel = pd.read_csv(OUTDIR / "topic_meeting_attention_outcomes.csv")
    data = panel.merge(metrics, on=["topic", "meeting"], how="inner")
    baseline_terms = [
        "papers_prior3",
        "neighbor_papers_prior3",
        "outcomes_prior3",
        POPULARITY_TERM,
    ]
    future_meetings = np.asarray(
        sorted(meeting for meeting in data["meeting"].unique() if meeting >= 16),
        dtype=int,
    )
    blocks = [block for block in np.array_split(future_meetings, 5) if len(block)]
    rows = []
    specifications = [
        (term, term, []) for term in CENTRALITY_TERMS
    ] + [
        (
            "topic_complexity_given_ubiquity",
            "topic_complexity_percentile",
            ["ubiquity_percentile"],
        ),
        (
            "topic_complexity_given_holder_breadth",
            "topic_complexity_percentile",
            ["holder_breadth_percentile"],
        ),
    ]
    for comparison, term, extra_baseline_terms in specifications:
        usable = data.dropna(subset=[term] + extra_baseline_terms).copy()
        for fold, block in enumerate(blocks, start=1):
            start = int(block.min())
            train = usable[usable["meeting"].lt(start)].copy()
            test = usable[usable["meeting"].isin(block)].copy()
            if train["meeting"].nunique() < 8 or test.empty:
                continue
            predictions = {}
            comparison_baseline = baseline_terms + extra_baseline_terms
            for label, terms in (
                ("baseline", comparison_baseline),
                ("augmented", comparison_baseline + [term]),
            ):
                transform = ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), terms),
                        (
                            "topic",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["topic"],
                        ),
                    ]
                )
                model = make_pipeline(
                    transform,
                    PoissonRegressor(alpha=1e-5, max_iter=2000),
                )
                model.fit(train[terms + ["topic"]], train["outcome_mass"])
                predictions[label] = model.predict(test[terms + ["topic"]])
            scored = test[["meeting", "topic", "outcome_mass"]].copy()
            scored["baseline"] = predictions["baseline"]
            scored["augmented"] = predictions["augmented"]
            for meeting, group in scored.groupby("meeting"):
                baseline_rank = float(
                    spearmanr(group["outcome_mass"], group["baseline"]).statistic
                )
                augmented_rank = float(
                    spearmanr(group["outcome_mass"], group["augmented"]).statistic
                )
                rows.append(
                    {
                        "term": term,
                        "comparison": comparison,
                        "fold": fold,
                        "meeting": int(meeting),
                        "baseline_rank_correlation": baseline_rank,
                        "augmented_rank_correlation": augmented_rank,
                        "delta_rank_correlation": augmented_rank - baseline_rank,
                    }
                )
    per_meeting = pd.DataFrame(rows)
    summary_rows = []
    rng = np.random.default_rng(SEED)
    for (comparison, term), group in per_meeting.groupby(["comparison", "term"]):
        differences = group["delta_rank_correlation"].dropna().to_numpy(float)
        bootstrap = np.asarray(
            [
                rng.choice(differences, len(differences), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ]
        )
        summary_rows.append(
            {
                "term": term,
                "comparison": comparison,
                "baseline_rank_correlation": float(
                    group["baseline_rank_correlation"].mean()
                ),
                "augmented_rank_correlation": float(
                    group["augmented_rank_correlation"].mean()
                ),
                "delta_rank_correlation": float(differences.mean()),
                "delta_ci_low": float(np.quantile(bootstrap, 0.025)),
                "delta_ci_high": float(np.quantile(bootstrap, 0.975)),
                "n_test_meetings": int(group["meeting"].nunique()),
                "training_rule": "all earlier meetings; five consecutive future blocks",
            }
        )
    per_meeting.to_csv(
        OUTDIR / "topic_output_centrality_cv_by_meeting.csv", index=False
    )
    return pd.DataFrame(summary_rows)


def write_report(
    models: pd.DataFrame,
    cv: pd.DataFrame,
    topic_models: pd.DataFrame,
    topic_cv: pd.DataFrame,
) -> None:
    adjusted = models[models["specification"].eq("popularity_adjusted")].copy()
    adjusted = adjusted.sort_values("p_value")
    cv_adjusted = cv[cv["comparison"].eq("add_to_full_baseline")].copy()
    cv_adjusted = cv_adjusted.sort_values("delta_auc", ascending=False)
    primary_topic_models = topic_models[
        topic_models["specification"].str.startswith("prior_meeting_add_")
    ].sort_values("p_value")
    primary_topic_cv = topic_cv[
        topic_cv["comparison"].eq(topic_cv["term"])
    ].sort_values("delta_rank_correlation", ascending=False)
    lines = [
        "# Do central paper concerns predict formal contribution?",
        "",
        "All concern-network quantities are calculated from meetings preceding the focal ATCM. The unit is a candidate paper--output pair. Centrality describes the paper's labelled concern, not the output concern, because the latter does not vary among papers competing around the same output.",
        "",
        "## Conditional estimates",
        "",
        "Each row adds one structural quantity to exact alignment, neighbouring-concern proximity, title overlap, Working Paper status, and prior concern-level document volume. Odds ratios are per one standard deviation in the concern's within-meeting rank. Intervals below are model-based; the meeting-held-out results are the predictive check.",
        "",
        "| Quantity | Odds ratio | 95% CI | BH-adjusted p |",
        "|---|---:|---:|---:|",
    ]
    for row in adjusted.itertuples(index=False):
        lines.append(
            f"| {row.focal_term.replace('_percentile', '').replace('_', ' ')} | "
            f"{row.odds_ratio:.3f} | {row.ci_low:.3f}--{row.ci_high:.3f} | "
            f"{row.p_value_bh:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out meetings",
            "",
            "The baseline contains exact alignment, neighbouring-concern proximity, title overlap, Working Paper status, and prior concern volume. Positive changes mean that adding the structural quantity ranked linked papers better in meetings excluded from model fitting.",
            "",
            "| Added quantity | Baseline AUC | Augmented AUC | Change | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in cv_adjusted.itertuples(index=False):
        lines.append(
            f"| {row.added_term.replace('_percentile', '').replace('_', ' ')} | "
            f"{row.baseline_auc:.3f} | {row.augmented_auc:.3f} | "
            f"{row.delta_auc:+.3f} | {row.delta_ci_low:+.3f}--{row.delta_ci_high:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Which concerns receive formal output?",
            "",
            "A second analysis uses concern--meeting rather than paper--output pairs. The full-sample model relates formal output at a meeting to centrality calculated before that meeting, alongside paper volume on the concern and its neighbours during the preceding three meetings, prior output, cumulative concern volume, and concern and meeting fixed effects.",
            "",
            "| Quantity | Full-sample IRR | 95% CI | BH-adjusted p | Later-meeting rank change | 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    topic_cv_lookup = primary_topic_cv.set_index("term")
    for row in primary_topic_models.itertuples(index=False):
        held_out = topic_cv_lookup.loc[row.predictor]
        lines.append(
            f"| {row.predictor.replace('_percentile', '').replace('_', ' ')} | "
            f"{row.incidence_rate_ratio:.3f} | {row.ci_low:.3f}--{row.ci_high:.3f} | "
            f"{row.p_value_bh:.3f} | {held_out.delta_rank_correlation:+.3f} | "
            f"{held_out.delta_ci_low:+.3f}--{held_out.delta_ci_high:+.3f} |"
        )
    complexity_given_ubiquity = topic_models.loc[
        topic_models["specification"].eq("topic_complexity_given_ubiquity")
    ].iloc[0]
    complexity_cv_given_ubiquity = topic_cv.loc[
        topic_cv["comparison"].eq("topic_complexity_given_ubiquity")
    ].iloc[0]
    lines.extend(
        [
            "",
            "The later-meeting column is the change in the mean within-meeting Spearman correlation when the metric is added. Models are trained only on earlier meetings. This prevents an in-sample association from being described as predictive when it does not travel forward in time.",
            "",
            f"Topic complexity is not independent of the simpler count of actors specialized in a concern. Once that ubiquity benchmark is included, the complexity IRR is {complexity_given_ubiquity.incidence_rate_ratio:.3f} ({complexity_given_ubiquity.ci_low:.3f}--{complexity_given_ubiquity.ci_high:.3f}) and its later-meeting rank gain is {complexity_cv_given_ubiquity.delta_rank_correlation:+.3f} ({complexity_cv_given_ubiquity.delta_ci_low:+.3f}--{complexity_cv_given_ubiquity.delta_ci_high:+.3f}).",
            "",
            "## Interpretation rule",
            "",
            "A useful structural predictor should have a stable conditional association and improve ranking in held-out meetings. A small p-value without a held-out gain is treated as description, not predictive evidence. Strength, closeness, eigenvector centrality, and PageRank are expected to overlap heavily; the correlation table should therefore be consulted before giving any one of them a distinct substantive interpretation.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    metrics, paper_categories = historical_metrics()
    metrics.to_csv(METRICS_PATH, index=False)

    panel = pd.read_csv(PANEL_PATH)
    data = attach_paper_topic(panel, metrics, paper_categories)
    data = usable_groups(data)

    correlation_terms = [POPULARITY_TERM] + CENTRALITY_TERMS
    correlations = (
        data[["meeting", "paper_topic"] + correlation_terms]
        .drop_duplicates(["meeting", "paper_topic"])[correlation_terms]
        .corr(method="spearman")
    )
    correlations.to_csv(CORRELATIONS_PATH)

    model_rows = []
    for term in CENTRALITY_TERMS:
        model_rows.append(
            fit_model(data, CORE_TERMS + [term], "not_popularity_adjusted", term)
        )
        model_rows.append(
            fit_model(
                data,
                CORE_TERMS + [POPULARITY_TERM, term],
                "popularity_adjusted",
                term,
            )
        )
    models = pd.DataFrame(model_rows)
    for specification in models["specification"].unique():
        mask = models["specification"].eq(specification)
        models.loc[mask, "p_value_bh"] = benjamini_hochberg(
            models.loc[mask, "p_value"]
        )
    models.to_csv(MODELS_PATH, index=False)

    cv_rows = []
    outcome_rows = []
    content_baseline = [
        "same_concern_mass",
        "title_overlap",
        "working_paper",
        POPULARITY_TERM,
    ]
    full_baseline = CORE_TERMS + [POPULARITY_TERM]
    _, nearby_summary = cross_validated_scores(
        data,
        content_baseline,
        "related_concern_proximity",
        "add_neighbouring_concerns_to_content_baseline",
    )
    cv_rows.append(nearby_summary)
    for term in CENTRALITY_TERMS:
        per_output, summary = cross_validated_scores(
            data, content_baseline, term, "add_to_content_baseline"
        )
        outcome_rows.append(per_output)
        cv_rows.append(summary)
        per_output, summary = cross_validated_scores(
            data, full_baseline, term, "add_to_full_baseline"
        )
        outcome_rows.append(per_output)
        cv_rows.append(summary)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(CV_PATH, index=False)
    pd.concat(outcome_rows, ignore_index=True).to_csv(
        OUTDIR / "outcome_topic_centrality_cv_by_output.csv", index=False
    )

    topic_models = fit_topic_output_models(metrics)
    topic_models.to_csv(TOPIC_MODELS_PATH, index=False)
    topic_cv = expanding_topic_output_cv(metrics)
    topic_cv.to_csv(TOPIC_CV_PATH, index=False)

    write_report(models, cv, topic_models, topic_cv)
    summary = {
        "design": {
            "unit": "candidate paper--output pair",
            "positive": "adoption-linked paper",
            "history_rule": "submissions from ATCMs before the focal meeting",
            "conditional_group": "formal output",
            "cv_group": "meeting",
            "n_folds": N_FOLDS,
            "n_outputs": int(data["outcome_id"].nunique()),
            "n_rows": int(len(data)),
        },
        "centrality_terms": CENTRALITY_TERMS,
        "primary_controls": CORE_TERMS + [POPULARITY_TERM],
        "outputs": {
            "metrics": str(METRICS_PATH.relative_to(ROOT)),
            "models": str(MODELS_PATH.relative_to(ROOT)),
            "correlations": str(CORRELATIONS_PATH.relative_to(ROOT)),
            "cross_validation": str(CV_PATH.relative_to(ROOT)),
            "topic_output_models": str(TOPIC_MODELS_PATH.relative_to(ROOT)),
            "topic_output_cross_validation": str(TOPIC_CV_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
