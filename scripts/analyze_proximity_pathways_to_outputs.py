#!/usr/bin/env python3
"""Test whether concern-space pathways locate later formal output.

Each proximity is the minimum of two conditional co-specialization
probabilities.  For every rolling origin, this script rebuilds the 45-concern
matrix from earlier papers, assigns each edge cost -log(phi), and finds the
maximum-product path between every concern pair.  It then asks whether direct
attention, one-step proximity, or the additional strength of multistep paths
improves held-out predictions of formal-output categories.

The inferred paths are potential routes implied by the concern space.  They
are not observed transmissions or documentary lineage.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fig01_space_of_concerns_topology import normalize_topic_key
from scripts.analyze_output_category_families import paper_family_relations
from scripts.hazard_conditional_logit import choose_period_col, sanitize_periods
from scripts.official_regular_atcm_outputs import (
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
)
from scripts.primary_concern_sensitivity import variants
from utils import (
    _split_multi_value,
    compute_product_space,
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
    standardize_index_labels,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_PANEL = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "output_family_topic_meeting_panel.csv"
)
OUTDIR = ROOT / "output" / "proximity_pathway_forecast"

TRAIN_START = 20
TEST_START = 29
TEST_END = 47
HISTORY_HORIZON = 5
PATH_DECAY = 0.5
LOGISTIC_C = 1.0
POISSON_ALPHA = 0.2
SEED = 20260821
TIMINGS = ("strict_lag", "retrospective_nowcast")
OUTPUT_TYPES = ("Measure", "Decision", "Resolution")


MODEL_FEATURES = {
    "output history": ("output_history",),
    "direct attention": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
    ),
    "one-step proximity": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "one_step_exposure",
    ),
    "strongest pathway": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "one_step_exposure",
        "multistep_increment",
    ),
    "random-walk diffusion": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "diffusion_exposure",
    ),
}


@dataclass(frozen=True)
class ConcernSpace:
    concerns: tuple[str, ...]
    family_of: tuple[str, ...]
    families: tuple[str, ...]
    actors_raw: frozenset[str]
    concerns_raw: frozenset[str]


def concern_space_metadata(submitted: pd.DataFrame) -> ConcernSpace:
    actors_raw = frozenset(extract_unique_countries(submitted))
    concerns_raw = frozenset(extract_unique_topics(submitted))
    full = standardize_index_labels(
        generate_interaction_matrix(submitted, set(actors_raw), set(concerns_raw))
    )
    concerns = tuple(full.index)
    crosswalk = {
        normalize_topic_key(concern): family
        for concern, family in PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.items()
    }
    family_of = tuple(crosswalk[normalize_topic_key(concern)] for concern in concerns)
    families = tuple(sorted(set(family_of)))
    if len(concerns) != 45 or len(families) != 15:
        raise AssertionError("Expected 45 concerns mapped into 15 output families")
    return ConcernSpace(concerns, family_of, families, actors_raw, concerns_raw)


def fine_attention(submitted: pd.DataFrame, period_col: str) -> pd.DataFrame:
    rows = []
    for record in submitted.drop_duplicates("paper id")[
        ["paper id", period_col, "category"]
    ].to_dict(orient="records"):
        meeting = pd.to_numeric(record[period_col], errors="coerce")
        concerns = _split_multi_value(record.get("category"), "\t")
        if pd.isna(meeting) or not concerns:
            continue
        weight = 1.0 / len(concerns)
        for concern in concerns:
            rows.append(
                {
                    "meeting": int(meeting),
                    "concern_key": normalize_topic_key(concern),
                    "paper_weight": weight,
                }
            )
    return (
        pd.DataFrame(rows)
        .groupby(["meeting", "concern_key"], as_index=False)["paper_weight"]
        .sum()
    )


def historical_proximity(
    submitted: pd.DataFrame,
    period_col: str,
    metadata: ConcernSpace,
    cutoff: int,
) -> np.ndarray:
    history = submitted[submitted[period_col].lt(cutoff)]
    interaction = generate_interaction_matrix(
        history,
        set(metadata.actors_raw),
        set(metadata.concerns_raw),
    )
    interaction = standardize_index_labels(interaction).reindex(
        index=metadata.concerns, fill_value=0.0
    )
    proximity = compute_product_space(get_rca(interaction)).reindex(
        index=metadata.concerns, columns=metadata.concerns, fill_value=0.0
    )
    values = proximity.to_numpy(dtype=float)
    values = 0.5 * (values + values.T)
    np.fill_diagonal(values, 0.0)
    return values


def maximum_product_paths(proximity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum path products and shortest-path predecessors."""
    costs = np.full_like(proximity, np.inf, dtype=float)
    positive = proximity > 0
    costs[positive] = -np.log(proximity[positive])
    np.fill_diagonal(costs, 0.0)
    distances, predecessors = shortest_path(
        costs,
        directed=False,
        return_predecessors=True,
    )
    products = np.exp(-distances)
    products[~np.isfinite(distances)] = 0.0
    np.fill_diagonal(products, 0.0)
    return products, predecessors


def row_normalize(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def diffusion_kernel(proximity: np.ndarray, decay: float = PATH_DECAY) -> np.ndarray:
    transition = row_normalize(proximity)
    identity = np.eye(len(transition))
    kernel = (1.0 - decay) * transition @ np.linalg.inv(identity - decay * transition)
    np.fill_diagonal(kernel, 0.0)
    return kernel


def reconstruct_path(source: int, target: int, predecessors: np.ndarray) -> list[int]:
    if source == target:
        return [source]
    if predecessors[source, target] < 0:
        return []
    path = [target]
    current = target
    while current != source:
        current = int(predecessors[source, current])
        if current < 0 or len(path) > predecessors.shape[0]:
            return []
        path.append(current)
    return list(reversed(path))


def family_exposures(
    paper_vector: np.ndarray,
    proximity: np.ndarray,
    best_paths: np.ndarray,
    diffusion: np.ndarray,
    metadata: ConcernSpace,
) -> pd.DataFrame:
    rows = []
    family_array = np.asarray(metadata.family_of)
    for family in metadata.families:
        targets = np.flatnonzero(family_array == family)
        sources = np.flatnonzero(family_array != family)
        source_attention = paper_vector[sources]
        direct_probability = proximity[np.ix_(sources, targets)].max(axis=1)
        pathway_probability = best_paths[np.ix_(sources, targets)].max(axis=1)
        diffusion_probability = diffusion[np.ix_(sources, targets)].sum(axis=1)
        one_step = float(source_attention @ direct_probability)
        strongest = float(source_attention @ pathway_probability)
        rows.append(
            {
                "topic": family,
                "one_step_exposure": one_step,
                "strongest_path_exposure": strongest,
                "multistep_increment": max(strongest - one_step, 0.0),
                "diffusion_exposure": float(source_attention @ diffusion_probability),
            }
        )
    return pd.DataFrame(rows)


def prepare_panel(
    base_panel: pd.DataFrame,
    submitted: pd.DataFrame,
    period_col: str,
    relations: pd.DataFrame,
    attention: pd.DataFrame,
    metadata: ConcernSpace,
    timing: str,
    permutation_seed: int | None = None,
    proximity_cache: dict[int, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    panel = base_panel.sort_values(["meeting", "topic"]).copy()
    for output in OUTPUT_TYPES:
        column = f"{output.lower()}_mass"
        panel[f"{output.lower()}_history"] = (
            panel.sort_values(["topic", "meeting"])
            .groupby("topic")[column]
            .transform(
                lambda values: (
                    values.shift(1).rolling(HISTORY_HORIZON, min_periods=1).sum()
                )
            )
        )
    base_by_meeting = {
        int(meeting): group.set_index("topic").reindex(metadata.families)
        for meeting, group in panel.groupby("meeting")
    }
    attention_lookup = attention.set_index(["meeting", "concern_key"])["paper_weight"]
    concern_keys = [normalize_topic_key(concern) for concern in metadata.concerns]
    family_array = np.asarray(metadata.family_of)
    actor_reach = (
        relations.groupby(["meeting", "family"])["actor"].nunique().rename("reach")
    )
    permutation = (
        None
        if permutation_seed is None
        else np.random.default_rng(permutation_seed).permutation(len(metadata.concerns))
    )
    rows = []
    maps: dict[int, dict] = {}
    for meeting in sorted(panel["meeting"].unique().astype(int)):
        attention_meeting = (
            meeting if timing == "retrospective_nowcast" else meeting - 1
        )
        paper_vector = np.asarray(
            [
                attention_lookup.get((attention_meeting, key), 0.0)
                for key in concern_keys
            ],
            dtype=float,
        )
        proximity = (
            historical_proximity(
                submitted, period_col, metadata, cutoff=attention_meeting
            )
            if proximity_cache is None
            else proximity_cache[attention_meeting].copy()
        )
        if permutation is not None:
            proximity = proximity[np.ix_(permutation, permutation)]
        best_paths, predecessors = maximum_product_paths(proximity)
        diffusion = diffusion_kernel(proximity)
        exposures = family_exposures(
            paper_vector, proximity, best_paths, diffusion, metadata
        ).set_index("topic")
        maps[meeting] = {
            "attention_meeting": attention_meeting,
            "paper_vector": paper_vector,
            "proximity": proximity,
            "best_paths": best_paths,
            "predecessors": predecessors,
        }
        focal = base_by_meeting[meeting]
        for family in metadata.families:
            source = focal.loc[family]
            target_indices = np.flatnonzero(family_array == family)
            row = {
                "topic": family,
                "meeting": meeting,
                "timing": timing,
                "attention_meeting": attention_meeting,
                "map_uses_meetings_before": attention_meeting,
                "direct_papers": float(paper_vector[target_indices].sum()),
                "direct_actor_reach": float(
                    actor_reach.get((attention_meeting, family), 0.0)
                ),
                "total_papers": float(paper_vector.sum()),
                **exposures.loc[family].to_dict(),
            }
            for output in OUTPUT_TYPES:
                stem = output.lower()
                mass = float(source[f"{stem}_mass"])
                row[f"{stem}_mass"] = mass
                row[f"{stem}_occurrence"] = int(mass > 0)
                row[f"{stem}_history"] = float(source[f"{stem}_history"])
            rows.append(row)
    return pd.DataFrame(rows), maps


def model_pipeline(target: str, features: tuple[str, ...]):
    transform = ColumnTransformer(
        [
            ("topic", OneHotEncoder(handle_unknown="ignore"), ["topic"]),
            ("numeric", StandardScaler(), list(features)),
        ]
    )
    estimator = (
        LogisticRegression(C=LOGISTIC_C, solver="lbfgs", max_iter=3_000)
        if target == "occurrence"
        else PoissonRegressor(
            alpha=POISSON_ALPHA,
            max_iter=3_000,
            tol=1e-9,
        )
    )
    return make_pipeline(transform, estimator)


def run_forecasts(
    panel: pd.DataFrame,
    model_names: list[str] | None = None,
    output_types: tuple[str, ...] = OUTPUT_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = list(MODEL_FEATURES) if model_names is None else model_names
    prediction_rows = []
    score_rows = []
    for output in output_types:
        stem = output.lower()
        for meeting in range(TEST_START, TEST_END + 1):
            train = panel[
                panel["meeting"].ge(TRAIN_START) & panel["meeting"].lt(meeting)
            ].copy()
            test = panel[panel["meeting"].eq(meeting)].copy()
            train["output_history"] = train[f"{stem}_history"]
            test["output_history"] = test[f"{stem}_history"]
            for model_name in selected:
                features = MODEL_FEATURES[model_name]
                design = ["topic", *features]
                occurrence_model = model_pipeline("occurrence", features)
                mass_model = model_pipeline("mass", features)
                occurrence_model.fit(train[design], train[f"{stem}_occurrence"])
                mass_model.fit(train[design], train[f"{stem}_mass"])
                probability = np.clip(
                    occurrence_model.predict_proba(test[design])[:, 1],
                    1e-12,
                    1 - 1e-12,
                )
                predicted_mass = np.maximum(mass_model.predict(test[design]), 1e-12)
                observed = test[f"{stem}_occurrence"].to_numpy(dtype=int)
                observed_mass = test[f"{stem}_mass"].to_numpy(dtype=float)
                prediction = test[
                    ["topic", "meeting", "timing", "attention_meeting"]
                ].copy()
                prediction["output_type"] = output
                prediction["model"] = model_name
                prediction["observed_occurrence"] = observed
                prediction["predicted_probability"] = probability
                prediction["observed_mass"] = observed_mass
                prediction["predicted_mass"] = predicted_mass
                prediction_rows.append(prediction)
                score_rows.append(
                    {
                        "timing": test["timing"].iloc[0],
                        "output_type": output,
                        "meeting": meeting,
                        "model": model_name,
                        "binary_log_loss": float(
                            log_loss(observed, probability, labels=[0, 1])
                        ),
                        "brier_score": float(brier_score_loss(observed, probability)),
                        "poisson_deviance": float(
                            mean_poisson_deviance(observed_mass, predicted_mass)
                        ),
                    }
                )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(score_rows)


def block_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
    block_length: int = 3,
) -> list[float]:
    n_values = len(values)
    n_blocks = math.ceil(n_values / block_length)
    offsets = np.arange(block_length)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        starts = rng.integers(0, n_values, n_blocks)
        indices = (starts[:, None] + offsets) % n_values
        means[draw] = values[indices.ravel()[:n_values]].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sign_flip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    draws = 20_000
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    randomized = np.abs((signs * values).mean(axis=1))
    return float((1 + (randomized >= observed).sum()) / (draws + 1))


def summarize(scores: pd.DataFrame, bootstrap_draws: int) -> dict:
    rng = np.random.default_rng(SEED)
    metrics = ("binary_log_loss", "brier_score", "poisson_deviance")
    result = {}
    for (timing, output), group in scores.groupby(["timing", "output_type"]):
        key = f"{timing}__{output}"
        result[key] = {"model_means": {}, "paired_comparisons": {}}
        for model, rows in group.groupby("model"):
            result[key]["model_means"][model] = {
                metric: float(rows[metric].mean()) for metric in metrics
            }
        comparisons = (
            ("one-step proximity", "direct attention"),
            ("strongest pathway", "direct attention"),
            ("strongest pathway", "one-step proximity"),
            ("random-walk diffusion", "direct attention"),
        )
        for candidate_name, baseline_name in comparisons:
            candidate = group[group["model"].eq(candidate_name)].set_index("meeting")
            baseline = group[group["model"].eq(baseline_name)].set_index("meeting")
            comparison = {}
            for metric in metrics:
                difference = (candidate[metric] - baseline[metric]).to_numpy(float)
                comparison[metric] = {
                    "mean_difference": float(difference.mean()),
                    "meeting_block_bootstrap_95_interval": block_interval(
                        difference, rng, bootstrap_draws
                    ),
                    "paired_sign_flip_p": sign_flip_p(difference, rng),
                    "meetings_better": int((difference < 0).sum()),
                    "meetings_total": len(difference),
                }
            result[key]["paired_comparisons"][
                f"{candidate_name}__vs__{baseline_name}"
            ] = comparison
    return result


def pathway_rows(
    panel: pd.DataFrame,
    maps: dict[int, dict],
    metadata: ConcernSpace,
    timing: str,
) -> pd.DataFrame:
    family_array = np.asarray(metadata.family_of)
    rows = []
    for output in OUTPUT_TYPES:
        stem = output.lower()
        active = panel[
            panel["meeting"].between(TEST_START, TEST_END) & panel[f"{stem}_mass"].gt(0)
        ]
        for record in active.itertuples(index=False):
            map_data = maps[int(record.meeting)]
            paper_vector = map_data["paper_vector"]
            proximity = map_data["proximity"]
            best = map_data["best_paths"]
            predecessors = map_data["predecessors"]
            targets = np.flatnonzero(family_array == record.topic)
            sources = np.flatnonzero(
                (family_array != record.topic) & (paper_vector > 0)
            )
            candidates = []
            for source in sources:
                for target in targets:
                    score = paper_vector[source] * best[source, target]
                    candidates.append((float(score), int(source), int(target)))
            for rank, (score, source, target) in enumerate(
                sorted(candidates, reverse=True)[:3], start=1
            ):
                path = reconstruct_path(source, target, predecessors)
                rows.append(
                    {
                        "timing": timing,
                        "meeting": int(record.meeting),
                        "attention_meeting": int(map_data["attention_meeting"]),
                        "output_type": output,
                        "output_family": record.topic,
                        "output_mass": float(getattr(record, f"{stem}_mass")),
                        "path_rank": rank,
                        "source_attention": float(paper_vector[source]),
                        "direct_probability": float(proximity[source, target]),
                        "pathway_probability": float(best[source, target]),
                        "weighted_pathway_score": score,
                        "path_length": max(len(path) - 1, 0),
                        "path": " -> ".join(metadata.concerns[index] for index in path),
                    }
                )
    return pd.DataFrame(rows)


def emergent_pathway_rows(
    maps: dict[int, dict], metadata: ConcernSpace, timing: str
) -> pd.DataFrame:
    """Export every pair whose best indirect path exceeds its direct edge."""
    rows = []
    for meeting, map_data in maps.items():
        if not TEST_START <= meeting <= TEST_END:
            continue
        proximity = map_data["proximity"]
        best = map_data["best_paths"]
        predecessors = map_data["predecessors"]
        for source in range(len(metadata.concerns)):
            for target in range(source + 1, len(metadata.concerns)):
                if best[source, target] <= proximity[source, target] + 1e-12:
                    continue
                path = reconstruct_path(source, target, predecessors)
                rows.append(
                    {
                        "timing": timing,
                        "meeting": meeting,
                        "attention_meeting": int(map_data["attention_meeting"]),
                        "source_concern": metadata.concerns[source],
                        "source_family": metadata.family_of[source],
                        "target_concern": metadata.concerns[target],
                        "target_family": metadata.family_of[target],
                        "direct_probability": float(proximity[source, target]),
                        "pathway_probability": float(best[source, target]),
                        "probability_gain": float(
                            best[source, target] - proximity[source, target]
                        ),
                        "path_length": max(len(path) - 1, 0),
                        "path": " -> ".join(metadata.concerns[index] for index in path),
                    }
                )
    return pd.DataFrame(rows)


def label_null(
    base_panel: pd.DataFrame,
    submitted: pd.DataFrame,
    period_col: str,
    relations: pd.DataFrame,
    attention: pd.DataFrame,
    metadata: ConcernSpace,
    observed_scores: pd.DataFrame,
    observed_maps: dict[int, dict],
    draws: int,
) -> pd.DataFrame:
    if draws <= 0:
        return pd.DataFrame()
    observed = observed_scores[
        observed_scores["timing"].eq("retrospective_nowcast")
        & observed_scores["output_type"].eq("Resolution")
    ]
    one_step = observed[observed["model"].eq("one-step proximity")].set_index("meeting")
    pathway = observed[observed["model"].eq("strongest pathway")].set_index("meeting")
    observed_difference = float(
        (pathway["binary_log_loss"] - one_step["binary_log_loss"]).mean()
    )
    rows = []
    proximity_cache = {
        int(map_data["attention_meeting"]): map_data["proximity"]
        for map_data in observed_maps.values()
    }
    for draw in range(draws):
        null_panel, _ = prepare_panel(
            base_panel,
            submitted,
            period_col,
            relations,
            attention,
            metadata,
            "retrospective_nowcast",
            permutation_seed=SEED + draw + 1,
            proximity_cache=proximity_cache,
        )
        _, null_scores = run_forecasts(
            null_panel,
            model_names=["one-step proximity", "strongest pathway"],
            output_types=("Resolution",),
        )
        null_one_step = null_scores[
            null_scores["model"].eq("one-step proximity")
        ].set_index("meeting")
        null_pathway = null_scores[
            null_scores["model"].eq("strongest pathway")
        ].set_index("meeting")
        null_difference = float(
            (null_pathway["binary_log_loss"] - null_one_step["binary_log_loss"]).mean()
        )
        rows.append(
            {
                "draw": draw,
                "null_mean_log_loss_difference_vs_observed_one_step": null_difference,
                "observed_pathway_difference_vs_one_step": observed_difference,
            }
        )
    result = pd.DataFrame(rows)
    result["lower_tail_p"] = (
        1
        + (
            result["null_mean_log_loss_difference_vs_observed_one_step"]
            <= observed_difference
        ).sum()
    ) / (1 + len(result))
    return result


def diagnostics(
    panels: list[pd.DataFrame], maps: dict[int, dict], metadata: ConcernSpace
) -> dict:
    latest = maps[max(maps)]
    proximity = latest["proximity"]
    best = latest["best_paths"]
    positive = proximity > 0
    improved = best > proximity + 1e-12
    return {
        "concerns": len(metadata.concerns),
        "output_families": len(metadata.families),
        "test_meetings": [TEST_START, TEST_END],
        "cells_per_output_and_timing": (TEST_END - TEST_START + 1)
        * len(metadata.families),
        "timing_valid": bool(
            all(
                panel["map_uses_meetings_before"].eq(panel["attention_meeting"]).all()
                for panel in panels
            )
            and (
                panels[0].loc[panels[0]["timing"].eq("strict_lag"), "attention_meeting"]
                < panels[0].loc[panels[0]["timing"].eq("strict_lag"), "meeting"]
            ).all()
        ),
        "latest_positive_pair_share": float(positive[np.triu_indices(45, 1)].mean()),
        "latest_pairs_with_stronger_indirect_path": int(
            improved[np.triu_indices(45, 1)].sum()
        ),
        "latest_pair_count": 990,
        "interpretation": (
            "Maximum-product paths rank potential routes implied by conditional "
            "co-specialization probabilities. They are not observed transitions."
        ),
        "crosswalk_warning": (
            "The 45 concerns enter 15 official output families through the "
            "author-defined hierarchy, which still requires independent blind validation."
        ),
        "metadata_timing_warning": (
            "Paper category memberships come from retrospective query responses "
            "without assignment timestamps."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-draws", type=int, default=50)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    submitted = variants()["fractional_multilabel"]
    period_col = choose_period_col(submitted)
    submitted = sanitize_periods(submitted, period_col)
    metadata = concern_space_metadata(submitted)
    attention = fine_attention(submitted, period_col)
    relations = paper_family_relations(submitted)
    base_panel = pd.read_csv(INPUT_PANEL)

    panels = []
    prediction_tables = []
    score_tables = []
    pathway_tables = []
    emergent_tables = []
    map_sets = {}
    for timing in TIMINGS:
        panel, maps = prepare_panel(
            base_panel,
            submitted,
            period_col,
            relations,
            attention,
            metadata,
            timing,
        )
        predictions, scores = run_forecasts(panel)
        panels.append(panel)
        prediction_tables.append(predictions)
        score_tables.append(scores)
        pathway_tables.append(pathway_rows(panel, maps, metadata, timing))
        emergent_tables.append(emergent_pathway_rows(maps, metadata, timing))
        map_sets[timing] = maps
    predictions = pd.concat(prediction_tables, ignore_index=True)
    scores = pd.concat(score_tables, ignore_index=True)
    summary = summarize(scores, args.bootstrap_draws)
    null = label_null(
        base_panel,
        submitted,
        period_col,
        relations,
        attention,
        metadata,
        scores,
        map_sets["retrospective_nowcast"],
        args.null_draws,
    )
    audit = diagnostics(
        panels,
        map_sets["retrospective_nowcast"],
        metadata,
    )

    pd.concat(panels, ignore_index=True).to_csv(
        OUTDIR / "analysis_panel.csv", index=False
    )
    predictions.to_csv(OUTDIR / "predictions.csv", index=False)
    scores.to_csv(OUTDIR / "meeting_scores.csv", index=False)
    pd.concat(pathway_tables, ignore_index=True).to_csv(
        OUTDIR / "observed_output_pathways.csv", index=False
    )
    pd.concat(emergent_tables, ignore_index=True).to_csv(
        OUTDIR / "emergent_pathways.csv", index=False
    )
    null.to_csv(OUTDIR / "label_null.csv", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTDIR / "diagnostics.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"diagnostics": audit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
