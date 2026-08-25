#!/usr/bin/env python3
"""Test whether directed documentary flows locate later formal output.

The unit is an output type, broad category, and ATCM meeting.  Rolling-origin
models predict both whether any output appears and its fractional count.  The
strict forecast uses papers from the preceding meeting; the retrospective
nowcast uses papers tabled at the focal meeting.  Both freeze the directed
transition operator before the paper-attention measurement.

The directed operator comes from actor entries into 15 broad paper families.
It is distinct from the symmetric co-specialization map.  One-step inflow and
discounted paths of any length enter as alternatives to symmetric neighboring
attention, always after direct attention has entered the model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_poisson_deviance,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.analyze_output_category_families import paper_family_relations
from scripts.primary_concern_sensitivity import variants
from utils import get_rca

ROOT = Path(__file__).resolve().parents[1]
INPUT_PANEL = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "output_family_topic_meeting_panel.csv"
)
OUTDIR = ROOT / "output" / "directed_attention_adoption"

WINDOW_MEETINGS = 5
RPA_THRESHOLD = 1.0
PRIOR_STRENGTH = 10.0
HISTORY_HORIZON = 5
TRAIN_START = 20
TEST_START = 29
TEST_END = 47
PRIMARY_LAMBDA = 0.5
LAMBDA_SENSITIVITY = (0.25, 0.5, 0.75)
LOGISTIC_C = 1.0
POISSON_ALPHA = 0.2
SEED = 20260821
OUTPUT_TYPES = ("Measure", "Decision", "Resolution")
TIMINGS = ("strict_lag", "retrospective_nowcast")


@dataclass(frozen=True)
class TransitionHistory:
    families: tuple[str, ...]
    meetings: tuple[int, ...]
    records: tuple[dict[str, np.ndarray | int], ...]


MODEL_FEATURES = {
    "output history": ("output_history",),
    "global activity": ("output_history", "total_papers"),
    "direct attention": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
    ),
    "target popularity": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "target_entry_prevalence",
    ),
    "symmetric neighbors": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "symmetric_inflow",
    ),
    "directed one-step": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "directed_one_step",
    ),
    "directed all-path": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "directed_all_paths",
    ),
    "symmetrized all-path": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "symmetrized_all_paths",
    ),
    "reversed all-path": (
        "output_history",
        "total_papers",
        "direct_papers",
        "direct_actor_reach",
        "reversed_all_paths",
    ),
}


def build_transition_history(relations: pd.DataFrame) -> TransitionHistory:
    """Build exact actor portfolio transitions at the output-family scale."""
    families = tuple(sorted(relations["family"].unique()))
    actors = tuple(sorted(relations["actor"].unique()))
    meetings = tuple(sorted(relations["meeting"].astype(int).unique()))
    states: list[tuple[int, np.ndarray]] = []
    for index in range(WINDOW_MEETINGS - 1, len(meetings)):
        window = meetings[index - WINDOW_MEETINGS + 1 : index + 1]
        counts = (
            relations[relations["meeting"].isin(window)]
            .groupby(["family", "actor"])["paper_weight"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=families, columns=actors, fill_value=0.0)
        )
        active = get_rca(counts).ge(RPA_THRESHOLD).to_numpy(dtype=bool)
        states.append((int(window[-1]), active))

    records: list[dict[str, np.ndarray | int]] = []
    for (_, previous), (end_meeting, current) in pairwise(states):
        n_families = len(families)
        exposures = np.zeros((n_families, n_families), dtype=float)
        entries = np.zeros_like(exposures)
        target_exposures = np.zeros(n_families, dtype=float)
        target_entries = np.zeros(n_families, dtype=float)
        for actor_index in range(len(actors)):
            held = previous[:, actor_index]
            if not held.any():
                continue
            at_risk = ~held
            entered = current[:, actor_index] & at_risk
            exposures += np.outer(held, at_risk)
            entries += np.outer(held, entered)
            target_exposures += at_risk
            target_entries += entered
        records.append(
            {
                "end_meeting": end_meeting,
                "exposures": exposures,
                "entries": entries,
                "target_exposures": target_exposures,
                "target_entries": target_entries,
            }
        )
    return TransitionHistory(families, meetings, tuple(records))


def row_normalize(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[~np.isfinite(result)] = 0.0
    result[result < 0] = 0.0
    totals = result.sum(axis=1, keepdims=True)
    return np.divide(result, totals, out=np.zeros_like(result), where=totals > 0)


def transition_operator(
    history: TransitionHistory, cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate positive transition lift using records completed before cutoff."""
    retained = [record for record in history.records if record["end_meeting"] < cutoff]
    n_families = len(history.families)
    exposures = sum(
        (np.asarray(record["exposures"]) for record in retained),
        np.zeros((n_families, n_families), dtype=float),
    )
    entries = sum(
        (np.asarray(record["entries"]) for record in retained),
        np.zeros((n_families, n_families), dtype=float),
    )
    target_exposures = sum(
        (np.asarray(record["target_exposures"]) for record in retained),
        np.zeros(n_families, dtype=float),
    )
    target_entries = sum(
        (np.asarray(record["target_entries"]) for record in retained),
        np.zeros(n_families, dtype=float),
    )
    base_rate = (target_entries + 0.5) / (target_exposures + 1.0)
    rates = (entries + PRIOR_STRENGTH * base_rate[np.newaxis, :]) / (
        exposures + PRIOR_STRENGTH
    )
    lift = np.maximum(rates - base_rate[np.newaxis, :], 0.0)
    np.fill_diagonal(lift, 0.0)
    return row_normalize(lift), base_rate


def path_kernel(operator: np.ndarray, decay: float) -> np.ndarray:
    """Return discounted transition paths with lengths one through infinity."""
    identity = np.eye(operator.shape[0])
    kernel = (1.0 - decay) * operator @ np.linalg.inv(identity - decay * operator)
    np.fill_diagonal(kernel, 0.0)
    return kernel


def permute_kernel(kernel: np.ndarray, permutation: np.ndarray | None) -> np.ndarray:
    if permutation is None:
        return kernel
    return kernel[np.ix_(permutation, permutation)]


def strength_preserving_rewire(
    operator: np.ndarray, rng: np.random.Generator, steps: int = 500
) -> np.ndarray:
    """Randomize directed weights while preserving row and column strengths."""
    result = operator.copy()
    n_rows = result.shape[0]
    accepted = 0
    attempts = 0
    while accepted < steps and attempts < steps * 20:
        attempts += 1
        r1, r2 = (int(value) for value in rng.choice(n_rows, 2, replace=False))
        c1, c2 = (int(value) for value in rng.choice(n_rows, 2, replace=False))
        if r1 in {c1, c2} or r2 in {c1, c2}:
            continue
        lower = max(-result[r1, c1], -result[r2, c2])
        upper = min(result[r1, c2], result[r2, c1])
        if upper - lower <= 1e-12:
            continue
        delta = rng.uniform(lower, upper)
        result[r1, c1] += delta
        result[r2, c2] += delta
        result[r1, c2] -= delta
        result[r2, c1] -= delta
        accepted += 1
    result[np.abs(result) < 1e-14] = 0.0
    if not np.allclose(result.sum(axis=1), operator.sum(axis=1), atol=1e-10):
        raise AssertionError("Strength null changed row strengths")
    if not np.allclose(result.sum(axis=0), operator.sum(axis=0), atol=1e-10):
        raise AssertionError("Strength null changed column strengths")
    return result


def prepare_panel(
    base_panel: pd.DataFrame,
    relations: pd.DataFrame,
    history: TransitionHistory,
    timing: str,
    decay: float = PRIMARY_LAMBDA,
    permutation: np.ndarray | None = None,
    rewire_seed: int | None = None,
) -> pd.DataFrame:
    """Add leakage-controlled direct, symmetric, and directed exposures."""
    if timing not in TIMINGS:
        raise ValueError(f"Unknown timing: {timing}")
    families = list(history.families)
    panel = base_panel.sort_values(["meeting", "topic"]).copy()
    mass_columns = {output: f"{output.lower()}_mass" for output in OUTPUT_TYPES}
    for output, column in mass_columns.items():
        panel[f"{output.lower()}_history"] = (
            panel.sort_values(["topic", "meeting"])
            .groupby("topic")[column]
            .transform(
                lambda values: (
                    values.shift(1).rolling(HISTORY_HORIZON, min_periods=1).sum()
                )
            )
        )

    attention_by_meeting = {
        int(meeting): group.set_index("topic").reindex(families)
        for meeting, group in panel.groupby("meeting")
    }
    actor_reach = (
        relations.groupby(["meeting", "family"])["actor"]
        .nunique()
        .rename("actor_reach")
    )
    rows: list[dict] = []
    for meeting in sorted(panel["meeting"].unique().astype(int)):
        attention_meeting = (
            meeting if timing == "retrospective_nowcast" else meeting - 1
        )
        if attention_meeting not in attention_by_meeting:
            continue
        attention = attention_by_meeting[attention_meeting]
        paper_vector = attention["paper_count"].to_numpy(dtype=float)
        symmetric_vector = attention["neighbor_papers"].to_numpy(dtype=float)
        operator, target_prevalence = transition_operator(
            history, cutoff=attention_meeting
        )
        if rewire_seed is not None:
            operator = strength_preserving_rewire(
                operator,
                np.random.default_rng(rewire_seed + attention_meeting),
            )
        directed_kernel = permute_kernel(operator, permutation)
        all_path_kernel = permute_kernel(path_kernel(operator, decay), permutation)
        symmetric = row_normalize((operator + operator.T) / 2.0)
        symmetric_kernel = permute_kernel(path_kernel(symmetric, decay), permutation)
        reversed = row_normalize(operator.T)
        reversed_kernel = permute_kernel(path_kernel(reversed, decay), permutation)
        one_step = paper_vector @ directed_kernel
        all_paths = paper_vector @ all_path_kernel
        symmetric_paths = paper_vector @ symmetric_kernel
        reversed_paths = paper_vector @ reversed_kernel
        focal = attention_by_meeting[meeting]
        for family_index, family in enumerate(families):
            source = focal.loc[family]
            row = {
                "topic": family,
                "meeting": meeting,
                "timing": timing,
                "attention_meeting": attention_meeting,
                "transition_cutoff": attention_meeting - 1,
                "direct_papers": float(paper_vector[family_index]),
                "direct_actor_reach": float(
                    actor_reach.get((attention_meeting, family), 0.0)
                ),
                "total_papers": float(paper_vector.sum()),
                "target_entry_prevalence": float(target_prevalence[family_index]),
                "symmetric_inflow": float(symmetric_vector[family_index]),
                "directed_one_step": float(one_step[family_index]),
                "directed_all_paths": float(all_paths[family_index]),
                "symmetrized_all_paths": float(symmetric_paths[family_index]),
                "reversed_all_paths": float(reversed_paths[family_index]),
            }
            for output, column in mass_columns.items():
                stem = output.lower()
                mass = float(source[column])
                row[f"{stem}_mass"] = mass
                row[f"{stem}_occurrence"] = int(mass > 0)
                row[f"{stem}_history"] = float(source[f"{stem}_history"])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["meeting", "topic"]).reset_index(drop=True)


def model_pipeline(target: str, features: tuple[str, ...], regularization: float):
    transform = ColumnTransformer(
        [
            ("topic", OneHotEncoder(handle_unknown="ignore"), ["topic"]),
            ("numeric", StandardScaler(), list(features)),
        ]
    )
    if target == "occurrence":
        estimator = LogisticRegression(
            C=regularization,
            solver="lbfgs",
            max_iter=3_000,
        )
    elif target == "mass":
        estimator = PoissonRegressor(
            alpha=regularization,
            max_iter=3_000,
            tol=1e-9,
        )
    else:
        raise ValueError(target)
    return make_pipeline(transform, estimator)


def run_forecasts(
    panel: pd.DataFrame,
    *,
    model_names: tuple[str, ...] | list[str] | None = None,
    output_types: tuple[str, ...] = OUTPUT_TYPES,
    logistic_c: float = LOGISTIC_C,
    poisson_alpha: float = POISSON_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_models = list(MODEL_FEATURES) if model_names is None else list(model_names)
    prediction_rows: list[pd.DataFrame] = []
    score_rows: list[dict] = []
    for output in output_types:
        stem = output.lower()
        for meeting in range(TEST_START, TEST_END + 1):
            train = panel[
                panel["meeting"].ge(TRAIN_START) & panel["meeting"].lt(meeting)
            ].copy()
            test = panel[panel["meeting"].eq(meeting)].copy()
            train["output_history"] = train[f"{stem}_history"]
            test["output_history"] = test[f"{stem}_history"]
            if len(test) != len(panel["topic"].unique()):
                raise AssertionError(f"Incomplete test panel at ATCM {meeting}")
            for model_name in selected_models:
                features = MODEL_FEATURES[model_name]
                design = ["topic", *features]
                occurrence_model = model_pipeline("occurrence", features, logistic_c)
                count_model = model_pipeline("mass", features, poisson_alpha)
                occurrence_model.fit(train[design], train[f"{stem}_occurrence"])
                count_model.fit(train[design], train[f"{stem}_mass"])
                probability = occurrence_model.predict_proba(test[design])[:, 1]
                predicted_mass = np.maximum(count_model.predict(test[design]), 1e-12)
                observed = test[f"{stem}_occurrence"].to_numpy(dtype=int)
                observed_mass = test[f"{stem}_mass"].to_numpy(dtype=float)
                probability = np.clip(probability, 1e-12, 1 - 1e-12)
                predictions = test[
                    [
                        "topic",
                        "meeting",
                        "timing",
                        "attention_meeting",
                        "transition_cutoff",
                    ]
                ].copy()
                predictions["output_type"] = output
                predictions["model"] = model_name
                predictions["observed_occurrence"] = observed
                predictions["predicted_probability"] = probability
                predictions["observed_mass"] = observed_mass
                predictions["predicted_mass"] = predicted_mass
                prediction_rows.append(predictions)
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
                        "roc_auc": (
                            float(roc_auc_score(observed, probability))
                            if np.unique(observed).size == 2
                            else float("nan")
                        ),
                        "average_precision": (
                            float(average_precision_score(observed, probability))
                            if observed.sum() > 0
                            else float("nan")
                        ),
                        "poisson_deviance": float(
                            mean_poisson_deviance(observed_mass, predicted_mass)
                        ),
                    }
                )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(score_rows)


def bootstrap_interval(
    values: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    block_length: int = 3,
) -> list[float]:
    n_values = len(values)
    sampled = np.empty(draws, dtype=float)
    offsets = np.arange(block_length)
    n_blocks = math.ceil(n_values / block_length)
    for draw in range(draws):
        starts = rng.integers(0, n_values, size=n_blocks)
        indices = (starts[:, np.newaxis] + offsets) % n_values
        sampled[draw] = values[indices.ravel()[:n_values]].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def sign_flip_p(
    values: np.ndarray, rng: np.random.Generator, draws: int = 20_000
) -> float:
    """Return a two-sided paired randomization probability for the mean."""
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    randomized = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(randomized >= observed)) / (draws + 1))


def summarize_scores(scores: pd.DataFrame, bootstrap_draws: int) -> dict:
    rng = np.random.default_rng(SEED)
    metrics = (
        "binary_log_loss",
        "brier_score",
        "roc_auc",
        "average_precision",
        "poisson_deviance",
    )
    summary: dict[str, dict] = {}
    for (timing, output), group in scores.groupby(["timing", "output_type"]):
        key = f"{timing}__{output}"
        summary[key] = {"model_means": {}, "paired_vs_direct": {}}
        for model, model_rows in group.groupby("model"):
            summary[key]["model_means"][model] = {
                metric: float(model_rows[metric].mean()) for metric in metrics
            }
        direct = group[group["model"].eq("direct attention")].set_index("meeting")
        for model in sorted(
            set(group["model"]) - {"direct attention", "output history"}
        ):
            candidate = group[group["model"].eq(model)].set_index("meeting")
            comparisons = {}
            for metric in metrics:
                difference = (
                    (candidate[metric] - direct[metric]).dropna().to_numpy(float)
                )
                comparisons[metric] = {
                    "mean_difference": float(difference.mean()),
                    "meeting_bootstrap_95_interval": bootstrap_interval(
                        difference, bootstrap_draws, rng
                    ),
                    "paired_sign_flip_p": sign_flip_p(difference, rng),
                    "meetings_better": int(
                        (difference < 0).sum()
                        if metric
                        in {"binary_log_loss", "brier_score", "poisson_deviance"}
                        else (difference > 0).sum()
                    ),
                    "meetings_total": len(difference),
                }
            summary[key]["paired_vs_direct"][model] = comparisons
        all_path = group[group["model"].eq("directed all-path")].set_index("meeting")
        summary[key]["all_path_vs_required_baselines"] = {}
        for baseline_name in ("symmetric neighbors", "directed one-step"):
            baseline = group[group["model"].eq(baseline_name)].set_index("meeting")
            baseline_comparisons = {}
            for metric in metrics:
                difference = (
                    (all_path[metric] - baseline[metric]).dropna().to_numpy(float)
                )
                baseline_comparisons[metric] = {
                    "mean_difference": float(difference.mean()),
                    "meeting_block_bootstrap_95_interval": bootstrap_interval(
                        difference, bootstrap_draws, rng
                    ),
                    "paired_sign_flip_p": sign_flip_p(difference, rng),
                }
            summary[key]["all_path_vs_required_baselines"][baseline_name] = (
                baseline_comparisons
            )
    return summary


def score_decomposition(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locate category contributions and leave-one-category-out dependence."""
    data = predictions.copy()
    observed = data["observed_occurrence"].to_numpy(dtype=float)
    probability = data["predicted_probability"].to_numpy(dtype=float)
    data["cell_log_loss"] = -(
        observed * np.log(probability) + (1.0 - observed) * np.log(1.0 - probability)
    )
    observed_mass = data["observed_mass"].to_numpy(dtype=float)
    predicted_mass = data["predicted_mass"].to_numpy(dtype=float)
    deviance = np.empty(len(data), dtype=float)
    positive = observed_mass > 0
    deviance[positive] = 2.0 * (
        observed_mass[positive]
        * np.log(observed_mass[positive] / predicted_mass[positive])
        - observed_mass[positive]
        + predicted_mass[positive]
    )
    deviance[~positive] = 2.0 * predicted_mass[~positive]
    data["cell_poisson_deviance"] = deviance

    keys = ["timing", "output_type", "meeting", "topic"]
    wide = data.pivot_table(
        index=keys,
        columns="model",
        values=["cell_log_loss", "cell_poisson_deviance"],
    )
    rows = []
    leave_one_out = []
    for model in ("symmetric neighbors", "directed one-step", "directed all-path"):
        for metric in ("cell_log_loss", "cell_poisson_deviance"):
            difference = wide[(metric, model)] - wide[(metric, "direct attention")]
            table = difference.rename("difference").reset_index()
            category = (
                table.groupby(["timing", "output_type", "topic"], as_index=False)[
                    "difference"
                ]
                .mean()
                .assign(model=model, metric=metric)
            )
            rows.append(category)
            for omitted in sorted(table["topic"].unique()):
                retained = table[~table["topic"].eq(omitted)]
                meeting_difference = retained.groupby(
                    ["timing", "output_type", "meeting"]
                )["difference"].mean()
                for (timing, output), values in meeting_difference.groupby(
                    level=["timing", "output_type"]
                ):
                    leave_one_out.append(
                        {
                            "timing": timing,
                            "output_type": output,
                            "omitted_topic": omitted,
                            "model": model,
                            "metric": metric,
                            "mean_difference_vs_direct": float(values.mean()),
                            "meetings_better": int((values < 0).sum()),
                            "meetings_total": len(values),
                        }
                    )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(leave_one_out)


def lambda_sensitivity(
    base_panel: pd.DataFrame,
    relations: pd.DataFrame,
    history: TransitionHistory,
    bootstrap_draws: int,
) -> pd.DataFrame:
    rows = []
    for decay in LAMBDA_SENSITIVITY:
        for timing in TIMINGS:
            panel = prepare_panel(base_panel, relations, history, timing, decay=decay)
            _, scores = run_forecasts(
                panel,
                model_names=["direct attention", "directed all-path"],
            )
            summary = summarize_scores(scores, bootstrap_draws)
            for output in OUTPUT_TYPES:
                comparison = summary[f"{timing}__{output}"]["paired_vs_direct"][
                    "directed all-path"
                ]
                for metric in ("binary_log_loss", "brier_score", "poisson_deviance"):
                    values = comparison[metric]
                    rows.append(
                        {
                            "decay": decay,
                            "timing": timing,
                            "output_type": output,
                            "metric": metric,
                            **values,
                        }
                    )
    return pd.DataFrame(rows)


def label_null(
    base_panel: pd.DataFrame,
    relations: pd.DataFrame,
    history: TransitionHistory,
    observed_scores: pd.DataFrame,
    draws: int,
) -> pd.DataFrame:
    """Shuffle category identities of the directed map and rerun Resolution tests."""
    if draws <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED + 1)
    observed = observed_scores[
        observed_scores["timing"].eq("retrospective_nowcast")
        & observed_scores["output_type"].eq("Resolution")
    ]
    direct = observed[observed["model"].eq("direct attention")].set_index("meeting")
    directed = observed[observed["model"].eq("directed all-path")].set_index("meeting")
    observed_differences = {
        metric: float((directed[metric] - direct[metric]).mean())
        for metric in ("binary_log_loss", "brier_score", "poisson_deviance")
    }
    rows = []
    for draw in range(draws):
        permutation = rng.permutation(len(history.families))
        panel = prepare_panel(
            base_panel,
            relations,
            history,
            "retrospective_nowcast",
            decay=PRIMARY_LAMBDA,
            permutation=permutation,
        )
        _, scores = run_forecasts(
            panel,
            model_names=["directed all-path"],
            output_types=("Resolution",),
        )
        shuffled = scores.set_index("meeting")
        for metric, observed_difference in observed_differences.items():
            null_difference = float((shuffled[metric] - direct[metric]).mean())
            rows.append(
                {
                    "draw": draw,
                    "metric": metric,
                    "null_mean_difference_vs_direct": null_difference,
                    "observed_mean_difference_vs_direct": observed_difference,
                }
            )
    result = pd.DataFrame(rows)
    for metric, indices in result.groupby("metric").groups.items():
        values = result.loc[indices, "null_mean_difference_vs_direct"]
        observed_value = result.loc[indices, "observed_mean_difference_vs_direct"].iloc[
            0
        ]
        result.loc[indices, "lower_tail_p"] = (
            1 + int((values <= observed_value).sum())
        ) / (1 + len(values))
    return result


def strength_null(
    base_panel: pd.DataFrame,
    relations: pd.DataFrame,
    history: TransitionHistory,
    observed_scores: pd.DataFrame,
    draws: int,
) -> pd.DataFrame:
    """Rewire directed weights while preserving every row and column strength."""
    if draws <= 0:
        return pd.DataFrame()
    observed = observed_scores[
        observed_scores["timing"].eq("retrospective_nowcast")
        & observed_scores["output_type"].eq("Resolution")
    ]
    direct = observed[observed["model"].eq("direct attention")].set_index("meeting")
    directed = observed[observed["model"].eq("directed all-path")].set_index("meeting")
    observed_difference = float(
        (directed["binary_log_loss"] - direct["binary_log_loss"]).mean()
    )
    rows = []
    for draw in range(draws):
        panel = prepare_panel(
            base_panel,
            relations,
            history,
            "retrospective_nowcast",
            rewire_seed=SEED + 10_000 * (draw + 1),
        )
        _, scores = run_forecasts(
            panel,
            model_names=["directed all-path"],
            output_types=("Resolution",),
        )
        shuffled = scores.set_index("meeting")
        null_difference = float(
            (shuffled["binary_log_loss"] - direct["binary_log_loss"]).mean()
        )
        rows.append(
            {
                "draw": draw,
                "null_mean_log_loss_difference_vs_direct": null_difference,
                "observed_mean_log_loss_difference_vs_direct": observed_difference,
            }
        )
    result = pd.DataFrame(rows)
    result["lower_tail_p"] = (
        1
        + int(
            (
                result["null_mean_log_loss_difference_vs_direct"] <= observed_difference
            ).sum()
        )
    ) / (1 + len(result))
    return result


def diagnostics(
    base_panel: pd.DataFrame,
    history: TransitionHistory,
    panels: list[pd.DataFrame],
) -> dict:
    latest, _ = transition_operator(history, TEST_END)
    return {
        "families": len(history.families),
        "panel_meetings": [
            int(base_panel["meeting"].min()),
            int(base_panel["meeting"].max()),
        ],
        "test_meetings": [TEST_START, TEST_END],
        "test_cells_per_output_and_timing": (TEST_END - TEST_START + 1)
        * len(history.families),
        "transition_records": len(history.records),
        "latest_operator_row_sum_range": [
            float(latest.sum(axis=1).min()),
            float(latest.sum(axis=1).max()),
        ],
        "latest_operator_asymmetry_mean_absolute": float(
            np.abs(latest - latest.T).mean()
        ),
        "timing_cutoffs_valid": bool(
            all(
                (panel["transition_cutoff"] < panel["attention_meeting"]).all()
                for panel in panels
            )
        ),
        "strict_attention_precedes_output": bool(
            all(
                (panel["attention_meeting"] < panel["meeting"]).all()
                for panel in panels
                if panel["timing"].iloc[0] == "strict_lag"
            )
        ),
        "crosswalk_warning": (
            "Paper concerns enter 15 author-defined output families; independent blind "
            "validation of this hierarchy remains required."
        ),
        "interpretation_limit": (
            "The models test predictive alignment between documentary flow and adopted "
            "output categories. They do not identify causation or paper-to-output lineage."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--strength-null-draws", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base_panel = pd.read_csv(INPUT_PANEL)
    relations = paper_family_relations(variants()["fractional_multilabel"])
    history = build_transition_history(relations)
    if set(history.families) != set(base_panel["topic"].unique()):
        raise AssertionError("Transition and output families differ")

    panels = [
        prepare_panel(base_panel, relations, history, timing) for timing in TIMINGS
    ]
    analysis_panel = pd.concat(panels, ignore_index=True)
    prediction_tables = []
    score_tables = []
    for panel in panels:
        predictions, scores = run_forecasts(panel)
        prediction_tables.append(predictions)
        score_tables.append(scores)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    scores = pd.concat(score_tables, ignore_index=True)
    summary = summarize_scores(scores, args.bootstrap_draws)
    category_contributions, leave_one_out = score_decomposition(predictions)
    sensitivity = lambda_sensitivity(
        base_panel, relations, history, args.bootstrap_draws
    )
    null = label_null(base_panel, relations, history, scores, args.null_draws)
    strength = strength_null(
        base_panel,
        relations,
        history,
        scores,
        args.strength_null_draws,
    )
    audit = diagnostics(base_panel, history, panels)

    analysis_panel.to_csv(OUTDIR / "analysis_panel.csv", index=False)
    predictions.to_csv(OUTDIR / "predictions.csv", index=False)
    scores.to_csv(OUTDIR / "meeting_scores.csv", index=False)
    category_contributions.to_csv(OUTDIR / "category_contributions.csv", index=False)
    leave_one_out.to_csv(OUTDIR / "leave_one_category_out.csv", index=False)
    sensitivity.to_csv(OUTDIR / "lambda_sensitivity.csv", index=False)
    null.to_csv(OUTDIR / "label_null.csv", index=False)
    strength.to_csv(OUTDIR / "strength_null.csv", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTDIR / "diagnostics.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"diagnostics": audit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
