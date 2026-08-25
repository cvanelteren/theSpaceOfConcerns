#!/usr/bin/env python3
"""Forecast where adopted ATCM output lands from earlier documentary attention.

The outcome is the distribution of official output-category weight across the
15 broad categories at each meeting. Forecasts are genuinely prospective:
each test meeting is fitted only on earlier meetings, and the history horizon
is selected using earlier expanding-window forecasts. The comparison asks
whether direct paper attention and actor reach add information beyond the
category's own output history. Neighbouring attention is a final ablation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.analyze_output_category_families import paper_family_relations
from scripts.primary_concern_sensitivity import variants


ROOT = Path(__file__).resolve().parents[1]
INPUT_PANEL = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "output_family_topic_meeting_panel.csv"
)
OUTDIR = ROOT / "output" / "attention_output_forecast"
PREDICTIONS_PATH = OUTDIR / "predictions.csv"
MEETING_SCORES_PATH = OUTDIR / "meeting_scores.csv"
SUMMARY_PATH = OUTDIR / "summary.json"
SENSITIVITY_PATH = OUTDIR / "regularization_sensitivity.csv"
AREA_EXCLUSION_SCORES_PATH = OUTDIR / "area_exclusion_meeting_scores.csv"
AREA_EXCLUSION_SUMMARY_PATH = OUTDIR / "area_exclusion_summary.json"
CATEGORY_CONTRIBUTIONS_PATH = OUTDIR / "category_score_contributions.csv"
OCCURRENCE_DIAGNOSTICS_PATH = OUTDIR / "occurrence_diagnostics.csv"
AREA_CATEGORY = "Area protection and management"

TRAIN_START = 20
INNER_START = 25
TEST_START = 29
TEST_END = 47
HISTORY_HORIZONS = (1, 3, 5, 8)
ATTENTION_HORIZONS = (1, 2, 3, 5)
PRIMARY_ALPHA = 0.2
ALPHA_SENSITIVITY = (0.1, 0.2, 0.5, 1.0)
BOOTSTRAP_DRAWS = 20_000
SEED = 20260815


@dataclass(frozen=True)
class Specification:
    model: str
    history_horizon: int | None
    attention_horizon: int | None

    @property
    def key(self) -> str:
        return (
            f"{self.model}__history_{self.history_horizon}"
            f"__attention_{self.attention_horizon}"
        )


def specifications() -> list[Specification]:
    rows: list[Specification] = []
    rows.extend(Specification("output history", h, None) for h in HISTORY_HORIZONS)
    rows.extend(Specification("direct attention", None, a) for a in ATTENTION_HORIZONS)
    for history in HISTORY_HORIZONS:
        for attention in ATTENTION_HORIZONS:
            rows.append(Specification("history + direct attention", history, attention))
            rows.append(
                Specification(
                    "history + direct + neighbouring attention", history, attention
                )
            )
    return rows


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(INPUT_PANEL).sort_values(["topic", "meeting"]).copy()
    relations = paper_family_relations(variants()["fractional_multilabel"])[
        ["family", "meeting", "actor"]
    ].drop_duplicates()

    for horizon in HISTORY_HORIZONS:
        panel[f"output_history_{horizon}"] = (
            panel.groupby("topic")["outcome_mass"]
            .transform(
                lambda values: values.shift(1).rolling(horizon, min_periods=1).sum()
            )
            .fillna(0.0)
        )

    for horizon in ATTENTION_HORIZONS:
        panel[f"paper_history_{horizon}"] = (
            panel.groupby("topic")["paper_count"]
            .transform(
                lambda values: values.shift(1).rolling(horizon, min_periods=1).sum()
            )
            .fillna(0.0)
        )
        panel[f"nearby_history_{horizon}"] = (
            panel.groupby("topic")["neighbor_papers"]
            .transform(
                lambda values: values.shift(1).rolling(horizon, min_periods=1).sum()
            )
            .fillna(0.0)
        )
        reach = []
        for record in panel[["topic", "meeting"]].itertuples(index=False):
            reach.append(
                relations[
                    relations["family"].eq(record.topic)
                    & relations["meeting"].ge(record.meeting - horizon)
                    & relations["meeting"].lt(record.meeting)
                ]["actor"].nunique()
            )
        panel[f"actor_reach_{horizon}"] = reach
    return panel


def numeric_features(specification: Specification) -> list[str]:
    features: list[str] = []
    if specification.history_horizon is not None:
        features.append(f"output_history_{specification.history_horizon}")
    if specification.attention_horizon is not None:
        horizon = specification.attention_horizon
        features.extend([f"paper_history_{horizon}", f"actor_reach_{horizon}"])
        if "neighbouring" in specification.model:
            features.append(f"nearby_history_{horizon}")
    return features


def fit_forecast(
    panel: pd.DataFrame,
    test_meeting: int,
    specification: Specification,
    alpha: float,
) -> pd.DataFrame:
    train = panel[
        panel["meeting"].ge(TRAIN_START) & panel["meeting"].lt(test_meeting)
    ].copy()
    test = panel[panel["meeting"].eq(test_meeting)].copy()
    features = numeric_features(specification)
    design = ["topic", *features]
    transform = ColumnTransformer(
        [
            ("topic", OneHotEncoder(handle_unknown="ignore"), ["topic"]),
            ("numeric", StandardScaler(), features),
        ]
    )
    model = make_pipeline(
        transform,
        PoissonRegressor(alpha=alpha, max_iter=2_000, tol=1e-9),
    )
    model.fit(train[design], train["outcome_mass"])
    prediction = np.maximum(model.predict(test[design]), 1e-12)
    result = test[["topic", "meeting", "outcome_mass"]].copy()
    result["predicted_count"] = prediction
    result["predicted_share"] = prediction / prediction.sum()
    result["observed_share"] = result["outcome_mass"] / result["outcome_mass"].sum()
    result["specification"] = specification.key
    result["model"] = specification.model
    result["history_horizon"] = specification.history_horizon
    result["attention_horizon"] = specification.attention_horizon
    result["alpha"] = alpha
    return result


def allocation_log_score(prediction: pd.DataFrame) -> float:
    return float(
        -np.sum(
            prediction["observed_share"].to_numpy()
            * np.log(prediction["predicted_share"].to_numpy())
        )
    )


def cached_forecasts(
    panel: pd.DataFrame,
    alpha: float,
    model_names: set[str] | None = None,
) -> dict[tuple[str, int], pd.DataFrame]:
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    selected_specs = [
        specification
        for specification in specifications()
        if model_names is None or specification.model in model_names
    ]
    for specification in selected_specs:
        for meeting in range(INNER_START, TEST_END + 1):
            cache[(specification.key, meeting)] = fit_forecast(
                panel, meeting, specification, alpha
            )
    return cache


def choose_specification(
    cache: dict[tuple[str, int], pd.DataFrame],
    model_name: str,
    outer_meeting: int,
) -> Specification:
    candidates = [item for item in specifications() if item.model == model_name]
    inner_meetings = range(INNER_START, outer_meeting)
    scored = []
    for candidate in candidates:
        values = [
            allocation_log_score(cache[(candidate.key, meeting)])
            for meeting in inner_meetings
        ]
        scored.append((float(np.mean(values)), candidate.key, candidate))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def meeting_metrics(prediction: pd.DataFrame) -> dict[str, float]:
    observed = prediction["outcome_mass"].to_numpy(dtype=float)
    predicted = prediction["predicted_count"].to_numpy(dtype=float)
    occurrence = observed > 0
    auc = (
        float(roc_auc_score(occurrence, predicted))
        if np.unique(occurrence).size == 2
        else float("nan")
    )
    top_three = np.argsort(predicted)[-3:]
    return {
        "allocation_log_score": allocation_log_score(prediction),
        "poisson_deviance": float(mean_poisson_deviance(observed, predicted)),
        "occurrence_auc": auc,
        "rank_correlation": float(spearmanr(observed, predicted).statistic),
        "top_three_output_share": float(observed[top_three].sum() / observed.sum()),
    }


def run_alpha(
    panel: pd.DataFrame,
    alpha: float,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_model_names = list(
        dict.fromkeys(specification.model for specification in specifications())
    )
    model_names = all_model_names if model_names is None else model_names
    cache = cached_forecasts(panel, alpha, set(model_names))
    prediction_tables = []
    score_rows = []
    for meeting in range(TEST_START, TEST_END + 1):
        for model_name in model_names:
            selected = choose_specification(cache, model_name, meeting)
            prediction = cache[(selected.key, meeting)].copy()
            prediction_tables.append(prediction)
            score_rows.append(
                {
                    "model": model_name,
                    "meeting": meeting,
                    "selected_history_horizon": selected.history_horizon,
                    "selected_attention_horizon": selected.attention_horizon,
                    "alpha": alpha,
                    **meeting_metrics(prediction),
                }
            )
    return pd.concat(prediction_tables, ignore_index=True), pd.DataFrame(score_rows)


def run_fixed_horizons(
    panel: pd.DataFrame, alpha: float, selected_scores: pd.DataFrame
) -> pd.DataFrame:
    """Change regularization while retaining the past-selected horizons."""
    score_rows = []
    for row in selected_scores.itertuples(index=False):
        history = (
            None
            if pd.isna(row.selected_history_horizon)
            else int(row.selected_history_horizon)
        )
        attention = (
            None
            if pd.isna(row.selected_attention_horizon)
            else int(row.selected_attention_horizon)
        )
        selected = Specification(row.model, history, attention)
        prediction = fit_forecast(panel, int(row.meeting), selected, alpha)
        score_rows.append(
            {
                "model": row.model,
                "meeting": int(row.meeting),
                "selected_history_horizon": history,
                "selected_attention_horizon": attention,
                "alpha": alpha,
                **meeting_metrics(prediction),
            }
        )
    return pd.DataFrame(score_rows)


def bootstrap_mean_difference(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def summarize(scores: pd.DataFrame) -> dict:
    grouped = scores.groupby("model")
    model_summary = grouped[
        [
            "allocation_log_score",
            "poisson_deviance",
            "occurrence_auc",
            "rank_correlation",
            "top_three_output_share",
        ]
    ].mean().to_dict(orient="index")
    baseline = scores[scores["model"].eq("output history")].set_index("meeting")
    comparisons = {}
    for model_name in sorted(set(scores["model"]) - {"output history"}):
        candidate = scores[scores["model"].eq(model_name)].set_index("meeting")
        difference = (
            candidate["allocation_log_score"] - baseline["allocation_log_score"]
        ).to_numpy()
        comparisons[model_name] = {
            "mean_log_score_difference_vs_output_history": float(difference.mean()),
            "meeting_bootstrap_95_interval": bootstrap_mean_difference(difference),
            "meetings_improved": int((difference < 0).sum()),
            "meetings_total": int(len(difference)),
        }
    return {
        "forecast_target": "conditional distribution of adopted output-category weight within each ATCM",
        "test_meetings": [TEST_START, TEST_END],
        "n_test_meetings": TEST_END - TEST_START + 1,
        "selection": "history horizons chosen using expanding forecasts from earlier meetings only",
        "primary_alpha": PRIMARY_ALPHA,
        "model_summary": model_summary,
        "paired_comparisons": comparisons,
    }


def write_prediction_diagnostics(predictions: pd.DataFrame) -> None:
    """Decompose forecast gains and separate occurrence from output amount."""
    clipped = predictions.copy()
    clipped["occurrence"] = clipped["outcome_mass"].gt(0).astype(float)
    clipped["occurrence_probability"] = 1.0 - np.exp(-clipped["predicted_count"])
    clipped["occurrence_probability"] = clipped["occurrence_probability"].clip(
        1e-12, 1.0 - 1e-12
    )
    occurrence_rows = []
    for model_name, group in clipped.groupby("model"):
        observed = group["occurrence"].to_numpy(float)
        probability = group["occurrence_probability"].to_numpy(float)
        occurrence_rows.append(
            {
                "model": model_name,
                "brier_score": float(np.mean((observed - probability) ** 2)),
                "binary_log_loss": float(
                    -np.mean(
                        observed * np.log(probability)
                        + (1.0 - observed) * np.log(1.0 - probability)
                    )
                ),
            }
        )
    pd.DataFrame(occurrence_rows).to_csv(OCCURRENCE_DIAGNOSTICS_PATH, index=False)

    index = ["meeting", "topic", "outcome_mass", "observed_share"]
    wide = predictions.pivot_table(
        index=index, columns="model", values="predicted_share"
    ).reset_index()
    baseline = wide["output history"]
    rows = []
    for model_name in sorted(set(predictions["model"]) - {"output history"}):
        wide["gain"] = wide["observed_share"] * np.log(
            wide[model_name] / baseline
        )
        category = (
            wide.groupby("topic", as_index=False)
            .agg(
                log_score_gain=("gain", "sum"),
                output_mass=("outcome_mass", "sum"),
                active_meetings=("outcome_mass", lambda values: int((values > 0).sum())),
            )
            .assign(model=model_name)
        )
        rows.append(category)
    pd.concat(rows, ignore_index=True).to_csv(CATEGORY_CONTRIBUTIONS_PATH, index=False)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    primary_predictions, primary_scores = run_alpha(panel, PRIMARY_ALPHA)
    primary_predictions.to_csv(PREDICTIONS_PATH, index=False)
    primary_scores.to_csv(MEETING_SCORES_PATH, index=False)
    write_prediction_diagnostics(primary_predictions)
    summary = summarize(primary_scores)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    sensitivity_rows = []
    for alpha in ALPHA_SENSITIVITY:
        _, scores = (
            (primary_predictions, primary_scores)
            if alpha == PRIMARY_ALPHA
            else (None, run_fixed_horizons(panel, alpha, primary_scores))
        )
        alpha_summary = summarize(scores)
        for model_name, values in alpha_summary["model_summary"].items():
            sensitivity_rows.append(
                {"alpha": alpha, "model": model_name, **values}
            )
    pd.DataFrame(sensitivity_rows).to_csv(SENSITIVITY_PATH, index=False)

    # Area protection and management contains an unusually large share of the
    # official output. This check asks whether direct attention still adds to
    # output recurrence when that outcome-category row is removed. It does not
    # use neighbouring attention, so no retained predictor includes the omitted
    # category through the broad map.
    no_area = panel[~panel["topic"].eq(AREA_CATEGORY)].copy()
    _, area_scores = run_alpha(
        no_area,
        PRIMARY_ALPHA,
        model_names=["output history", "history + direct attention"],
    )
    area_scores.to_csv(AREA_EXCLUSION_SCORES_PATH, index=False)
    area_summary = summarize(area_scores)
    area_summary["excluded_outcome_category"] = AREA_CATEGORY
    AREA_EXCLUSION_SUMMARY_PATH.write_text(
        json.dumps(area_summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
