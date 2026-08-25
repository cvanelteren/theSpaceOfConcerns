#!/usr/bin/env python3
"""Test the documentary-attention -> Resolution -> later-Measure sequence.

The analysis uses the canonical 15-category by meeting panel. Stage A relates
current paper attention to current Resolution mass. Stage B relates current
Resolution mass to Measure mass in subsequent meetings while controlling for
current paper attention and the preceding five meetings of Resolution and
Measure history. Category and origin-meeting fixed effects enter both PPML
models. The estimates describe a sequential association, not causal mediation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.official_regular_atcm_outputs import (
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
    load_official_regular_outputs,
)


OUTDIR = ROOT / "output" / "attention_output_signal"
DATA = ROOT / "data" / "document-summary-multilabel.parquet"
PANEL_PATH = OUTDIR / "attention_resolution_measure_sequence_panel.csv"
COEFFICIENT_PATH = OUTDIR / "attention_resolution_measure_sequence_coefficients.csv"
MODEL_PATH = OUTDIR / "attention_resolution_measure_sequence_models.csv"
SUMMARY_PATH = OUTDIR / "attention_resolution_measure_sequence_summary.json"

HORIZONS = (1, 3, 5, 10)
PRIMARY_HORIZON = 5
HISTORY = 5
AREA_CATEGORY = "Area protection and management"
START_MEETING = 19
END_MEETING = 47
LOCAL_K = 5
LOG_TWO = math.log(2.0)
Z_975 = float(stats.norm.ppf(0.975))


def prior_sum(values: pd.Series, horizon: int) -> pd.Series:
    return values.shift(1).rolling(horizon, min_periods=horizon).sum()


def values(value: object, separator: str | None = None) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raw = str(value).split(separator or ",")
    return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def paper_family_relations() -> pd.DataFrame:
    raw = pd.read_parquet(DATA)
    rows: list[dict[str, object]] = []
    for paper_id, group in raw.groupby("paper_id", sort=False):
        first = group.iloc[0]
        concerns = values(first["category"], "\t")
        actors = values(first["parties"])
        meeting = int(first["meeting_number"])
        if not concerns or not actors or meeting < START_MEETING or meeting > END_MEETING:
            continue
        family_weights: dict[str, float] = {}
        for concern in concerns:
            family = PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.get(concern)
            if family is None:
                raise KeyError(f"Unmapped paper concern: {concern}")
            family_weights[family] = family_weights.get(family, 0.0) + 1.0 / len(concerns)
        for family_name, weight in family_weights.items():
            for actor in actors:
                rows.append(
                    {
                        "paper_id": int(paper_id),
                        "meeting": meeting,
                        "family": family_name,
                        "actor": actor,
                        "paper_weight": weight,
                    }
                )
    relations = pd.DataFrame(rows)
    if relations.empty:
        raise ValueError("No paper-family relations were constructed")
    return relations


def proximity(counts: pd.DataFrame) -> pd.DataFrame:
    actor_total = counts.sum(axis=0).replace(0, np.nan)
    actor_share = counts.divide(actor_total, axis=1)
    family_share = counts.sum(axis=1) / counts.to_numpy(float).sum()
    active = actor_share.divide(family_share.replace(0, np.nan), axis=0).fillna(0).ge(1)
    matrix = active.to_numpy(dtype=int)
    overlap = matrix @ matrix.T
    holders = matrix.sum(axis=1)
    denominator = np.maximum.outer(holders, holders)
    phi = np.divide(
        overlap,
        denominator,
        out=np.zeros_like(overlap, dtype=float),
        where=denominator > 0,
    )
    np.fill_diagonal(phi, 1.0)
    return pd.DataFrame(phi, index=counts.index, columns=counts.index)


def local_weight_matrix(phi: pd.DataFrame) -> np.ndarray:
    matrix = phi.to_numpy(float).copy()
    np.fill_diagonal(matrix, 0.0)
    weights = np.zeros_like(matrix)
    for row in range(len(matrix)):
        order = np.argsort(matrix[row])[::-1]
        keep = [index for index in order if matrix[row, index] > 0][:LOCAL_K]
        if keep:
            weights[row, keep] = matrix[row, keep]
            weights[row] /= weights[row].sum()
    return weights


def add_sequence_features(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["topic", "meeting"]).copy()
    grouped = data.groupby("topic", sort=False)
    data["resolution_prior5"] = grouped["resolution_mass"].transform(
        lambda values: prior_sum(values, HISTORY)
    )
    data["measure_prior5"] = grouped["measure_mass"].transform(
        lambda values: prior_sum(values, HISTORY)
    )
    for horizon in HORIZONS:
        future = [grouped["measure_mass"].shift(-offset) for offset in range(1, horizon + 1)]
        data[f"measure_next{horizon}"] = pd.concat(future, axis=1).sum(
            axis=1, min_count=horizon
        )
    return data


def build_sequence_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    relations = paper_family_relations()
    families = sorted(set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values()))
    meetings = list(range(START_MEETING, END_MEETING + 1))
    index = pd.MultiIndex.from_product([families, meetings], names=["topic", "meeting"])
    papers = relations.drop_duplicates(["paper_id", "family"])
    paper_count = (
        papers.groupby(["family", "meeting"])["paper_weight"]
        .sum()
        .rename_axis(["topic", "meeting"])
        .reindex(index, fill_value=0.0)
        .rename("paper_count")
        .reset_index()
    )
    outputs = load_official_regular_outputs()
    output_rows: list[dict[str, object]] = []
    for output in outputs.itertuples(index=False):
        categories = list(dict.fromkeys(output.official_categories))
        weight = 1.0 / len(categories)
        for category in categories:
            output_rows.append(
                {
                    "topic": category,
                    "meeting": int(output.meeting),
                    "instrument": output.instrument,
                    "weight": weight,
                }
            )
    output_long = pd.DataFrame(output_rows)
    panel = paper_count.copy()
    for instrument in ("Measure", "Decision", "Resolution"):
        column = f"{instrument.lower()}_mass"
        mass = (
            output_long[output_long["instrument"].eq(instrument)]
            .groupby(["topic", "meeting"])["weight"]
            .sum()
            .reindex(index, fill_value=0.0)
            .rename(column)
            .reset_index()
        )
        panel = panel.merge(mass, on=["topic", "meeting"], validate="one_to_one")
    panel["outcome_mass"] = panel[["measure_mass", "decision_mass", "resolution_mass"]].sum(axis=1)
    actors = sorted(relations["actor"].unique())
    topic_index = {topic: position for position, topic in enumerate(families)}
    panel["neighbor_papers"] = 0.0
    for meeting in meetings:
        history = relations[relations["meeting"].lt(meeting)]
        counts = (
            history.groupby(["family", "actor"])["paper_weight"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=families, columns=actors, fill_value=0.0)
        )
        weights = local_weight_matrix(proximity(counts))
        rows = panel["meeting"].eq(meeting)
        current = panel.loc[rows].set_index("topic")["paper_count"].reindex(families).to_numpy(float)
        nearby = weights @ current
        panel.loc[rows, "neighbor_papers"] = [nearby[topic_index[topic]] for topic in panel.loc[rows, "topic"]]
    panel = panel.sort_values(["topic", "meeting"]).reset_index(drop=True)
    expected = {"measure_mass": 277, "decision_mass": 135, "resolution_mass": 172}
    observed = {column: float(panel[column].sum()) for column in expected}
    for column, target in expected.items():
        if not np.isclose(observed[column], target, atol=1e-7):
            raise AssertionError(f"{column} allocation mass {observed[column]} != {target}")
    validation = {
        "n_papers_post_1995": int(relations["paper_id"].nunique()),
        "n_categories": len(families),
        "n_meetings": len(meetings),
        "meeting_range": [min(meetings), max(meetings)],
        "n_category_meetings": int(len(panel)),
        "fractional_output_mass": observed,
        "paper_weighting": "one total unit per paper across archive concerns",
        "output_weighting": "equal fractional weight across official instrument categories",
    }
    return panel, validation


def safe(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def design_matrix(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {"intercept": np.ones(len(data), dtype=float)}
    topics = sorted(data["topic"].unique())
    meetings = sorted(data["meeting"].astype(int).unique())
    for topic in topics[1:]:
        columns[f"fe_topic__{safe(topic)}"] = data["topic"].eq(topic).to_numpy(float)
    for meeting in meetings[1:]:
        columns[f"fe_meeting__{meeting}"] = data["meeting"].eq(meeting).to_numpy(float)
    for predictor in predictors:
        columns[f"slope__{predictor}"] = np.log1p(data[predictor].to_numpy(float))
    return pd.DataFrame(columns, index=data.index, dtype=float)


def fit_ppml(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> tuple[object, np.ndarray, pd.DataFrame]:
    required = [outcome, *predictors]
    model_data = data.dropna(subset=required).copy()
    if (model_data[required] < 0).any().any():
        raise ValueError("Sequential model variables must be nonnegative")
    design = design_matrix(model_data, predictors)
    fitted = sm.GLM(
        model_data[outcome].to_numpy(float),
        design,
        family=sm.families.Poisson(),
    ).fit(maxiter=500, tol=1e-10)
    if not bool(fitted.converged):
        raise RuntimeError(f"PPML did not converge for {outcome}")
    covariance, _, _ = cov_cluster_2groups(
        fitted,
        pd.Categorical(model_data["topic"]).codes,
        pd.Categorical(model_data["meeting"]).codes,
        use_correction=True,
    )
    covariance = np.asarray(covariance, dtype=float)
    if not np.isfinite(covariance).all():
        raise RuntimeError(f"Non-finite clustered covariance for {outcome}")
    return fitted, covariance, design


def coefficient_rows(
    fitted: object,
    covariance: np.ndarray,
    design: pd.DataFrame,
    data: pd.DataFrame,
    stage: str,
    outcome: str,
    predictors: list[str],
    horizon: int,
    sensitivity: str,
) -> list[dict[str, object]]:
    index = {name: position for position, name in enumerate(design.columns)}
    rows = []
    for predictor in predictors:
        position = index[f"slope__{predictor}"]
        beta = float(fitted.params.iloc[position])
        variance = max(float(covariance[position, position]), 0.0)
        se = math.sqrt(variance)
        z_value = beta / se if se > 0 else float("nan")
        p_value = float(2 * stats.norm.sf(abs(z_value))) if se > 0 else float("nan")
        low = beta - Z_975 * se
        high = beta + Z_975 * se
        rows.append(
            {
                "stage": stage,
                "outcome": outcome,
                "horizon": horizon,
                "sensitivity": sensitivity,
                "predictor": predictor,
                "beta_log1p": beta,
                "clustered_se": se,
                "z_value": z_value,
                "p_value": p_value,
                "ratio_per_doubling_plus_one": math.exp(LOG_TWO * beta),
                "doubling_ci_low": math.exp(LOG_TWO * low),
                "doubling_ci_high": math.exp(LOG_TWO * high),
                "n_topic_meetings": int(len(data)),
                "n_topics": int(data["topic"].nunique()),
                "n_origin_meetings": int(data["meeting"].nunique()),
            }
        )
    return rows


def model_record(
    fitted: object,
    data: pd.DataFrame,
    stage: str,
    outcome: str,
    horizon: int,
    sensitivity: str,
    specification: str,
) -> dict[str, object]:
    return {
        "stage": stage,
        "outcome": outcome,
        "horizon": horizon,
        "sensitivity": sensitivity,
        "specification": specification,
        "n_topic_meetings": int(len(data)),
        "n_topics": int(data["topic"].nunique()),
        "n_origin_meetings": int(data["meeting"].nunique()),
        "outcome_mass": float(data[outcome].sum()),
        "deviance": float(fitted.deviance),
        "aic": float(fitted.aic),
    }


def run_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_records: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    sensitivities = {
        "all_categories": panel,
        "exclude_area_protection": panel[panel["topic"].ne(AREA_CATEGORY)].copy(),
    }
    stage_a_predictors = ["paper_count", "neighbor_papers", "resolution_prior5"]
    stage_b_predictors = [
        "resolution_mass",
        "paper_count",
        "neighbor_papers",
        "resolution_prior5",
        "measure_prior5",
    ]
    reduced_b_predictors = [
        "paper_count",
        "neighbor_papers",
        "resolution_prior5",
        "measure_prior5",
    ]

    for sensitivity, subset in sensitivities.items():
        for horizon in HORIZONS:
            outcome = f"measure_next{horizon}"
            required = [
                outcome,
                "resolution_mass",
                "paper_count",
                "neighbor_papers",
                "resolution_prior5",
                "measure_prior5",
            ]
            data = subset.dropna(subset=required).copy()

            stage_a_fit, stage_a_cov, stage_a_design = fit_ppml(
                data, "resolution_mass", stage_a_predictors
            )
            coefficient_records.extend(
                coefficient_rows(
                    stage_a_fit,
                    stage_a_cov,
                    stage_a_design,
                    data,
                    "A_attention_to_resolution",
                    "resolution_mass",
                    stage_a_predictors,
                    horizon,
                    sensitivity,
                )
            )
            model_records.append(
                model_record(
                    stage_a_fit,
                    data,
                    "A_attention_to_resolution",
                    "resolution_mass",
                    horizon,
                    sensitivity,
                    "full",
                )
            )

            stage_b_fit, stage_b_cov, stage_b_design = fit_ppml(
                data, outcome, stage_b_predictors
            )
            coefficient_records.extend(
                coefficient_rows(
                    stage_b_fit,
                    stage_b_cov,
                    stage_b_design,
                    data,
                    "B_resolution_to_future_measure",
                    outcome,
                    stage_b_predictors,
                    horizon,
                    sensitivity,
                )
            )
            model_records.append(
                model_record(
                    stage_b_fit,
                    data,
                    "B_resolution_to_future_measure",
                    outcome,
                    horizon,
                    sensitivity,
                    "with_current_resolution",
                )
            )

            reduced_fit, _, _ = fit_ppml(data, outcome, reduced_b_predictors)
            model_records.append(
                model_record(
                    reduced_fit,
                    data,
                    "B_resolution_to_future_measure",
                    outcome,
                    horizon,
                    sensitivity,
                    "without_current_resolution",
                )
            )

    return pd.DataFrame(coefficient_records), pd.DataFrame(model_records)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel, validation = build_sequence_panel()
    panel = add_sequence_features(panel)
    panel.to_csv(PANEL_PATH, index=False)
    coefficients, models = run_models(panel)
    coefficients.to_csv(COEFFICIENT_PATH, index=False)
    models.to_csv(MODEL_PATH, index=False)

    primary = coefficients[
        coefficients["horizon"].eq(PRIMARY_HORIZON)
        & coefficients["sensitivity"].eq("all_categories")
    ]
    attention = primary[
        primary["stage"].eq("A_attention_to_resolution")
        & primary["predictor"].eq("paper_count")
    ].iloc[0]
    resolution = primary[
        primary["stage"].eq("B_resolution_to_future_measure")
        & primary["predictor"].eq("resolution_mass")
    ].iloc[0]
    pathway_supported = bool(
        attention["beta_log1p"] > 0
        and attention["p_value"] < 0.05
        and resolution["beta_log1p"] > 0
        and resolution["p_value"] < 0.05
    )
    summary = {
        "estimand": (
            "sequential association: current paper attention -> current Resolution "
            "mass -> later Measure mass, conditional on current attention and five-meeting "
            "Resolution and Measure histories"
        ),
        "primary_horizon_meetings": PRIMARY_HORIZON,
        "sensitivity_horizons_meetings": [value for value in HORIZONS if value != PRIMARY_HORIZON],
        "fixed_effects": ["official output category", "origin meeting"],
        "covariance": "two-way clustered by category and origin meeting",
        "causal_claim": False,
        "pathway_supported_at_primary_horizon": pathway_supported,
        "primary_attention_to_resolution": attention.to_dict(),
        "primary_resolution_to_future_measure": resolution.to_dict(),
        "validation": validation,
        "outputs": {
            "panel": str(PANEL_PATH.relative_to(ROOT)),
            "coefficients": str(COEFFICIENT_PATH.relative_to(ROOT)),
            "models": str(MODEL_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(coefficients.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
