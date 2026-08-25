#!/usr/bin/env python3
"""Confirmatory comparison of documentary histories across ATCM output types.

This script implements the analysis gate in ``MULTILABEL_RESULTS_REWRITE_PLAN.md``.
It uses the corrected fractional-multilabel paper panel and fractional allocations
from the official ATS instrument categories.  The
five-meeting model stacks Measures, Decisions, and Resolutions, estimates separate
slopes on a common unstandardized log(1+x) scale, and absorbs concern-by-type and
meeting-by-type fixed effects.  It also reports:

* the six-degree-of-freedom omnibus test of slope heterogeneity;
* all nine pairwise slope contrasts with Holm correction;
* type-specific slopes on a one-plus-doubling scale;
* the pooled focal-versus-nearby contrast;
* horizon profiles estimated on an identical set of meetings; and
* a leave-one-meeting-cluster-out jackknife sensitivity for the five-meeting model.

The main covariance is finite-sample-corrected two-way clustering by concern and
meeting.  The meeting jackknife is a small-meeting-cluster sensitivity; it is not a
two-dimensional wild-cluster bootstrap and does not replace the main covariance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "formal_outputs"
    / "attention_fractional_multilabel__coding_fractional_multilabel"
    / "topic_meeting_panel.csv"
)
REFERENCE_PATH = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "formal_output_instrument_sensitivity.csv"
)
PREDICTIONS_PATH = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "fractional_multilabel"
    / "outcome_topic_predictions.csv"
)
WEIGHTS_PATH = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "fractional_multilabel"
    / "outcome_topic_probabilities.csv"
)
TITLE_WEIGHTS_PATH = WEIGHTS_PATH.with_name(
    "outcome_topic_probabilities_title_model.csv"
)
OUTDIR = ROOT / "output" / "category_treatment_comparison"

TYPE_SLOPES_PATH = OUTDIR / "output_type_interaction_type_slopes.csv"
CONTRASTS_PATH = OUTDIR / "output_type_interaction_pairwise_contrasts.csv"
OMNIBUS_PATH = OUTDIR / "output_type_interaction_omnibus.csv"
POOLED_PATH = OUTDIR / "output_type_interaction_pooled.csv"
HORIZONS_PATH = OUTDIR / "output_type_interaction_common_support_horizons.csv"
VARYING_HORIZONS_PATH = OUTDIR / "output_type_interaction_varying_support_horizons.csv"
TITLE_SENSITIVITY_PATH = OUTDIR / "output_title_model_same_support_sensitivity.csv"
SMALL_CLUSTER_PATH = OUTDIR / "output_type_interaction_small_cluster_sensitivity.csv"
SUMMARY_PATH = OUTDIR / "output_type_interaction_summary.json"

HORIZONS = (1, 2, 3, 5, 8, 10)
PRIMARY_HORIZON = 5
INSTRUMENTS = ("Measure", "Decision", "Resolution")
INSTRUMENT_COLUMNS = {
    "Measure": "measure_mass",
    "Decision": "decision_mass",
    "Resolution": "resolution_mass",
}
EXPECTED_OUTPUTS = {"Measure": 277, "Decision": 135, "Resolution": 172}
PREDICTORS = ("focal", "nearby", "prior_output")
PAIRWISE_TYPES = (
    ("Measure", "Decision"),
    ("Measure", "Resolution"),
    ("Resolution", "Decision"),
)
LOG_TWO = math.log(2.0)
Z_975 = float(stats.norm.ppf(0.975))


def rolling_prior(values: pd.Series, horizon: int) -> pd.Series:
    """Sum exactly ``horizon`` meetings preceding the focal meeting."""
    return values.shift(1).rolling(horizon, min_periods=horizon).sum()


def add_prior_stocks(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["topic", "meeting"]).copy()
    for horizon in HORIZONS:
        data[f"focal_prior{horizon}"] = data.groupby("topic")[
            "paper_count"
        ].transform(lambda values, h=horizon: rolling_prior(values, h))
        data[f"nearby_prior{horizon}"] = data.groupby("topic")[
            "neighbor_papers"
        ].transform(lambda values, h=horizon: rolling_prior(values, h))
        data[f"pooled_output_prior{horizon}"] = data.groupby("topic")[
            "outcome_mass"
        ].transform(lambda values, h=horizon: rolling_prior(values, h))
        for instrument, column in INSTRUMENT_COLUMNS.items():
            slug = instrument.lower()
            data[f"{slug}_output_prior{horizon}"] = data.groupby("topic")[
                column
            ].transform(lambda values, h=horizon: rolling_prior(values, h))
    return data


def load_and_validate_panel() -> tuple[pd.DataFrame, dict]:
    panel = pd.read_csv(PANEL_PATH)
    required = {
        "topic",
        "meeting",
        "paper_count",
        "neighbor_papers",
        "outcome_mass",
        *INSTRUMENT_COLUMNS.values(),
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    if panel.duplicated(["topic", "meeting"]).any():
        raise ValueError("Panel contains duplicate concern-meeting rows")
    topics = sorted(panel["topic"].unique())
    meetings = sorted(panel["meeting"].astype(int).unique())
    expected_rows = len(topics) * len(meetings)
    if len(panel) != expected_rows:
        raise ValueError(f"Panel is incomplete: {len(panel)} != {expected_rows}")
    output_totals = {
        instrument: float(panel[column].sum())
        for instrument, column in INSTRUMENT_COLUMNS.items()
    }
    for instrument, expected in EXPECTED_OUTPUTS.items():
        if not np.isclose(output_totals[instrument], expected, atol=1e-7):
            raise ValueError(
                f"{instrument} allocation mass {output_totals[instrument]} != {expected}"
            )
    total_mass = float(panel["outcome_mass"].sum())
    if not np.isclose(total_mass, sum(EXPECTED_OUTPUTS.values()), atol=1e-7):
        raise ValueError(f"Total output allocation mass {total_mass} != 584")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    weights = pd.read_csv(WEIGHTS_PATH)
    if predictions["outcome_id"].nunique() != 584 or len(predictions) != 584:
        raise ValueError("Expected 584 unique official output allocations")
    if (
        "allocation_source" not in predictions
        or not predictions["allocation_source"].eq(
            "official_ats_instrument_category"
        ).all()
    ):
        raise ValueError("Primary outputs must use official ATS instrument categories")
    if weights.duplicated(["outcome_id", "topic"]).any():
        raise ValueError("Output allocation weights contain duplicate output-topic rows")
    weight_sums = weights.groupby("outcome_id")["probability"].sum()
    if len(weight_sums) != 584 or not np.allclose(
        weight_sums.to_numpy(float), 1.0, atol=1e-9
    ):
        raise ValueError("Each output's official category weights must sum to one")
    if set(weights["outcome_id"]) != set(predictions["outcome_id"]):
        raise ValueError("Prediction and allocation-weight output inventories differ")
    eligible_topics = sorted(
        panel.groupby("topic")["outcome_mass"]
        .sum()
        .loc[lambda values: values.gt(0)]
        .index
    )
    if len(eligible_topics) != 15:
        raise ValueError(
            "Expected 15 concern-space counterparts of official output categories, "
            f"got {len(eligible_topics)}"
        )
    panel = panel[panel["topic"].isin(eligible_topics)].copy()
    panel = add_prior_stocks(panel)

    # Directly verify that a lag-one stock excludes the focal meeting.
    ordered = panel.sort_values(["topic", "meeting"])
    expected_lag = ordered.groupby("topic")["paper_count"].shift(1)
    observed_lag = ordered["focal_prior1"]
    comparable = expected_lag.notna()
    if not np.allclose(
        expected_lag[comparable].to_numpy(float),
        observed_lag[comparable].to_numpy(float),
    ):
        raise ValueError("Lag-one focal stock contains focal-meeting information")

    validation = {
        "panel_path": str(PANEL_PATH.relative_to(ROOT)),
        "n_topics_in_full_paper_space": len(topics),
        "n_output_eligible_topics": len(eligible_topics),
        "output_eligible_topics": eligible_topics,
        "n_meetings": len(meetings),
        "meeting_min": int(min(meetings)),
        "meeting_max": int(max(meetings)),
        "n_topic_meetings": int(len(panel)),
        "output_allocation_mass_by_type": output_totals,
        "total_output_allocation_mass": total_mass,
        "n_unique_officially_categorized_outputs": int(
            predictions["outcome_id"].nunique()
        ),
        "allocation_source": "official_ats_instrument_category",
        "allocation_weights_sum_to_one_per_output": True,
        "predictions_path": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "weights_path": str(WEIGHTS_PATH.relative_to(ROOT)),
        "lag_one_excludes_focal_meeting": True,
    }
    return panel, validation


def stack_instruments(
    panel: pd.DataFrame,
    horizon: int,
    meetings: Iterable[int] | None = None,
) -> pd.DataFrame:
    allowed = None if meetings is None else {int(value) for value in meetings}
    pieces = []
    for instrument in INSTRUMENTS:
        slug = instrument.lower()
        columns = [
            "topic",
            "meeting",
            INSTRUMENT_COLUMNS[instrument],
            f"focal_prior{horizon}",
            f"nearby_prior{horizon}",
            f"{slug}_output_prior{horizon}",
        ]
        piece = panel[columns].copy()
        if allowed is not None:
            piece = piece[piece["meeting"].astype(int).isin(allowed)].copy()
        piece = piece.dropna(subset=columns[3:])
        piece = piece.rename(
            columns={
                INSTRUMENT_COLUMNS[instrument]: "output_mass",
                f"focal_prior{horizon}": "focal",
                f"nearby_prior{horizon}": "nearby",
                f"{slug}_output_prior{horizon}": "prior_output",
            }
        )
        piece["instrument"] = instrument
        pieces.append(piece)
    stacked = pd.concat(pieces, ignore_index=True)
    for predictor in PREDICTORS:
        if (stacked[predictor] < 0).any():
            raise ValueError(f"Negative values found in {predictor}")
        stacked[f"log_{predictor}"] = np.log1p(stacked[predictor].to_numpy(float))
    return stacked.sort_values(["instrument", "topic", "meeting"]).reset_index(
        drop=True
    )


def pooled_panel(
    panel: pd.DataFrame,
    horizon: int,
    meetings: Iterable[int] | None = None,
) -> pd.DataFrame:
    columns = [
        "topic",
        "meeting",
        "outcome_mass",
        f"focal_prior{horizon}",
        f"nearby_prior{horizon}",
        f"pooled_output_prior{horizon}",
    ]
    data = panel[columns].copy()
    if meetings is not None:
        allowed = {int(value) for value in meetings}
        data = data[data["meeting"].astype(int).isin(allowed)].copy()
    data = data.dropna(subset=columns[3:]).rename(
        columns={
            "outcome_mass": "output_mass",
            f"focal_prior{horizon}": "focal",
            f"nearby_prior{horizon}": "nearby",
            f"pooled_output_prior{horizon}": "prior_output",
        }
    )
    for predictor in PREDICTORS:
        data[f"log_{predictor}"] = np.log1p(data[predictor].to_numpy(float))
    return data.sort_values(["topic", "meeting"]).reset_index(drop=True)


def _safe_token(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def build_design(data: pd.DataFrame, stacked: bool) -> pd.DataFrame:
    """Build an explicit full-rank fixed-effect design without a global intercept."""
    columns: dict[str, np.ndarray] = {}
    topics = sorted(data["topic"].unique())
    meetings = sorted(data["meeting"].astype(int).unique())
    if stacked:
        instruments = list(INSTRUMENTS)
        for instrument in instruments:
            mask = data["instrument"].eq(instrument).to_numpy(float)
            slug = instrument.lower()
            columns[f"intercept__{slug}"] = mask
            for topic in topics[1:]:
                columns[f"fe_topic__{slug}__{_safe_token(topic)}"] = mask * data[
                    "topic"
                ].eq(topic).to_numpy(float)
            for meeting in meetings[1:]:
                columns[f"fe_meeting__{slug}__{meeting}"] = mask * data[
                    "meeting"
                ].eq(meeting).to_numpy(float)
            for predictor in PREDICTORS:
                columns[f"slope__{slug}__{predictor}"] = (
                    mask * data[f"log_{predictor}"].to_numpy(float)
                )
    else:
        columns["intercept"] = np.ones(len(data), dtype=float)
        for topic in topics[1:]:
            columns[f"fe_topic__{_safe_token(topic)}"] = data["topic"].eq(
                topic
            ).to_numpy(float)
        for meeting in meetings[1:]:
            columns[f"fe_meeting__{meeting}"] = data["meeting"].eq(meeting).to_numpy(
                float
            )
        for predictor in PREDICTORS:
            columns[f"slope__pooled__{predictor}"] = data[
                f"log_{predictor}"
            ].to_numpy(float)
    design = pd.DataFrame(columns, index=data.index, dtype=float)
    if design.isna().any().any():
        raise ValueError("Design matrix contains missing values")
    return design


def fit_ppml(
    data: pd.DataFrame,
    stacked: bool,
    clustered: bool = True,
) -> tuple[object, np.ndarray | None, pd.DataFrame]:
    design = build_design(data, stacked=stacked)
    outcome = data["output_mass"].to_numpy(float)
    fitted = sm.GLM(outcome, design, family=sm.families.Poisson()).fit(
        maxiter=300, tol=1e-10
    )
    if not bool(fitted.converged):
        raise RuntimeError("PPML model did not converge")
    covariance = None
    if clustered:
        topic_groups = pd.Categorical(data["topic"]).codes
        meeting_groups = pd.Categorical(data["meeting"]).codes
        covariance, _, _ = cov_cluster_2groups(
            fitted,
            topic_groups,
            meeting_groups,
            use_correction=True,
        )
        covariance = np.asarray(covariance, dtype=float)
        if not np.isfinite(covariance).all():
            raise RuntimeError("Cluster covariance contains non-finite values")
    return fitted, covariance, design


def slope_name(instrument: str, predictor: str) -> str:
    return f"slope__{instrument.lower()}__{predictor}"


def coefficient_and_variance(
    fitted: object,
    covariance: np.ndarray,
    design: pd.DataFrame,
    weights: dict[str, float],
) -> tuple[float, float]:
    vector = np.zeros(design.shape[1], dtype=float)
    index = {name: position for position, name in enumerate(design.columns)}
    for name, value in weights.items():
        vector[index[name]] = value
    estimate = float(vector @ np.asarray(fitted.params, dtype=float))
    variance = float(vector @ covariance @ vector)
    if variance < -1e-10:
        raise RuntimeError(f"Contrast covariance is negative: {variance}")
    return estimate, max(variance, 0.0)


def ratio_record(
    estimate: float,
    variance: float,
    critical: float = Z_975,
    p_distribution: str = "normal",
    degrees_freedom: int | None = None,
) -> dict:
    standard_error = math.sqrt(variance)
    statistic = estimate / standard_error if standard_error > 0 else np.nan
    if p_distribution == "t" and degrees_freedom is not None:
        p_value = float(2 * stats.t.sf(abs(statistic), degrees_freedom))
    else:
        p_value = float(2 * stats.norm.sf(abs(statistic)))
    low = estimate - critical * standard_error
    high = estimate + critical * standard_error
    return {
        "log_slope": estimate,
        "se": standard_error,
        "z_or_t": statistic,
        "p_value": p_value,
        "ratio_per_doubling_plus_one": math.exp(LOG_TWO * estimate),
        "doubling_ci_low": math.exp(LOG_TWO * low),
        "doubling_ci_high": math.exp(LOG_TWO * high),
    }


def extract_type_slopes(
    fitted: object,
    covariance: np.ndarray,
    design: pd.DataFrame,
    horizon: int,
    n_rows: int,
    n_meetings: int,
    support: str,
) -> pd.DataFrame:
    rows = []
    for instrument in INSTRUMENTS:
        for predictor in PREDICTORS:
            name = slope_name(instrument, predictor)
            estimate, variance = coefficient_and_variance(
                fitted, covariance, design, {name: 1.0}
            )
            rows.append(
                {
                    "instrument": instrument,
                    "predictor": predictor,
                    "horizon_meetings": horizon,
                    "support": support,
                    **ratio_record(estimate, variance),
                    "n_rows": n_rows,
                    "n_meetings": n_meetings,
                    "covariance": "two-way concern and meeting; finite-sample correction",
                }
            )
    return pd.DataFrame(rows)


def pairwise_contrasts(
    fitted: object,
    covariance: np.ndarray,
    design: pd.DataFrame,
    horizon: int,
    n_rows: int,
    n_meetings: int,
) -> pd.DataFrame:
    rows = []
    for left, right in PAIRWISE_TYPES:
        for predictor in PREDICTORS:
            estimate, variance = coefficient_and_variance(
                fitted,
                covariance,
                design,
                {
                    slope_name(left, predictor): 1.0,
                    slope_name(right, predictor): -1.0,
                },
            )
            rows.append(
                {
                    "instrument_left": left,
                    "instrument_right": right,
                    "predictor": predictor,
                    "horizon_meetings": horizon,
                    **ratio_record(estimate, variance),
                    "n_rows": n_rows,
                    "n_meetings": n_meetings,
                }
            )
    table = pd.DataFrame(rows)
    reject, adjusted, _, _ = multipletests(
        table["p_value"].to_numpy(float), alpha=0.05, method="holm"
    )
    table["p_holm_nine_contrasts"] = adjusted
    table["reject_holm_0_05"] = reject
    table["contrast_scale"] = (
        "ratio of output-rate ratios for a doubling of one plus the predictor"
    )
    return table


def omnibus_test(
    fitted: object,
    covariance: np.ndarray,
    design: pd.DataFrame,
    horizon: int,
    n_rows: int,
    n_meetings: int,
) -> pd.DataFrame:
    names = list(design.columns)
    positions = {name: index for index, name in enumerate(names)}
    rows = []
    baseline = "Decision"
    for instrument in ("Measure", "Resolution"):
        for predictor in PREDICTORS:
            vector = np.zeros(len(names), dtype=float)
            vector[positions[slope_name(instrument, predictor)]] = 1.0
            vector[positions[slope_name(baseline, predictor)]] = -1.0
            rows.append(vector)
    restriction = np.vstack(rows)
    beta = np.asarray(fitted.params, dtype=float)
    difference = restriction @ beta
    restricted_covariance = restriction @ covariance @ restriction.T
    restricted_covariance = 0.5 * (
        restricted_covariance + restricted_covariance.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(restricted_covariance)
    minimum_eigenvalue = float(eigenvalues.min())
    common = {
        "test": "all six predictor-by-output-type interactions equal zero",
        "horizon_meetings": horizon,
        "n_restrictions": 6,
        "n_rows": n_rows,
        "n_meetings": n_meetings,
        "restricted_covariance_min_eigenvalue": minimum_eigenvalue,
    }
    rows = []
    if minimum_eigenvalue >= -1e-10:
        rank = int(np.linalg.matrix_rank(restricted_covariance))
        statistic = float(
            difference @ np.linalg.pinv(restricted_covariance) @ difference
        )
        rows.append(
            {
                **common,
                "method": "two-way cluster Wald",
                "status": "valid",
                "wald_chi2": statistic,
                "df_numerator": rank,
                "df_denominator": np.nan,
                "p_value": float(stats.chi2.sf(statistic, rank)),
                "covariance": (
                    "two-way concern and meeting; finite-sample correction"
                ),
            }
        )
    else:
        rows.append(
            {
                **common,
                "method": "two-way cluster Wald",
                "status": "not reported: restricted covariance is not positive semidefinite",
                "wald_chi2": np.nan,
                "df_numerator": 6,
                "df_denominator": np.nan,
                "p_value": np.nan,
                "covariance": (
                    "two-way concern and meeting; finite-sample correction"
                ),
            }
        )
        # Multiway cluster estimates can be indefinite in finite samples.  The
        # eigenvalue-clipped projection is reported only as a transparent
        # sensitivity, not silently substituted for the raw covariance.
        clipped_values = np.maximum(eigenvalues, 0.0)
        projected = (eigenvectors * clipped_values) @ eigenvectors.T
        rank = int(np.linalg.matrix_rank(projected))
        statistic = float(difference @ np.linalg.pinv(projected) @ difference)
        rows.append(
            {
                **common,
                "method": "two-way cluster Wald after zero-clipping negative eigenvalues",
                "status": "PSD-projection sensitivity",
                "wald_chi2": statistic,
                "df_numerator": rank,
                "df_denominator": np.nan,
                "p_value": float(stats.chi2.sf(statistic, rank)),
                "covariance": (
                    "PSD projection of finite-sample-corrected two-way covariance"
                ),
            }
        )
    return pd.DataFrame(rows)


def pooled_results(
    data: pd.DataFrame,
    fitted: object,
    covariance: np.ndarray,
    design: pd.DataFrame,
    horizon: int,
    support: str,
) -> pd.DataFrame:
    rows = []
    for predictor in PREDICTORS:
        name = slope_name("Pooled", predictor)
        estimate, variance = coefficient_and_variance(
            fitted, covariance, design, {name: 1.0}
        )
        rows.append(
            {
                "result": "slope",
                "predictor": predictor,
                "horizon_meetings": horizon,
                "support": support,
                **ratio_record(estimate, variance),
            }
        )
    estimate, variance = coefficient_and_variance(
        fitted,
        covariance,
        design,
        {
            slope_name("Pooled", "focal"): 1.0,
            slope_name("Pooled", "nearby"): -1.0,
        },
    )
    rows.append(
        {
            "result": "direct_contrast",
            "predictor": "focal_minus_nearby",
            "horizon_meetings": horizon,
            "support": support,
            **ratio_record(estimate, variance),
        }
    )
    table = pd.DataFrame(rows)
    table["n_rows"] = len(data)
    table["n_meetings"] = data["meeting"].nunique()
    table["covariance"] = "two-way concern and meeting; finite-sample correction"
    return table


def common_support_horizons(panel: pd.DataFrame) -> pd.DataFrame:
    max_horizon = max(HORIZONS)
    support_meetings = sorted(
        panel.loc[
            panel[f"focal_prior{max_horizon}"].notna(), "meeting"
        ].astype(int).unique()
    )
    support_label = f"common ATCM {min(support_meetings)}-{max(support_meetings)}"
    tables = []
    for horizon in HORIZONS:
        stacked = stack_instruments(panel, horizon, support_meetings)
        fitted, covariance, design = fit_ppml(stacked, stacked=True)
        table = extract_type_slopes(
            fitted,
            covariance,
            design,
            horizon,
            len(stacked),
            stacked["meeting"].nunique(),
            support_label,
        )
        table.insert(0, "model", "stacked_type")
        tables.append(table)

        pooled = pooled_panel(panel, horizon, support_meetings)
        pooled_fit, pooled_covariance, pooled_design = fit_ppml(
            pooled, stacked=False
        )
        pooled_table = pooled_results(
            pooled,
            pooled_fit,
            pooled_covariance,
            pooled_design,
            horizon,
            support_label,
        )
        pooled_table = pooled_table[pooled_table["result"].eq("slope")].copy()
        pooled_table.insert(0, "instrument", "Pooled")
        pooled_table.insert(0, "model", "pooled")
        tables.append(pooled_table)
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined["common_meeting_min"] = min(support_meetings)
    combined["common_meeting_max"] = max(support_meetings)
    combined["common_n_meetings"] = len(support_meetings)
    return combined


def varying_support_horizons(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit pooled histories on every meeting available to each window length."""
    tables = []
    for horizon in HORIZONS:
        pooled = pooled_panel(panel, horizon)
        fitted, covariance, design = fit_ppml(pooled, stacked=False)
        tables.append(
            pooled_results(
                pooled,
                fitted,
                covariance,
                design,
                horizon,
                f"all eligible meetings for {horizon}-meeting window",
            )
        )
    return pd.concat(tables, ignore_index=True)


def title_model_same_support_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    """Reallocate title-model mass over the same 15 official-category concerns."""
    weights = pd.read_csv(TITLE_WEIGHTS_PATH)
    topics = sorted(panel["topic"].unique())
    meetings = sorted(panel["meeting"].astype(int).unique())
    weights = weights[weights["topic"].isin(topics)].copy()
    retained = weights.groupby("outcome_id")["probability"].transform("sum")
    if retained.le(0).any():
        raise ValueError("Every title-model output must retain mass on the 15 concerns")
    weights["probability"] /= retained
    index = pd.MultiIndex.from_product([topics, meetings], names=["topic", "meeting"])
    columns = {
        "outcome_mass": weights.groupby(["topic", "meeting"])["probability"].sum()
    }
    for instrument, column in INSTRUMENT_COLUMNS.items():
        columns[column] = (
            weights[weights["instrument"].eq(instrument)]
            .groupby(["topic", "meeting"])["probability"]
            .sum()
        )
    mass = pd.concat(columns, axis=1).reindex(index, fill_value=0).reset_index()
    replace = ["outcome_mass", *INSTRUMENT_COLUMNS.values()]
    sensitivity = panel.drop(columns=[column for column in replace if column in panel])
    sensitivity = sensitivity.merge(mass, on=["topic", "meeting"], how="left")
    sensitivity[replace] = sensitivity[replace].fillna(0.0)
    sensitivity = add_prior_stocks(sensitivity)
    pooled = pooled_panel(sensitivity, PRIMARY_HORIZON)
    fitted, covariance, design = fit_ppml(pooled, stacked=False)
    return pooled_results(
        pooled,
        fitted,
        covariance,
        design,
        PRIMARY_HORIZON,
        "title-model allocations renormalized over the same 15 concerns",
    )


def meeting_jackknife(
    stacked: pd.DataFrame,
    full_fitted: object,
    full_design: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Leave one ATCM out, refit, and form a meeting-cluster jackknife covariance."""
    names = [
        slope_name(instrument, predictor)
        for instrument in INSTRUMENTS
        for predictor in PREDICTORS
    ]
    full = pd.Series(full_fitted.params, index=full_design.columns).reindex(names)
    meetings = sorted(stacked["meeting"].astype(int).unique())
    draws = []
    for meeting in meetings:
        subset = stacked[~stacked["meeting"].eq(meeting)].reset_index(drop=True)
        fitted, _, design = fit_ppml(subset, stacked=True, clustered=False)
        draws.append(pd.Series(fitted.params, index=design.columns).reindex(names))
    draw_table = pd.DataFrame(draws, index=meetings, columns=names)
    if draw_table.isna().any().any():
        raise RuntimeError("Meeting jackknife lost one or more slope coefficients")
    values = draw_table.to_numpy(float)
    centered = values - values.mean(axis=0, keepdims=True)
    n_clusters = len(meetings)
    covariance = (n_clusters - 1) / n_clusters * centered.T @ centered
    degrees_freedom = n_clusters - 1
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    index = {name: position for position, name in enumerate(names)}
    rows = []
    for instrument in INSTRUMENTS:
        for predictor in PREDICTORS:
            name = slope_name(instrument, predictor)
            position = index[name]
            record = ratio_record(
                float(full[name]),
                float(covariance[position, position]),
                critical=critical,
                p_distribution="t",
                degrees_freedom=degrees_freedom,
            )
            rows.append(
                {
                    "result": "type_slope",
                    "instrument_left": instrument,
                    "instrument_right": "",
                    "predictor": predictor,
                    **record,
                }
            )
    contrast_positions = []
    for left, right in PAIRWISE_TYPES:
        for predictor in PREDICTORS:
            vector = np.zeros(len(names), dtype=float)
            vector[index[slope_name(left, predictor)]] = 1.0
            vector[index[slope_name(right, predictor)]] = -1.0
            estimate = float(vector @ full.to_numpy(float))
            variance = float(vector @ covariance @ vector)
            record = ratio_record(
                estimate,
                max(variance, 0.0),
                critical=critical,
                p_distribution="t",
                degrees_freedom=degrees_freedom,
            )
            contrast_positions.append(len(rows))
            rows.append(
                {
                    "result": "pairwise_contrast",
                    "instrument_left": left,
                    "instrument_right": right,
                    "predictor": predictor,
                    **record,
                }
            )
    table = pd.DataFrame(rows)
    table["p_holm_nine_contrasts"] = np.nan
    table["reject_holm_0_05"] = False
    p_values = table.loc[contrast_positions, "p_value"].to_numpy(float)
    reject, adjusted, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    table.loc[contrast_positions, "p_holm_nine_contrasts"] = adjusted
    table.loc[contrast_positions, "reject_holm_0_05"] = reject
    table["horizon_meetings"] = PRIMARY_HORIZON
    table["n_meeting_clusters"] = n_clusters
    table["degrees_freedom"] = degrees_freedom
    table["method"] = "leave-one-meeting-cluster-out jackknife"

    # Omnibus F sensitivity on the same six differences used by the main test.
    restriction_rows = []
    for instrument in ("Measure", "Resolution"):
        for predictor in PREDICTORS:
            vector = np.zeros(len(names), dtype=float)
            vector[index[slope_name(instrument, predictor)]] = 1.0
            vector[index[slope_name("Decision", predictor)]] = -1.0
            restriction_rows.append(vector)
    restriction = np.vstack(restriction_rows)
    differences = restriction @ full.to_numpy(float)
    restricted_covariance = restriction @ covariance @ restriction.T
    rank = int(np.linalg.matrix_rank(restricted_covariance))
    wald = float(
        differences @ np.linalg.pinv(restricted_covariance) @ differences
    )
    f_statistic = wald / rank
    omnibus = {
        "method": "leave-one-meeting-cluster-out jackknife",
        "wald": wald,
        "f_statistic": f_statistic,
        "df_numerator": rank,
        "df_denominator": degrees_freedom,
        "p_value": float(stats.f.sf(f_statistic, rank, degrees_freedom)),
        "n_meeting_clusters": n_clusters,
        "limitation": (
            "small-meeting-cluster sensitivity only; not a two-dimensional "
            "wild-cluster bootstrap and not a replacement for two-way clustering"
        ),
    }
    return table, omnibus


def validate_against_separate_models(type_slopes: pd.DataFrame) -> dict:
    if not REFERENCE_PATH.exists():
        return {
            "reference_path": str(REFERENCE_PATH.relative_to(ROOT)),
            "reference_found": False,
        }
    reference = pd.read_csv(REFERENCE_PATH)
    reference = reference[
        reference["attention_treatment"].eq("fractional_multilabel")
        & reference["output_coding_treatment"].eq("fractional_multilabel")
        & reference["horizon_meetings"].eq(PRIMARY_HORIZON)
    ].copy()
    predictor_lookup = {
        "papers_prior5": "focal",
        "nearby_prior5": "nearby",
        "measure_prior5": "prior_output",
        "decision_prior5": "prior_output",
        "resolution_prior5": "prior_output",
    }
    reference["predictor_clean"] = reference["predictor"].map(predictor_lookup)
    reference = reference.dropna(subset=["predictor_clean"])
    merged = type_slopes.merge(
        reference[
            ["instrument", "predictor_clean", "ratio_per_doubling_plus_one"]
        ],
        left_on=["instrument", "predictor"],
        right_on=["instrument", "predictor_clean"],
        suffixes=("_stacked", "_separate"),
        validate="one_to_one",
    )
    difference = (
        merged["ratio_per_doubling_plus_one_stacked"]
        - merged["ratio_per_doubling_plus_one_separate"]
    ).abs()
    maximum = float(difference.max())
    tolerance = 1e-4
    if maximum > tolerance:
        raise RuntimeError(
            f"Stacked and separate point estimates differ unexpectedly: {maximum}"
        )
    return {
        "reference_path": str(REFERENCE_PATH.relative_to(ROOT)),
        "reference_found": True,
        "n_slopes_compared": int(len(merged)),
        "maximum_absolute_doubling_ratio_difference": maximum,
        "tolerance": tolerance,
        "passed_tolerance": True,
    }


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel, validation = load_and_validate_panel()

    primary = stack_instruments(panel, PRIMARY_HORIZON)
    fitted, covariance, design = fit_ppml(primary, stacked=True)
    support = (
        f"all eligible ATCMs {int(primary['meeting'].min())}-"
        f"{int(primary['meeting'].max())}"
    )
    type_slopes = extract_type_slopes(
        fitted,
        covariance,
        design,
        PRIMARY_HORIZON,
        len(primary),
        primary["meeting"].nunique(),
        support,
    )
    contrasts = pairwise_contrasts(
        fitted,
        covariance,
        design,
        PRIMARY_HORIZON,
        len(primary),
        primary["meeting"].nunique(),
    )
    omnibus = omnibus_test(
        fitted,
        covariance,
        design,
        PRIMARY_HORIZON,
        len(primary),
        primary["meeting"].nunique(),
    )

    pooled = pooled_panel(panel, PRIMARY_HORIZON)
    pooled_fit, pooled_covariance, pooled_design = fit_ppml(pooled, stacked=False)
    pooled_table = pooled_results(
        pooled,
        pooled_fit,
        pooled_covariance,
        pooled_design,
        PRIMARY_HORIZON,
        f"all eligible ATCMs {int(pooled['meeting'].min())}-{int(pooled['meeting'].max())}",
    )

    horizon_table = common_support_horizons(panel)
    varying_horizon_table = varying_support_horizons(panel)
    title_sensitivity = title_model_same_support_sensitivity(panel)
    small_cluster, small_cluster_omnibus = meeting_jackknife(
        primary, fitted, design
    )
    omnibus = pd.concat(
        [
            omnibus,
            pd.DataFrame(
                [
                    {
                        "test": (
                            "all six predictor-by-output-type interactions equal zero"
                        ),
                        "horizon_meetings": PRIMARY_HORIZON,
                        "n_restrictions": 6,
                        "n_rows": len(primary),
                        "n_meetings": primary["meeting"].nunique(),
                        "restricted_covariance_min_eigenvalue": np.nan,
                        "method": small_cluster_omnibus["method"],
                        "status": "small-meeting-cluster sensitivity",
                        "wald_chi2": small_cluster_omnibus["wald"],
                        "df_numerator": small_cluster_omnibus["df_numerator"],
                        "df_denominator": small_cluster_omnibus["df_denominator"],
                        "p_value": small_cluster_omnibus["p_value"],
                        "covariance": (
                            "leave-one-meeting-cluster-out jackknife covariance"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    reference_validation = validate_against_separate_models(type_slopes)

    type_slopes.to_csv(TYPE_SLOPES_PATH, index=False)
    contrasts.to_csv(CONTRASTS_PATH, index=False)
    omnibus.to_csv(OMNIBUS_PATH, index=False)
    pooled_table.to_csv(POOLED_PATH, index=False)
    horizon_table.to_csv(HORIZONS_PATH, index=False)
    varying_horizon_table.to_csv(VARYING_HORIZONS_PATH, index=False)
    title_sensitivity.to_csv(TITLE_SENSITIVITY_PATH, index=False)
    small_cluster.to_csv(SMALL_CLUSTER_PATH, index=False)

    summary = {
        "analysis": "pooled official-category PPML with exploratory output-type interactions",
        "status": "complete",
        "input_representation": {
            "paper_category_treatment": "fractional_multilabel",
            "output_allocation": "fractional official ATS instrument categories",
        },
        "primary_specification": {
            "horizon_meetings": PRIMARY_HORIZON,
            "predictors": [
                "log1p focal papers",
                "log1p proximity-weighted papers on five nearest concerns",
                "log1p earlier same-type output allocation weight",
            ],
            "fixed_effects": "concern by output type; meeting by output type",
            "covariance": "two-way concern and meeting with finite-sample correction",
            "n_rows": int(len(primary)),
            "n_topics": int(primary["topic"].nunique()),
            "n_meetings": int(primary["meeting"].nunique()),
            "meeting_min": int(primary["meeting"].min()),
            "meeting_max": int(primary["meeting"].max()),
            "converged": bool(fitted.converged),
        },
        "validation": {**validation, "stacked_matches_separate": reference_validation},
        "omnibus": records(omnibus),
        "type_slopes": records(type_slopes),
        "pairwise_contrasts": records(contrasts),
        "pooled": records(pooled_table),
        "common_support": {
            "meeting_min": int(horizon_table["common_meeting_min"].iloc[0]),
            "meeting_max": int(horizon_table["common_meeting_max"].iloc[0]),
            "n_meetings": int(horizon_table["common_n_meetings"].iloc[0]),
        },
        "small_cluster_sensitivity": {
            "omnibus": small_cluster_omnibus,
            "rows_path": str(SMALL_CLUSTER_PATH.relative_to(ROOT)),
        },
        "outputs": {
            "type_slopes": str(TYPE_SLOPES_PATH.relative_to(ROOT)),
            "pairwise_contrasts": str(CONTRASTS_PATH.relative_to(ROOT)),
            "omnibus": str(OMNIBUS_PATH.relative_to(ROOT)),
            "pooled": str(POOLED_PATH.relative_to(ROOT)),
            "common_support_horizons": str(HORIZONS_PATH.relative_to(ROOT)),
            "varying_support_horizons": str(
                VARYING_HORIZONS_PATH.relative_to(ROOT)
            ),
            "title_model_same_support_sensitivity": str(
                TITLE_SENSITIVITY_PATH.relative_to(ROOT)
            ),
            "small_cluster_sensitivity": str(SMALL_CLUSTER_PATH.relative_to(ROOT)),
        },
        "interpretation_boundary": (
            "Associational comparison of official-category adopted-output mass; "
            "not a paper-output lineage, causal mechanism, legal-effect test, or "
            "implementation analysis."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {TYPE_SLOPES_PATH}")
    print(f"Wrote {CONTRASTS_PATH}")
    print(f"Wrote {OMNIBUS_PATH}")
    print(f"Wrote {POOLED_PATH}")
    print(f"Wrote {HORIZONS_PATH}")
    print(f"Wrote {VARYING_HORIZONS_PATH}")
    print(f"Wrote {TITLE_SENSITIVITY_PATH}")
    print(f"Wrote {SMALL_CLUSTER_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
