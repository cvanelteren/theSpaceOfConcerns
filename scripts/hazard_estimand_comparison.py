#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Make repo-root imports work when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

WINDOW_YEARS = 5
RCA_THRESHOLD = 1.0

DATA_PATHS = [
    Path("antarctic-database-go/data/processed/document-summary.parquet"),
    Path("antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"),
    Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
    Path("document-summary.csv"),
]

MODE_ORDER = ["aggregate", "instantaneous", "cumulative_lagged"]

OUT_CSV = Path("output/hazard_estimand_comparison.csv")
OUT_JSON = Path("output/hazard_estimand_comparison_meta.json")


def load_data_with_fallback():
    last_err = None
    for path in DATA_PATHS:
        if not path.exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_err = exc
    if last_err is not None:
        raise RuntimeError("Failed to load ATS data from fallback paths.") from last_err
    raise FileNotFoundError("No known ATS data path exists.")


def sanitize_years(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=[year_col]).copy()
    out[year_col] = out[year_col].astype(int)
    return out


def build_periods(year_min: int, year_max: int, window: int) -> list[tuple[int, int]]:
    return [(y - window + 1, y) for y in range(year_min + window - 1, year_max + 1)]


def build_window_interaction(
    submitted_df: pd.DataFrame,
    year_col: str,
    year_start: int,
    year_end: int,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    members_order: list[str],
) -> pd.DataFrame:
    window_df = submitted_df[
        (submitted_df[year_col] >= int(year_start))
        & (submitted_df[year_col] <= int(year_end))
    ]
    interaction = generate_interaction_matrix(
        window_df, all_members_raw, all_topics_raw
    )
    interaction = standardize_index_labels(interaction)
    if interaction.index.has_duplicates:
        interaction = interaction.groupby(level=0).sum()
    return interaction.reindex(index=topics_order, columns=members_order, fill_value=0)


def distance_from_interaction(
    interaction: pd.DataFrame, topics_order: list[str]
) -> np.ndarray:
    # Use the legacy 5-year hazard definition: raw distance = 1 - phi.
    rca = get_rca(interaction)
    phi = compute_product_space(rca).reindex(
        index=topics_order, columns=topics_order, fill_value=0.0
    )
    dist = 1.0 - phi.to_numpy(dtype=float)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def fit_member_period_logit(panel_df: pd.DataFrame) -> dict[str, float] | None:
    # prev_diversity is constant within a member-period panel, so it is not
    # separately identified in this estimand.
    features = ["distance", "topic_popularity"]
    df = panel_df.dropna(subset=features + ["adopted"]).copy()
    if df.empty:
        return None

    n_adopted = int(df["adopted"].sum())
    if n_adopted == 0 or n_adopted == len(df):
        return None

    X = df[features].to_numpy()
    y = df["adopted"].to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=4000, solver="lbfgs")
    model.fit(X_std, y)

    coef = model.coef_[0].astype(float)
    return {
        "distance_coef": float(coef[0]),
        "topic_popularity_coef": float(coef[1]),
        "intercept": float(model.intercept_[0]),
        "n_obs": int(len(df)),
        "n_adopted": int(n_adopted),
    }


def fit_pooled_glm(panel_df: pd.DataFrame, *, standardize: bool) -> pd.DataFrame:
    features = ["distance", "prev_diversity", "topic_popularity"]
    df = panel_df.dropna(subset=features + ["adopted", "member"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["term", "coef", "std_err"])
    if int(df["adopted"].sum()) == 0 or int(df["adopted"].sum()) == len(df):
        return pd.DataFrame(columns=["term", "coef", "std_err"])

    X = df[features].copy()
    if standardize:
        X = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    X = sm.add_constant(X, has_constant="add")
    y = df["adopted"].astype(float).to_numpy()
    groups = df["member"].astype(str).to_numpy()

    glm = sm.GLM(y, X, family=sm.families.Binomial())
    try:
        res = glm.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        res = glm.fit()

    return pd.DataFrame(
        {
            "term": list(res.params.index),
            "coef": np.asarray(res.params.values, dtype=float),
            "std_err": np.asarray(res.bse, dtype=float),
        }
    )


def extract_term(df: pd.DataFrame, term: str) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    row = df[df["term"] == term]
    if row.empty:
        return float("nan"), float("nan")
    return float(row["coef"].iloc[0]), float(row["std_err"].iloc[0])


def main() -> None:
    counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback()
    year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns and "meeting_year" in submitted_df.columns:
        year_col = "meeting_year"
    if year_col not in submitted_df.columns:
        raise KeyError("No meeting year or year column found in source data.")
    submitted_df = sanitize_years(submitted_df, year_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    all_members_raw = set(members_raw)
    all_topics_raw = set(topics_raw)

    year_min = int(submitted_df[year_col].min())
    year_max = int(submitted_df[year_col].max())
    periods = build_periods(year_min, year_max, WINDOW_YEARS)

    interaction_by_period: list[pd.DataFrame] = []
    active_by_period: list[pd.DataFrame] = []
    for start, end in periods:
        interaction = build_window_interaction(
            submitted_df=submitted_df,
            year_col=year_col,
            year_start=int(start),
            year_end=int(end),
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics,
            members_order=members,
        )
        interaction_by_period.append(interaction)
        active_by_period.append((get_rca(interaction) >= RCA_THRESHOLD))

    aggregate_interaction = counts_df.reindex(
        index=topics, columns=members, fill_value=0
    )
    aggregate_dist = distance_from_interaction(aggregate_interaction, topics)

    result_rows = []
    panel_rows = []
    for mode in MODE_ORDER:
        for t in range(1, len(periods)):
            period_end = int(periods[t][1])
            prev_end = int(periods[t - 1][1])
            prev_active = active_by_period[t - 1]
            curr_active = active_by_period[t]

            if mode == "aggregate":
                dist = aggregate_dist
            elif mode == "instantaneous":
                dist = distance_from_interaction(interaction_by_period[t - 1], topics)
            elif mode == "cumulative_lagged":
                cumulative_interaction = build_window_interaction(
                    submitted_df=submitted_df,
                    year_col=year_col,
                    year_start=year_min,
                    year_end=prev_end,
                    all_members_raw=all_members_raw,
                    all_topics_raw=all_topics_raw,
                    topics_order=topics,
                    members_order=members,
                )
                dist = distance_from_interaction(cumulative_interaction, topics)
            else:  # pragma: no cover
                raise ValueError(f"Unknown mode: {mode}")

            prev_topic_popularity = prev_active.sum(axis=1) / max(len(members), 1)

            for member in members:
                prev_mask = prev_active[member].to_numpy(dtype=bool)
                curr_mask = curr_active[member].to_numpy(dtype=bool)
                if not prev_mask.any():
                    continue

                at_risk = ~prev_mask
                if not at_risk.any():
                    continue
                adopted = curr_mask & at_risk

                prev_indices = np.where(prev_mask)[0]
                min_dist = dist[:, prev_indices].min(axis=1)
                prev_diversity = float(len(prev_indices))

                rows = []
                for idx, topic in enumerate(topics):
                    if not at_risk[idx]:
                        continue
                    rows.append(
                        {
                            "mode": mode,
                            "period_end": period_end,
                            "member": member,
                            "topic": topic,
                            "adopted": int(adopted[idx]),
                            "distance": float(min_dist[idx]),
                            "prev_diversity": prev_diversity,
                            "topic_popularity": float(prev_topic_popularity.loc[topic]),
                        }
                    )

                if not rows:
                    continue
                member_period_panel = pd.DataFrame(rows)
                panel_rows.append(member_period_panel)

                fit = fit_member_period_logit(member_period_panel)
                if fit is None:
                    continue
                result_rows.append(
                    {"mode": mode, "period_end": period_end, "member": member, **fit}
                )

    results_df = pd.DataFrame(result_rows)
    if results_df.empty:
        raise RuntimeError("No valid member-period hazard models were fitted.")
    panel_df = pd.concat(panel_rows, ignore_index=True)

    period_summary = (
        results_df.groupby(["mode", "period_end"])
        .agg(
            n_models=("distance_coef", "size"),
            median_distance_coef=("distance_coef", "median"),
        )
        .reset_index()
    )

    comparison_rows = []
    for mode in MODE_ORDER:
        mode_panel = panel_df[panel_df["mode"] == mode].copy()
        mode_results = results_df[results_df["mode"] == mode].copy()
        mode_periods = period_summary[period_summary["mode"] == mode].copy()

        pooled_raw = fit_pooled_glm(mode_panel, standardize=False)
        pooled_std = fit_pooled_glm(mode_panel, standardize=True)
        pooled_raw_coef, pooled_raw_se = extract_term(pooled_raw, "distance")
        pooled_std_coef, pooled_std_se = extract_term(pooled_std, "distance")

        comparison_rows.append(
            {
                "mode": mode,
                "pooled_raw_distance_coef_with_controls": pooled_raw_coef,
                "pooled_raw_distance_std_err": pooled_raw_se,
                "pooled_standardized_distance_coef_with_controls": pooled_std_coef,
                "pooled_standardized_distance_std_err": pooled_std_se,
                "member_window_global_median_distance_coef": float(
                    mode_results["distance_coef"].median()
                ),
                "period_median_of_medians_distance_coef": float(
                    mode_periods["median_distance_coef"].median()
                ),
                "n_member_window_models": int(len(mode_results)),
                "n_panel_rows": int(len(mode_panel)),
                "n_periods": int(mode_periods["period_end"].nunique()),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUT_CSV, index=False)

    meta = {
        "window_years": WINDOW_YEARS,
        "rca_threshold": RCA_THRESHOLD,
        "distance_definition": "legacy_raw_one_minus_phi",
        "member_window_estimator": (
            "sklearn logistic regression on standardized distance and topic_popularity"
        ),
        "pooled_estimators": [
            "binomial GLM raw distance with controls",
            "binomial GLM standardized distance with controls",
        ],
        "data_paths": [str(path) for path in DATA_PATHS],
        "year_col": year_col,
        "n_topics": int(len(topics)),
        "n_members": int(len(members)),
        "year_min": year_min,
        "year_max": year_max,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(comparison_df.to_string(index=False))
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
