#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (  # noqa: E402
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

warnings.filterwarnings("ignore")

WINDOW_YEARS = 5
RCA_THRESHOLD = 1.0
MODE_ORDER = ["aggregate", "instantaneous", "cumulative_lagged"]

DATA_PATHS = [
    Path("antarctic-database-go/data/processed/document-summary.parquet"),
    Path("antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"),
    Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
    Path("document-summary.csv"),
]

OUT_CSV = Path("output/hazard_conditional_logit_summary.csv")
OUT_JSON = Path("output/hazard_conditional_logit_meta.json")


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


def phi_from_interaction(interaction: pd.DataFrame, topics_order: list[str]) -> np.ndarray:
    rca = get_rca(interaction)
    phi = compute_product_space(rca).reindex(
        index=topics_order, columns=topics_order, fill_value=0.0
    )
    arr = phi.to_numpy(dtype=float)
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 1.0)
    return arr


def build_conditional_logit_panel() -> tuple[pd.DataFrame, dict[str, int | str]]:
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

    aggregate_phi = phi_from_interaction(
        counts_df.reindex(index=topics, columns=members, fill_value=0), topics
    )

    panel_rows = []
    for mode in MODE_ORDER:
        for t in range(1, len(periods)):
            prev_end = int(periods[t - 1][1])
            period_end = int(periods[t][1])
            prev_active = active_by_period[t - 1]
            curr_active = active_by_period[t]

            if mode == "aggregate":
                phi = aggregate_phi
            elif mode == "instantaneous":
                phi = phi_from_interaction(interaction_by_period[t - 1], topics)
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
                phi = phi_from_interaction(cumulative_interaction, topics)
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
                if int(adopted.sum()) == 0:
                    continue

                prev_indices = np.where(prev_mask)[0]
                max_phi = phi[:, prev_indices].max(axis=1)
                min_raw_distance = 1.0 - max_phi
                group = f"{member}::{period_end}"

                for idx, topic in enumerate(topics):
                    if not at_risk[idx]:
                        continue
                    panel_rows.append(
                        {
                            "mode": mode,
                            "group": group,
                            "member": member,
                            "period_end": period_end,
                            "topic": topic,
                            "adopted": int(adopted[idx]),
                            "distance": float(min_raw_distance[idx]),
                            "topic_popularity": float(prev_topic_popularity.loc[topic]),
                        }
                    )

    panel_df = pd.DataFrame(panel_rows)
    meta = {
        "window_years": WINDOW_YEARS,
        "rca_threshold": RCA_THRESHOLD,
        "distance_definition": "raw_one_minus_max_phi_to_prior_portfolio",
        "group_definition": "member-period with at least one adoption event",
        "year_col": year_col,
        "year_min": year_min,
        "year_max": year_max,
        "n_topics": int(len(topics)),
        "n_members": int(len(members)),
        "n_panel_rows": int(len(panel_df)),
        "n_groups": int(panel_df["group"].nunique()),
    }
    return panel_df, meta


def main() -> None:
    panel_df, meta = build_conditional_logit_panel()
    rows = []
    for mode in MODE_ORDER:
        df = panel_df[panel_df["mode"] == mode].copy()
        model = ConditionalLogit(
            df["adopted"].astype(int),
            df[["distance", "topic_popularity"]],
            groups=df["group"],
        )
        res = model.fit(disp=False, maxiter=200)
        distance_coef = float(res.params["distance"])
        distance_std_err = float(res.bse["distance"])
        distance_ci_low_95 = float(distance_coef - 1.96 * distance_std_err)
        distance_ci_high_95 = float(distance_coef + 1.96 * distance_std_err)
        topic_popularity_coef = float(res.params["topic_popularity"])
        topic_popularity_std_err = float(res.bse["topic_popularity"])
        rows.append(
            {
                "mode": mode,
                "n_rows": int(len(df)),
                "n_groups": int(df["group"].nunique()),
                "distance_coef": distance_coef,
                "distance_std_err": distance_std_err,
                "distance_ci_low_95": distance_ci_low_95,
                "distance_ci_high_95": distance_ci_high_95,
                "distance_odds_ratio_per_unit": float(np.exp(distance_coef)),
                "distance_odds_ratio_per_0_1": float(np.exp(0.1 * distance_coef)),
                "distance_odds_ratio_per_0_1_ci_low_95": float(
                    np.exp(0.1 * distance_ci_low_95)
                ),
                "distance_odds_ratio_per_0_1_ci_high_95": float(
                    np.exp(0.1 * distance_ci_high_95)
                ),
                "topic_popularity_coef": topic_popularity_coef,
                "topic_popularity_std_err": topic_popularity_std_err,
                "log_likelihood": float(res.llf),
            }
        )

    summary_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
