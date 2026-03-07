#!/usr/bin/env python3
"""Estimate dominant-regime transition stability over rolling windows.

Outputs:
- output/fig45_regime_transition_summary_window{W}.csv
- output/fig45_regime_transition_matrix_counts_window{W}.csv
- output/fig45_regime_transition_matrix_row_normalized_window{W}.csv
- output/fig45_regime_member_window_profiles_window{W}.csv

The script reuses the fixed topic->region map from fig45 and computes
member-period dominant regimes from region signal max(RPA - 1, 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

DATA_PATHS = [
    "antarctic-database-go/data/processed/document-summary.parquet",
    "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet",
    "Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv",
    "document-summary.csv",
]


def _load_data_with_fallback(paths: list[str]):
    last_err = None
    for p in paths:
        try:
            return load_data(p), p
        except Exception as exc:  # pragma: no cover
            last_err = exc
    raise RuntimeError("Failed to load ATS data from fallback paths") from last_err


def _sanitize_years(df_in: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=[year_col]).copy()
    out[year_col] = out[year_col].astype(int)
    return out


def _build_window_interaction(
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
    inter = generate_interaction_matrix(window_df, all_members_raw, all_topics_raw)
    inter = standardize_index_labels(inter)
    if inter.index.has_duplicates:
        inter = inter.groupby(level=0).sum()
    return inter.reindex(index=topics_order, columns=members_order, fill_value=0)


def _member_period_profiles(
    submitted_df: pd.DataFrame,
    year_col: str,
    topics: list[str],
    members: list[str],
    topic_regions_zero_based: np.ndarray,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    window_years: int,
    rca_threshold: float,
    min_active_topics: int,
) -> pd.DataFrame:
    year_min = int(submitted_df[year_col].min())
    year_max = int(submitted_df[year_col].max())
    records: list[dict] = []

    for end in range(year_min + window_years - 1, year_max + 1):
        start = end - window_years + 1
        inter = _build_window_interaction(
            submitted_df=submitted_df,
            year_col=year_col,
            year_start=start,
            year_end=end,
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics,
            members_order=members,
        )
        rca = get_rca(inter).reindex(index=topics, columns=members, fill_value=0.0)
        arr = rca.to_numpy(dtype=float)
        active = arr > float(rca_threshold)
        region_signal = np.clip(arr - float(rca_threshold), 0.0, None)

        for col_idx, member in enumerate(members):
            k_active = int(active[:, col_idx].sum())
            if k_active < int(min_active_topics):
                continue
            region_weights = np.bincount(
                topic_regions_zero_based,
                weights=region_signal[:, col_idx],
                minlength=3,
            ).astype(float)
            total = float(region_weights.sum())
            if total <= 0:
                continue
            shares = region_weights / total
            dom_idx = int(np.argmax(shares))
            records.append(
                {
                    "period_end": int(end),
                    "period_start": int(start),
                    "member": member,
                    "k_active": int(k_active),
                    "dominant_region": int(dom_idx + 1),
                    "dominant_region_share": float(shares[dom_idx]),
                    "region_1_share": float(shares[0]),
                    "region_2_share": float(shares[1]),
                    "region_3_share": float(shares[2]),
                }
            )

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["member", "period_end"]).reset_index(drop=True)
    return out


def _build_transitions(profile_df: pd.DataFrame) -> pd.DataFrame:
    trans: list[dict] = []
    for member, g in profile_df.groupby("member", sort=False):
        g = g.sort_values("period_end")
        d_reg = dict(zip(g["period_end"], g["dominant_region"]))
        d_share = dict(zip(g["period_end"], g["dominant_region_share"]))
        years = sorted(d_reg)
        for t in years:
            t_next = t + 1
            if t_next not in d_reg:
                continue
            r_from = int(d_reg[t])
            r_to = int(d_reg[t_next])
            trans.append(
                {
                    "member": member,
                    "period_end": int(t),
                    "period_end_next": int(t_next),
                    "from_region": r_from,
                    "to_region": r_to,
                    "same_region": int(r_from == r_to),
                    "adjacent_or_same": int(abs(r_to - r_from) <= 1),
                    "far_jump_1_to_3": int(abs(r_to - r_from) == 2),
                    "from_dominant_share": float(d_share[t]),
                }
            )
    out = pd.DataFrame(trans)
    if not out.empty:
        out = out.sort_values(["member", "period_end"]).reset_index(drop=True)
    return out


def run(window_years: int, rca_threshold: float, min_active_topics: int) -> None:
    (counts_df, submitted_df, members_raw, topics_raw), data_path = (
        _load_data_with_fallback(DATA_PATHS)
    )
    year_col = "meeting_year" if "meeting_year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns:
        raise KeyError("No meeting_year or year column found in source data.")
    submitted_df = _sanitize_years(submitted_df, year_col)

    metacluster_df = pd.read_csv("output/fig45_topic_metaclusters.csv")
    topic_region_df = (
        metacluster_df[["topic", "region_id"]]
        .dropna()
        .drop_duplicates(subset=["topic"])
        .copy()
    )
    topics = topic_region_df["topic"].astype(str).tolist()
    topic_regions = (topic_region_df["region_id"].astype(int) - 1).to_numpy(dtype=int)

    members = counts_df.columns.astype(str).tolist()
    all_members_raw = set(members_raw)
    all_topics_raw = set(topics_raw)

    profiles = _member_period_profiles(
        submitted_df=submitted_df,
        year_col=year_col,
        topics=topics,
        members=members,
        topic_regions_zero_based=topic_regions,
        all_members_raw=all_members_raw,
        all_topics_raw=all_topics_raw,
        window_years=window_years,
        rca_threshold=rca_threshold,
        min_active_topics=min_active_topics,
    )
    transitions = _build_transitions(profiles)

    if transitions.empty:
        raise RuntimeError("No regime transitions available with current settings.")

    mat_counts = pd.crosstab(transitions["from_region"], transitions["to_region"])
    mat_norm = mat_counts.div(mat_counts.sum(axis=1), axis=0)
    summary = pd.DataFrame(
        [
            {
                "window_years": int(window_years),
                "rca_threshold": float(rca_threshold),
                "min_active_topics": int(min_active_topics),
                "data_path": data_path,
                "n_member_period_rows": int(len(profiles)),
                "n_members": int(profiles["member"].nunique()),
                "n_periods": int(profiles["period_end"].nunique()),
                "n_transitions": int(len(transitions)),
                "same_region_rate": float(transitions["same_region"].mean()),
                "adjacent_or_same_rate": float(transitions["adjacent_or_same"].mean()),
                "far_jump_rate": float(transitions["far_jump_1_to_3"].mean()),
                "mean_from_dominant_share": float(
                    transitions["from_dominant_share"].mean()
                ),
            }
        ]
    )

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"window{int(window_years)}"
    profiles.to_csv(
        out_dir / f"fig45_regime_member_window_profiles_{suffix}.csv", index=False
    )
    transitions.to_csv(
        out_dir / f"fig45_regime_member_window_transitions_{suffix}.csv", index=False
    )
    summary.to_csv(
        out_dir / f"fig45_regime_transition_summary_{suffix}.csv", index=False
    )
    mat_counts.to_csv(
        out_dir / f"fig45_regime_transition_matrix_counts_{suffix}.csv", index=True
    )
    mat_norm.to_csv(
        out_dir / f"fig45_regime_transition_matrix_row_normalized_{suffix}.csv",
        index=True,
    )

    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dominant-regime transition stability."
    )
    parser.add_argument(
        "--window-years",
        type=int,
        default=5,
        help="Rolling window length in years (default: 5).",
    )
    parser.add_argument(
        "--rca-threshold",
        type=float,
        default=1.0,
        help="RCA threshold for active/signal computation (default: 1.0).",
    )
    parser.add_argument(
        "--min-active-topics",
        type=int,
        default=3,
        help="Minimum active topics required to include a member-period (default: 3).",
    )
    args = parser.parse_args()
    run(
        window_years=int(args.window_years),
        rca_threshold=float(args.rca_threshold),
        min_active_topics=int(args.min_active_topics),
    )


if __name__ == "__main__":
    main()
