#!/usr/bin/env python3
"""Estimate dominant-regime transition stability over rolling windows.

Outputs:
- output/fig45_regime_transition_summary_{TIME_UNIT}_window{W}.csv
- output/fig45_regime_transition_matrix_counts_{TIME_UNIT}_window{W}.csv
- output/fig45_regime_transition_matrix_row_normalized_{TIME_UNIT}_window{W}.csv
- output/fig45_regime_member_window_profiles_{TIME_UNIT}_window{W}.csv

For `time_unit=year`, the legacy `..._window{W}.csv` filenames are also written
for backwards compatibility.

The script reuses the fixed topic->region map from fig45 and computes
member-period dominant regimes from region signal max(RPA - 1, 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
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


def _sanitize_int_column(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col]).copy()
    out[col] = out[col].astype(int)
    return out


def _build_periods_for_unit(
    submitted_df: pd.DataFrame,
    *,
    year_col: str,
    meeting_col: str | None,
    time_unit: str,
    window_size: int,
) -> list[tuple[int, int]]:
    time_unit = str(time_unit).strip().lower()
    window_size = max(1, int(window_size))

    if time_unit == "year":
        year_min = int(submitted_df[year_col].min())
        year_max = int(submitted_df[year_col].max())
        return [
            (end - window_size + 1, end)
            for end in range(year_min + window_size - 1, year_max + 1)
        ]

    if time_unit == "meeting":
        if meeting_col is None or meeting_col not in submitted_df.columns:
            raise KeyError(
                "No meeting-number column found in source data for time_unit='meeting'."
            )
        values = (
            pd.to_numeric(submitted_df[meeting_col], errors="coerce")
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        return [
            (int(values[idx - window_size + 1]), int(values[idx]))
            for idx in range(window_size - 1, len(values))
        ]

    raise ValueError(f"Unknown time_unit={time_unit!r}. Use meeting|year.")


def _build_window_interaction(
    submitted_df: pd.DataFrame,
    time_col: str,
    window_start: int,
    window_end: int,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    members_order: list[str],
) -> pd.DataFrame:
    window_df = submitted_df[
        (submitted_df[time_col] >= int(window_start))
        & (submitted_df[time_col] <= int(window_end))
    ]
    inter = generate_interaction_matrix(window_df, all_members_raw, all_topics_raw)
    inter = standardize_index_labels(inter)
    if inter.index.has_duplicates:
        inter = inter.groupby(level=0).sum()
    return inter.reindex(index=topics_order, columns=members_order, fill_value=0)


def _member_period_profiles(
    submitted_df: pd.DataFrame,
    time_col: str,
    periods: list[tuple[int, int]],
    topics: list[str],
    members: list[str],
    topic_regions_zero_based: np.ndarray,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    time_unit: str,
    window_size: int,
    rca_threshold: float,
    min_active_topics: int,
) -> pd.DataFrame:
    records: list[dict] = []

    for period_index, (start, end) in enumerate(periods):
        inter = _build_window_interaction(
            submitted_df=submitted_df,
            time_col=time_col,
            window_start=start,
            window_end=end,
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
                    "time_unit": str(time_unit),
                    "window_size": int(window_size),
                    "period_index": int(period_index),
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
        g = g.sort_values("period_index").reset_index(drop=True)
        period_indices = g["period_index"].to_numpy(dtype=int)
        for idx in range(len(g) - 1):
            if period_indices[idx + 1] != period_indices[idx] + 1:
                continue
            row = g.iloc[idx]
            row_next = g.iloc[idx + 1]
            r_from = int(row["dominant_region"])
            r_to = int(row_next["dominant_region"])
            trans.append(
                {
                    "member": member,
                    "time_unit": str(row["time_unit"]),
                    "window_size": int(row["window_size"]),
                    "period_index": int(row["period_index"]),
                    "period_index_next": int(row_next["period_index"]),
                    "period_start": int(row["period_start"]),
                    "period_end": int(row["period_end"]),
                    "period_start_next": int(row_next["period_start"]),
                    "period_end_next": int(row_next["period_end"]),
                    "from_region": r_from,
                    "to_region": r_to,
                    "same_region": int(r_from == r_to),
                    "adjacent_or_same": int(abs(r_to - r_from) <= 1),
                    "far_jump_1_to_3": int(abs(r_to - r_from) == 2),
                    "from_dominant_share": float(row["dominant_region_share"]),
                }
            )
    out = pd.DataFrame(trans)
    if not out.empty:
        out = out.sort_values(["member", "period_end"]).reset_index(drop=True)
    return out


def _output_suffix(time_unit: str, window_size: int) -> str:
    return f"{str(time_unit).strip().lower()}_window{int(window_size)}"


def _write_output_variants(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str,
    time_unit: str,
    window_size: int,
    index: bool,
) -> None:
    suffixes = [_output_suffix(time_unit, window_size)]
    if str(time_unit).strip().lower() == "year":
        suffixes.append(f"window{int(window_size)}")
    for suffix in suffixes:
        df.to_csv(out_dir / f"{stem}_{suffix}.csv", index=index)


def run(
    *,
    time_unit: str,
    window_size: int,
    rca_threshold: float,
    min_active_topics: int,
) -> None:
    (counts_df, submitted_df, members_raw, topics_raw), data_path = (
        _load_data_with_fallback(DATA_PATHS)
    )
    year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns:
        raise KeyError("No meeting year or year column found in source data.")
    submitted_df = _sanitize_years(submitted_df, year_col)
    meeting_col = (
        "meeting number" if "meeting number" in submitted_df.columns else "meeting_number"
    )
    if meeting_col in submitted_df.columns:
        submitted_df = _sanitize_int_column(submitted_df, meeting_col)
    else:
        meeting_col = None

    time_unit = str(time_unit).strip().lower()
    periods = _build_periods_for_unit(
        submitted_df=submitted_df,
        year_col=year_col,
        meeting_col=meeting_col,
        time_unit=time_unit,
        window_size=window_size,
    )
    time_col = meeting_col if time_unit == "meeting" else year_col

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
        time_col=time_col,
        periods=periods,
        topics=topics,
        members=members,
        topic_regions_zero_based=topic_regions,
        all_members_raw=all_members_raw,
        all_topics_raw=all_topics_raw,
        time_unit=time_unit,
        window_size=window_size,
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
                "time_unit": str(time_unit),
                "window_size": int(window_size),
                "window_years": int(window_size),
                "rca_threshold": float(rca_threshold),
                "min_active_topics": int(min_active_topics),
                "data_path": data_path,
                "n_member_period_rows": int(len(profiles)),
                "n_members": int(profiles["member"].nunique()),
                "n_periods": int(profiles["period_index"].nunique()),
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
    _write_output_variants(
        profiles,
        out_dir=out_dir,
        stem="fig45_regime_member_window_profiles",
        time_unit=time_unit,
        window_size=window_size,
        index=False,
    )
    _write_output_variants(
        transitions,
        out_dir=out_dir,
        stem="fig45_regime_member_window_transitions",
        time_unit=time_unit,
        window_size=window_size,
        index=False,
    )
    _write_output_variants(
        summary,
        out_dir=out_dir,
        stem="fig45_regime_transition_summary",
        time_unit=time_unit,
        window_size=window_size,
        index=False,
    )
    _write_output_variants(
        mat_counts,
        out_dir=out_dir,
        stem="fig45_regime_transition_matrix_counts",
        time_unit=time_unit,
        window_size=window_size,
        index=True,
    )
    _write_output_variants(
        mat_norm,
        out_dir=out_dir,
        stem="fig45_regime_transition_matrix_row_normalized",
        time_unit=time_unit,
        window_size=window_size,
        index=True,
    )

    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dominant-regime transition stability."
    )
    parser.add_argument(
        "--time-unit",
        choices=["meeting", "year"],
        default="year",
        help="Rolling window unit to use (default: year).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Rolling window length in the selected time unit (default: 5).",
    )
    parser.add_argument(
        "--window-years",
        type=int,
        dest="window_size",
        help="Backward-compatible alias for --window-size.",
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
        time_unit=str(args.time_unit),
        window_size=int(args.window_size),
        rca_threshold=float(args.rca_threshold),
        min_active_topics=int(args.min_active_topics),
    )


if __name__ == "__main__":
    main()
