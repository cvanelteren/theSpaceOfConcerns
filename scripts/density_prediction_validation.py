#!/usr/bin/env python3
"""Prospective validation that density helps rank an actor's next concern.

Builds transitions between disjoint 5-year windows, using a cumulative-lagged
space constructed only from documents available through the preceding window.
The candidate set contains only concerns already observed by that point.  The
panel therefore evaluates which established concern is selected, conditional
on an actor expanding its portfolio; it does not predict whether expansion
occurs or the first appearance of a new archive label.

- ``max_phi``: strongest proximity to the prior portfolio (1 - hazard distance)
- ``density``: Hidalgo-style proximity-weighted portfolio share,
  sum_j phi(i,j) * held_j / sum_j phi(i,j), diagonal zeroed
- ``popularity``: share of actors specialized in the topic in the prior window

and evaluates within-choice-set prediction of actual new adoptions against
those scores and a uniform-random baseline.

Outputs:
- output/density_prediction_validation_summary.csv
- output/density_prediction_validation_choice_sets.csv
- output/density_prediction_validation_meta.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hazard_conditional_logit import (  # noqa: E402
    WINDOW_YEARS,
    build_window_interaction,
    load_data_with_fallback,
    phi_from_interaction,
    sanitize_years,
)
from utils import get_rca  # noqa: E402

warnings.filterwarnings("ignore")

RCA_THRESHOLD = 1.0
N_TIEBREAK_SEEDS = 100
ERA_SPLIT_YEAR = 1990

OUT_SUMMARY = Path("output/density_prediction_validation_summary.csv")
OUT_CHOICE_SETS = Path("output/density_prediction_validation_choice_sets.csv")
OUT_META = Path("output/density_prediction_validation_meta.json")
OUT_PANEL = Path("output/density_prediction_panel.parquet")

SCORES = ["density", "max_phi", "popularity"]


def build_disjoint_periods(
    year_min: int, year_max: int, window: int
) -> list[tuple[int, int]]:
    """Return consecutive, non-overlapping windows, retaining complete blocks."""
    periods = []
    start = year_min
    while start + window - 1 <= year_max:
        periods.append((start, start + window - 1))
        start += window
    return periods


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def recall_at_k(
    scores: np.ndarray, labels: np.ndarray, k: int, n_seeds: int
) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0 or k <= 0:
        return np.nan
    k = min(k, len(scores))
    hits = np.zeros(n_seeds)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        tiebreak = rng.random(len(scores))
        order = np.lexsort((tiebreak, -scores))
        hits[seed] = labels[order[:k]].sum()
    return float(hits.mean() / n_pos)


def top_decile_adoption_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    n = len(scores)
    if n < 10:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    k = max(1, n // 10)
    return float(labels[order[:k]].mean())


def build_prediction_panel() -> tuple[pd.DataFrame, dict]:
    counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback()
    year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns and "meeting_year" in submitted_df.columns:
        year_col = "meeting_year"
    submitted_df = sanitize_years(submitted_df, year_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    all_members_raw = set(members_raw)
    all_topics_raw = set(topics_raw)
    n_topics = len(topics)

    year_min = int(submitted_df[year_col].min())
    year_max = int(submitted_df[year_col].max())
    periods = build_disjoint_periods(year_min, year_max, WINDOW_YEARS)
    first_appearance = (
        submitted_df.groupby("category")[year_col].min().reindex(topics).astype(int)
    )

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

    rows = []
    for t in range(1, len(periods)):
        prev_end = int(periods[t - 1][1])
        period_end = int(periods[t][1])
        prev_active = active_by_period[t - 1]
        curr_active = active_by_period[t]

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
        phi_density = phi.copy()
        np.fill_diagonal(phi_density, 0.0)
        phi_row_sums = phi_density.sum(axis=1)

        n_prior_submitters = int(
            (interaction_by_period[t - 1].sum(axis=0) > 0).sum()
        )
        prev_topic_popularity = prev_active.sum(axis=1) / max(n_prior_submitters, 1)
        historically_available = first_appearance.to_numpy() <= prev_end

        for member in members:
            prev_mask = prev_active[member].to_numpy(dtype=bool)
            curr_mask = curr_active[member].to_numpy(dtype=bool)
            if not prev_mask.any():
                continue
            at_risk = (~prev_mask) & historically_available
            if not at_risk.any():
                continue
            adopted = curr_mask & at_risk
            if int(adopted.sum()) == 0:
                continue

            prev_indices = np.where(prev_mask)[0]
            max_phi = phi[:, prev_indices].max(axis=1)
            density_num = phi_density[:, prev_indices].sum(axis=1)
            density = np.where(
                phi_row_sums > 0, density_num / np.maximum(phi_row_sums, 1e-12), 0.0
            )

            for idx in range(n_topics):
                if not at_risk[idx]:
                    continue
                rows.append(
                    {
                        "group": f"{member}::{period_end}",
                        "member": member,
                        "period_end": period_end,
                        "topic": topics[idx],
                        "adopted": int(adopted[idx]),
                        "max_phi": float(max_phi[idx]),
                        "density": float(density[idx]),
                        "popularity": float(prev_topic_popularity.iloc[idx]),
                    }
                )

    panel_df = pd.DataFrame(rows)
    meta = {
        "window_years": WINDOW_YEARS,
        "windows": "disjoint_complete_blocks",
        "rca_threshold": RCA_THRESHOLD,
        "space": "cumulative_lagged_prior_information_only",
        "candidate_rule": "first archive appearance on or before prior-window end",
        "prediction_target": (
            "which established concern is adopted, conditional on portfolio expansion"
        ),
        "density_definition": "sum_j phi_ij held_j / sum_j phi_ij, diagonal zeroed",
        "max_phi_definition": "max proximity to any held topic (1 - hazard distance)",
        "group_definition": "member-transition with at least one eligible adoption event",
        "year_min": year_min,
        "year_max": year_max,
        "n_topics": n_topics,
        "n_members": len(members),
        "n_panel_rows": int(len(panel_df)),
        "n_groups": int(panel_df["group"].nunique()),
        "n_adopted_rows": int(panel_df["adopted"].sum()),
        "era_split_year": ERA_SPLIT_YEAR,
        "n_tiebreak_seeds": N_TIEBREAK_SEEDS,
    }
    return panel_df, meta


def evaluate(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_rows = []
    for group, gdf in panel_df.groupby("group", sort=False):
        scores = {s: gdf[s].to_numpy(dtype=float) for s in SCORES}
        labels = gdf["adopted"].to_numpy(dtype=int)
        n_at_risk = int(len(gdf))
        n_adopted = int(labels.sum())
        row = {
            "group": group,
            "member": gdf["member"].iloc[0],
            "period_end": int(gdf["period_end"].iloc[0]),
            "n_at_risk": n_at_risk,
            "n_adopted": n_adopted,
            "random_recall_at_k": n_adopted / n_at_risk,
        }
        for s in SCORES:
            row[f"auc_{s}"] = auc_from_scores(scores[s], labels)
            row[f"recall_at_k_{s}"] = recall_at_k(
                scores[s], labels, n_adopted, N_TIEBREAK_SEEDS
            )
            row[f"top_decile_rate_{s}"] = top_decile_adoption_rate(scores[s], labels)
        fused = 0.5 * stats.rankdata(
            scores["density"], method="average"
        ) + 0.5 * stats.rankdata(scores["popularity"], method="average")
        row["auc_density_plus_popularity"] = auc_from_scores(fused, labels)
        row["recall_at_k_density_plus_popularity"] = recall_at_k(
            fused, labels, n_adopted, N_TIEBREAK_SEEDS
        )
        row["top_decile_rate_density_plus_popularity"] = top_decile_adoption_rate(
            fused, labels
        )
        set_rows.append(row)

    sets_df = pd.DataFrame(set_rows)

    summary_rows = []
    base_rate = float(panel_df["adopted"].mean())

    def add(metric, score, value, **extra):
        summary_rows.append({"metric": metric, "score": score, "value": value, **extra})

    for s in SCORES + ["density_plus_popularity"]:
        add("mean_auc", s, float(sets_df[f"auc_{s}"].mean()))
        add(
            "mean_auc_era1",
            s,
            float(sets_df.loc[sets_df["period_end"] <= ERA_SPLIT_YEAR, f"auc_{s}"].mean()),
        )
        add(
            "mean_auc_era2",
            s,
            float(sets_df.loc[sets_df["period_end"] > ERA_SPLIT_YEAR, f"auc_{s}"].mean()),
        )
        add("mean_recall_at_k", s, float(sets_df[f"recall_at_k_{s}"].mean()))
        add(
            "mean_top_decile_rate",
            s,
            float(sets_df[f"top_decile_rate_{s}"].mean()),
        )

    add("mean_recall_at_k", "random", float(sets_df["random_recall_at_k"].mean()))
    add("base_adoption_rate", "none", base_rate)

    pooled = panel_df.groupby("group").apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "adopted": g["adopted"].sum(),
            }
        ),
        include_groups=False,
    )
    micro_auc = {}
    for s in SCORES:
        num = 0.0
        den = 0.0
        for _, gdf in panel_df.groupby("group", sort=False):
            a = auc_from_scores(gdf[s].to_numpy(float), gdf["adopted"].to_numpy(int))
            if np.isnan(a):
                continue
            n_pos = int(gdf["adopted"].sum())
            n_neg = int(len(gdf) - n_pos)
            num += a * n_pos * n_neg
            den += n_pos * n_neg
        micro_auc[s] = num / den if den else np.nan
        add("micro_auc", s, float(micro_auc[s]))

    w_dens = sets_df["auc_density"]
    w_pop = sets_df["auc_popularity"]
    w_maxphi = sets_df["auc_max_phi"]
    w_fused = sets_df["auc_density_plus_popularity"]
    valid = ~(w_dens.isna() | w_pop.isna())
    add(
        "wilcoxon_auc_density_vs_popularity",
        "density",
        float(
            stats.wilcoxon(
                w_dens[valid], w_pop[valid], zero_method="wilcox", correction=False
            ).pvalue
        ),
        note="paired across choice sets",
    )
    add(
        "wilcoxon_auc_fused_vs_popularity",
        "density_plus_popularity",
        float(
            stats.wilcoxon(
                w_fused[valid], w_pop[valid], zero_method="wilcox", correction=False
            ).pvalue
        ),
        note="paired across choice sets",
    )
    r_dens = sets_df["recall_at_k_density"]
    r_pop = sets_df["recall_at_k_popularity"]
    r_rand = sets_df["random_recall_at_k"]
    add(
        "wilcoxon_recall_density_vs_popularity",
        "density",
        float(
            stats.wilcoxon(r_dens, r_pop, zero_method="wilcox", correction=False).pvalue
        ),
        note="paired across choice sets",
    )
    add(
        "wilcoxon_recall_density_vs_random",
        "density",
        float(
            stats.wilcoxon(r_dens, r_rand, zero_method="wilcox", correction=False).pvalue
        ),
        note="paired across choice sets",
    )
    add(
        "mean_lift_recall_density_over_random",
        "density",
        float((r_dens / r_rand).mean()),
    )
    add(
        "mean_lift_recall_popularity_over_random",
        "popularity",
        float((r_pop / r_rand).mean()),
    )
    add(
        "share_sets_density_beats_popularity_auc",
        "density",
        float((w_dens > w_pop).mean()),
    )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, sets_df


def main() -> None:
    panel_df, meta = build_prediction_panel()
    panel_df.to_parquet(OUT_PANEL, index=False)
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_PANEL}")
    summary_df, sets_df = evaluate(panel_df)

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    sets_df.to_csv(OUT_CHOICE_SETS, index=False)
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary_df.to_string(index=False))
    print(f"\nWrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_CHOICE_SETS}")
    print(f"Wrote: {OUT_META}")


if __name__ == "__main__":
    main()
