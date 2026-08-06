#!/usr/bin/env python3
"""In-repo recomputation of the two circularity checks behind Fig 3C.

The locality coefficient could be circular because phi is estimated from the
same co-specialization the hazard model predicts. The manuscript answers with
two stricter constructions of the space, previously transcribed into the
specifications table from an out-of-repo run:

  * leave-one-actor-out: each actor's distances come from a cumulative-lagged
    space re-estimated with that actor's own submissions removed (its column of
    the interaction matrix zeroed) before RPA and proximity estimation, while
    the outcome panel stays fixed;
  * fractional co-sponsorship: the cumulative-lagged space is rebuilt counting
    a co-sponsored paper as 1/n for each of its n sponsors instead of once per
    sponsor.

Both reuse the conditional-logit specification of
``scripts/hazard_conditional_logit.py`` verbatim; only the space changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

import scripts.hazard_conditional_logit as hcl
from utils import (
    _is_excluded_topic_label,
    _normalize_topic_label,
    _split_multi_value,
)

WINDOW_YEARS = hcl.WINDOW_YEARS
RCA_THRESHOLD = hcl.RCA_THRESHOLD
CHECK_ORDER = ["leave_one_actor_out", "fractional"]

OUT_CSV = Path("output/hazard_circularity_coefficients.csv")
OUT_JSON = Path("output/hazard_circularity_coefficients_meta.json")


def build_fractional_interaction(
    submitted_df: pd.DataFrame,
    year_col: str,
    year_start: int,
    year_end: int,
    topics_order: list[str],
    members_order: list[str],
) -> pd.DataFrame:
    win = submitted_df[
        (submitted_df[year_col] >= int(year_start))
        & (submitted_df[year_col] <= int(year_end))
    ]
    df = win.dropna(subset=["category", "submitted by"]).copy()
    df["category"] = df["category"].apply(lambda v: _split_multi_value(v, "\t"))
    df = df.explode("category")
    df["category"] = df["category"].apply(_normalize_topic_label)
    df = df[~df["category"].apply(_is_excluded_topic_label)]
    df["submitted by"] = df["submitted by"].apply(_split_multi_value)
    df = df.explode("submitted by")
    df = df.dropna(subset=["category", "submitted by"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(0, index=topics_order, columns=members_order)

    if "paper id" in df.columns:
        df = df.drop_duplicates(subset=["paper id", "category", "submitted by"])
        n_sponsors = df.groupby("paper id")["submitted by"].nunique()
        df["w"] = 1.0 / df["paper id"].map(n_sponsors)
    else:  # pragma: no cover
        df = df.drop_duplicates(subset=["category", "submitted by"])
        df["w"] = 1.0

    matrix = df.pivot_table(
        index="submitted by", columns="category", values="w", aggfunc="sum"
    )
    matrix = matrix.reindex(
        index=sorted(members_order), columns=sorted(topics_order), fill_value=0.0
    )
    return matrix.T


def build_panels() -> tuple[pd.DataFrame, dict]:
    counts_df, submitted_df, members_raw, topics_raw = hcl.load_data_with_fallback()
    year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns and "meeting_year" in submitted_df.columns:
        year_col = "meeting_year"
    submitted_df = hcl.sanitize_years(submitted_df, year_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    year_min = int(submitted_df[year_col].min())
    year_max = int(submitted_df[year_col].max())
    periods = hcl.build_periods(year_min, year_max, WINDOW_YEARS)

    active_by_period = []
    for start, end in periods:
        interaction = hcl.build_window_interaction(
            submitted_df, year_col, start, end,
            set(members_raw), set(topics_raw), topics, members,
        )
        active_by_period.append(hcl.get_rca(interaction) >= RCA_THRESHOLD)

    panel_rows = []
    for t in range(1, len(periods)):
        prev_end = int(periods[t - 1][1])
        period_end = int(periods[t][1])
        prev_active = active_by_period[t - 1]
        curr_active = active_by_period[t]
        prev_topic_popularity = prev_active.sum(axis=1) / max(len(members), 1)

        cumulative = hcl.build_window_interaction(
            submitted_df, year_col, year_min, prev_end,
            set(members_raw), set(topics_raw), topics, members,
        )
        fractional = build_fractional_interaction(
            submitted_df, year_col, year_min, prev_end, topics, members,
        )
        phi_frac = hcl.phi_from_interaction(fractional, topics)

        phis = {"fractional": phi_frac}
        for member in members:
            prev_mask = prev_active[member].to_numpy(dtype=bool)
            if not prev_mask.any():
                continue
            at_risk = ~prev_mask
            adopted = curr_active[member].to_numpy(dtype=bool) & at_risk
            if not at_risk.any() or int(adopted.sum()) == 0:
                continue

            excluded = cumulative.copy()
            excluded[member] = 0
            phis["leave_one_actor_out"] = hcl.phi_from_interaction(
                excluded, topics
            )

            prev_indices = np.where(prev_mask)[0]
            group = f"{member}::{period_end}"
            for mode in CHECK_ORDER:
                max_phi = phis[mode][:, prev_indices].max(axis=1)
                distance = 1.0 - max_phi
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
                            "distance": float(distance[idx]),
                            "topic_popularity": float(
                                prev_topic_popularity.loc[topic]
                            ),
                        }
                    )

    panel_df = pd.DataFrame(panel_rows)
    meta = {
        "window_years": WINDOW_YEARS,
        "rca_threshold": RCA_THRESHOLD,
        "space": "cumulative_lagged",
        "outcome_panel": "fixed, full-count RCA",
        "n_panel_rows": int(len(panel_df)),
        "n_groups": int(panel_df["group"].nunique()),
    }
    return panel_df, meta


def main() -> None:
    panel_df, meta = build_panels()
    rows = []
    for mode in CHECK_ORDER:
        df = panel_df[panel_df["mode"] == mode].copy()
        model = ConditionalLogit(
            df["adopted"].astype(int),
            df[["distance", "topic_popularity"]],
            groups=df["group"],
        )
        res = model.fit(disp=False, maxiter=200)
        coef = float(res.params["distance"])
        se = float(res.bse["distance"])
        rows.append(
            {
                "mode": mode,
                "n_rows": int(len(df)),
                "n_groups": int(df["group"].nunique()),
                "distance_coef": coef,
                "distance_std_err": se,
                "distance_ci_low_95": coef - 1.96 * se,
                "distance_ci_high_95": coef + 1.96 * se,
                "distance_odds_ratio_per_0_1": float(np.exp(0.1 * coef)),
            }
        )

    summary = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
