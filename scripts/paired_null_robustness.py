#!/usr/bin/env python3
"""Dependence- and popularity-robust versions of the paired displacement test.

The headline paired test (portfolio_displacement.py) judges each transition
against the pooled full-history space with a uniform redraw from topics the
actor did not hold. Two weaknesses: the pooled space uses future information,
and the uniform draw ignores that observed entry concentrates on popular
topics, which sit closer to everything in a dense space.

This recomputes the same paired statistic under a 2x2 design:

- space: pooled full-history (as before) vs cumulative-lagged (built only from
  documents up to the end of the previous window, matching the hazard);
- null: uniform redraw vs popularity-weighted redraw (draw probability
  proportional to the number of actors specialized in the topic in the
  previous window, plus one so topics with no prior holders stay available).

It also repeats all four cells after restricting the redraw pool to concerns
that had appeared in the archive by the end of the current window. A final,
fully prospective specification admits only concerns already observed by the
end of the preceding window. That last specification is the manuscript's
headline matched comparison.

Outputs:
- output/paired_null_robustness.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hazard_conditional_logit import (  # noqa: E402
    RCA_THRESHOLD,
    WINDOW_MEETINGS,
    build_periods,
    build_window_interaction,
    choose_period_col,
    load_data_with_fallback,
    phi_from_interaction,
    sanitize_periods,
    topic_first_appearance,
)
from portfolio_displacement import _displacement, get_active  # noqa: E402
from utils import get_rca  # noqa: E402

N_NULL_DRAWS = 200
N_ACTOR_BOOTSTRAP = 2000
SEED = 1991

OUT_JSON = Path("output/paired_null_robustness.json")
OUT_STRICT_CSV = Path("output/paired_null_strict_events.csv")


def summarize(moved: pd.DataFrame, prefix: str) -> dict:
    delta = moved["displacement"] - moved["null_mean"]
    return {
        f"{prefix}_share_nearer_than_null": float((delta < 0).mean()),
        f"{prefix}_observed_median": float(moved["displacement"].median()),
        f"{prefix}_null_median": float(moved["null_mean"].median()),
        f"{prefix}_wilcoxon_p": float(stats.wilcoxon(delta, alternative="less").pvalue),
    }


def actor_bootstrap_share(moved: pd.DataFrame) -> tuple[float, float]:
    """Cluster-bootstrap the share statistic by submitting actor."""
    by_actor = (
        moved.assign(
            _nearer=(moved["displacement"] < moved["null_mean"]).astype(float)
        )
        .groupby("member")["_nearer"]
        .agg(["sum", "count"])
    )
    successes = by_actor["sum"].to_numpy(dtype=float)
    totals = by_actor["count"].to_numpy(dtype=float)
    n_actors = len(by_actor)
    rng = np.random.default_rng(SEED + 2)
    draws = np.empty(N_ACTOR_BOOTSTRAP, dtype=float)
    for draw in range(N_ACTOR_BOOTSTRAP):
        sampled = rng.integers(0, n_actors, size=n_actors)
        draws[draw] = float(successes[sampled].sum() / totals[sampled].sum())
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback()
    period_col = choose_period_col(submitted_df)
    submitted_df = sanitize_periods(submitted_df, period_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    period_min = int(submitted_df[period_col].min())
    period_max = int(submitted_df[period_col].max())
    periods = build_periods(period_min, period_max, WINDOW_MEETINGS)
    n_topics = len(topics)
    topic_first_year = pd.Series(
        topic_first_appearance(submitted_df, period_col)
    ).reindex(topics)
    if topic_first_year.isna().any():
        missing = topic_first_year[topic_first_year.isna()].index.tolist()
        raise ValueError(f"Missing first-appearance year for topics: {missing}")
    topic_first_year_arr = topic_first_year.to_numpy(dtype=int)

    active_by_period = []
    for start, end in periods:
        interaction = build_window_interaction(
            submitted_df=submitted_df,
            period_col=period_col,
            period_start=int(start),
            period_end=int(end),
            all_members_raw=set(members_raw),
            all_topics_raw=set(topics_raw),
            topics_order=topics,
            members_order=members,
        )
        active_by_period.append(get_active(interaction))

    phi_pooled = phi_from_interaction(
        counts_df.reindex(index=topics, columns=members, fill_value=0), topics
    )

    rng = np.random.default_rng(SEED)
    rng_available = np.random.default_rng(SEED + 1)
    rng_prior = np.random.default_rng(SEED + 3)

    records = {
        "pooled_uniform": [],
        "pooled_popularity": [],
        "lagged_uniform": [],
        "lagged_popularity": [],
        "pooled_uniform_available": [],
        "pooled_popularity_available": [],
        "lagged_uniform_available": [],
        "lagged_popularity_available": [],
        "lagged_popularity_prior": [],
    }

    for t in range(1, len(periods)):
        prev_end = int(periods[t - 1][1])
        cumulative_interaction = build_window_interaction(
            submitted_df=submitted_df,
            period_col=period_col,
            period_start=period_min,
            period_end=prev_end,
            all_members_raw=set(members_raw),
            all_topics_raw=set(topics_raw),
            topics_order=topics,
            members_order=members,
        )
        phi_lagged = phi_from_interaction(cumulative_interaction, topics)

        prev_active = active_by_period[t - 1]
        curr_active = active_by_period[t]
        popularity = prev_active.sum(axis=1).to_numpy(dtype=float)

        for member in members:
            prior_idx = np.flatnonzero(prev_active[member].to_numpy())
            new_idx = np.flatnonzero(curr_active[member].to_numpy())
            if prior_idx.size == 0 or new_idx.size == 0:
                continue
            added_idx = np.setdiff1d(new_idx, prior_idx)
            available = np.setdiff1d(np.arange(n_topics), prior_idx)
            appeared = np.flatnonzero(topic_first_year_arr <= int(periods[t][1]))
            available_observed = np.setdiff1d(appeared, prior_idx)
            appeared_prior = np.flatnonzero(topic_first_year_arr <= prev_end)
            available_prior = np.setdiff1d(appeared_prior, prior_idx)
            added_prior = np.intersect1d(added_idx, appeared_prior)
            if added_idx.size == 0 or available.size < added_idx.size:
                continue
            if available_observed.size < added_idx.size:
                continue

            weights = popularity[available] + 1.0
            weights = weights / weights.sum()
            weights_observed = popularity[available_observed] + 1.0
            weights_observed = weights_observed / weights_observed.sum()
            if added_prior.size and available_prior.size >= added_prior.size:
                weights_prior = popularity[available_prior] + 1.0
                weights_prior = weights_prior / weights_prior.sum()
                prior_obs = _displacement(added_prior, prior_idx, phi_lagged)
                prior_draws = np.array(
                    [
                        _displacement(
                            rng_prior.choice(
                                available_prior,
                                size=added_prior.size,
                                replace=False,
                                p=weights_prior,
                            ),
                            prior_idx,
                            phi_lagged,
                        )
                        for _ in range(N_NULL_DRAWS)
                    ]
                )
                records["lagged_popularity_prior"].append(
                    {
                        "member": member,
                        "period_end": int(periods[t][1]),
                        "n_adopted_prior_available": int(added_prior.size),
                        "displacement": prior_obs,
                        "null_mean": float(prior_draws.mean()),
                    }
                )

            for space_key, phi in (("pooled", phi_pooled), ("lagged", phi_lagged)):
                obs = _displacement(added_idx, prior_idx, phi)
                uniform_draws = np.array(
                    [
                        _displacement(
                            rng.choice(available, size=added_idx.size, replace=False),
                            prior_idx,
                            phi,
                        )
                        for _ in range(N_NULL_DRAWS)
                    ]
                )
                weighted_draws = np.array(
                    [
                        _displacement(
                            rng.choice(
                                available,
                                size=added_idx.size,
                                replace=False,
                                p=weights,
                            ),
                            prior_idx,
                            phi,
                        )
                        for _ in range(N_NULL_DRAWS)
                    ]
                )
                uniform_available_draws = np.array(
                    [
                        _displacement(
                            rng_available.choice(
                                available_observed,
                                size=added_idx.size,
                                replace=False,
                            ),
                            prior_idx,
                            phi,
                        )
                        for _ in range(N_NULL_DRAWS)
                    ]
                )
                weighted_available_draws = np.array(
                    [
                        _displacement(
                            rng_available.choice(
                                available_observed,
                                size=added_idx.size,
                                replace=False,
                                p=weights_observed,
                            ),
                            prior_idx,
                            phi,
                        )
                        for _ in range(N_NULL_DRAWS)
                    ]
                )
                row = {
                    "member": member,
                    "period_end": int(periods[t][1]),
                    "displacement": obs,
                }
                row_u = dict(row, null_mean=float(uniform_draws.mean()))
                row_w = dict(row, null_mean=float(weighted_draws.mean()))
                records[f"{space_key}_uniform"].append(row_u)
                records[f"{space_key}_popularity"].append(row_w)
                records[f"{space_key}_uniform_available"].append(
                    dict(row, null_mean=float(uniform_available_draws.mean()))
                )
                records[f"{space_key}_popularity_available"].append(
                    dict(row, null_mean=float(weighted_available_draws.mean()))
                )

    meta = {
        "window_meetings": WINDOW_MEETINGS,
        "period_col": period_col,
        "rca_threshold": RCA_THRESHOLD,
        "n_null_draws": N_NULL_DRAWS,
        "seed": SEED,
        "n_actor_bootstrap": N_ACTOR_BOOTSTRAP,
        "popularity_weights": "previous-window holder counts + 1, renormalized",
        "historical_availability_rule": (
            "topic has at least one archive document by current-window end"
        ),
        "prospective_availability_rule": (
            "topic has at least one archive document by prior-window end"
        ),
    }
    for key, rows in records.items():
        moved = pd.DataFrame(rows)
        meta[f"{key}_n_transitions"] = int(len(moved))
        meta.update(summarize(moved, key))
        low, high = actor_bootstrap_share(moved)
        meta[f"{key}_share_actor_bootstrap_ci_low"] = float(low)
        meta[f"{key}_share_actor_bootstrap_ci_high"] = float(high)

    strict = pd.DataFrame(records["lagged_popularity_prior"])
    strict.to_csv(OUT_STRICT_CSV, index=False)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_STRICT_CSV}")


if __name__ == "__main__":
    main()
