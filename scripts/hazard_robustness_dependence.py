#!/usr/bin/env python3
"""Dependence-robust inference for the locality hazard.

The main conditional logit uses rolling 5-year windows advanced one year at a
time, so adjacent choice sets overlap heavily and the naive standard errors are
anti-conservative. This script produces the three validity checks the manuscript
reports:

1. Disjoint blocks: non-overlapping 5-meeting periods and cumulative-lagged
   space reduce the document reuse caused by rolling windows. Successive
   actor transitions still share the middle block, and their cumulative maps
   are nested, so this check does not make choice sets independent.
2. Actor-level bootstrap on the fully prospective rolling panel: resample
   actors with replacement and re-estimate, so within-actor overlap is
   absorbed into the sampling distribution for the manuscript's primary
   risk set.
3. States-only panel: restrict the risk sets to states, testing whether the
   distance effect is driven by mandate-driven expert bodies.

Also reports the rolling-panel estimate with a topic-age control (years since a
topic first appeared in the archive), which addresses the objection that old
topics are both more proximate and more likely to be entered.

Outputs:
- output/hazard_robustness_dependence.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hazard_conditional_logit import (  # noqa: E402
    RCA_THRESHOLD,
    WINDOW_MEETINGS,
    build_conditional_logit_panel,
    build_window_interaction,
    choose_period_col,
    load_data_with_fallback,
    phi_from_interaction,
    sanitize_periods,
    topic_first_appearance,
)
from utils import get_rca  # noqa: E402

warnings.filterwarnings("ignore")

N_BOOTSTRAP = 200
N_JOBS = 8
SEED = 20260806

STATE_NAME_MAP = {
    "Czech Republic": "Czechia",
    "North Korea": "Korea (DPRK)",
    "South Korea": "Korea (ROK)",
    "Russia": "Russian Federation",
    "Turkey": "Türkiye",
}

OUT_JSON = Path("output/hazard_robustness_dependence.json")
_BOOTSTRAP_PRIOR_PIECES: dict[str, pd.DataFrame] = {}


def fit_distance(panel_df: pd.DataFrame, covariates: list[str]) -> dict:
    counts = panel_df.groupby("group")["adopted"].agg(["sum", "count"])
    informative = counts[(counts["sum"] > 0) & (counts["sum"] < counts["count"])].index
    panel_df = panel_df[panel_df["group"].isin(informative)].copy()
    model = ConditionalLogit(
        panel_df["adopted"].astype(int),
        panel_df[covariates],
        groups=panel_df["group"],
    )
    res = model.fit(disp=False, maxiter=200)
    beta = float(res.params["distance"])
    se = float(res.bse["distance"])
    return {
        "distance_coef": beta,
        "distance_se": se,
        "odds_ratio_per_0_1": float(np.exp(0.1 * beta)),
        "odds_ratio_per_0_1_ci_low": float(np.exp(0.1 * (beta - 1.96 * se))),
        "odds_ratio_per_0_1_ci_high": float(np.exp(0.1 * (beta + 1.96 * se))),
        "n_rows": int(len(panel_df)),
        "n_groups": int(panel_df["group"].nunique()),
    }


def add_topic_age(panel_df: pd.DataFrame, submitted_df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    first_period = topic_first_appearance(submitted_df, period_col)
    out = panel_df.copy()
    out["topic_age"] = out.apply(
        lambda r: max(
            0.0,
            float(r["period_end"] - first_period.get(r["topic"], r["period_end"])),
        ),
        axis=1,
    )
    return out


def fit_actor_bootstrap_beta(
    drawn: np.ndarray,
    replicate: int,
) -> float:
    """Refit one actor-block bootstrap replicate; return NaN on failure."""
    parts = []
    for slot, member in enumerate(drawn):
        piece = _BOOTSTRAP_PRIOR_PIECES[str(member)].copy()
        piece["group"] = piece["group"] + f"::prior_b{replicate}_r{slot}"
        parts.append(piece)
    boot_panel = pd.concat(parts, ignore_index=True)
    counts = boot_panel.groupby("group")["adopted"].agg(["sum", "count"])
    informative = counts[
        counts["sum"].gt(0) & counts["sum"].lt(counts["count"])
    ].index
    boot_panel = boot_panel[boot_panel["group"].isin(informative)]
    try:
        model = ConditionalLogit(
            boot_panel["adopted"].astype(int),
            boot_panel[["distance", "topic_popularity"]],
            groups=boot_panel["group"],
        )
        fitted = model.fit(disp=False, maxiter=200)
        return float(fitted.params["distance"])
    except Exception:
        return float("nan")


def build_nonoverlapping_panel() -> pd.DataFrame:
    """Five-meeting blocks that reduce, but do not eliminate, dependence."""
    counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback()
    period_col = choose_period_col(submitted_df)
    submitted_df = sanitize_periods(submitted_df, period_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    period_min = int(submitted_df[period_col].min())
    period_max = int(submitted_df[period_col].max())
    first_appearance = topic_first_appearance(submitted_df, period_col)

    periods = [
        (period, period + WINDOW_MEETINGS - 1)
        for period in range(period_min, period_max + 1, WINDOW_MEETINGS)
        if period + WINDOW_MEETINGS - 1 <= period_max
    ]

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
        active_by_period.append(get_rca(interaction) >= RCA_THRESHOLD)

    rows = []
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
        phi = phi_from_interaction(cumulative_interaction, topics)
        prev_active = active_by_period[t - 1]
        curr_active = active_by_period[t]
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
            group = f"{member}::{periods[t][1]}"
            for idx, topic in enumerate(topics):
                if not at_risk[idx]:
                    continue
                if first_appearance.get(topic, prev_end + 1) > prev_end:
                    continue
                rows.append(
                    {
                        "group": group,
                        "member": member,
                        "period_end": int(periods[t][1]),
                        "topic": topic,
                        "adopted": int(adopted[idx]),
                        "distance": float(1.0 - max_phi[idx]),
                        "topic_popularity": float(prev_topic_popularity.loc[topic]),
                    }
                )
    return pd.DataFrame(rows)


def load_state_set() -> set[str]:
    mem = pd.read_csv(PROJECT_ROOT / "membership.csv")
    states = set()
    for country in mem["country"]:
        states.add(STATE_NAME_MAP.get(country, country))
    return states


def main() -> None:
    results: dict = {
        "window_meetings": WINDOW_MEETINGS,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
    }

    panel_df, meta = build_conditional_logit_panel()
    rolling = panel_df[panel_df["mode"] == "cumulative_lagged"].copy()
    results["rolling_baseline"] = fit_distance(rolling, ["distance", "topic_popularity"])

    counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback()
    period_col = choose_period_col(submitted_df)
    submitted_df = sanitize_periods(submitted_df, period_col)
    first_period = topic_first_appearance(submitted_df, period_col)
    rolling["first_period"] = rolling["topic"].map(first_period)
    rolling_prior = rolling[
        rolling["first_period"] <= rolling["period_end"] - 1
    ].copy()
    results["rolling_prior_available"] = fit_distance(
        rolling_prior, ["distance", "topic_popularity"]
    )
    rolling_age = add_topic_age(rolling_prior, submitted_df, period_col)
    results["rolling_topic_age_control"] = fit_distance(
        rolling_age, ["distance", "topic_popularity", "topic_age"]
    )

    rng = np.random.default_rng(SEED)
    prior_members = sorted(rolling_prior["member"].unique())
    prior_pieces = {
        member: rolling_prior.loc[rolling_prior["member"].eq(member)].copy()
        for member in prior_members
    }
    draws = [
        rng.choice(prior_members, size=len(prior_members), replace=True)
        for _ in range(N_BOOTSTRAP)
    ]
    global _BOOTSTRAP_PRIOR_PIECES
    _BOOTSTRAP_PRIOR_PIECES = prior_pieces
    with mp.get_context("fork").Pool(processes=N_JOBS) as pool:
        prior_boot_betas = np.asarray(
            pool.starmap(
                fit_actor_bootstrap_beta,
                [(drawn, replicate) for replicate, drawn in enumerate(draws)],
                chunksize=1,
            ),
            dtype=float,
        )
    prior_boot_betas = prior_boot_betas[np.isfinite(prior_boot_betas)]
    prior_beta = results["rolling_prior_available"]["distance_coef"]
    results["rolling_prior_actor_bootstrap"] = {
        "n_successful_replicates": int(len(prior_boot_betas)),
        "distance_coef_mean": float(prior_boot_betas.mean()),
        "distance_coef_sd": float(prior_boot_betas.std(ddof=1)),
        "odds_ratio_per_0_1": float(np.exp(0.1 * prior_beta)),
        "odds_ratio_per_0_1_ci_low": float(
            np.exp(0.1 * np.percentile(prior_boot_betas, 2.5))
        ),
        "odds_ratio_per_0_1_ci_high": float(
            np.exp(0.1 * np.percentile(prior_boot_betas, 97.5))
        ),
    }

    states = load_state_set()
    states_panel = rolling_prior[rolling_prior["member"].isin(states)].copy()
    results["states_only"] = fit_distance(states_panel, ["distance", "topic_popularity"])
    results["states_only"]["n_states_in_panel"] = int(states_panel["member"].nunique())

    nonoverlap_panel = build_nonoverlapping_panel()
    results["nonoverlapping_windows"] = fit_distance(
        nonoverlap_panel, ["distance", "topic_popularity"]
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
