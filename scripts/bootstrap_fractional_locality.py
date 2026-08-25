#!/usr/bin/env python3
"""Actor-history bootstrap for the primary fractional multi-label locality result.

Each draw resamples complete submitting-actor histories.  The rolling actor--
concern matrices, specialization portfolios, prior concern geometry, available
concerns, and conditional-choice panel are then rebuilt inside the draw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hazard_conditional_logit import (  # noqa: E402
    RCA_THRESHOLD,
    WINDOW_MEETINGS,
    build_periods,
    build_window_interaction,
    choose_period_col,
    phi_from_interaction,
    sanitize_periods,
)
from primary_concern_sensitivity import variants  # noqa: E402
from utils import (  # noqa: E402
    extract_unique_countries,
    extract_unique_topics,
    get_rca,
    standardize_index_labels,
)

OUTDIR = ROOT / "output" / "category_treatment_comparison" / "fractional_multilabel"


def source_matrices() -> tuple[list[str], list[int], list[tuple[int, int]], list[np.ndarray]]:
    data = variants()["fractional_multilabel"]
    period_col = choose_period_col(data)
    clean = sanitize_periods(data, period_col)
    actor_raw = extract_unique_countries(clean)
    topic_raw = extract_unique_topics(clean)
    actors = sorted({str(actor) for actor in actor_raw})
    topics = sorted(
        standardize_index_labels(pd.DataFrame(index=sorted(topic_raw))).index
    )
    meetings = list(range(int(clean[period_col].min()), int(clean[period_col].max()) + 1))
    periods = build_periods(min(meetings), max(meetings), WINDOW_MEETINGS)
    matrices: list[np.ndarray] = []
    for meeting in meetings:
        matrix = build_window_interaction(
            clean,
            period_col,
            meeting,
            meeting,
            set(actor_raw),
            set(topic_raw),
            topics,
            actors,
        )
        matrices.append(matrix.to_numpy(dtype=float))
    return topics, meetings, periods, matrices


def bootstrap_panel(
    topics: list[str],
    meetings: list[int],
    periods: list[tuple[int, int]],
    meeting_matrices: list[np.ndarray],
    sampled_columns: np.ndarray,
) -> pd.DataFrame:
    columns = [f"actor_{index}" for index in range(len(sampled_columns))]
    sampled = [matrix[:, sampled_columns] for matrix in meeting_matrices]
    by_meeting = dict(zip(meetings, sampled))
    period_arrays = [
        np.sum([by_meeting[m] for m in meetings if start <= m <= end], axis=0)
        for start, end in periods
    ]
    period_frames = [pd.DataFrame(array, index=topics, columns=columns) for array in period_arrays]
    active = [get_rca(frame).ge(RCA_THRESHOLD) for frame in period_frames]

    rows: list[dict] = []
    for t in range(1, len(periods)):
        prev_end = periods[t - 1][1]
        history = np.sum(
            [by_meeting[m] for m in meetings if m <= prev_end], axis=0
        )
        history_frame = pd.DataFrame(history, index=topics, columns=columns)
        phi = phi_from_interaction(history_frame, topics)
        available = history.sum(axis=1) > 0
        prev_active = active[t - 1]
        curr_active = active[t]
        popularity = prev_active.sum(axis=1).to_numpy(dtype=float) / len(columns)
        for actor_index, actor in enumerate(columns):
            held = prev_active.iloc[:, actor_index].to_numpy(dtype=bool)
            if not held.any():
                continue
            current = curr_active.iloc[:, actor_index].to_numpy(dtype=bool)
            at_risk = (~held) & available
            adopted = current & at_risk
            if not adopted.any() or adopted.sum() == at_risk.sum():
                continue
            distances = 1.0 - phi[:, np.flatnonzero(held)].max(axis=1)
            for topic_index in np.flatnonzero(at_risk):
                rows.append(
                    {
                        "group": f"{actor}::{periods[t][1]}",
                        "adopted": int(adopted[topic_index]),
                        "distance": float(distances[topic_index]),
                        "topic_popularity": float(popularity[topic_index]),
                    }
                )
    return pd.DataFrame(rows)


def fit_draw(panel: pd.DataFrame) -> tuple[float, int, int]:
    result = ConditionalLogit(
        panel["adopted"].astype(int),
        panel[["distance", "topic_popularity"]],
        groups=panel["group"],
    ).fit(disp=False, maxiter=200)
    beta = float(result.params["distance"])
    return beta, int(panel["group"].nunique()), int(len(panel))


def one_draw(
    draw: int,
    sampled_columns: np.ndarray,
    topics: list[str],
    meetings: list[int],
    periods: list[tuple[int, int]],
    matrices: list[np.ndarray],
) -> dict:
    try:
        panel = bootstrap_panel(topics, meetings, periods, matrices, sampled_columns)
        beta, groups, risk_rows = fit_draw(panel)
        return {
            "draw": draw + 1,
            "distance_beta": beta,
            "odds_ratio_per_0_1": float(np.exp(0.1 * beta)),
            "actor_periods": groups,
            "risk_rows": risk_rows,
            "fit_ok": True,
        }
    except Exception as error:
        return {
            "draw": draw + 1,
            "distance_beta": np.nan,
            "odds_ratio_per_0_1": np.nan,
            "actor_periods": 0,
            "risk_rows": 0,
            "fit_ok": False,
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    topics, meetings, periods, matrices = source_matrices()
    n_actors = matrices[0].shape[1]
    rng = np.random.default_rng(args.seed)
    samples = [rng.integers(0, n_actors, size=n_actors) for _ in range(args.draws)]
    rows = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(one_draw)(draw, sample, topics, meetings, periods, matrices)
        for draw, sample in enumerate(samples)
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    draws = pd.DataFrame(rows)
    draws.to_csv(OUTDIR / "locality_actor_history_bootstrap.csv", index=False)
    valid = draws.loc[draws["fit_ok"], "odds_ratio_per_0_1"].dropna()
    valid_beta = draws.loc[draws["fit_ok"], "distance_beta"].dropna()
    identity_panel = bootstrap_panel(
        topics, meetings, periods, matrices, np.arange(n_actors)
    )
    original_beta, _, _ = fit_draw(identity_panel)
    beta_q_low, beta_q_high = valid_beta.quantile([0.025, 0.975])
    basic_beta_low = 2 * original_beta - beta_q_high
    basic_beta_high = 2 * original_beta - beta_q_low
    summary = {
        "category_treatment": "fractional_multilabel",
        "resampling_unit": "complete submitting-actor history",
        "rebuilt_within_draw": [
            "rolling actor-concern matrices",
            "RPA portfolios",
            "prior concern geometry",
            "historical availability",
            "conditional-choice panel",
        ],
        "requested_draws": int(args.draws),
        "successful_draws": int(len(valid)),
        "seed": int(args.seed),
        "parallel_jobs": int(args.jobs),
        "original_distance_beta": float(original_beta),
        "original_odds_ratio_per_0_1": float(np.exp(0.1 * original_beta)),
        "odds_ratio_per_0_1_median": float(valid.median()),
        "odds_ratio_per_0_1_percentile_ci_low": float(valid.quantile(0.025)),
        "odds_ratio_per_0_1_percentile_ci_high": float(valid.quantile(0.975)),
        "odds_ratio_per_0_1_basic_ci_low": float(np.exp(0.1 * basic_beta_low)),
        "odds_ratio_per_0_1_basic_ci_high": float(np.exp(0.1 * basic_beta_high)),
    }
    (OUTDIR / "locality_actor_history_bootstrap.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
