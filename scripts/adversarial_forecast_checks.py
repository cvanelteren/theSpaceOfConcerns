#!/usr/bin/env python3
"""Adversarial checks for the Resolution attention nowcast.

The primary analysis is a same-meeting pseudoprospective nowcast.  These checks
ask whether its gain survives dependence-aware uncertainty, whether strictly
earlier attention carries the same signal, whether future attention performs
similarly, and whether the category alignment of current papers matters.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.analyze_output_category_families import (
    family_phi_by_meeting,
    local_weight_matrix,
    paper_family_relations,
)
from scripts.analyze_resolution_attention_forecast import (
    OUTPUT_COLUMNS,
    PRIMARY_NETWORK_K,
    add_features,
    forecast_meetings,
    model_features,
    paired_summary,
)
from scripts.forecast_output_allocation import BOOTSTRAP_DRAWS, SEED
from scripts.primary_concern_sensitivity import variants


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "scientific_checks"
SUMMARY_PATH = OUTDIR / "forecast_adversarial_summary.json"
SCORES_PATH = OUTDIR / "forecast_adversarial_meeting_scores.csv"
NULL_PATH = OUTDIR / "forecast_current_label_null.csv"
LAGGED_TYPE_SCORES_PATH = OUTDIR / "forecast_lagged_type_scores.csv"
LAGGED_TYPE_SUMMARY_PATH = OUTDIR / "forecast_lagged_type_summary.csv"
PERMUTATIONS = 200
OUTPUT = "resolution_mass"
NETWORK = f"neighbor_papers_k{PRIMARY_NETWORK_K}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help="Run deterministic and by-output checks without category permutations.",
    )
    return parser.parse_args()


def score_models(panel: pd.DataFrame, test_end: int = 47) -> pd.DataFrame:
    history = model_features(OUTPUT, include_focal=False)
    specifications = {
        "output history": history,
        "strictly lagged direct attention": [
            *history,
            "paper_history_1",
            "actor_reach_1",
        ],
        "strictly lagged direct + network attention": [
            *history,
            "paper_history_1",
            "actor_reach_1",
            "nearby_history_1",
        ],
        "same-meeting direct attention": model_features(OUTPUT),
        "same-meeting direct + network attention": model_features(
            OUTPUT, neighbor=NETWORK
        ),
    }
    rows = []
    for name, features in specifications.items():
        rows.append(
            forecast_meetings(panel, OUTPUT, features, test_end=test_end).assign(
                model=name
            )
        )
    return pd.concat(rows, ignore_index=True)


def paired_difference(
    scores: pd.DataFrame, candidate: str, baseline: str
) -> np.ndarray:
    wide = scores.pivot(index="meeting", columns="model", values="allocation_log_score")
    return (wide[candidate] - wide[baseline]).to_numpy(float)


def moving_block_interval(values: np.ndarray, block: int) -> list[float]:
    rng = np.random.default_rng(SEED + 100 * block)
    starts = np.arange(len(values) - block + 1)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled: list[float] = []
        while len(sampled) < len(values):
            start = int(rng.choice(starts))
            sampled.extend(values[start : start + block])
        draws.append(float(np.mean(sampled[: len(values)])))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def exact_sign_flip(values: np.ndarray) -> dict[str, float]:
    observed = float(values.mean())
    means = sign_matrix(len(values)) @ values / len(values)
    return {
        "observed_mean": observed,
        "one_sided_p_lower": float(np.mean(means <= observed)),
        "two_sided_p": float(np.mean(np.abs(means) >= abs(observed))),
    }


@lru_cache(maxsize=2)
def sign_matrix(size: int) -> np.ndarray:
    rows = np.arange(2**size, dtype=np.uint32)[:, None]
    bits = (rows >> np.arange(size, dtype=np.uint32)) & 1
    return (2 * bits.astype(np.int8)) - 1


def add_future_attention(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["topic", "meeting"]).copy()
    for source, target in (
        ("paper_count", "future_paper_count"),
        ("current_actor_reach", "future_actor_reach"),
        (NETWORK, "future_network_attention"),
    ):
        result[target] = result.groupby("topic")[source].shift(-1)
    return result


def lagged_type_comparison(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_tables = []
    summaries = []
    for instrument, output in OUTPUT_COLUMNS.items():
        history = model_features(output, include_focal=False)
        specifications = {
            "output history": history,
            "strictly lagged direct attention": [
                *history,
                "paper_history_1",
                "actor_reach_1",
            ],
            "strictly lagged direct + network attention": [
                *history,
                "paper_history_1",
                "actor_reach_1",
                "nearby_history_1",
            ],
        }
        scores = {
            name: forecast_meetings(panel, output, features)
            for name, features in specifications.items()
        }
        for name, table in scores.items():
            score_tables.append(table.assign(instrument=instrument, model=name))
        for name in (
            "strictly lagged direct attention",
            "strictly lagged direct + network attention",
        ):
            summaries.append(
                {
                    "instrument": instrument,
                    "comparison": f"{name} vs output history",
                    **paired_summary(scores[name], scores["output history"]),
                }
            )
        summaries.append(
            {
                "instrument": instrument,
                "comparison": "strictly lagged network vs lagged direct",
                **paired_summary(
                    scores["strictly lagged direct + network attention"],
                    scores["strictly lagged direct attention"],
                ),
            }
        )
    return pd.concat(score_tables, ignore_index=True), pd.DataFrame(summaries)


def permuted_current_attention(
    panel: pd.DataFrame,
    topics: list[str],
    weights: dict[int, np.ndarray],
    lagged_scores: pd.Series,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED + 10_000 + seed)
    shuffled = panel.copy()
    for meeting, row_index in shuffled.groupby("meeting").groups.items():
        indices = list(row_index)
        ordered = (
            shuffled.loc[indices]
            .set_index("topic")
            .reindex(topics)
        )
        permutation = rng.permutation(len(topics))
        paper_count = ordered["paper_count"].to_numpy(float)[permutation]
        actor_reach = ordered["current_actor_reach"].to_numpy(float)[permutation]
        network = weights[int(meeting)] @ paper_count
        values = pd.DataFrame(
            {
                "topic": topics,
                "paper_count": paper_count,
                "current_actor_reach": actor_reach,
                NETWORK: network,
            }
        ).set_index("topic")
        topic_order = shuffled.loc[indices, "topic"]
        shuffled.loc[indices, "paper_count"] = values.loc[
            topic_order, "paper_count"
        ].to_numpy()
        shuffled.loc[indices, "current_actor_reach"] = values.loc[
            topic_order, "current_actor_reach"
        ].to_numpy()
        shuffled.loc[indices, NETWORK] = values.loc[topic_order, NETWORK].to_numpy()

    candidate = forecast_meetings(
        shuffled,
        OUTPUT,
        model_features(OUTPUT, neighbor=NETWORK),
    )
    difference = (
        candidate.set_index("meeting")["allocation_log_score"] - lagged_scores
    ).to_numpy(float)
    return {"permutation": seed, "mean_difference": float(difference.mean())}


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = add_features()
    scores = score_models(panel)
    lagged_type_scores, lagged_type_summary = lagged_type_comparison(panel)
    lagged_type_scores.to_csv(LAGGED_TYPE_SCORES_PATH, index=False)
    lagged_type_summary.to_csv(LAGGED_TYPE_SUMMARY_PATH, index=False)

    future_panel = add_future_attention(panel).dropna(
        subset=["future_paper_count", "future_actor_reach", "future_network_attention"]
    )
    future_features = [
        *model_features(OUTPUT, include_focal=False),
        "paper_history_1",
        "actor_reach_1",
        "future_paper_count",
        "future_actor_reach",
        "future_network_attention",
    ]
    future_scores = forecast_meetings(
        future_panel, OUTPUT, future_features, test_end=46
    ).assign(model="future-meeting direct + network attention")
    comparable = pd.concat(
        [scores[scores["meeting"].le(46)], future_scores], ignore_index=True
    )
    all_scores = pd.concat([scores, future_scores], ignore_index=True)
    all_scores.to_csv(SCORES_PATH, index=False)

    comparisons = {}
    for candidate, baseline in (
        ("strictly lagged direct attention", "output history"),
        ("strictly lagged direct + network attention", "output history"),
        ("same-meeting direct attention", "output history"),
        ("same-meeting direct + network attention", "output history"),
        (
            "same-meeting direct + network attention",
            "strictly lagged direct + network attention",
        ),
    ):
        values = paired_difference(scores, candidate, baseline)
        comparisons[f"{candidate} vs {baseline}"] = {
            "mean_difference": float(values.mean()),
            "meetings_improved": int((values < 0).sum()),
            "meetings": int(len(values)),
            "moving_block_95_intervals": {
                str(block): moving_block_interval(values, block)
                for block in (1, 2, 3, 4)
            },
            "exact_sign_flip": exact_sign_flip(values),
        }

    future_values = paired_difference(
        comparable,
        "future-meeting direct + network attention",
        "output history",
    )
    comparisons["future-meeting direct + network attention vs output history"] = {
        "mean_difference": float(future_values.mean()),
        "meetings_improved": int((future_values < 0).sum()),
        "meetings": int(len(future_values)),
        "moving_block_95_intervals": {
            str(block): moving_block_interval(future_values, block)
            for block in (1, 2, 3, 4)
        },
        "exact_sign_flip": exact_sign_flip(future_values),
    }

    relations = paper_family_relations(variants()["fractional_multilabel"])
    topics = sorted(panel["topic"].unique())
    meetings = sorted(panel["meeting"].unique())
    phi = family_phi_by_meeting(relations, topics, meetings)
    weights = {
        meeting: local_weight_matrix(phi[meeting], k=PRIMARY_NETWORK_K)
        for meeting in meetings
    }
    lagged_scores = scores[
        scores["model"].eq("strictly lagged direct + network attention")
    ].set_index("meeting")["allocation_log_score"]
    if args.skip_null and NULL_PATH.exists():
        null = pd.read_csv(NULL_PATH)
    else:
        null = pd.DataFrame(
            Parallel(n_jobs=4)(
                delayed(permuted_current_attention)(
                    panel, topics, weights, lagged_scores, seed
                )
                for seed in range(PERMUTATIONS)
            )
        )
        null.to_csv(NULL_PATH, index=False)
    observed = comparisons[
        "same-meeting direct + network attention vs strictly lagged direct + network attention"
    ]["mean_difference"]
    null_summary = {
        "null_mean": float(null["mean_difference"].mean()),
        "null_95_interval": np.quantile(
            null["mean_difference"], [0.025, 0.975]
        ).tolist(),
        "observed_mean_difference": float(observed),
        "permutation_p_lower": float(
            (1 + np.sum(null["mean_difference"] <= observed)) / (1 + len(null))
        ),
        "permutations": int(len(null)),
    }
    summary = {
        "interpretation": (
            "Pseudoprospective rolling-origin checks using a database snapshot "
            "collected after all evaluated meetings; not a contemporaneously "
            "registered forecast."
        ),
        "comparisons": comparisons,
        "strictly_lagged_by_output_type": lagged_type_summary.to_dict(
            orient="records"
        ),
        "current_attention_category_label_null": null_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
