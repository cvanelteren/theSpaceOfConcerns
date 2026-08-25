#!/usr/bin/env python3
"""Nested rolling evaluation of attention forecasts by ATS output type.

For every outer ATCM, model settings are chosen only from forecasts of earlier
meetings.  The selected model is then refitted on all meetings available before
the outer ATCM and scored once on that ATCM.  This keeps the full ATCM 29--47
evaluation while preventing its outcomes from selecting the reported horizons,
regularization, or neighbourhood size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.analyze_resolution_attention_forecast import (
    ALPHAS,
    ATTENTION_WINDOWS,
    HISTORY_WINDOWS,
    NETWORK_K,
    OUTPUT_COLUMNS,
    OUTDIR,
    TEST_END,
    TEST_START,
    add_features,
    forecast_meetings,
    instrument_contrasts,
    model_features,
    paired_summary,
)


INNER_START = 25
N_JOBS = 8
SCORES_PATH = OUTDIR / "nested_type_meeting_scores.csv"
SUMMARY_PATH = OUTDIR / "nested_type_summary.csv"
SELECTED_PATH = OUTDIR / "nested_selected_specifications.csv"
CONTRAST_PATH = OUTDIR / "nested_type_contrast.csv"


@dataclass(frozen=True)
class Candidate:
    model: str
    history: int
    attention: int | None
    k: int | None
    alpha: float

    @property
    def key(self) -> str:
        return (
            f"{self.model}|h={self.history}|a={self.attention}"
            f"|k={self.k}|alpha={self.alpha:g}"
        )


def candidates(model: str) -> list[Candidate]:
    rows: list[Candidate] = []
    if model == "output history":
        for history in HISTORY_WINDOWS:
            for alpha in ALPHAS:
                rows.append(Candidate(model, history, None, None, alpha))
    elif model == "history + direct attention":
        for history in HISTORY_WINDOWS:
            for attention in ATTENTION_WINDOWS:
                for alpha in ALPHAS:
                    rows.append(Candidate(model, history, attention, None, alpha))
    elif model == "history + direct + nearby attention":
        for history in HISTORY_WINDOWS:
            for attention in ATTENTION_WINDOWS:
                for k in NETWORK_K:
                    for alpha in ALPHAS:
                        rows.append(Candidate(model, history, attention, k, alpha))
    else:
        raise ValueError(f"Unknown model: {model}")
    return rows


def features(output: str, candidate: Candidate) -> list[str]:
    if candidate.model == "output history":
        return model_features(output, history=candidate.history, include_focal=False)
    neighbor = None if candidate.k is None else f"neighbor_papers_k{candidate.k}"
    if candidate.attention is None:
        raise AssertionError("Attention models require an attention horizon")
    return model_features(
        output,
        history=candidate.history,
        attention=candidate.attention,
        neighbor=neighbor,
    )


def evaluate_candidate(
    panel: pd.DataFrame, output: str, candidate: Candidate
) -> pd.DataFrame:
    table = forecast_meetings(
        panel,
        output,
        features(output, candidate),
        alpha=candidate.alpha,
        test_start=INNER_START,
        test_end=TEST_END,
    )
    return table.assign(
        candidate=candidate.key,
        history=candidate.history,
        attention=candidate.attention,
        k=candidate.k,
        alpha=candidate.alpha,
        model=candidate.model,
    )


def nested_model_scores(
    panel: pd.DataFrame, instrument: str, output: str, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = candidates(model)
    evaluated = pd.concat(
        Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_candidate)(panel, output, candidate)
            for candidate in grid
        ),
        ignore_index=True,
    )
    outer_rows = []
    selected_rows = []
    for outer in range(TEST_START, TEST_END + 1):
        inner = evaluated[
            evaluated["meeting"].ge(INNER_START)
            & evaluated["meeting"].lt(outer)
        ]
        means = (
            inner.groupby("candidate", as_index=False)["allocation_log_score"]
            .mean()
            .sort_values(["allocation_log_score", "candidate"])
        )
        if means.empty:
            raise AssertionError(f"No inner forecasts available before ATCM {outer}")
        chosen = str(means.iloc[0]["candidate"])
        row = evaluated[
            evaluated["candidate"].eq(chosen)
            & evaluated["meeting"].eq(outer)
        ]
        if len(row) != 1:
            raise AssertionError(
                f"Expected one outer score for {instrument}, {model}, ATCM {outer}"
            )
        record = row.iloc[0].to_dict()
        outer_rows.append(
            {
                "instrument": instrument,
                "meeting": outer,
                "model": model,
                "allocation_log_score": float(record["allocation_log_score"]),
            }
        )
        selected_rows.append(
            {
                "instrument": instrument,
                "meeting": outer,
                "model": model,
                "candidate": chosen,
                "inner_meetings": int(outer - INNER_START),
                "inner_mean_score": float(means.iloc[0]["allocation_log_score"]),
                "history": int(record["history"]),
                "attention": (
                    np.nan if pd.isna(record["attention"]) else int(record["attention"])
                ),
                "k": np.nan if pd.isna(record["k"]) else int(record["k"]),
                "alpha": float(record["alpha"]),
            }
        )
    return pd.DataFrame(outer_rows), pd.DataFrame(selected_rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instrument in OUTPUT_COLUMNS:
        tables = {
            model: group[["meeting", "allocation_log_score"]]
            for model, group in scores[scores["instrument"].eq(instrument)].groupby(
                "model"
            )
        }
        baseline = tables["output history"]
        for model in (
            "history + direct attention",
            "history + direct + nearby attention",
        ):
            rows.append(
                {
                    "instrument": instrument,
                    "comparison": f"{model} vs output history",
                    **paired_summary(tables[model], baseline),
                }
            )
        rows.append(
            {
                "instrument": instrument,
                "comparison": "nearby attention vs direct attention",
                **paired_summary(
                    tables["history + direct + nearby attention"],
                    tables["history + direct attention"],
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = add_features()
    score_tables = []
    selected_tables = []
    for instrument, output in OUTPUT_COLUMNS.items():
        for model in (
            "output history",
            "history + direct attention",
            "history + direct + nearby attention",
        ):
            print(f"evaluating {instrument}: {model}", flush=True)
            scores, selected = nested_model_scores(
                panel, instrument, output, model
            )
            score_tables.append(scores)
            selected_tables.append(selected)
    scores = pd.concat(score_tables, ignore_index=True)
    selected = pd.concat(selected_tables, ignore_index=True)
    summary = summarize(scores)
    scores.to_csv(SCORES_PATH, index=False)
    selected.to_csv(SELECTED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    instrument_contrasts(scores).to_csv(CONTRAST_PATH, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
