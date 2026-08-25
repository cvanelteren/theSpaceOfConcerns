#!/usr/bin/env python3
"""Conditional-choice sensitivity to historically available concern labels.

The base conditional-choice panel contains the 45 labels found anywhere in the
archive. This script compares that construction with the historical risk-set
rule used for the manuscript's primary estimate and with a stricter prospective
rule:

1. a concern has appeared by the end of the current window; and
2. a concern has appeared by the end of the previous window.

The first rule admits concerns that emerge during the focal window. The second
is stricter and uses only labels already observed when that window begins.
"""

from __future__ import annotations

import json
import sys
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
    build_conditional_logit_panel,
    choose_period_col,
    load_data_with_fallback,
    topic_first_appearance,
)

OUT_CSV = Path("output/hazard_availability_sensitivity.csv")
OUT_JSON = Path("output/hazard_availability_sensitivity_meta.json")


def fit_variant(df: pd.DataFrame, name: str) -> dict[str, float | int | str]:
    group_counts = df.groupby("group")["adopted"].agg(["sum", "count"])
    valid_groups = group_counts[
        (group_counts["sum"] > 0) & (group_counts["sum"] < group_counts["count"])
    ].index
    fitted = df[df["group"].isin(valid_groups)].copy()
    result = ConditionalLogit(
        fitted["adopted"].astype(int),
        fitted[["distance", "topic_popularity"]],
        groups=fitted["group"],
    ).fit(disp=False, maxiter=200)
    coefficient = float(result.params["distance"])
    standard_error = float(result.bse["distance"])
    low = coefficient - 1.96 * standard_error
    high = coefficient + 1.96 * standard_error
    return {
        "risk_set": name,
        "n_rows": int(len(fitted)),
        "n_groups": int(fitted["group"].nunique()),
        "distance_coefficient": coefficient,
        "distance_standard_error": standard_error,
        "odds_ratio_per_0_1": float(np.exp(0.1 * coefficient)),
        "odds_ratio_per_0_1_ci_low_95": float(np.exp(0.1 * low)),
        "odds_ratio_per_0_1_ci_high_95": float(np.exp(0.1 * high)),
    }


def main() -> None:
    panel, panel_meta = build_conditional_logit_panel()
    counts, submitted, *_ = load_data_with_fallback()
    period_col = choose_period_col(submitted)
    submitted = submitted.copy()
    submitted[period_col] = pd.to_numeric(submitted[period_col], errors="coerce")
    first_period = pd.Series(
        topic_first_appearance(submitted, period_col)
    ).reindex(counts.index)
    if first_period.isna().any():
        missing = first_period[first_period.isna()].index.tolist()
        raise ValueError(f"Missing first-appearance year for topics: {missing}")

    cumulative = panel[panel["mode"] == "cumulative_lagged"].copy()
    cumulative["first_period"] = cumulative["topic"].map(
        first_period.astype(int).to_dict()
    )
    variants = {
        "all_eventual_topics": pd.Series(True, index=cumulative.index),
        "appeared_by_current_window_end": (
            cumulative["first_period"] <= cumulative["period_end"]
        ),
        "appeared_by_prior_window_end": (
            cumulative["first_period"] <= cumulative["period_end"] - 1
        ),
    }
    rows = [fit_variant(cumulative[mask], name) for name, mask in variants.items()]
    summary = pd.DataFrame(rows)
    meta = {
        "window_meetings": panel_meta["window_meetings"],
        "period_col": period_col,
        "space": "cumulative_lagged",
        "full_panel_rows": int(len(cumulative)),
        "future_topic_rows": int(
            (cumulative["first_period"] > cumulative["period_end"]).sum()
        ),
        "future_topic_share": float(
            (cumulative["first_period"] > cumulative["period_end"]).mean()
        ),
        "current_window_rule": (
            "topic has at least one archive document by current-window end"
        ),
        "prior_window_rule": (
            "topic has at least one archive document before current-window end"
        ),
        "interval_note": "model-based 95% intervals",
    }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
