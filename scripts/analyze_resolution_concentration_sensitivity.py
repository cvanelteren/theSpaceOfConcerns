#!/usr/bin/env python3
"""Check whether the Resolution forecast gain is confined to broad meetings.

This is a descriptive, post hoc sensitivity. It does not identify whether legal
form, subject breadth, drafting practice, or consensus requirements produce the
output-type difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_resolution_attention_forecast import (
    BOOTSTRAP_DRAWS,
    OUTPUT_COLUMNS,
    SEED,
    TYPE_SCORES_PATH,
    add_features,
)


OUTDIR = Path("output/attention_output_signal")
MEETING_PATH = OUTDIR / "resolution_category_concentration_sensitivity.csv"
SUMMARY_PATH = OUTDIR / "resolution_category_concentration_sensitivity.json"


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    panel = add_features()
    scores = pd.read_csv(TYPE_SCORES_PATH)
    resolution = scores[scores["instrument"].eq("Resolution")].pivot(
        index="meeting", columns="model", values="allocation_log_score"
    )
    resolution["score_change"] = (
        resolution["history + direct + network attention"]
        - resolution["output history"]
    )

    rows = []
    output = OUTPUT_COLUMNS["Resolution"]
    for meeting, group in panel.groupby("meeting"):
        mass = group[output].to_numpy(float)
        if mass.sum() <= 0:
            continue
        shares = mass / mass.sum()
        positive = shares[shares > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        rows.append(
            {
                "meeting": int(meeting),
                "resolution_entropy": entropy,
                "effective_categories": float(np.exp(entropy)),
                "active_categories": int((shares > 0).sum()),
                "dominant_category_share": float(shares.max()),
            }
        )

    meeting = resolution[["score_change"]].join(
        pd.DataFrame(rows).set_index("meeting"), how="inner"
    ).reset_index()
    meeting = meeting[meeting["meeting"].between(29, 47)].copy()
    median_entropy = float(meeting["resolution_entropy"].median())
    meeting["entropy_half"] = np.where(
        meeting["resolution_entropy"].le(median_entropy), "lower", "higher"
    )
    meeting.to_csv(MEETING_PATH, index=False)

    rng = np.random.default_rng(SEED)
    lower = meeting.loc[meeting["entropy_half"].eq("lower"), "score_change"].to_numpy(float)
    higher = meeting.loc[meeting["entropy_half"].eq("higher"), "score_change"].to_numpy(float)
    lower_ci = bootstrap_mean(lower, rng)
    higher_ci = bootstrap_mean(higher, rng)
    difference_draws = (
        rng.choice(higher, size=(BOOTSTRAP_DRAWS, len(higher)), replace=True).mean(axis=1)
        - rng.choice(lower, size=(BOOTSTRAP_DRAWS, len(lower)), replace=True).mean(axis=1)
    )

    entropy = meeting["resolution_entropy"].to_numpy(float)
    change = meeting["score_change"].to_numpy(float)
    correlation = float(np.corrcoef(change, entropy)[0, 1])
    correlation_draws = []
    for indices in rng.integers(0, len(meeting), size=(BOOTSTRAP_DRAWS, len(meeting))):
        if np.std(change[indices]) > 0 and np.std(entropy[indices]) > 0:
            correlation_draws.append(float(np.corrcoef(change[indices], entropy[indices])[0, 1]))

    summary = {
        "scope": "post hoc descriptive sensitivity across Resolution test meetings ATCM 29-47",
        "meetings": int(len(meeting)),
        "median_entropy": median_entropy,
        "lower_entropy_half": {
            "meetings": int(len(lower)),
            "mean_score_change": float(lower.mean()),
            "meeting_bootstrap_95_interval": list(lower_ci),
        },
        "higher_entropy_half": {
            "meetings": int(len(higher)),
            "mean_score_change": float(higher.mean()),
            "meeting_bootstrap_95_interval": list(higher_ci),
        },
        "higher_minus_lower": {
            "mean_difference": float(higher.mean() - lower.mean()),
            "meeting_bootstrap_95_interval": [
                float(value) for value in np.quantile(difference_draws, [0.025, 0.975])
            ],
        },
        "score_change_entropy_correlation": {
            "pearson_r": correlation,
            "meeting_bootstrap_95_interval": [
                float(value)
                for value in np.quantile(correlation_draws, [0.025, 0.975])
            ],
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {MEETING_PATH}")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

