#!/usr/bin/env python3
"""Prospective locality under three observable definitions of portfolio entry.

Every specification uses the same five-meeting chronology, a concern space
built only from meetings already observed, and concerns that had appeared by
the end of the preceding ATCM.  The outcomes differ:

``relative_crossing``
    RPA crosses from below 1 to at least 1 in the next rolling window.
``new_document_after_gap``
    The actor submits on the concern in the focal meeting after no papers on it
    in the preceding five-meeting window.
``crossing_with_count_increase``
    RPA crosses 1 and the actor's paper count rises between windows.

The latter two checks ensure that locality is not produced only by a changing
system-wide denominator in the relative-share measure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    load_data_with_fallback,
    phi_from_interaction,
    sanitize_periods,
    topic_first_appearance,
)
from utils import get_rca  # noqa: E402

OUT_CSV = ROOT / "output/entry_definition_robustness.csv"
OUT_JSON = ROOT / "output/entry_definition_robustness.json"
OUT_WINDOW_CSV = ROOT / "output/entry_window_sensitivity.csv"


def fit(panel: pd.DataFrame, outcome: str) -> dict[str, float | int | str]:
    subset = panel[panel["eligible_" + outcome]].copy()
    subset["adopted"] = subset[outcome].astype(int)
    counts = subset.groupby("group")["adopted"].agg(["sum", "count"])
    informative = counts[
        counts["sum"].gt(0) & counts["sum"].lt(counts["count"])
    ].index
    subset = subset[subset["group"].isin(informative)].copy()
    result = ConditionalLogit(
        subset["adopted"],
        subset[["distance", "topic_popularity"]],
        groups=subset["group"],
    ).fit(disp=False, maxiter=300)
    beta = float(result.params["distance"])
    se = float(result.bse["distance"])
    return {
        "entry_definition": outcome,
        "events": int(subset["adopted"].sum()),
        "actor_meeting_choice_sets": int(subset["group"].nunique()),
        "risk_rows": int(len(subset)),
        "distance_beta": beta,
        "distance_se": se,
        "odds_ratio_per_0_1_distance": float(np.exp(0.1 * beta)),
        "odds_ratio_ci_low_95": float(np.exp(0.1 * (beta - 1.96 * se))),
        "odds_ratio_ci_high_95": float(np.exp(0.1 * (beta + 1.96 * se))),
    }


def build_panel(
    window_meetings: int = WINDOW_MEETINGS,
) -> tuple[pd.DataFrame, dict]:
    counts, submitted, members_raw, topics_raw = load_data_with_fallback()
    period_col = choose_period_col(submitted)
    submitted = sanitize_periods(submitted, period_col)
    topics = counts.index.tolist()
    members = counts.columns.tolist()
    period_min = int(submitted[period_col].min())
    period_max = int(submitted[period_col].max())
    periods = build_periods(period_min, period_max, window_meetings)
    first = topic_first_appearance(submitted, period_col)

    interaction_by_period = []
    active_by_period = []
    for start, end in periods:
        interaction = build_window_interaction(
            submitted,
            period_col,
            start,
            end,
            set(members_raw),
            set(topics_raw),
            topics,
            members,
        )
        interaction_by_period.append(interaction)
        active_by_period.append(get_rca(interaction).ge(RCA_THRESHOLD))

    rows: list[dict] = []
    for index in range(1, len(periods)):
        prev_end = int(periods[index - 1][1])
        focal_meeting = int(periods[index][1])
        prior_counts = interaction_by_period[index - 1]
        current_counts = interaction_by_period[index]
        prior_active = active_by_period[index - 1]
        current_active = active_by_period[index]
        cumulative = build_window_interaction(
            submitted,
            period_col,
            period_min,
            prev_end,
            set(members_raw),
            set(topics_raw),
            topics,
            members,
        )
        phi = phi_from_interaction(cumulative, topics)
        popularity = prior_active.sum(axis=1) / max(len(members), 1)

        for member in members:
            held = prior_active[member].to_numpy(dtype=bool)
            if not held.any():
                continue
            distances = 1.0 - phi[:, np.flatnonzero(held)].max(axis=1)
            before = prior_counts[member].to_numpy(dtype=float)
            after = current_counts[member].to_numpy(dtype=float)
            current = current_active[member].to_numpy(dtype=bool)
            for topic_index, topic in enumerate(topics):
                if first.get(topic, focal_meeting + 1) > prev_end:
                    continue
                not_specialized = not bool(held[topic_index])
                no_recent_document = before[topic_index] <= 0
                relative_crossing = not_specialized and bool(current[topic_index])
                new_document = no_recent_document and after[topic_index] > 0
                crossing_with_increase = (
                    relative_crossing and after[topic_index] > before[topic_index]
                )
                rows.append(
                    {
                        "group": f"{member}::{focal_meeting}",
                        "member": member,
                        "meeting": focal_meeting,
                        "topic": topic,
                        "distance": float(distances[topic_index]),
                        "topic_popularity": float(popularity.loc[topic]),
                        "eligible_relative_crossing": not_specialized,
                        "relative_crossing": relative_crossing,
                        "eligible_new_document_after_gap": no_recent_document,
                        "new_document_after_gap": new_document,
                        "eligible_crossing_with_count_increase": not_specialized,
                        "crossing_with_count_increase": crossing_with_increase,
                    }
                )

    panel = pd.DataFrame(rows)
    meta = {
        "period_col": period_col,
        "window_meetings": window_meetings,
        "space": "cumulative_lagged_through_preceding_meeting",
        "availability": "concern appeared by end of preceding meeting",
        "specialization_threshold": "RPA >= 1",
        "covariates": ["distance", "previous_window_holder_share"],
    }
    return panel, meta


def main() -> None:
    panel, meta = build_panel()
    outcomes = [
        "relative_crossing",
        "new_document_after_gap",
        "crossing_with_count_increase",
    ]
    summary = pd.DataFrame([fit(panel, outcome) for outcome in outcomes])

    event_sets = {
        outcome: set(
            map(
                tuple,
                panel.loc[panel[outcome], ["member", "meeting", "topic"]].to_numpy(),
            )
        )
        for outcome in outcomes
    }
    overlap = {
        f"{left}__{right}": {
            "intersection": len(event_sets[left] & event_sets[right]),
            "union": len(event_sets[left] | event_sets[right]),
            "jaccard": (
                len(event_sets[left] & event_sets[right])
                / max(len(event_sets[left] | event_sets[right]), 1)
            ),
        }
        for left in outcomes
        for right in outcomes
        if left < right
    }
    payload = {**meta, "overlap": overlap, "rows": summary.to_dict(orient="records")}

    window_rows = []
    for window in (3, 5, 7, 10):
        window_panel, _ = (
            (panel, meta) if window == WINDOW_MEETINGS else build_panel(window)
        )
        for outcome in ("relative_crossing", "new_document_after_gap"):
            row = fit(window_panel, outcome)
            row["window_meetings"] = window
            window_rows.append(row)
    window_summary = pd.DataFrame(window_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    window_summary.to_csv(OUT_WINDOW_CSV, index=False)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(overlap, indent=2))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_WINDOW_CSV}")


if __name__ == "__main__":
    main()
