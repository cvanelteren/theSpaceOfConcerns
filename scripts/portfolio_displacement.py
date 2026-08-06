#!/usr/bin/env python3
r"""Portfolio displacement in the space of concerns, without any mode partition.

The mode-transition statistic ("82% of transitions stay in the same mode") is a
property of a three-way cut that the manuscript itself describes as an
interpretive division of a continuum. This computes the same claim -- movement
is small -- directly in phi-space, where nothing has to be discretized.

For each actor and each pair of adjacent 5-year windows we take the specialized
portfolio S_t = {topics with RPA >= 1} and measure

    D_at = mean_{i in A_t} min_{j in S_{a,t-1}} ( 1 - phi(i, j) )

which is the same 1 - max-proximity distance the adoption hazard uses, averaged
over a set of topics instead of evaluated per candidate. The null redraws a set
of the same size uniformly from the topics the actor did not already hold, so it
holds breadth fixed and varies only *where* the actor went.

A_t is reported two ways. The primary measure takes A_t = S_t \ S_{t-1}, the
topics genuinely newly specialized in, so displacement cannot be inflated by an
actor simply keeping what it had -- that is persistence, measured separately.
The secondary measure takes A_t = S_t, the whole standing portfolio, which is
the closer analogue of the mode-transition statistic since that too counts
staying put as staying put.
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

from hazard_conditional_logit import (  # noqa: E402
    RCA_THRESHOLD,
    WINDOW_YEARS,
    build_periods,
    build_window_interaction,
    load_data_with_fallback,
    phi_from_interaction,
    sanitize_years,
)

N_NULL_DRAWS = 200
SEED = 1991

OUT_CSV = Path("output/portfolio_displacement.csv")
OUT_JSON = Path("output/portfolio_displacement_meta.json")
OUT_MOVES = Path("output/portfolio_displacement_moves.csv")


def _displacement(new_idx: np.ndarray, prior_idx: np.ndarray, phi: np.ndarray) -> float:
    """Mean over the new portfolio of distance to the nearest prior holding."""
    if new_idx.size == 0 or prior_idx.size == 0:
        return float("nan")
    return float(np.mean(1.0 - phi[np.ix_(new_idx, prior_idx)].max(axis=1)))


def _nearest_prior(
    added_idx: np.ndarray, prior_idx: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each newly entered topic, the prior holding it is closest to.

    This is the same quantity the displacement average is built from, kept
    per-move rather than collapsed, so a figure can draw the individual step an
    actor took instead of only its mean length. Returns (source index into the
    topic list, distance) aligned with ``added_idx``.
    """
    block = phi[np.ix_(added_idx, prior_idx)]
    nearest = prior_idx[block.argmax(axis=1)]
    return nearest, 1.0 - block.max(axis=1)


def build_displacement_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback()
    year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
    submitted_df = sanitize_years(submitted_df, year_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    periods = build_periods(
        int(submitted_df[year_col].min()),
        int(submitted_df[year_col].max()),
        WINDOW_YEARS,
    )

    active_by_period = []
    for start, end in periods:
        interaction = build_window_interaction(
            submitted_df=submitted_df,
            year_col=year_col,
            year_start=int(start),
            year_end=int(end),
            all_members_raw=set(members_raw),
            all_topics_raw=set(topics_raw),
            topics_order=topics,
            members_order=members,
        )
        active_by_period.append(get_active(interaction))

    # Pooled geometry, matching the descriptive panels elsewhere in the paper.
    phi = phi_from_interaction(
        counts_df.reindex(index=topics, columns=members, fill_value=0), topics
    )
    n_topics = len(topics)
    rng = np.random.default_rng(SEED)

    rows = []
    moves = []
    for t in range(1, len(periods)):
        prev_active = active_by_period[t - 1]
        curr_active = active_by_period[t]
        for member in members:
            prior_idx = np.flatnonzero(prev_active[member].to_numpy())
            new_idx = np.flatnonzero(curr_active[member].to_numpy())
            if prior_idx.size == 0 or new_idx.size == 0:
                continue

            added_idx = np.setdiff1d(new_idx, prior_idx)
            available = np.setdiff1d(np.arange(n_topics), prior_idx)
            retained = np.intersect1d(new_idx, prior_idx).size / new_idx.size

            row = {
                "member": member,
                "period_end": int(periods[t][1]),
                "prior_breadth": int(prior_idx.size),
                "breadth": int(new_idx.size),
                "n_added": int(added_idx.size),
                "retained_fraction": float(retained),
                "displacement_portfolio": _displacement(new_idx, prior_idx, phi),
                "displacement": float("nan"),
                "null_displacement_mean": float("nan"),
                "null_displacement_p05": float("nan"),
            }

            # Primary measure: only the topics the actor actually moved into.
            if added_idx.size and available.size >= added_idx.size:
                null_draws = np.array(
                    [
                        _displacement(
                            rng.choice(available, size=added_idx.size, replace=False),
                            prior_idx,
                            phi,
                        )
                        for _ in range(N_NULL_DRAWS)
                    ]
                )
                row["displacement"] = _displacement(added_idx, prior_idx, phi)
                row["null_displacement_mean"] = float(np.mean(null_draws))
                row["null_displacement_p05"] = float(np.percentile(null_draws, 5))

                source_idx, step_distance = _nearest_prior(added_idx, prior_idx, phi)
                for dest, source, distance in zip(added_idx, source_idx, step_distance):
                    moves.append(
                        {
                            "member": member,
                            "period_end": int(periods[t][1]),
                            "from_topic": topics[int(source)],
                            "to_topic": topics[int(dest)],
                            "distance": float(distance),
                        }
                    )

            rows.append(row)

    panel = pd.DataFrame(rows)
    move_panel = pd.DataFrame(moves)
    panel["delta_vs_null"] = panel["displacement"] - panel["null_displacement_mean"]
    moved = panel.dropna(subset=["displacement"])
    sign_test = stats.binomtest(
        int((moved["delta_vs_null"] < 0).sum()), int(len(moved)), 0.5, alternative="greater"
    )
    meta = {
        "window_years": WINDOW_YEARS,
        "rca_threshold": RCA_THRESHOLD,
        "n_null_draws": N_NULL_DRAWS,
        "seed": SEED,
        "phi": "pooled full-history space of concerns",
        "distance": "1 - max phi to prior specialized portfolio, averaged over a topic set",
        "primary_set": "newly specialized topics only (S_t minus S_{t-1})",
        "null": "size-matched uniform redraw from topics not already held",
        "n_transitions": int(len(panel)),
        "n_transitions_with_entry": int(len(moved)),
        "n_members": int(panel["member"].nunique()),
        "share_no_entry": float(1.0 - len(moved) / max(len(panel), 1)),
        "observed_median": float(moved["displacement"].median()),
        "null_median": float(moved["null_displacement_mean"].median()),
        "share_below_null_p05": float(
            (moved["displacement"] < moved["null_displacement_p05"]).mean()
        ),
        "share_nearer_than_null": float((moved["delta_vs_null"] < 0).mean()),
        "median_delta_vs_null": float(moved["delta_vs_null"].median()),
        "sign_test_p": float(sign_test.pvalue),
        "wilcoxon_p": float(
            stats.wilcoxon(moved["delta_vs_null"], alternative="less").pvalue
        ),
        "portfolio_measure_median": float(panel["displacement_portfolio"].median()),
        "n_moves": int(len(move_panel)),
    }
    return panel, move_panel, meta


def get_active(interaction: pd.DataFrame) -> pd.DataFrame:
    from utils import get_rca

    return get_rca(interaction) >= RCA_THRESHOLD


def main() -> None:
    panel, move_panel, meta = build_displacement_panel()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_CSV, index=False)
    move_panel.to_csv(OUT_MOVES, index=False)
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_CSV} ({len(panel)} transitions)")
    print(f"Wrote {OUT_MOVES} ({len(move_panel)} individual moves)")
    for key, value in meta.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
