#!/usr/bin/env python3
"""Assemble the locality coefficient across every construction of the space.

Panel C of the dynamics figure exists to answer one objection: phi is estimated
from the same co-specialization the hazard model predicts, so locality could be
circular. The answer is that the coefficient survives constructions that break
the circularity. That answer is only legible if all the specifications sit on
one axis, which means they have to live in one table.

Provenance is tracked per row. The three space definitions are read from the
computed conditional-logit output; the two circularity checks are read from
``scripts/hazard_circularity_checks.py`` output, recomputed in-repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMPUTED_CSV = Path("output/fig15_hazard_space_pooled_coefficients.csv")
CIRCULARITY_CSV = Path("output/hazard_circularity_coefficients.csv")
OUT_CSV = Path("output/hazard_locality_specifications.csv")

# Space definitions, keyed by the `mode` column of the computed CSV.
SPACE_LABELS = {
    "cumulative_lagged": ("Cumulative-lagged space", "primary", 0),
    "instantaneous": ("Previous-window space", "space definition", 1),
    "aggregate": ("Pooled full-history space", "space definition", 2),
}

CIRCULARITY_LABELS = {
    "leave_one_actor_out": ("Leave-one-actor-out space", "circularity check", 3),
    "fractional": ("Fractional co-sponsorship counting", "circularity check", 4),
}


def build() -> pd.DataFrame:
    computed = pd.read_csv(COMPUTED_CSV)
    rows = []
    for _, row in computed.iterrows():
        mode = str(row["mode"])
        if mode not in SPACE_LABELS:
            continue
        label, group, order = SPACE_LABELS[mode]
        rows.append(
            {
                "label": label,
                "group": group,
                "order": order,
                "distance_coef": float(row["distance_coef"]),
                "distance_ci_low_95": float(row["distance_ci_low_95"]),
                "distance_ci_high_95": float(row["distance_ci_high_95"]),
                "odds_ratio_per_0_1": float(row["distance_odds_ratio_per_0_1"]),
                "n_rows": int(row["n_rows"]),
                "n_groups": int(row["n_groups"]),
                "source": str(COMPUTED_CSV),
            }
        )
    circularity = pd.read_csv(CIRCULARITY_CSV)
    for _, row in circularity.iterrows():
        mode = str(row["mode"])
        if mode not in CIRCULARITY_LABELS:
            continue
        label, group, order = CIRCULARITY_LABELS[mode]
        rows.append(
            {
                "label": label,
                "group": group,
                "order": order,
                "distance_coef": float(row["distance_coef"]),
                "distance_ci_low_95": float(row["distance_ci_low_95"]),
                "distance_ci_high_95": float(row["distance_ci_high_95"]),
                "odds_ratio_per_0_1": float(
                    row["distance_odds_ratio_per_0_1"]
                ),
                "n_rows": int(row["n_rows"]),
                "n_groups": int(row["n_groups"]),
                "source": str(CIRCULARITY_CSV),
            }
        )

    out = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    return out


def main() -> None:
    table = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
