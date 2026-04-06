#!/usr/bin/env python3
"""Supplementary Figure S07.

Re-estimates the local topic-adoption hazard under alternative concern-space
constructions using the same 5-year conditional-logit specification as the
main text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import hazard_conditional_logit as hcl

MODE_ORDER = ["aggregate", "instantaneous", "cumulative_lagged"]
MODE_LABELS = {
    "aggregate": "Aggregate (full history)",
    "instantaneous": "Instantaneous (window t-1)",
    "cumulative_lagged": "Cumulative-lagged (<= t-1)",
}
MODE_COLORS = {
    "aggregate": "#1f77b4",
    "instantaneous": "#d62728",
    "cumulative_lagged": "#2ca02c",
}

OUT_PDF = Path("figures/figS07_hazard_space_sensitivity.pdf")
OUT_PNG = Path("figures/figS07_hazard_space_sensitivity.png")
OUT_SUMMARY = Path("output/fig15_hazard_space_conditional_logit_summary.csv")
OUT_SUMMARY_META = Path("output/fig15_hazard_space_conditional_logit_meta.json")
OUT_MEMBER_PERIOD = Path("output/fig15_hazard_space_member_period_coefficients.csv")
OUT_PERIOD_SUMMARY = Path("output/fig15_hazard_space_period_summary.csv")
OUT_POOLED_COEF = Path("output/fig15_hazard_space_pooled_coefficients.csv")
OUT_META = Path("output/fig15_hazard_space_meta.json")


def load_or_build_summary() -> tuple[pd.DataFrame, dict]:
    if not hcl.OUT_CSV.exists() or not hcl.OUT_JSON.exists():
        hcl.main()
    summary_df = pd.read_csv(hcl.OUT_CSV)
    required_cols = {
        "distance_ci_low_95",
        "distance_ci_high_95",
        "distance_odds_ratio_per_0_1_ci_low_95",
        "distance_odds_ratio_per_0_1_ci_high_95",
    }
    if not required_cols.issubset(summary_df.columns):
        hcl.main()
        summary_df = pd.read_csv(hcl.OUT_CSV)
    meta = json.loads(hcl.OUT_JSON.read_text(encoding="utf-8"))
    return summary_df, meta


def main() -> None:
    summary_df, hazard_meta = load_or_build_summary()
    summary_df = summary_df.copy()
    summary_df["mode"] = pd.Categorical(
        summary_df["mode"], categories=MODE_ORDER, ordered=True
    )
    summary_df = summary_df.sort_values("mode").reset_index(drop=True)
    summary_df["estimand"] = "5y_member_period_conditional_logit_raw_1_minus_phi"
    summary_df["distance_definition"] = "raw_one_minus_max_phi_to_prior_portfolio"

    OUT_POOLED_COEF.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    summary_df.to_csv(OUT_MEMBER_PERIOD, index=False)
    summary_df.to_csv(OUT_PERIOD_SUMMARY, index=False)
    summary_df.to_csv(OUT_POOLED_COEF, index=False)

    meta = {
        "estimand": "5y_member_period_conditional_logit_raw_1_minus_phi",
        "group_definition": hazard_meta.get("group_definition"),
        "window_years": hazard_meta.get("window_years"),
        "distance_definition": hazard_meta.get("distance_definition"),
        "source_summary_csv": str(hcl.OUT_CSV),
        "modes": MODE_ORDER,
    }
    OUT_SUMMARY_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    y_pos = np.arange(len(summary_df))

    for i, row in summary_df.iterrows():
        mode = str(row["mode"])
        color = MODE_COLORS[mode]

        coef = float(row["distance_coef"])
        ci_low = float(row["distance_ci_low_95"])
        ci_high = float(row["distance_ci_high_95"])
        ax1.errorbar(
            x=coef,
            y=i,
            xerr=np.array([[coef - ci_low], [ci_high - coef]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.0,
            markersize=6.0,
        )

        odds = float(row["distance_odds_ratio_per_0_1"])
        odds_low = float(row["distance_odds_ratio_per_0_1_ci_low_95"])
        odds_high = float(row["distance_odds_ratio_per_0_1_ci_high_95"])
        ax2.errorbar(
            x=odds,
            y=i,
            xerr=np.array([[odds - odds_low], [odds_high - odds]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.0,
            markersize=6.0,
        )

    labels = [MODE_LABELS[str(mode)] for mode in summary_df["mode"].tolist()]

    ax1.axvline(0.0, color="#666666", ls="--", lw=1.0)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Conditional-logit distance coefficient")
    ax1.set_title("Distance Effect")
    ax1.grid(alpha=0.2, linewidth=0.6, axis="x")

    ax2.axvline(1.0, color="#666666", ls="--", lw=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Odds ratio per 0.1 increase in raw distance")
    ax2.set_title("Substantive Magnitude")
    ax2.grid(alpha=0.2, linewidth=0.6, axis="x")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(summary_df.to_string(index=False))
    print(f"\nWrote: {OUT_PDF}")
    print(f"Wrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_SUMMARY_META}")
    print(f"Wrote: {OUT_MEMBER_PERIOD}")
    print(f"Wrote: {OUT_PERIOD_SUMMARY}")
    print(f"Wrote: {OUT_POOLED_COEF}")
    print(f"Wrote: {OUT_META}")


if __name__ == "__main__":
    main()
