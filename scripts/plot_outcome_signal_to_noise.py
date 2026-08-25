#!/usr/bin/env python3
"""Plot the formal-output signal-to-noise diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle


OUTDIR = ROOT / "output" / "outcome_linkage"
FIGURE = ROOT / "figures" / "exploratory_outcome_signal_to_noise"

COLORS = {
    "full": "#9B4F5C",
    "exact": "#7A6396",
    "nearby": "#3F8780",
    "neutral": "#657A9A",
}


def pick(
    table: pd.DataFrame, specification: str, term: str
) -> pd.Series:
    rows = table[
        table["specification"].eq(specification) & table["term"].eq(term)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {specification!r}, {term!r}")
    return rows.iloc[0]


def main() -> None:
    models = pd.read_csv(OUTDIR / "outcome_signal_to_noise_models.csv")
    ranks = pd.read_csv(OUTDIR / "outcome_signal_to_noise_summary.csv")

    model_specs = [
        ("Exact, nearby, and title", "exact_nearby_and_title"),
        ("Add paper type", "recommended_add_paper_type"),
        ("Add paper type and\nconcern connectedness", "add_paper_type_and_hubness"),
        ("Only papers in a\nknown lineage", "known_lineage_controls"),
        ("Only high-confidence\noutcome labels", "high_consensus_confidence"),
        ("Include secondary\noutcome concern", "primary_0.75_secondary_0.25"),
    ]
    y_model = np.arange(len(model_specs))[::-1]

    rank_specs = [
        ("All meeting papers", "all_same_meeting_papers"),
        ("Same paper type", "same_paper_type_controls"),
        ("Only known-lineage papers", "papers_known_to_enter_any_lineage"),
    ]
    rank_terms = [
        ("Full map", "expected_proximity", COLORS["full"]),
        ("Exact concern", "same_concern_mass", COLORS["exact"]),
    ]
    nearby_specs = {
        "all_same_meeting_papers": "nonexact_candidates_only",
        "same_paper_type_controls": "nonexact_same_paper_type_controls",
        "papers_known_to_enter_any_lineage": "nonexact_papers_known_to_enter_any_lineage",
    }

    fig, axs = uplt.subplots(ncols=2, refwidth=3.15, refheight=3.05, share=False)

    ax = axs[0]
    offsets = {"exact": 0.11, "nearby": -0.11}
    for key, term, label in (
        ("exact", "same_concern_mass", "Exact concern"),
        ("nearby", "related_concern_proximity", "Nearby concerns"),
    ):
        estimates, lower, upper = [], [], []
        for _, specification in model_specs:
            row = pick(models, specification, term)
            estimates.append(row.estimate)
            lower.append(row.ci_low)
            upper.append(row.ci_high)
        estimates = np.asarray(estimates)
        errors = np.vstack([estimates - lower, np.asarray(upper) - estimates])
        ax.errorbar(
            estimates,
            y_model + offsets[key],
            xerr=errors,
            fmt="o",
            ms=6.5,
            lw=1.8,
            capsize=3,
            color=COLORS[key],
            label=label,
            zorder=3,
        )
    ax.axvline(1, color="#9AA7B5", lw=1.2, ls="--", zorder=1)
    ax.format(
        title="Which association survives cleaner comparisons?",
        xlabel="Odds ratio for documented contribution\n(per 1-SD increase)",
        xscale="log",
        xlim=(0.58, 2.75),
        xticks=[0.75, 1, 1.5, 2, 2.5],
        xticklabels=["0.75", "1", "1.5", "2", "2.5"],
        yticks=y_model,
        yticklabels=[label for label, _ in reversed(model_specs)],
        grid=False,
    )
    ax.legend(loc="b", frame=False, ncols=2)

    ax = axs[1]
    y_rank = np.arange(len(rank_specs))[::-1]
    offsets_rank = {"Full map": 0.18, "Exact concern": 0.0, "Nearby only": -0.18}
    for label, term, color in rank_terms:
        estimates, lower, upper = [], [], []
        for _, specification in rank_specs:
            row = pick(ranks, specification, term)
            estimates.append(row.estimate)
            lower.append(row.ci_low)
            upper.append(row.ci_high)
        estimates = np.asarray(estimates)
        errors = np.vstack([estimates - lower, np.asarray(upper) - estimates])
        ax.errorbar(
            estimates,
            y_rank + offsets_rank[label],
            xerr=errors,
            fmt="o",
            ms=6.5,
            lw=1.8,
            capsize=3,
            color=color,
            label=label,
            zorder=3,
        )

    estimates, lower, upper = [], [], []
    for _, specification in rank_specs:
        row = pick(
            ranks,
            nearby_specs[specification],
            "related_concern_proximity",
        )
        estimates.append(row.estimate)
        lower.append(row.ci_low)
        upper.append(row.ci_high)
    estimates = np.asarray(estimates)
    errors = np.vstack([estimates - lower, np.asarray(upper) - estimates])
    ax.errorbar(
        estimates,
        y_rank + offsets_rank["Nearby only"],
        xerr=errors,
        fmt="o",
        ms=6.5,
        lw=1.8,
        capsize=3,
        color=COLORS["nearby"],
        label="Nearby only",
        zorder=3,
    )
    ax.axvline(0.5, color="#9AA7B5", lw=1.2, ls="--", zorder=1)
    ax.format(
        title="What ranks a linked paper above a control?",
        xlabel="Linked paper ranks closer (%)",
        xlim=(0.36, 0.74),
        xticks=[0.4, 0.5, 0.6, 0.7],
        xticklabels=["40%", "50%", "60%", "70%"],
        yticks=y_rank,
        yticklabels=[label for label, _ in reversed(rank_specs)],
        grid=False,
    )
    ax.legend(loc="b", frame=False, ncols=3)

    axs.format(abc="a", abcloc="ul", grid=False)
    figstyle.apply_typography(axs)
    fig.save(FIGURE.with_suffix(".pdf"))
    fig.save(FIGURE.with_suffix(".png"), dpi=360)
    uplt.close(fig)


if __name__ == "__main__":
    main()
