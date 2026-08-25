#!/usr/bin/env python3
"""Supplementary diagnostics for within-meeting outcome alignment.

Two panels, one claim each.

  A. Concern proximity ranks the paper that reached adoption above a randomly
     chosen paper submitted to the same meeting. Papers that only joined the
     discussion are ranked at chance.
  B. That discrimination is not a restatement of the Secretariat's concern
     label. Holding constant the probability that the instrument is about the
     paper's own concern, proximity through related concerns still identifies
     documented contributors -- and neither term does so for discussion-only
     papers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "outcome_linkage"
FIGDIR = ROOT / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle


ADOPTION_COLOR = figstyle.ADOPTION
DISCUSSION_COLOR = figstyle.DISCUSSION
GEOMETRY_COLOR = figstyle.NEARBY
LABEL_COLOR = figstyle.FOCAL


def save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.save(FIGDIR / f"{stem}.pdf")
    fig.save(FIGDIR / f"{stem}.png", dpi=260)
    uplt.close(fig)


def errorbar_dot(ax, x, y, low, high, color, size=56) -> None:
    ax.errorbar(
        [x], [y], xerr=np.array([[x - low], [high - x]]),
        fmt="none", ecolor=color, elinewidth=2.0, capsize=3.0, zorder=2,
    )
    ax.scatter(
        [x], [y], s=size, color=color, edgecolor="white", linewidth=0.65, zorder=3,
    )


def make_figure() -> None:
    auc = pd.read_csv(OUTDIR / "space_discrimination_auc_consensus.csv").set_index("comparison")
    race = pd.read_csv(OUTDIR / "space_discrimination_race_consensus.csv")

    fig, axes = uplt.subplots(ncols=2, refwidth=3.05, refaspect=0.92, share=False, wspace=12.5)
    ax_a, ax_b = axes

    # --- A. Discrimination within the meeting's opportunity set -----------
    rows_a = [
        (
            "Adoption-linked papers, full proximity",
            "Linked to adoption",
            ADOPTION_COLOR,
        ),
        (
            "Discussion-only papers, full proximity",
            "Discussion only",
            DISCUSSION_COLOR,
        ),
        (
            "Adoption-linked papers, excluding site administration",
            "Without routine\nsite files",
            ADOPTION_COLOR,
        ),
        (
            "Adoption-linked papers, title overlap below 0.15",
            "Low title overlap",
            ADOPTION_COLOR,
        ),
        (
            "Adoption-linked papers, off-label geometry, exact matches removed",
            "Different concern\nlabel",
            GEOMETRY_COLOR,
        ),
        (
            "Discussion-only papers, off-label geometry, exact matches removed",
            "Discussion, different\nconcern label",
            DISCUSSION_COLOR,
        ),
    ]
    present = [row for row in rows_a if row[0] in auc.index]
    y = np.arange(len(present))[::-1]
    for yi, (key, label, color) in zip(y, present):
        record = auc.loc[key]
        errorbar_dot(
            ax_a, record["auc"], yi, record["ci_low"], record["ci_high"], color
        )
        ax_a.text(
            0.855, yi, f"n={int(record['outcomes'])}", ha="right", va="center",
            fontsize=figstyle.FS_ANNOT, color=figstyle.MUTED,
        )
    ax_a.axvline(0.5, color=figstyle.REFERENCE, lw=0.9, ls="--")
    ax_a.text(
        0.502, y.max() + 0.5, "Chance", color=figstyle.MUTED,
        fontsize=figstyle.FS_ANNOT, ha="left", va="center",
    )
    ax_a.format(
        title="The pattern is specific to adoption",
        xlabel="Probability the linked paper ranks higher",
        xlim=(0.40, 0.86), xformatter="{x:.0%}", xlocator=[0.5, 0.6, 0.7, 0.8],
        ylim=(y.min() - 0.7, y.max() + 0.8),
        yticks=y, yticklabels=[label for _, label, _ in present],
        grid=False, titlesize=figstyle.FS_TITLE,
    )

    # --- B. Geometry against label, within meetings -----------------------
    term_label = {
        "same_concern_mass": "Same concern",
        "related_concern_proximity": "Nearest concerns",
    }
    term_color = {
        "same_concern_mass": LABEL_COLOR,
        "related_concern_proximity": GEOMETRY_COLOR,
    }
    blocks = [
        ("adoption_linked_papers", "Papers linked to adoption"),
        (
            "adoption_linked_papers_title_overlap_controlled",
            "After accounting for title overlap",
        ),
        ("discussion_only_papers", "Papers linked only to discussion"),
    ]
    positions: list[tuple[float, str, str]] = []
    headings: list[tuple[float, str]] = []
    cursor = 0.0
    for specification, heading in blocks:
        headings.append((cursor, heading))
        cursor -= 0.8
        for term in ("same_concern_mass", "related_concern_proximity"):
            positions.append((cursor, specification, term))
            cursor -= 1.0
        cursor -= 0.6

    for yi, specification, term in positions:
        record = race[
            race["specification"].eq(specification) & race["term"].eq(term)
        ]
        if record.empty:
            continue
        record = record.iloc[0]
        # Prefer the outcome-cluster bootstrap interval where it converged.
        low = record.get("outcome_bootstrap_ci_low")
        high = record.get("outcome_bootstrap_ci_high")
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = record["ci_low"], record["ci_high"]
        color = term_color[term]
        if specification == "discussion_only_papers":
            color = DISCUSSION_COLOR
        effect = 100 * (record["odds_ratio"] - 1)
        low_effect = 100 * (low - 1)
        high_effect = 100 * (high - 1)
        errorbar_dot(ax_b, effect, yi, low_effect, high_effect, color)
        ax_b.text(
            137, yi, f"{effect:.0f}%", ha="right", va="center",
            fontsize=figstyle.FS_ANNOT, color=figstyle.MUTED,
        )
    ax_b.axvline(0.0, color=figstyle.REFERENCE, lw=0.9, ls="--")
    ax_b.format(
        title="The concern label explains more than nearby concerns",
        xlabel="Change in odds of a documented link (%)",
        xlim=(-25, 145), xlocator=[0, 50, 100],
        ylim=(cursor + 0.35, 0.65),
        yticks=[yi for yi, _, _ in positions],
        yticklabels=[term_label[term] for _, _, term in positions],
        grid=False, titlesize=figstyle.FS_TITLE,
    )
    for yi, heading in headings:
        ax_b.text(
            -18, yi, heading, ha="left", va="center",
            fontsize=figstyle.FS_ANNOT, color=figstyle.PRIMARY, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1.5},
        )

    fig.format(
        abc="a", abcloc="ul", abcsize=figstyle.FS_PANEL,
        suptitle="Papers linked to adoption sit closer to the resulting output",
        suptitlesize=14,
    )
    figstyle.apply_typography(axes)
    save(fig, "figS_selective_translation_diagnostics")


def main() -> None:
    make_figure()


if __name__ == "__main__":
    main()
