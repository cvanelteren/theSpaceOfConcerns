#!/usr/bin/env python3
"""The outcome-pathway figure: how Measures are assembled, when, and where.

One three-panel figure, as specified in `MEASURE_PATHWAYS_NEXT_STEPS.md`:

  A. what each Measure is built from (pathway composition);
  B. how long the formal inheritance takes (lags, strong versus contextual);
  C. whether those links are unusually close in the concern space.

The concern map establishes the geometry once, elsewhere. Nothing here is
overlaid on it; this figure tests what flows through it.
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


RANDOM_SEED = 20260813

PATHWAY_LABELS = {
    "paper_only": "Paper only",
    "outcome_only": "Formal predecessor only",
    "both": "Both",
    "neither": "Neither recovered",
}
PATHWAY_COLORS = {
    "paper_only": "#2B7A78",
    "outcome_only": "#B36B2C",
    "both": "#7B3294",
    "neither": "#B7BDC6",
}

# Panel C stacks two different question, so the rows are grouped rather than
# listed: the first block is about reaching any instrument, the second is about
# reaching a Measure specifically. Without the split a reader assumes five
# comparable rows, which they are not -- the matched pools differ in size and
# composition, so each point is only interpretable against its own null tick.
EDGE_SET_LABELS = {
    "paper_to_outcome_adoption": "Adoption-linked",
    "paper_to_outcome_discussion": "Discussion only",
    "strong_transformation": "Earlier instrument,\ntransforms",
    "contextual_reference": "Earlier instrument,\ncites or recalls",
    "paper_to_intermediate_to_measure_adoption": "Paper to intermediate\nto measure",
}
EDGE_SET_BLOCKS = [
    (
        "Paper into any instrument",
        ["paper_to_outcome_adoption", "paper_to_outcome_discussion"],
    ),
    (
        "Into a measure",
        [
            "strong_transformation",
            "contextual_reference",
            "paper_to_intermediate_to_measure_adoption",
        ],
    ),
]
EDGE_SET_COLORS = {
    "paper_to_outcome_adoption": "#2B7A78",
    "paper_to_outcome_discussion": "#8A94A3",
    "strong_transformation": "#B36B2C",
    "contextual_reference": "#C4699E",
    "paper_to_intermediate_to_measure_adoption": "#7B3294",
}


def save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.save(FIGDIR / f"{stem}.pdf")
    fig.save(FIGDIR / f"{stem}.png", dpi=260)
    uplt.close(fig)


def bootstrap_target_means(
    edges: pd.DataFrame, n_bootstrap: int = 5000
) -> pd.DataFrame:
    """Target-balanced mean percentile with a target-bootstrap interval.

    Resampling targets, not edges, keeps the interval consistent with the
    permutation test, which also averages targets equally.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    usable = edges[edges["matched_pool"] > 1]
    for edge_set, subset in usable.groupby("edge_set"):
        values = (
            subset.groupby("target_id")["matched_percentile"].mean().to_numpy(dtype=float)
        )
        draws = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
        )
        rows.append(
            {
                "edge_set": edge_set,
                "mean": float(values.mean()),
                "low": float(np.quantile(draws, 0.025)),
                "high": float(np.quantile(draws, 0.975)),
                "n_targets": int(len(values)),
                "n_edges": int(len(subset)),
            }
        )
    return pd.DataFrame(rows).set_index("edge_set")


def make_pathway_figure() -> None:
    inventory = pd.read_csv(OUTDIR / "measure_pathway_inventory.csv")
    edge_audit = pd.read_csv(OUTDIR / "measure_edge_audit.csv")
    spatial_edges = pd.read_csv(OUTDIR / "measure_spatial_continuity_edges.csv")
    spatial_tests = pd.read_csv(OUTDIR / "measure_spatial_continuity_tests.csv").set_index(
        "edge_set"
    )

    fig, axes = uplt.subplots(ncols=3, refwidth=2.72, refaspect=0.95, share=False, wspace=10.5)
    ax_a, ax_b, ax_c = axes

    # --- A. Pathway composition ------------------------------------------
    order = ["outcome_only", "neither", "both", "paper_only"]
    counts = inventory["pathway"].value_counts()
    total = int(len(inventory))
    y = np.arange(len(order))[::-1]
    for yi, pathway in zip(y, order):
        count = int(counts.get(pathway, 0))
        # UltraPlot's barh takes the bar length as the second positional and
        # the bar thickness as `width`.
        ax_a.barh(
            [yi], [count], width=0.62, color=PATHWAY_COLORS[pathway],
            edgecolor="white", linewidth=0.6,
        )
        ax_a.text(
            count + total * 0.015, yi, f"{count}  ({count / total:.0%})",
            va="center", ha="left", fontsize=figstyle.FS_ANNOT, color=figstyle.TEXT,
        )
    ax_a.format(
        title="Most measures inherit from formal precedent",
        xlabel="Measures (of 279)",
        xlim=(0, total * 0.78),
        yticks=y, yticklabels=[PATHWAY_LABELS[pathway] for pathway in order],
        grid=False, titlesize=figstyle.FS_TITLE,
    )

    # --- B. Time to Measure ----------------------------------------------
    lag_groups = [
        ("strong_transformation", "Strong transformation", "#B36B2C"),
        ("contextual_reference", "Contextual reference", "#C4699E"),
    ]
    bins = np.arange(0, 60, 3)
    centers = (bins[:-1] + bins[1:]) / 2
    curves = []
    for relation_class, label, color in lag_groups:
        lags = (
            edge_audit[edge_audit["relation_class"].eq(relation_class)]["lag_years"]
            .dropna()
            .to_numpy(dtype=float)
        )
        density, _ = np.histogram(lags, bins=bins, density=True)
        curves.append((label, color, density, float(np.median(lags)), len(lags)))
    # Annotation heights are placed against the realised peak, so they cannot
    # land outside the axes when the data change.
    peak = max(float(density.max()) for _, _, density, _, _ in curves)
    top = peak * 1.22

    # The published exposure window, for comparison with the observed lags.
    ax_b.axvspan(1, 3, color="#98A2B3", alpha=0.18, zorder=0)
    ax_b.text(
        4.0, top * 0.03, "Published\n1--3 y window", color=figstyle.MUTED,
        fontsize=figstyle.FS_ANNOT, ha="left", va="bottom",
    )
    for index, (label, color, density, median, count) in enumerate(curves):
        ax_b.plot(centers, density, color=color, lw=2.0, label=f"{label} (n={count})")
        ax_b.fill_between(centers, 0, density, color=color, alpha=0.16)
        ax_b.axvline(median, color=color, lw=1.1, ls="--", alpha=0.85)
        ax_b.text(
            median + 1.2, top * (0.70 if index == 0 else 0.52),
            f"Median {median:.0f} y", color=color, fontsize=figstyle.FS_ANNOT,
            ha="left", va="center",
        )
    ax_b.legend(loc="ur", ncols=1, frame=False)
    ax_b.format(
        title="Formal inheritance is slower than the tested window",
        xlabel="Years from predecessor to measure",
        ylabel="Density of edges",
        xlim=(0, 57), ylim=(0, top), grid=False, titlesize=figstyle.FS_TITLE,
    )

    # --- C. Spatial continuity -------------------------------------------
    summary = bootstrap_target_means(spatial_edges)

    # Lay the two blocks out top to bottom with a gap between them, so the
    # grouping is visible without a second axes.
    positions: list[tuple[float, str]] = []
    block_headings: list[tuple[float, str]] = []
    cursor = 0.0
    for heading, members in EDGE_SET_BLOCKS:
        present = [name for name in members if name in summary.index]
        if not present:
            continue
        block_headings.append((cursor, heading))
        cursor -= 0.75
        for name in present:
            positions.append((cursor, name))
            cursor -= 1.0
        cursor -= 0.55

    for yi, edge_set in positions:
        row = summary.loc[edge_set]
        color = EDGE_SET_COLORS[edge_set]
        ax_c.errorbar(
            [row["mean"]], [yi],
            xerr=np.array([[row["mean"] - row["low"]], [row["high"] - row["mean"]]]),
            fmt="none", ecolor=color, elinewidth=2.0, capsize=3.0, zorder=2,
        )
        ax_c.scatter(
            [row["mean"]], [yi], s=54, color=color,
            edgecolor="white", linewidth=0.65, zorder=3,
        )
        null_mean = float(spatial_tests.loc[edge_set, "null_mean"])
        ax_c.scatter([null_mean], [yi], s=26, marker="|", color="#98A2B3", zorder=2)
        ax_c.text(
            0.895, yi, f"n={int(row['n_targets'])}", ha="right", va="center",
            fontsize=figstyle.FS_ANNOT, color=figstyle.MUTED,
        )

    ax_c.axvline(0.5, color="#98A2B3", lw=0.9, ls="--")
    ax_c.format(
        title="Observed lineage links are concern-proximate",
        xlabel="Matched-null proximity percentile",
        xlim=(0.44, 0.90), xformatter="{x:.0%}", xlocator=[0.5, 0.6, 0.7, 0.8],
        ylim=(cursor + 0.35, 0.6),
        yticks=[yi for yi, _ in positions],
        yticklabels=[EDGE_SET_LABELS[name] for _, name in positions],
        grid=False, titlesize=figstyle.FS_TITLE,
    )
    for yi, heading in block_headings:
        # Offset clear of the panel letter, which sits in the upper left.
        ax_c.text(
            0.495, yi, heading, ha="left", va="center",
            fontsize=figstyle.FS_ANNOT, color=figstyle.PRIMARY, fontweight="bold",
        )

    fig.format(
        abc="A", abcloc="ul", abcsize=figstyle.FS_PANEL,
        suptitle="Measures follow long formal lineages, not recent attention",
        suptitlesize=14,
    )
    figstyle.apply_typography(axes)
    save(fig, "exploratory_measure_pathways")


def main() -> None:
    make_pathway_figure()


if __name__ == "__main__":
    main()
