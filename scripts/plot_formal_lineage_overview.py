#!/usr/bin/env python3
"""Plot the verified formal lineage of post-1995 ATS Measures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import ultraplot as uplt

from scripts.official_regular_atcm_outputs import load_official_regular_outputs


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "output"
    / "attention_output_signal"
    / "measure_formal_predecessor_citations.csv"
)
OUTPUT = ROOT / "figures" / "figS_formal_lineage_overview.pdf"
PREVIEW = Path("/tmp/figS_formal_lineage_overview.png")
AREA_CATEGORY = "Area protection and management"

COLORS = {
    "source": "#176D78",
    "target": "#C9872B",
    "area": "#176D78",
    "other": "#C9872B",
    "connector": "#B8B5AD",
}


def load_links() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load verified prior-instrument citations and the official inventory."""
    if not INPUT.exists():
        raise FileNotFoundError(
            "Run `python -m scripts.audit_resolution_measure_ladders` first"
        )
    links = pd.read_csv(INPUT)
    links = links[
        links["citation_resolves_to_inventory"] & links["citation_is_prior"]
    ].copy()
    outputs = load_official_regular_outputs()
    if links.empty:
        raise AssertionError("The verified formal lineage table is empty")
    return links, outputs


def predecessor_summary(links: pd.DataFrame) -> pd.DataFrame:
    """Count distinct predecessors and Measures connected to each layer."""
    order = ["Measure", "Decision", "Resolution"]
    summary = (
        links.groupby("predecessor_type")
        .agg(
            cited_predecessors=("predecessor_id", "nunique"),
            citing_measures=("measure_id", "nunique"),
            citation_edges=("measure_id", "size"),
        )
        .reindex(order)
        .reset_index()
    )
    if summary.isna().any().any():
        raise AssertionError("Every predecessor layer must occur in the citation audit")
    return summary


def resolution_timelines(
    links: pd.DataFrame, outputs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return source metadata and yearly target counts for Resolution lineages."""
    resolution_links = links[links["predecessor_type"].eq("Resolution")].copy()
    metadata = outputs.set_index("output_id")
    sources = (
        resolution_links.groupby("predecessor_id", as_index=False)
        .agg(
            resolution_year=("predecessor_year", "first"),
            resolution_number=("predecessor_number", "first"),
            cited_measures=("measure_id", "nunique"),
            first_measure_year=("measure_year", "min"),
            last_measure_year=("measure_year", "max"),
        )
        .rename(columns={"predecessor_id": "resolution_id"})
    )
    sources["area_protection"] = sources["resolution_id"].map(
        lambda identifier: AREA_CATEGORY
        in metadata.loc[identifier, "official_categories"]
    )
    sources = sources.sort_values(
        ["resolution_year", "resolution_number"]
    ).reset_index(drop=True)
    yearly_targets = (
        resolution_links.groupby(
            ["predecessor_id", "measure_year"], as_index=False
        )["measure_id"]
        .nunique()
        .rename(
            columns={
                "predecessor_id": "resolution_id",
                "measure_id": "measures_in_year",
            }
        )
    )
    return sources, yearly_targets


def main() -> None:
    links, outputs = load_links()
    summary = predecessor_summary(links)
    sources, yearly_targets = resolution_timelines(links, outputs)

    fig, axs = uplt.subplots(
        ncols=2,
        width_ratios=(0.9, 1.8),
        figwidth=7.2,
        share=False,
        span=False,
    )

    ax = axs[0]
    for row_index, row in summary.iterrows():
        ax.plot(
            [row.cited_predecessors, row.citing_measures],
            [row_index, row_index],
            color=COLORS["connector"],
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(
            row.cited_predecessors,
            row_index,
            s=44,
            marker="o",
            facecolor="white",
            edgecolor=COLORS["source"],
            linewidth=1.4,
            label="Cited predecessors" if row_index == 0 else None,
            zorder=3,
        )
        ax.scatter(
            row.citing_measures,
            row_index,
            s=42,
            marker="s",
            color=COLORS["target"],
            edgecolor="white",
            linewidth=0.5,
            label="Citing Measures" if row_index == 0 else None,
            zorder=3,
        )
        ax.text(
            row.cited_predecessors - 5,
            row_index + 0.17,
            str(int(row.cited_predecessors)),
            color=COLORS["source"],
            ha="right",
            va="bottom",
            fontsize=7.5,
        )
        ax.text(
            row.citing_measures + 5,
            row_index + 0.17,
            str(int(row.citing_measures)),
            color=COLORS["target"],
            ha="left",
            va="bottom",
            fontsize=7.5,
        )
    ax.legend(loc="b", ncols=1, frame=False)
    ax.format(
        xlabel="Distinct instruments",
        xlim=(-8, 255),
        xticks=[0, 50, 100, 150, 200, 250],
        yticks=range(len(summary)),
        yticklabels=[f"Prior {value}" for value in summary["predecessor_type"]],
        ylim=(-0.55, len(summary) - 0.45),
        grid=False,
    )
    ax.invert_yaxis()

    ax = axs[1]
    for row_index, source in sources.iterrows():
        color = COLORS["area"] if source.area_protection else COLORS["other"]
        ax.plot(
            [source.resolution_year, source.last_measure_year],
            [row_index, row_index],
            color=COLORS["connector"],
            linewidth=1.2,
            zorder=1,
        )
        ax.scatter(
            source.resolution_year,
            row_index,
            s=34,
            marker="D",
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        targets = yearly_targets[
            yearly_targets["resolution_id"].eq(source.resolution_id)
        ]
        ax.scatter(
            targets["measure_year"],
            [row_index] * len(targets),
            s=16 + 8 * targets["measures_in_year"],
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.text(
            2026.2,
            row_index,
            f"{int(source.cited_measures)}",
            color=color,
            ha="left",
            va="center",
            fontsize=7.3,
        )
    ax.scatter(
        [],
        [],
        marker="o",
        color=COLORS["area"],
        label="Area protection",
    )
    ax.scatter(
        [],
        [],
        marker="o",
        color=COLORS["other"],
        label="Other concern",
    )
    ax.legend(loc="b", ncols=2, frame=False)
    ax.format(
        xlabel="Year",
        xlim=(1993.5, 2029.0),
        xticks=[1995, 2000, 2005, 2010, 2015, 2020, 2025],
        yticks=range(len(sources)),
        yticklabels=sources["resolution_id"].tolist(),
        ylim=(-0.65, len(sources) - 0.35),
        grid=False,
    )
    ax.invert_yaxis()
    axs.format(abc="a", abcloc="ul")
    fig.save(OUTPUT, dpi=600)
    fig.save(PREVIEW, dpi=220)
    print(f"Saved {OUTPUT}")
    print(f"Saved {PREVIEW}")


if __name__ == "__main__":
    main()
