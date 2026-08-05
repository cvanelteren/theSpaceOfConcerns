"""Main-text Figure 2: how actors are positioned in, and move through, the
space of concerns.

Top row is the population-level claim. Panel A is the cross-section: each
actor's RPA-weighted mean position on the 1D space-of-concerns ordering,

    c(a) = sum_i rho(a,i) * x_i / sum_i rho(a,i),

where x_i is topic i's coordinate on the MDS-scaled axis, grouped by dominant
mode. Panel B is movement across all actors: the row-normalized dominant-mode
transition matrix over rolling windows.

Bottom row is the same claim at individual scale: mode shares window by window
for three actors with contrasting institutional histories.

Note that the partition is defined on the same axis panel A plots, so panel A
describes how actor mass distributes along that ordering rather than
independently establishing that three modes exist. The manuscript states this
directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt


TOPIC_FP = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
ACTOR_FP = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")
REGION_FP = Path("output/fig45_portfolio_space_ridgelines_region_summary.csv")
PROFILE_TEMPLATE = "output/fig45_regime_member_window_profiles_year_window{window}.csv"

OUT_PDF = Path("figures/fig02_positioning.pdf")
OUT_PNG = Path("figures/fig02_positioning.png")

DEFAULT_ACTORS = ["Australia", "Netherlands", "Ukraine"]
WINDOW_SIZE = 5

REGION_COLORS = {1: "#D55E00", 2: "#0072B2", 3: "#009E73"}
MODE_SHORT = {1: "Mode 1", 2: "Mode 2", 3: "Mode 3"}
MODE_GLOSS = {
    1: "Coordination\n& exchange",
    2: "Compliance\n& management",
    3: "Strategy\n& resources",
}
TEXT_COLOR = "#374151"


def _jitter(n: int, width: float = 0.14, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def build_figure(actors: list[str], window_size: int) -> uplt.Figure:
    actor_df = pd.read_csv(ACTOR_FP)
    region_df = pd.read_csv(REGION_FP)
    profile_df = pd.read_csv(PROFILE_TEMPLATE.format(window=int(window_size)))

    missing = [a for a in actors if a not in set(profile_df["member"])]
    if missing:
        raise ValueError(f"Actors absent from rolling profiles: {missing}")

    n = len(actors)
    array = [[1, 1, 2, 2]] + [[3 + i] * 4 for i in range(n)]
    # Square-ish layout: panels A & B on top (4.2 wide), trajectories below
    fig, axs = uplt.subplots(
        array,
        figsize=(10.0, 5.0 + 0.8 * n),
        hratios=(2.6, *([0.6] * n)),
        sharex=False,
        sharey=False,
        wspace=5.5,
        hspace=(5.0, *([0.2] * (n - 1))),
    )
    ax_cent, ax_trans = axs[0], axs[1]
    share_axes = axs[2:]

    # --- Panel A: cross-sectional position by dominant mode ---------------
    for _, row in region_df.sort_values("region_id").iterrows():
        rid = int(row["region_id"])
        ax_cent.axhspan(
            float(row["boundary_left"]), float(row["boundary_right"]),
            color=REGION_COLORS[rid], alpha=0.08, zorder=0,
        )
    for rid in [1, 2, 3]:
        vals = actor_df.loc[
            actor_df["dominant_region"] == rid, "centroid_xplot_raw_rca"
        ].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        xj = np.full(vals.size, float(rid)) + _jitter(vals.size, seed=rid)
        ax_cent.scatter(
            xj, vals, s=34, color=REGION_COLORS[rid], alpha=0.9,
            edgecolor="black", linewidth=0.4, zorder=3, absolute_size=True,
        )
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        ax_cent.vlines(rid, q1, q3, color=REGION_COLORS[rid], lw=3.0, zorder=2)
        ax_cent.hlines(
            float(np.median(vals)), rid - 0.16, rid + 0.16,
            color="black", lw=1.2, zorder=5,
        )
    ax_cent.format(
        ylim=(0.0, 1.0), xlim=(0.5, 3.5), yminorticks=[],
        ylocator=np.linspace(0.0, 1.0, 6), xlocator=[1, 2, 3],
        xformatter=["Coordination", "Compliance", "Strategy"], xlabel="",
        ylabel="Concern-space position (RPA-weighted)", grid=False,
    )
    ax_cent.grid(axis="y", alpha=0.22, linewidth=0.7)
    # Increase label font size
    for label in ax_cent.get_yticklabels():
        label.set_fontsize(9)
    for label in ax_cent.get_xticklabels():
        label.set_fontsize(9)
    ax_cent.yaxis.label.set_fontsize(10)
    for _, row in region_df.sort_values("region_id").iterrows():
        rid = int(row["region_id"])
        ax_cent.text(
            0.5, float(row["anchor_x"]),
            MODE_GLOSS[rid],
            transform=ax_cent.get_yaxis_transform(),
            ha="center", va="bottom", fontsize=6.8,
            color=REGION_COLORS[rid], zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )

    # --- Panel B: pairwise portfolio complementarity -----------------------
    pair_df = pd.read_csv("output/fig27_pairwise_portfolio_complementarity_pairs.csv")
    x_med = float(pair_df["overlap_jaccard_rpa"].median())
    y_med = float(pair_df["exclusive_phi_proximity"].median())
    comp_colors = {
        "Complementary": "teal7",
        "Aligned": "orange7",
        "Separate / weak": "gray6",
    }
    for label, group in pair_df.groupby("pair_type", sort=False):
        ax_trans.scatter(
            group["overlap_jaccard_rpa"],
            group["exclusive_phi_proximity"],
            s=12,
            alpha=0.72 if label != "Separate / weak" else 0.32,
            c=comp_colors[label],
            edgecolor="none",
            label=label,
            zorder=3 if label != "Separate / weak" else 2,
        )
    ax_trans.axvline(x_med, color="black", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax_trans.axhline(y_med, color="black", lw=0.8, ls="--", alpha=0.5, zorder=1)
    
    xlim = ax_trans.get_xlim()
    ylim = ax_trans.get_ylim()
    ax_trans.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), y_med + 0.01 * (ylim[1] - ylim[0]), "Complementary", color="teal8", fontsize=7, weight="bold")
    ax_trans.text(x_med + 0.01 * (xlim[1] - xlim[0]), y_med + 0.01 * (ylim[1] - ylim[0]), "Aligned", color="orange8", fontsize=7, weight="bold")
    ax_trans.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), ylim[0] + 0.02 * (ylim[1] - ylim[0]), "Separate / weak", color="gray7", fontsize=7, weight="bold")

    # Add labels for top complementary pairs (Tufte-style, minimal but visible)
    top_labels = (
        pair_df.loc[pair_df["pair_type"] == "Complementary"]
        .sort_values(["exclusive_phi_proximity", "complementarity_score"], ascending=[False, False])
        .head(6)
        .copy()
    )
    for idx, (_, row) in enumerate(top_labels.iterrows()):
        dx = 0.005 + (idx % 2) * 0.002
        dy = 0.002 + (idx % 3) * 0.003
        ax_trans.text(
            row["overlap_jaccard_rpa"] + dx,
            row["exclusive_phi_proximity"] + dy,
            row["pair_label"],
            fontsize=6.5,
            color="black",
            alpha=0.88,
            weight="normal",
        )

    ax_trans.format(
        xlabel=r"Overlap (Jaccard on $RPA>1$ topics)",
        ylabel=r"Exclusive proximity in $\phi$",
        grid=False,
    )
    ax_trans.grid(alpha=0.1, color="black")
    # Increase label font sizes
    ax_trans.xaxis.label.set_fontsize(10)
    ax_trans.yaxis.label.set_fontsize(10)
    for label in ax_trans.get_xticklabels() + ax_trans.get_yticklabels():
        label.set_fontsize(9)
    ax_trans.legend(loc="lower right", ncols=1, frame=False, fontsize=8)

    # --- Panels C-E: mode shares over time --------------------------------
    xmin = float(profile_df["period_end"].min())
    xmax = float(profile_df["period_end"].max())
    for idx, (actor, ax) in enumerate(zip(actors, share_axes)):
        sub = profile_df[profile_df["member"] == actor].sort_values("period_end")
        years = sub["period_end"].to_numpy(dtype=float)
        stack = [sub[f"region_{rid}_share"].to_numpy(dtype=float) for rid in [1, 2, 3]]
        ax.stackplot(
            years, stack, colors=[REGION_COLORS[r] for r in [1, 2, 3]],
            alpha=0.85, baseline="zero",
        )
        ax.format(
            ylim=(0.0, 1.0), ylocator=[0.5], yformatter=[actor],
            yminorticks=[], grid=True, ylabel="",
            xlim=(xmin, xmax),
        )
        ax.grid(axis="y", alpha=0.15, linewidth=0.6)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", labelsize=9)
        if idx < len(actors) - 1:
            ax.set_xticklabels([])
    # No shared "Mode share" ylabel: the actor names occupy the y tick slot
    share_axes[-1].format(xlabel="Window end year")
    share_axes[-1].xaxis.label.set_fontsize(10)

    axs.format(abc="[A]", abcloc="ul")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default=",".join(DEFAULT_ACTORS))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    actors = [a.strip() for a in args.actors.split(",") if a.strip()]
    fig = build_figure(actors, args.window_size)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PNG, dpi=300)
    fig.save(OUT_PDF)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
