"""Visualize actor movement through the ATS space of concerns over time.

Supports both:
- a single-actor view with regime-share panel
- a multi-actor trajectory view with end-of-line flag annotations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from utils import load_flag

REGION_FP = Path("output/fig45_portfolio_space_ridgelines_region_summary.csv")
PROFILE_TEMPLATE = "output/fig45_regime_member_window_profiles_year_window{window}.csv"

FIG_DIR = Path("figures")
OUT_DIR = Path("output")

REGION_COLORS = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
}
TEXT_COLOR = "#374151"
LINE_COLOR = "#111827"
MULTI_LINE_COLORS = ["#111827", "#6b7280", "#cbd5e1", "#4b5563", "#e5e7eb"]


def _actor_slug(actor: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(actor).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "actor"


def _actors_slug(actors: list[str]) -> str:
    return "_".join(_actor_slug(actor) for actor in actors)


def _compute_actor_windows(
    actor: str,
    *,
    window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_df = pd.read_csv(REGION_FP)
    profile_fp = Path(PROFILE_TEMPLATE.format(window=int(window_size)))
    if not profile_fp.exists():
        raise FileNotFoundError(f"Missing rolling profile file: {profile_fp}")

    profile_df = pd.read_csv(profile_fp)
    summary_df = profile_df.loc[profile_df["member"] == actor].copy()
    if summary_df.empty:
        raise ValueError(f"Actor {actor!r} not found in {profile_fp.name}.")

    anchor_map = region_df.set_index("region_id")["anchor_x"].to_dict()
    summary_df["centroid_xplot"] = (
        summary_df["region_1_share"].to_numpy(dtype=float) * float(anchor_map[1])
        + summary_df["region_2_share"].to_numpy(dtype=float) * float(anchor_map[2])
        + summary_df["region_3_share"].to_numpy(dtype=float) * float(anchor_map[3])
    )
    summary_df = summary_df.sort_values("period_end").reset_index(drop=True)
    return summary_df, region_df


def _add_regime_background(ax, region_df: pd.DataFrame) -> None:
    for _, row in region_df.sort_values("region_id").iterrows():
        rid = int(row["region_id"])
        ax.axhspan(
            float(row["boundary_left"]),
            float(row["boundary_right"]),
            color=REGION_COLORS[rid],
            alpha=0.08,
            zorder=0,
        )
        ax.axhline(
            float(row["anchor_x"]),
            color=REGION_COLORS[rid],
            lw=0.8,
            alpha=0.35,
            zorder=1,
        )


def _format_right_axis(ax, region_df: pd.DataFrame) -> None:
    region_labels = region_df.sort_values("region_id")
    ax_right = ax.twinx()
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_yticks(region_labels["anchor_x"].to_numpy(dtype=float))
    ax_right.set_yticklabels(region_labels["anchor_topic"].tolist())
    ax_right.tick_params(axis="y", colors=TEXT_COLOR, labelsize=8, length=0)
    ax_right.spines["right"].set_visible(False)


def _annotate_regime_changes(ax, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    change_mask = summary_df["dominant_region"].ne(
        summary_df["dominant_region"].shift(1)
    )
    changed = summary_df.loc[change_mask].copy()
    for _, row in changed.iterrows():
        ax.text(
            float(row["period_end"]),
            float(row["centroid_xplot"]) + 0.018,
            str(int(row["period_end"])),
            color=TEXT_COLOR,
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=5,
        )


def _load_flag_image(actor: str):
    return load_flag(actor, save=False, base="./assets/flags")


def _add_flag(ax, x: float, y: float, actor: str, dx_pts: float, dy_pts: float) -> None:
    img = _load_flag_image(actor)
    if img is not None:
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.025),
            (x, y),
            xybox=(dx_pts, dy_pts),
            xycoords="data",
            boxcoords="offset points",
            box_alignment=(0.5, 0.5),
            frameon=False,
            zorder=6,
        )
        ax.add_artist(ab)


def _add_corner_flag(ax, actor: str) -> None:
    img = _load_flag_image(actor)
    if img is None:
        return
    ab = AnnotationBbox(
        OffsetImage(img, zoom=0.03),
        (0.015, 0.96),
        xycoords="axes fraction",
        box_alignment=(0.0, 1.0),
        frameon=False,
        zorder=8,
    )
    ax.add_artist(ab)


def _make_single_actor_figure(
    actor: str,
    summary_df: pd.DataFrame,
    region_df: pd.DataFrame,
    *,
    window_size: int,
) -> uplt.Figure:
    fig, axs = uplt.subplots(
        nrows=2,
        figsize=(10.8, 6.4),
        hratios=(2.4, 0.5),
        sharex=True,
        sharey=False,
    )
    ax_top, ax_bottom = axs

    _add_regime_background(ax_top, region_df)

    ax_top.plot(
        summary_df["period_end"].to_numpy(dtype=float),
        summary_df["centroid_xplot"].to_numpy(dtype=float),
        color=LINE_COLOR,
        lw=1.9,
        alpha=0.95,
        zorder=3,
    )
    centroid_sizes = 34.0 + 6.0 * summary_df["k_active"].to_numpy(dtype=float)
    centroid_colors = summary_df["dominant_region"].map(REGION_COLORS).tolist()
    ax_top.scatter(
        summary_df["period_end"].to_numpy(dtype=float),
        summary_df["centroid_xplot"].to_numpy(dtype=float),
        s=centroid_sizes,
        c=centroid_colors,
        edgecolor="white",
        lw=0.7,
        zorder=4,
        absolute_size=True,
    )
    _annotate_regime_changes(ax_top, summary_df)

    first_row = summary_df.iloc[0]
    last_row = summary_df.iloc[-1]
    ax_top.annotate(
        f"start {int(first_row['period_start'])}-{int(first_row['period_end'])}",
        (float(first_row["period_end"]), float(first_row["centroid_xplot"])),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=7,
        color=TEXT_COLOR,
    )
    ax_top.annotate(
        f"end {int(last_row['period_start'])}-{int(last_row['period_end'])}",
        (float(last_row["period_end"]), float(last_row["centroid_xplot"])),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=7,
        color=TEXT_COLOR,
    )

    ax_top.format(
        ylabel="Concern-space position\n(anchor-weighted)",
        ylim=(0.0, 1.0),
        xmargin=0.02,
        yminorticks=[],
        ylocator=np.linspace(0.0, 1.0, 6),
        grid=False,
    )
    _format_right_axis(ax_top, region_df)

    years = summary_df["period_end"].to_numpy(dtype=float)
    shares = [
        summary_df[f"region_{rid}_share"].to_numpy(dtype=float) for rid in [1, 2, 3]
    ]
    colors = [REGION_COLORS[rid] for rid in [1, 2, 3]]
    ax_bottom.stackplot(years, shares, colors=colors, alpha=0.82, baseline="zero")
    ax_bottom.format(
        xlabel="Window end year",
        ylabel="Region share",
        ylim=(0.0, 1.0),
        ylocator=np.linspace(0.0, 1.0, 5),
        yminorticks=[],
        grid=False,
    )
    return fig


def _make_multi_actor_figure(
    actor_summaries: dict[str, pd.DataFrame],
    region_df: pd.DataFrame,
    *,
    window_size: int,
) -> uplt.Figure:
    n_actors = len(actor_summaries)
    fig, axs = uplt.subplots(
        nrows=1 + n_actors,
        figsize=(11.4, 5.8 + 0.7 * n_actors),
        hratios=(2.8, *([0.65] * n_actors)),
        sharex=True,
        sharey=False,
    )
    ax = axs[0]
    share_axes = axs[1:]
    axs.format(abc="a")

    _add_regime_background(ax, region_df)

    actor_items = list(actor_summaries.items())
    for idx, (actor, summary_df) in enumerate(actor_items):
        x = summary_df["period_end"].to_numpy(dtype=float)
        y = summary_df["centroid_xplot"].to_numpy(dtype=float)
        sizes = 22.0 + 4.0 * summary_df["k_active"].to_numpy(dtype=float)
        marker_colors = summary_df["dominant_region"].map(REGION_COLORS).tolist()
        ax.plot(
            x,
            y,
            color=MULTI_LINE_COLORS[idx % len(MULTI_LINE_COLORS)],
            lw=2.1,
            alpha=0.98,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            s=sizes,
            c=marker_colors,
            edgecolor="white",
            lw=0.6,
            zorder=4,
            absolute_size=True,
        )

    ax.format(
        xlabel="Window end year",
        ylabel="Concern-space position\n(anchor-weighted)",
        ylim=(0.0, 1.0),
        xmargin=0.02,
        yminorticks=[],
        ylocator=np.linspace(0.0, 1.0, 6),
        grid=False,
    )
    _format_right_axis(ax, region_df)

    share_axes.format(
        ylim=(0.0, 1.0),
        ylocator=[0, 0.5, 1.0],
        yminorticks=[],
        ylabel="Mode share",
        grid=False,
    )
    for share_ax, (actor, summary_df) in zip(share_axes, actor_items):
        years = summary_df["period_end"].to_numpy(dtype=float)
        shares = [
            summary_df[f"region_{rid}_share"].to_numpy(dtype=float) for rid in [1, 2, 3]
        ]
        colors = [REGION_COLORS[rid] for rid in [1, 2, 3]]
        share_ax.stackplot(years, shares, colors=colors, alpha=0.85, baseline="zero")

        _add_corner_flag(share_ax, actor)

    share_axes[-1].format(xlabel="Window end year")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", default="Australia", help="Actor to visualize.")
    parser.add_argument(
        "--actors",
        default="",
        help="Comma-separated list of actors for a multi-actor trajectory figure.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Rolling window size in years (default: 5).",
    )
    args = parser.parse_args()

    actors = (
        [item.strip() for item in args.actors.split(",") if item.strip()]
        if args.actors.strip()
        else [args.actor]
    )

    if len(actors) == 1:
        actor = actors[0]
        summary_df, region_df = _compute_actor_windows(
            actor, window_size=args.window_size
        )
        if summary_df.empty:
            raise RuntimeError(f"No active windows found for actor {actor!r}.")

        slug = _actor_slug(actor)
        fig_name = f"fig46_actor_trajectory_{slug}_window{args.window_size}"
        fig_png = FIG_DIR / f"{fig_name}.png"
        fig_pdf = FIG_DIR / f"{fig_name}.pdf"
        summary_csv = OUT_DIR / f"{fig_name}_summary.csv"

        fig_png.parent.mkdir(parents=True, exist_ok=True)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)

        fig = _make_single_actor_figure(
            actor, summary_df, region_df, window_size=args.window_size
        )
        fig.save(fig_png, dpi=300)
        fig.save(fig_pdf)
        print(f"Wrote {fig_png}")
        print(f"Wrote {fig_pdf}")
        print(f"Wrote {summary_csv}")
        return

    actor_summaries: dict[str, pd.DataFrame] = {}
    region_df: pd.DataFrame | None = None
    for actor in actors:
        summary_df, region_df_current = _compute_actor_windows(
            actor, window_size=args.window_size
        )
        if summary_df.empty:
            raise RuntimeError(f"No active windows found for actor {actor!r}.")
        actor_summaries[actor] = summary_df
        if region_df is None:
            region_df = region_df_current

    assert region_df is not None

    slug = _actors_slug(actors)
    fig_name = f"fig46_actor_trajectory_{slug}_window{args.window_size}"
    fig_png = FIG_DIR / f"{fig_name}.png"
    fig_pdf = FIG_DIR / f"{fig_name}.pdf"
    summary_csv = OUT_DIR / f"{fig_name}_summary.csv"

    fig_png.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(actor_summaries.values(), ignore_index=True).to_csv(
        summary_csv, index=False
    )

    fig = _make_multi_actor_figure(
        actor_summaries, region_df, window_size=args.window_size
    )
    fig.save(fig_png, dpi=300)
    fig.save(fig_pdf)
    print(f"Wrote {fig_png}")
    print(f"Wrote {fig_pdf}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
