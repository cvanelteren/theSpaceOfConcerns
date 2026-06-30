"""Supplementary Figure S05. Tests the retain-and-enter model under year-balanced thinning of the archive. Checks that the mechanism result is not driven only by dense late-period data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import ultraplot as uplt


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "figures"

SUMMARY_JSON = OUTPUT_DIR / "actor_topic_modeling_starter_summary.json"
SUBSAMPLE_RAW_CSV = OUTPUT_DIR / "actor_topic_modeling_starter_subsample_validation.csv"

OUT_PDF = FIG_DIR / "figS05_split_support_thinning.pdf"
OUT_PNG = FIG_DIR / "figS05_split_support_thinning.png"

MODEL_COLORS = {
    "Direct allocation": "#c44e52",
    "Single-rule support": "#4c72b0",
    "Retain-and-enter": "#2a9d55",
    "Observed": "#111111",
}


def _load_summary() -> dict[str, float]:
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def _aggregate_subsample_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for keep_fraction, group in df.groupby("keep_fraction", sort=True):
        rows.extend(
            [
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Direct allocation",
                    "metric": "active_corr",
                    "mean": float(group["corr_mean_active_topics"].mean()),
                    "sd": float(group["corr_mean_active_topics"].std(ddof=1)),
                },
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Single-rule support",
                    "metric": "active_corr",
                    "mean": float(group["corr_mean_active_topics_two_stage"].mean()),
                    "sd": float(group["corr_mean_active_topics_two_stage"].std(ddof=1)),
                },
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Retain-and-enter",
                    "metric": "active_corr",
                    "mean": float(group["corr_mean_active_topics_split"].mean()),
                    "sd": float(group["corr_mean_active_topics_split"].std(ddof=1)),
                },
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Direct allocation",
                    "metric": "pop_corr",
                    "mean": float(group["corr_mean_topic_popularity"].mean()),
                    "sd": float(group["corr_mean_topic_popularity"].std(ddof=1)),
                },
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Single-rule support",
                    "metric": "pop_corr",
                    "mean": float(group["corr_mean_topic_popularity_two_stage"].mean()),
                    "sd": float(group["corr_mean_topic_popularity_two_stage"].std(ddof=1)),
                },
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Retain-and-enter",
                    "metric": "pop_corr",
                    "mean": float(group["corr_mean_topic_popularity_split"].mean()),
                    "sd": float(group["corr_mean_topic_popularity_split"].std(ddof=1)),
                },
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Observed",
                    "metric": "entry_rank",
                    "mean": float(group["mean_entry_phi_rank_observed"].mean()),
                    "sd": float(group["mean_entry_phi_rank_observed"].std(ddof=1)),
                },
                {
                    "keep_fraction": float(keep_fraction),
                    "model": "Retain-and-enter",
                    "metric": "entry_rank",
                    "mean": float(group["mean_entry_phi_rank_split"].mean()),
                    "sd": float(group["mean_entry_phi_rank_split"].std(ddof=1)),
                },
            ]
        )
    return pd.DataFrame(rows)


def _append_full_data_points(df: pd.DataFrame, summary: dict[str, float]) -> pd.DataFrame:
    full_rows = [
        {
            "keep_fraction": 1.0,
            "model": "Direct allocation",
            "metric": "active_corr",
            "mean": float(summary["corr_mean_active_topics"]),
            "sd": 0.0,
        },
        {
            "keep_fraction": 1.0,
            "model": "Single-rule support",
            "metric": "active_corr",
            "mean": float(summary["corr_mean_active_topics_two_stage"]),
            "sd": 0.0,
        },
        {
            "keep_fraction": 1.0,
            "model": "Retain-and-enter",
            "metric": "active_corr",
            "mean": float(summary["corr_mean_active_topics_split"]),
            "sd": 0.0,
        },
        {
            "keep_fraction": 1.0,
            "model": "Direct allocation",
            "metric": "pop_corr",
            "mean": float(summary["corr_mean_topic_popularity"]),
            "sd": 0.0,
        },
        {
            "keep_fraction": 1.0,
            "model": "Single-rule support",
            "metric": "pop_corr",
            "mean": float(summary["corr_mean_topic_popularity_two_stage"]),
            "sd": 0.0,
        },
        {
            "keep_fraction": 1.0,
            "model": "Retain-and-enter",
            "metric": "pop_corr",
            "mean": float(summary["corr_mean_topic_popularity_split"]),
            "sd": 0.0,
        },
        {
            "keep_fraction": 1.0,
            "model": "Observed",
            "metric": "entry_rank",
            "mean": float(summary["mean_entry_phi_rank_observed"]),
            "sd": 0.0,
        },
        {
            "keep_fraction": 1.0,
            "model": "Retain-and-enter",
            "metric": "entry_rank",
            "mean": float(summary["mean_entry_phi_rank_split"]),
            "sd": 0.0,
        },
    ]
    return pd.concat([df, pd.DataFrame(full_rows)], ignore_index=True)


def _plot_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    metric_df = df[df["metric"] == metric].copy()
    for model in metric_df["model"].drop_duplicates():
        d = metric_df[metric_df["model"] == model].sort_values("keep_fraction")
        color = MODEL_COLORS[model]
        ax.errorbar(
            d["keep_fraction"].to_numpy(),
            d["mean"].to_numpy(),
            yerr=d["sd"].to_numpy(),
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            capsize=2.8,
            color=color,
            label=model,
        )

    ax.format(
        title=title,
        xlabel="Fraction of raw submissions retained",
        ylabel=ylabel,
        grid=True,
        xlim=(0.36, 1.04),
        xlocator=[0.4, 0.6, 0.8, 1.0],
    )


def build_figure():
    summary = _load_summary()
    subsample_raw = pd.read_csv(SUBSAMPLE_RAW_CSV)
    plot_df = _aggregate_subsample_results(subsample_raw)
    plot_df = _append_full_data_points(plot_df, summary)

    fig, axs = uplt.subplots(ncols=3, refwidth=3.6, refheight=2.6, share=False)
    axs.format(abc="[A]")

    _plot_metric(
        axs[0],
        plot_df,
        metric="active_corr",
        title="Breadth Tracking",
        ylabel="Correlation with observed mean active topics",
    )
    axs[0].set_ylim(0.42, 0.92)

    _plot_metric(
        axs[1],
        plot_df,
        metric="pop_corr",
        title="Topic Popularity Tracking",
        ylabel="Correlation with observed mean topic popularity",
    )
    axs[1].set_ylim(0.94, 0.99)

    _plot_metric(
        axs[2],
        plot_df,
        metric="entry_rank",
        title=r"Local Entry in $\Phi$",
        ylabel="Mean rank of newly entered topics",
    )
    axs[2].axhline(0.5, color="#888888", linestyle="--", linewidth=1.0)
    axs[2].set_ylim(0.49, 0.71)

    handle_map = {}
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handle_map[label] = handle
    fig.legend(
        list(handle_map.values()),
        list(handle_map.keys()),
        ncols=4,
        loc="b",
        frame=False,
    )
    fig.format(
        suptitle=(
            "Retain-and-Enter Validation Under Year-Balanced Thinning of the Raw ATS Corpus"
        )
    )
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    print("Wrote", OUT_PDF)
    print("Wrote", OUT_PNG)


if __name__ == "__main__":
    main()
