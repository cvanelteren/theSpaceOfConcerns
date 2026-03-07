from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import ultraplot as uplt


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "figures"

SUMMARY_PATH = OUTPUT_DIR / "actor_topic_modeling_starter_summary.json"
PROCESS_HISTORY = OUTPUT_DIR / "actor_topic_modeling_starter_process_uncertainty_history.csv"
PROCESS_ENTRY = OUTPUT_DIR / "actor_topic_modeling_starter_process_uncertainty_entry.csv"

OUT_PDF = FIG_DIR / "fig04_split_support_validation.pdf"
OUT_PNG = FIG_DIR / "fig04_split_support_validation.png"


COLORS = {
    "observed": "#111111",
    "one_stage": "#c44e52",
    "two_stage": "#4c72b0",
    "split_support": "#2a9d55",
    "baseline": "#9a9a9a",
}


def _load_summary() -> pd.Series:
    return pd.read_json(SUMMARY_PATH, typ="series")


def _load_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _draw_step_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    facecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + h - 0.06,
        title,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
    )
    ax.text(
        x + 0.02,
        y + h - 0.12,
        "\n".join(lines),
        fontsize=8.7,
        va="top",
        ha="left",
        color="#222222",
        linespacing=1.25,
    )


def build_figure() -> plt.Figure:
    summary = _load_summary()
    process_history = _load_history(PROCESS_HISTORY)
    process_entry = _load_history(PROCESS_ENTRY)
    hist_one = process_history[process_history["model"] == "one_stage"].sort_values("window_end")
    hist_two = process_history[process_history["model"] == "two_stage"].sort_values("window_end")
    hist_split = process_history[process_history["model"] == "split_support"].sort_values("window_end")

    years = hist_split["window_end"]
    observed_breadth = hist_split["mean_active_topics_obs"]
    observed_popularity = hist_split["mean_topic_popularity_obs"]

    layout = [
        [1, 1, 1],
        [2, 3, 4],
    ]
    fig, axs = uplt.subplots(layout, share=0, refnum=2, hratios=(1.05, 1.0))
    ax_a, ax_b, ax_c, ax_d = axs
    axs.format(abc="[A]")

    # Panel A: sequential model schematic
    ax_a.format(
        title="Sequential Retain-and-Enter Model",
        xlim=(0, 1),
        ylim=(0, 1),
        xlocator=[],
        ylocator=[],
    )
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    box_w = 0.27
    box_h = 0.63
    y0 = 0.15
    xs = [0.03, 0.355, 0.68]
    _draw_step_box(
        ax_a,
        xs[0],
        y0,
        box_w,
        box_h,
        "1. Retain Prior Topics",
        [
            "Keep topics already in the portfolio.",
            "Retention is stronger for previously",
            "important topics.",
            r"Uses prior topic weight $s_{ai,t-1}$",
            rf"$\lambda_{{ret}}={summary['lambda_retention_mle']:.2f}$",
        ],
        facecolor="#efe7da",
    )
    _draw_step_box(
        ax_a,
        xs[1],
        y0,
        box_w,
        box_h,
        "2. Enter Nearby Topics",
        [
            r"Add topics that sit near the prior",
            r"support in the pooled concern space $\Phi$.",
            r"Local fit = mean $\phi(i,j)$ to",
            r"previously held topics.",
            r"New topics are more likely if they are",
            r"close to the prior support.",
            rf"$\beta_{{ent}}={summary['beta_entry_mle']:.2f}$",
        ],
        facecolor="#ddeaf5",
    )
    _draw_step_box(
        ax_a,
        xs[2],
        y0,
        box_w,
        box_h,
        "3. Allocate Budget",
        [
            "Condition on observed active actors",
            r"and observed budgets $K_{at}$.",
            "Allocate within the selected support.",
            r"Uses prior shares and local fit.",
            rf"$\rho={summary['rho_mle']:.2f}$, $\beta={summary['beta_mle']:.2f}$",
        ],
        facecolor="#e2f0e5",
    )

    ax_a.text(
        0.98,
        0.95,
        (
            "Simulation conditions:\n"
            "active actor set, budgets $K_{at}$, and pooled $\\Phi$ are fixed;\n"
            "a scalar intercept shift calibrates support levels."
        ),
        fontsize=8.4,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"),
        zorder=9,
    )

    for x_left, x_right in zip(xs[:-1], xs[1:]):
        arrow = FancyArrowPatch(
            (x_left + box_w + 0.02, y0 + box_h / 2),
            (x_right - 0.02, y0 + box_h / 2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color="#444444",
            zorder=10,
        )
        ax_a.add_patch(arrow)

    # Panel B: actor breadth over time
    ax_b.plot(years, observed_breadth, color=COLORS["observed"], lw=2.2, label="Observed")
    ax_b.fill_between(
        years,
        hist_one["mean_active_topics_q05"],
        hist_one["mean_active_topics_q95"],
        color=COLORS["one_stage"],
        alpha=0.35,
        lw=0,
    )
    ax_b.plot(
        years,
        hist_one["mean_active_topics_mean"],
        color=COLORS["one_stage"],
        lw=1.6,
        alpha=0.9,
        linestyle="--",
        label="Direct allocation",
    )
    ax_b.fill_between(
        years,
        hist_two["mean_active_topics_q05"],
        hist_two["mean_active_topics_q95"],
        color=COLORS["two_stage"],
        alpha=0.35,
        lw=0,
    )
    ax_b.plot(
        years,
        hist_two["mean_active_topics_mean"],
        color=COLORS["two_stage"],
        lw=1.6,
        alpha=0.9,
        linestyle=":",
        label="Single-rule support",
    )
    ax_b.fill_between(
        years,
        hist_split["mean_active_topics_q05"],
        hist_split["mean_active_topics_q95"],
        color=COLORS["split_support"],
        alpha=0.35,
        lw=0,
    )
    ax_b.plot(
        years,
        hist_split["mean_active_topics_mean"],
        color=COLORS["split_support"],
        lw=2.0,
        alpha=0.95,
        label="Retain-and-enter",
    )
    ax_b.format(
        title="Mean Active Topics Per Actor",
        xlabel="Year",
        ylabel="Topics",
        grid=True,
    )
    ax_b.text(
        0.98,
        0.97,
        (
            f"Observed mean = {summary['mean_active_topics_obs_avg']:.2f}\n"
            f"Split model = {summary['mean_active_topics_split_avg']:.2f}\n"
            f"$r$ = {summary['corr_mean_active_topics_split']:.3f}\n"
            f"Band = 5--95% over {int(summary['process_uncertainty_reps'])} runs"
        ),
        transform=ax_b.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"),
    )

    # Panel C: topic popularity over time
    ax_c.plot(
        years,
        observed_popularity,
        color=COLORS["observed"],
        lw=2.2,
        label="Observed",
    )
    ax_c.fill_between(
        years,
        hist_one["mean_topic_popularity_q05"],
        hist_one["mean_topic_popularity_q95"],
        color=COLORS["one_stage"],
        alpha=0.35,
        lw=0,
    )
    ax_c.plot(
        years,
        hist_one["mean_topic_popularity_mean"],
        color=COLORS["one_stage"],
        lw=1.6,
        alpha=0.9,
        linestyle="--",
        label="Direct allocation",
    )
    ax_c.fill_between(
        years,
        hist_two["mean_topic_popularity_q05"],
        hist_two["mean_topic_popularity_q95"],
        color=COLORS["two_stage"],
        alpha=0.35,
        lw=0,
    )
    ax_c.plot(
        years,
        hist_two["mean_topic_popularity_mean"],
        color=COLORS["two_stage"],
        lw=1.6,
        alpha=0.9,
        linestyle=":",
        label="Single-rule support",
    )
    ax_c.fill_between(
        years,
        hist_split["mean_topic_popularity_q05"],
        hist_split["mean_topic_popularity_q95"],
        color=COLORS["split_support"],
        alpha=0.35,
        lw=0,
    )
    ax_c.plot(
        years,
        hist_split["mean_topic_popularity_mean"],
        color=COLORS["split_support"],
        lw=2.0,
        alpha=0.95,
        label="Retain-and-enter",
    )
    ax_c.format(
        title="Mean Topic Popularity",
        xlabel="Year",
        ylabel="Active actors per topic",
        grid=True,
    )
    ax_c.text(
        0.98,
        0.97,
        (
            f"Observed mean = {summary['mean_topic_popularity_obs_avg']:.2f}\n"
            f"Split model = {summary['mean_topic_popularity_split_avg']:.2f}\n"
            f"$r$ = {summary['corr_mean_topic_popularity_split']:.3f}\n"
            f"Band = 5--95% over {int(summary['process_uncertainty_reps'])} runs"
        ),
        transform=ax_c.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"),
    )

    # Panel D: Phi-local entry comparison
    entry_one = process_entry[process_entry["model"] == "one_stage"].iloc[0]
    entry_two = process_entry[process_entry["model"] == "two_stage"].iloc[0]
    entry_split = process_entry[process_entry["model"] == "split_support"].iloc[0]
    entry_obs = process_entry[process_entry["model"] == "observed"].iloc[0]
    models = ["Direct allocation", "Single-rule support", "Retain-and-enter"]
    values = [
        float(entry_one["mean_entry_phi_rank_mean"]),
        float(entry_two["mean_entry_phi_rank_mean"]),
        float(entry_split["mean_entry_phi_rank_mean"]),
    ]
    yerr = [
        [
            float(entry_one["mean_entry_phi_rank_mean"] - entry_one["mean_entry_phi_rank_q05"]),
            float(entry_two["mean_entry_phi_rank_mean"] - entry_two["mean_entry_phi_rank_q05"]),
            float(entry_split["mean_entry_phi_rank_mean"] - entry_split["mean_entry_phi_rank_q05"]),
        ],
        [
            float(entry_one["mean_entry_phi_rank_q95"] - entry_one["mean_entry_phi_rank_mean"]),
            float(entry_two["mean_entry_phi_rank_q95"] - entry_two["mean_entry_phi_rank_mean"]),
            float(entry_split["mean_entry_phi_rank_q95"] - entry_split["mean_entry_phi_rank_mean"]),
        ],
    ]
    colors = [COLORS["one_stage"], COLORS["two_stage"], COLORS["split_support"]]
    ax_d.bar(
        models,
        values,
        yerr=yerr,
        color=colors,
        width=0.62,
        edgecolor="#333333",
        linewidth=0.6,
        capsize=3,
        error_kw=dict(ecolor="#333333", lw=0.8),
    )
    ax_d.axhline(
        float(entry_obs["mean_entry_phi_rank_mean"]),
        color=COLORS["observed"],
        lw=1.8,
        linestyle="-",
        label="Observed",
    )
    ax_d.axhline(
        0.5,
        color=COLORS["baseline"],
        lw=1.2,
        linestyle="--",
        label="Random baseline",
    )
    ax_d.format(
        title=r"Local Entry In $\Phi$-Space",
        ylabel="Mean rank of entered topics",
        ylim=(0.45, 0.76),
        grid="y",
    )
    ax_d.text(
        0.98,
        0.97,
        (
            f"Observed = {entry_obs['mean_entry_phi_rank_mean']:.3f}\n"
            f"Split model = {entry_split['mean_entry_phi_rank_mean']:.3f}\n"
            f"5--95% = [{entry_split['mean_entry_phi_rank_q05']:.3f}, {entry_split['mean_entry_phi_rank_q95']:.3f}]"
        ),
        transform=ax_d.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"),
    )

    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=2.2),
        Line2D([0], [0], color=COLORS["one_stage"], lw=1.6, linestyle="--"),
        Line2D([0], [0], color=COLORS["two_stage"], lw=1.6, linestyle=":"),
        Line2D([0], [0], color=COLORS["split_support"], lw=2.0),
    ]
    labels = ["Observed", "Direct allocation", "Single-rule support", "Retain-and-enter"]
    fig.legend(handles, labels, loc="b", ncols=4, frame=False)
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Wrote", OUT_PDF)
    print("Wrote", OUT_PNG)


if __name__ == "__main__":
    main()
