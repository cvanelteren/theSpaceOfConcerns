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
SLIDE_DIR = PROJECT_ROOT / "slides"

SUMMARY_PATH = OUTPUT_DIR / "actor_topic_modeling_starter_summary.json"
PROCESS_HISTORY = OUTPUT_DIR / "actor_topic_modeling_starter_process_uncertainty_history.csv"
PROCESS_ENTRY = OUTPUT_DIR / "actor_topic_modeling_starter_process_uncertainty_entry.csv"

OUT_PDF = FIG_DIR / "fig04_split_support_validation.pdf"
OUT_PNG = FIG_DIR / "fig04_split_support_validation.png"

SLIDE_OUTPUTS = {
    "observed_only": {
        "title": "Observed Data",
        "label": "Observed",
        "stem": "fig04_observed_only_validation_slide",
    },
    "one_stage": {
        "title": "Model 1: Direct Allocation",
        "label": "Direct allocation",
        "stem": "fig04_direct_allocation_validation_slide",
    },
    "two_stage": {
        "title": "Model 2: Pooled Support",
        "label": "Pooled support",
        "stem": "fig04_pooled_support_validation_slide",
    },
    "split_support": {
        "title": "Model 3: Split Support",
        "label": "Split support",
        "stem": "fig04_split_support_validation_slide",
    },
}

STAGE_SEQUENCE = {
    "observed_only": [],
    "one_stage": ["one_stage"],
    "two_stage": ["one_stage", "two_stage"],
    "split_support": ["one_stage", "two_stage", "split_support"],
}

MODEL_DISPLAY = {
    "one_stage": "Direct allocation",
    "two_stage": "Single-rule support",
    "split_support": "Retain-and-enter",
}

MODEL_SHORT = {
    "one_stage": "Direct",
    "two_stage": "Single-rule",
    "split_support": "Retain-enter",
}


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


def _metric_map(model_key: str) -> dict[str, str]:
    return {
        "one_stage": {
            "breadth_mean": "mean_active_topics_mean",
            "breadth_q05": "mean_active_topics_q05",
            "breadth_q95": "mean_active_topics_q95",
            "breadth_avg": "mean_active_topics_sim_avg",
            "breadth_corr": "corr_mean_active_topics",
            "pop_mean": "mean_topic_popularity_mean",
            "pop_q05": "mean_topic_popularity_q05",
            "pop_q95": "mean_topic_popularity_q95",
            "pop_avg": "mean_topic_popularity_sim_avg",
            "pop_corr": "corr_mean_topic_popularity",
        },
        "two_stage": {
            "breadth_mean": "mean_active_topics_mean",
            "breadth_q05": "mean_active_topics_q05",
            "breadth_q95": "mean_active_topics_q95",
            "breadth_avg": "mean_active_topics_two_stage_avg",
            "breadth_corr": "corr_mean_active_topics_two_stage",
            "pop_mean": "mean_topic_popularity_mean",
            "pop_q05": "mean_topic_popularity_q05",
            "pop_q95": "mean_topic_popularity_q95",
            "pop_avg": "mean_topic_popularity_two_stage_avg",
            "pop_corr": "corr_mean_topic_popularity_two_stage",
        },
        "split_support": {
            "breadth_mean": "mean_active_topics_mean",
            "breadth_q05": "mean_active_topics_q05",
            "breadth_q95": "mean_active_topics_q95",
            "breadth_avg": "mean_active_topics_split_avg",
            "breadth_corr": "corr_mean_active_topics_split",
            "pop_mean": "mean_topic_popularity_mean",
            "pop_q05": "mean_topic_popularity_q05",
            "pop_q95": "mean_topic_popularity_q95",
            "pop_avg": "mean_topic_popularity_split_avg",
            "pop_corr": "corr_mean_topic_popularity_split",
        },
    }[model_key]


def _plot_breadth_panel(
    ax: plt.Axes,
    summary: pd.Series,
    observed: pd.DataFrame,
    process_history: pd.DataFrame,
    active_models: list[str],
    focus_model: str,
    *,
    show_title: bool = True,
    show_inset: bool = True,
) -> None:
    years = observed["window_end"]
    ax.plot(years, observed["mean_active_topics_obs"], color=COLORS["observed"], lw=2.3)
    for model_key in active_models:
        metrics = _metric_map(model_key)
        history = process_history[process_history["model"] == model_key].sort_values("window_end")
        ax.fill_between(
            years,
            history[metrics["breadth_q05"]],
            history[metrics["breadth_q95"]],
            color=COLORS[model_key],
            alpha=0.35,
            lw=0,
        )
        ax.plot(years, history[metrics["breadth_mean"]], color=COLORS[model_key], lw=2.0)
    focus_metrics = _metric_map(focus_model)
    ax.format(
        title="Actor Breadth" if show_title else "",
        xlabel="Year",
        ylabel="Active topics" if not show_title else "Active topics per actor",
        grid=True,
        ylim=(0.0, 9.0) if not show_title else None,
    )
    if show_inset:
        ax.text(
            0.98,
            0.97,
            (
                f"Observed mean = {summary['mean_active_topics_obs_avg']:.2f}\n"
                f"{MODEL_DISPLAY[focus_model]} = {summary[focus_metrics['breadth_avg']]:.2f}\n"
                f"$r$ = {summary[focus_metrics['breadth_corr']]:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"),
        )


def _plot_popularity_panel(
    ax: plt.Axes,
    summary: pd.Series,
    observed: pd.DataFrame,
    process_history: pd.DataFrame,
    active_models: list[str],
    focus_model: str,
    *,
    show_title: bool = True,
    show_inset: bool = True,
) -> None:
    years = observed["window_end"]
    ax.plot(years, observed["mean_topic_popularity_obs"], color=COLORS["observed"], lw=2.3)
    for model_key in active_models:
        metrics = _metric_map(model_key)
        history = process_history[process_history["model"] == model_key].sort_values("window_end")
        ax.fill_between(
            years,
            history[metrics["pop_q05"]],
            history[metrics["pop_q95"]],
            color=COLORS[model_key],
            alpha=0.35,
            lw=0,
        )
        ax.plot(years, history[metrics["pop_mean"]], color=COLORS[model_key], lw=2.0)
    focus_metrics = _metric_map(focus_model)
    ax.format(
        title="Topic Popularity" if show_title else "",
        xlabel="Year",
        ylabel="Actors per topic" if not show_title else "Active actors per topic",
        grid=True,
        ylim=(0.0, 7.0) if not show_title else None,
    )
    if show_inset:
        ax.text(
            0.98,
            0.97,
            (
                f"Observed mean = {summary['mean_topic_popularity_obs_avg']:.2f}\n"
                f"{MODEL_DISPLAY[focus_model]} = {summary[focus_metrics['pop_avg']]:.2f}\n"
                f"$r$ = {summary[focus_metrics['pop_corr']]:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"),
        )


def _plot_entry_panel(
    ax: plt.Axes,
    process_entry: pd.DataFrame,
    active_models: list[str],
    focus_model: str,
    *,
    show_title: bool = True,
    show_inset: bool = True,
) -> None:
    entry_obs = process_entry[process_entry["model"] == "observed"].iloc[0]
    x = [0, 1, 2]
    keys = ["one_stage", "two_stage", "split_support"]
    labels = [MODEL_SHORT[key] if key in active_models else "" for key in keys]
    ax.axhline(float(entry_obs["mean_entry_phi_rank_mean"]), color=COLORS["observed"], lw=1.8)
    ax.axhline(0.5, color=COLORS["baseline"], lw=1.2, linestyle="--")
    for xpos, model_key in zip(x, keys):
        if model_key not in active_models:
            continue
        entry_model = process_entry[process_entry["model"] == model_key].iloc[0]
        model_value = float(entry_model["mean_entry_phi_rank_mean"])
        yerr = [
            [float(model_value - entry_model["mean_entry_phi_rank_q05"])],
            [float(entry_model["mean_entry_phi_rank_q95"] - model_value)],
        ]
        ax.bar(
            [xpos],
            [model_value],
            yerr=yerr,
            color=[COLORS[model_key]],
            width=0.62,
            edgecolor="#333333",
            linewidth=0.6,
            capsize=3,
            error_kw=dict(ecolor="#333333", lw=0.8),
        )
    focus_entry = process_entry[process_entry["model"] == focus_model].iloc[0]
    ax.format(
        title=r"Local Entry In $\Phi$-Space" if show_title else "",
        ylabel="Locality rank" if not show_title else "Mean rank of entered topics",
        ylim=(0.45, 0.76),
        xlim=(-0.5, 2.5),
        xticks=x,
        xticklabels=labels,
        grid="y",
    )
    if show_inset:
        ax.text(
            0.98,
            0.97,
            (
                f"Observed = {entry_obs['mean_entry_phi_rank_mean']:.3f}\n"
                f"{MODEL_DISPLAY[focus_model]} = {focus_entry['mean_entry_phi_rank_mean']:.3f}\n"
                f"5--95% = [{focus_entry['mean_entry_phi_rank_q05']:.3f}, {focus_entry['mean_entry_phi_rank_q95']:.3f}]"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"),
        )


def _build_model_slide(
    summary: pd.Series,
    process_history: pd.DataFrame,
    process_entry: pd.DataFrame,
    model_key: str,
) -> plt.Figure:
    config = SLIDE_OUTPUTS[model_key]
    observed = process_history[process_history["model"] == "split_support"].sort_values("window_end")
    active_models = STAGE_SEQUENCE[model_key]

    fig, axs = uplt.subplots(ncols=3, share=0, refnum=2)
    ax_b, ax_c, ax_d = axs
    _plot_breadth_panel(
        ax_b,
        summary,
        observed,
        process_history,
        active_models,
        focus_model="one_stage" if model_key == "observed_only" else model_key,
        show_title=False,
        show_inset=False,
    )
    _plot_popularity_panel(
        ax_c,
        summary,
        observed,
        process_history,
        active_models,
        focus_model="one_stage" if model_key == "observed_only" else model_key,
        show_title=False,
        show_inset=False,
    )
    _plot_entry_panel(
        ax_d,
        process_entry,
        active_models,
        focus_model="one_stage" if model_key == "observed_only" else model_key,
        show_title=False,
        show_inset=False,
    )

    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=2.2),
        *[Line2D([0], [0], color=COLORS[key], lw=2.0) for key in active_models],
        Line2D([0], [0], color=COLORS["baseline"], lw=1.2, linestyle="--"),
    ]
    labels = ["Observed", *[MODEL_DISPLAY[key] for key in active_models], "Random baseline"]
    fig.legend(handles, labels, loc="b", ncols=len(labels), frame=False)
    return fig


def _build_metric_slide(
    summary: pd.Series,
    process_history: pd.DataFrame,
    process_entry: pd.DataFrame,
    model_key: str,
    metric: str,
) -> plt.Figure:
    config = SLIDE_OUTPUTS[model_key]
    active_models = STAGE_SEQUENCE[model_key]
    observed = process_history[process_history["model"] == "split_support"].sort_values("window_end")
    fig, ax = uplt.subplots(refnum=2)
    focus_model = "one_stage" if model_key == "observed_only" else model_key
    if metric == "breadth":
        _plot_breadth_panel(
            ax,
            summary,
            observed,
            process_history,
            active_models,
            focus_model,
            show_title=False,
            show_inset=False,
        )
    elif metric == "popularity":
        _plot_popularity_panel(
            ax,
            summary,
            observed,
            process_history,
            active_models,
            focus_model,
            show_title=False,
            show_inset=False,
        )
    elif metric == "entry":
        _plot_entry_panel(
            ax,
            process_entry,
            active_models,
            focus_model,
            show_title=False,
            show_inset=False,
        )
    else:
        raise ValueError(metric)
    return fig


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
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Wrote", OUT_PDF)
    print("Wrote", OUT_PNG)

    summary = _load_summary()
    process_history = _load_history(PROCESS_HISTORY)
    process_entry = _load_history(PROCESS_ENTRY)
    for model_key, config in SLIDE_OUTPUTS.items():
        slide_fig = _build_model_slide(summary, process_history, process_entry, model_key)
        pdf_path = SLIDE_DIR / f"{config['stem']}.pdf"
        png_path = SLIDE_DIR / f"{config['stem']}.png"
        slide_fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        slide_fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(slide_fig)
        print("Wrote", pdf_path)
        print("Wrote", png_path)
        for metric in ("breadth", "popularity", "entry"):
            metric_fig = _build_metric_slide(summary, process_history, process_entry, model_key, metric)
            metric_pdf = SLIDE_DIR / f"{config['stem']}_{metric}.pdf"
            metric_png = SLIDE_DIR / f"{config['stem']}_{metric}.png"
            metric_fig.savefig(metric_pdf, dpi=300, bbox_inches="tight")
            metric_fig.savefig(metric_png, dpi=300, bbox_inches="tight", transparent=True)
            plt.close(metric_fig)
            print("Wrote", metric_pdf)
            print("Wrote", metric_png)


if __name__ == "__main__":
    main()
