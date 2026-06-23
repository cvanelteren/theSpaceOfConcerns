from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "figures"
SLIDE_DIR = PROJECT_ROOT / "slides"

SUMMARY_PATH = OUTPUT_DIR / "actor_topic_modeling_starter_summary.json"
PROCESS_HISTORY = (
    OUTPUT_DIR / "actor_topic_modeling_starter_process_uncertainty_history.csv"
)
PROCESS_ENTRY = (
    OUTPUT_DIR / "actor_topic_modeling_starter_process_uncertainty_entry.csv"
)

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
    "one_stage": "Direct\nallocation",
    "two_stage": "Single\nsupport",
    "split_support": "Retain-and-adopt",
}

MODEL_SHORT = {
    "one_stage": "Direct\nallocation",
    "two_stage": "Single\nsupport",
    "split_support": "Retain-\nand-adopt",
}


COLORS = {
    "observed": "#111111",
    "one_stage": "#c44e52",
    "two_stage": "#4c72b0",
    "split_support": "#2a9d55",
    "baseline": "#9a9a9a",
}

SCHEMATIC_MARGIN = 0.16
SCHEMATIC_SPAN = 0.68
RIGHT_STATE_SCALE = 0.40
CURRENT_INSET_BOUNDS = (0.02, 0.08, 0.33, 0.80)
FINAL_INSET_BOUNDS = (0.72, 0.11, 0.23, 0.70)
STAGE_LABEL_Y_OFFSET = 0.0
DIRECT_STAGE_X = 0.55
SUPPORT_STAGE_X = 0.46
ALLOCATE_STAGE_X = 0.61
STAGE_BOX_HALF_WIDTHS = {
    "Choose support": 0.056,
    "Retain + adopt": 0.062,
    "Allocate": 0.046,
}
ROW_Y = (0.74, 0.50, 0.26)
ROW_INSET_Y = (0.85, 0.50, 0.158)
ROW_LABEL_X = 0.395


def _load_summary() -> pd.Series:
    return pd.read_json(SUMMARY_PATH, typ="series")


def _load_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _time_axis_label(summary: pd.Series) -> str:
    time_unit = str(summary.get("time_unit", "year")).strip().lower()
    return "Meeting" if time_unit == "meeting" else "Year"


SCHEMATIC_GRAPH = nx.florentine_families_graph().copy()
_SCHEMATIC_LAYOUT = nx.kamada_kawai_layout(SCHEMATIC_GRAPH)
_xs = [xy[0] for xy in _SCHEMATIC_LAYOUT.values()]
_ys = [xy[1] for xy in _SCHEMATIC_LAYOUT.values()]
_xmin, _xmax = min(_xs), max(_xs)
_ymin, _ymax = min(_ys), max(_ys)
SCHEMATIC_POS = {
    node: (
        SCHEMATIC_MARGIN + SCHEMATIC_SPAN * ((xy[0] - _xmin) / (_xmax - _xmin)),
        SCHEMATIC_MARGIN + SCHEMATIC_SPAN * ((xy[1] - _ymin) / (_ymax - _ymin)),
    )
    for node, xy in _SCHEMATIC_LAYOUT.items()
}
SCHEMATIC_EDGES = list(SCHEMATIC_GRAPH.edges())


def _draw_portfolio_schematic(
    ax: plt.Axes,
    *,
    active_nodes: dict[str, float],
    title: str,
    subtitle: str,
    accent: str,
    added_nodes: set[str] | None = None,
    time_label: str | None = None,
    show_reach_outline: bool = False,
    reach_label: str | None = None,
) -> None:
    added_nodes = added_nodes or set()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.patch.set_visible(False)
    ax.text(0.02, 0.98, title, ha="left", va="top", fontsize=10.5, fontweight="bold")
    ax.text(0.02, 0.90, subtitle, ha="left", va="top", fontsize=8.3, color="#444444")

    reachable_nodes = set()
    if show_reach_outline:
        for u, v in SCHEMATIC_EDGES:
            if u in active_nodes and v not in active_nodes:
                reachable_nodes.add(v)
            if v in active_nodes and u not in active_nodes:
                reachable_nodes.add(u)
        _draw_reach_hull(ax, reachable_nodes)

    for u, v in SCHEMATIC_EDGES:
        x0, y0 = SCHEMATIC_POS[u]
        x1, y1 = SCHEMATIC_POS[v]
        edge_color = "#d4d4d4"
        edge_width = 1.0
        if u in active_nodes and v in active_nodes:
            edge_color = accent
            edge_width = 1.0 + 2.2 * min(active_nodes[u], active_nodes[v])
        (line,) = ax.plot(
            [x0, x1],
            [y0, y1],
            color=edge_color,
            lw=edge_width,
            solid_capstyle="round",
            zorder=1,
            clip_on=True,
        )
        line.set_clip_path(ax.patch)

    for node, (x, y) in SCHEMATIC_POS.items():
        value = active_nodes.get(node, 0.0)
        radius = 0.020 if value <= 0 else 0.024 + 0.027 * value
        face = "#efefef"
        edge = "#bbbbbb"
        lw = 1.0
        if value > 0:
            face = "#1a1a1a"  # retained/original topics stay black
            edge = "#2f2f2f"
            lw = 0.8
        if node in added_nodes:
            face = accent  # only newly adopted topics carry the model color
            edge = "#111111"
            lw = 1.6
        if node in reachable_nodes:
            reach_patch = Circle(
                (x, y),
                radius=radius * 1.8,
                facecolor="none",
                edgecolor="#a6a6a6",
                lw=1.1,
                ls=(0, (3, 2)),
                zorder=2,
                clip_on=True,
            )
            reach_patch.set_clip_path(ax.patch)
            ax.add_patch(reach_patch)
        node_patch = Circle(
            (x, y),
            radius=radius,
            facecolor=face,
            edgecolor=edge,
            lw=lw,
            zorder=3,
            clip_on=True,
        )
        node_patch.set_clip_path(ax.patch)
        ax.add_patch(node_patch)

    if show_reach_outline and reach_label:
        ax.text(
            0.88,
            0.30,
            reach_label,
            ha="right",
            va="center",
            fontsize=7.8,
            color="#666666",
            zorder=0,
        )

    if time_label is not None:
        ax.text(
            0.96,
            0.16,
            time_label,
            ha="right",
            va="center",
            fontsize=10,
            color="#666666",
        )


_SCHEMATIC_X_CENTER = sum(x for x, _ in SCHEMATIC_POS.values()) / len(SCHEMATIC_POS)
_SCHEMATIC_Y_CENTER = sum(y for _, y in SCHEMATIC_POS.values()) / len(SCHEMATIC_POS)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 2:
        return points
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def _chaikin_closed(points: np.ndarray, n_iter: int = 2) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    for _ in range(n_iter):
        new_pts = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            new_pts.append(0.75 * p0 + 0.25 * p1)
            new_pts.append(0.25 * p0 + 0.75 * p1)
        pts = np.asarray(new_pts, dtype=float)
    return pts


def _draw_reach_hull(ax: plt.Axes, reachable_nodes: set[str]) -> None:
    if len(reachable_nodes) < 3:
        return
    pts = np.array([SCHEMATIC_POS[node] for node in reachable_nodes], dtype=float)
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return
    centroid = hull.mean(axis=0)
    expanded = centroid + 1.22 * (hull - centroid)
    smooth = _chaikin_closed(expanded, n_iter=2)
    patch = Polygon(
        smooth,
        closed=True,
        facecolor="#d2d2d2",
        edgecolor="#8b8b8b",
        linewidth=1.0,
        alpha=0.28,
        joinstyle="round",
        zorder=1.8,
        clip_on=True,
    )
    patch.set_clip_path(ax.patch)
    ax.add_patch(patch)


def _draw_network_state(
    ax: plt.Axes,
    *,
    active_nodes: dict[str, float],
    accent: str,
    added_nodes: set[str] | None = None,
    x_center: float,
    y_center: float,
    scale: float,
) -> None:
    added_nodes = added_nodes or set()
    for u, v in SCHEMATIC_EDGES:
        x0, y0 = SCHEMATIC_POS[u]
        x1, y1 = SCHEMATIC_POS[v]
        x0 = x_center + (x0 - _SCHEMATIC_X_CENTER) * scale
        y0 = y_center + (y0 - _SCHEMATIC_Y_CENTER) * scale
        x1 = x_center + (x1 - _SCHEMATIC_X_CENTER) * scale
        y1 = y_center + (y1 - _SCHEMATIC_Y_CENTER) * scale
        edge_color = "#d4d4d4"
        edge_width = 0.9
        if u in active_nodes and v in active_nodes:
            edge_color = accent
            edge_width = 1.0 + 2.0 * min(active_nodes[u], active_nodes[v])
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=edge_color,
            lw=edge_width,
            solid_capstyle="round",
            zorder=1,
        )

    for node, (x, y) in SCHEMATIC_POS.items():
        x = x_center + (x - _SCHEMATIC_X_CENTER) * scale
        y = y_center + (y - _SCHEMATIC_Y_CENTER) * scale
        value = active_nodes.get(node, 0.0)
        radius = scale * (0.036 if value <= 0 else 0.041 + 0.027 * value)
        face = "#efefef"
        edge = "#bbbbbb"
        lw = 0.9
        if value > 0:
            face = "#1a1a1a"  # retained/original topics stay black
            edge = "#2f2f2f"
            lw = 0.8
        if node in added_nodes:
            face = accent  # only newly adopted topics carry the model color
            edge = "#111111"
            lw = 1.5
        ax.add_patch(
            Circle(
                (x, y), radius=radius, facecolor=face, edgecolor=edge, lw=lw, zorder=3
            )
        )


def _draw_stage_label(
    ax: plt.Axes, x: float, y: float, text: str, color: str = "#555555"
) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.2,
        color=color,
        bbox=dict(facecolor="white", edgecolor="#d0d0d0", boxstyle="round,pad=0.18"),
        zorder=12,
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
        history = process_history[process_history["model"] == model_key].sort_values(
            "window_end"
        )
        ax.fill_between(
            years,
            history[metrics["breadth_q05"]],
            history[metrics["breadth_q95"]],
            color=COLORS[model_key],
            alpha=0.35,
            lw=0,
        )
        ax.plot(
            years, history[metrics["breadth_mean"]], color=COLORS[model_key], lw=2.0
        )
    focus_metrics = _metric_map(focus_model)
    ax.format(
        title="Actor Breadth" if show_title else "",
        xlabel=_time_axis_label(summary),
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
            bbox=dict(
                facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"
            ),
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
    ax.plot(
        years, observed["mean_topic_popularity_obs"], color=COLORS["observed"], lw=2.3
    )
    for model_key in active_models:
        metrics = _metric_map(model_key)
        history = process_history[process_history["model"] == model_key].sort_values(
            "window_end"
        )
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
        xlabel=_time_axis_label(summary),
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
            bbox=dict(
                facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"
            ),
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
    ax.axhline(
        float(entry_obs["mean_entry_phi_rank_mean"]), color=COLORS["observed"], lw=1.8
    )
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
    ax.tick_params(axis="x", labelsize=7.5 if show_title else 8)
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
            bbox=dict(
                facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25"
            ),
        )


def _build_model_slide(
    summary: pd.Series,
    process_history: pd.DataFrame,
    process_entry: pd.DataFrame,
    model_key: str,
) -> plt.Figure:
    config = SLIDE_OUTPUTS[model_key]
    observed = process_history[process_history["model"] == "split_support"].sort_values(
        "window_end"
    )
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
    labels = [
        "Observed",
        *[MODEL_DISPLAY[key] for key in active_models],
        "Random baseline",
    ]
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
    observed = process_history[process_history["model"] == "split_support"].sort_values(
        "window_end"
    )
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
    hist_one = process_history[process_history["model"] == "one_stage"].sort_values(
        "window_end"
    )
    hist_two = process_history[process_history["model"] == "two_stage"].sort_values(
        "window_end"
    )
    hist_split = process_history[
        process_history["model"] == "split_support"
    ].sort_values("window_end")

    years = hist_split["window_end"]
    observed_breadth = hist_split["mean_active_topics_obs"]
    observed_popularity = hist_split["mean_topic_popularity_obs"]

    layout = [
        [1, 1, 1],
        [2, 3, 4],
    ]
    fig, axs = uplt.subplots(layout, share=0, refnum=2, hratios=(1.05, 1.0))
    ax_a, ax_b, ax_c, ax_d = axs
    axs.format(abc="[A]", abcloc="ul")

    # Panel A: portfolio evolution schematic
    ax_a.format(
        xlim=(0, 1),
        ylim=(0, 1),
        xlocator=[],
        ylocator=[],
    )
    ax_a.set_axis_off()
    ax_a.text(
        0.5,
        0.99,
        "Three alternative updates from the same portfolio at $t$",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111111",
    )

    current_portfolio = {"Medici": 0.38, "Ridolfi": 0.30, "Tornabuoni": 0.26}
    direct_portfolio = {
        "Medici": 0.15,
        "Ridolfi": 0.13,
        "Tornabuoni": 0.12,
        "Albizzi": 0.11,
        "Barbadori": 0.09,
        "Guadagni": 0.12,
        "Salviati": 0.09,
        "Strozzi": 0.10,
    }
    pooled_support = {
        "Medici": 0.28,
        "Ridolfi": 0.22,
        "Tornabuoni": 0.18,
        "Guadagni": 0.17,
        "Salviati": 0.15,
    }
    pooled_portfolio = {
        "Medici": 0.28,
        "Ridolfi": 0.21,
        "Tornabuoni": 0.17,
        "Guadagni": 0.20,
        "Salviati": 0.14,
    }
    retained_portfolio = {"Medici": 0.36, "Ridolfi": 0.29, "Tornabuoni": 0.24}
    split_portfolio = {
        "Medici": 0.31,
        "Ridolfi": 0.25,
        "Tornabuoni": 0.19,
        "Guadagni": 0.13,
    }

    current_inset = ax_a.inset_axes(list(CURRENT_INSET_BOUNDS), zoom=0)
    _draw_portfolio_schematic(
        current_inset,
        active_nodes=current_portfolio,
        title="Current portfolio",
        subtitle="Observed support.",
        accent=COLORS["observed"],
        added_nodes=set(),
        time_label=r"$t$",
        show_reach_outline=True,
        reach_label="reachable\nin one step",
    )

    band_x0, band_x1 = FINAL_INSET_BOUNDS[0] - 0.02, 0.98
    band_y0, band_h = 0.83, 0.08
    ax_a.add_patch(
        Rectangle(
            (band_x0, band_y0),
            band_x1 - band_x0,
            band_h,
            facecolor="#ececec",
            edgecolor="none",
            zorder=2,
        )
    )
    ax_a.text(
        (band_x0 + band_x1) / 2,
        band_y0 + band_h / 2,
        r"$t+1$",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
        zorder=3,
    )

    final_x0, final_y0, final_w, final_h = FINAL_INSET_BOUNDS
    final_xc = final_x0 + final_w / 2
    final_inset = ax_a.inset_axes(list(FINAL_INSET_BOUNDS), zoom=0)
    final_inset.set_axis_off()
    final_inset.set_xlim(0, 1)
    final_inset.set_ylim(0, 1)
    final_inset.set_aspect("equal", adjustable="box")

    rows = [
        {
            "y": ROW_Y[0],
            "y_inset": ROW_INSET_Y[0],
            "color": COLORS["one_stage"],
            "label": "Direct allocation",
            "stages": [("Allocate", DIRECT_STAGE_X), ("t+1", final_xc)],
            "portfolio": direct_portfolio,
            "added": {"Albizzi", "Barbadori", "Guadagni", "Salviati", "Strozzi"},
        },
        {
            "y": ROW_Y[1],
            "y_inset": ROW_INSET_Y[1],
            "color": COLORS["two_stage"],
            "label": "Single support",
            "stages": [
                ("Choose support", SUPPORT_STAGE_X),
                ("Allocate", ALLOCATE_STAGE_X),
                ("t+1", final_xc),
            ],
            "portfolio": pooled_portfolio,
            "support": pooled_support,
            "added": {"Guadagni", "Salviati"},
        },
        {
            "y": ROW_Y[2],
            "y_inset": ROW_INSET_Y[2],
            "color": COLORS["split_support"],
            "label": "Retain-and-adopt",
            "stages": [
                ("Retain + adopt", SUPPORT_STAGE_X),
                ("Allocate", ALLOCATE_STAGE_X),
                ("t+1", final_xc),
            ],
            "portfolio": split_portfolio,
            "support": retained_portfolio,
            "added": {"Guadagni"},
        },
    ]

    branch_start_x = CURRENT_INSET_BOUNDS[0] + CURRENT_INSET_BOUNDS[2]
    for row in rows:
        y = row["y"]
        color = row["color"]
        ax_a.text(
            ROW_LABEL_X,
            y + 0.09,
            row["label"],
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=color,
        )
        prev = (branch_start_x, y)
        for stage, x_stage in row["stages"]:
            stage_half_width = STAGE_BOX_HALF_WIDTHS.get(stage, 0.06)
            target_x = x_stage - stage_half_width if stage != "t+1" else final_x0 - 0.02
            target = (target_x, y)
            arrow = FancyArrowPatch(
                prev,
                target,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#444444",
                zorder=10,
                connectionstyle="arc3,rad=0.0",
            )
            ax_a.add_patch(arrow)
            if stage != "t+1":
                _draw_stage_label(ax_a, x_stage, y + STAGE_LABEL_Y_OFFSET, stage)
                prev = (x_stage + stage_half_width, y)
            else:
                _draw_network_state(
                    final_inset,
                    active_nodes=row["portfolio"],
                    accent=color,
                    added_nodes=row["added"],
                    x_center=0.50,
                    y_center=row["y_inset"],
                    scale=RIGHT_STATE_SCALE,
                )
                prev = (final_x0 + final_w, y)

    ax_a.text(
        0.02,
        0.03,
        "Observed actors, budgets $K_{at}$, and the pooled concern space $\\Phi$ are fixed.\nThe three rules differ only in how they update support from $t$ to $t+1$.",
        fontsize=8.5,
        ha="left",
        va="bottom",
        color="#333333",
    )

    # Panel B: actor breadth over time
    ax_b.plot(
        years, observed_breadth, color=COLORS["observed"], lw=2.2, label="Observed"
    )
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
        label="Retain-and-adopt",
    )
    ax_b.format(
        title="",
        xlabel=_time_axis_label(summary),
        ylabel="Mean active topics per actor",
        grid=True,
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
        label="Retain-and-adopt",
    )
    ax_c.format(
        title="",
        xlabel=_time_axis_label(summary),
        ylabel="Mean active actors per topic",
        grid=True,
    )
    # Panel D: Phi-local entry comparison
    entry_one = process_entry[process_entry["model"] == "one_stage"].iloc[0]
    entry_two = process_entry[process_entry["model"] == "two_stage"].iloc[0]
    entry_split = process_entry[process_entry["model"] == "split_support"].iloc[0]
    entry_obs = process_entry[process_entry["model"] == "observed"].iloc[0]
    models = [
        MODEL_DISPLAY["one_stage"],
        MODEL_DISPLAY["two_stage"],
        MODEL_DISPLAY["split_support"],
    ]
    values = [
        float(entry_one["mean_entry_phi_rank_mean"]),
        float(entry_two["mean_entry_phi_rank_mean"]),
        float(entry_split["mean_entry_phi_rank_mean"]),
    ]
    yerr = [
        [
            float(
                entry_one["mean_entry_phi_rank_mean"]
                - entry_one["mean_entry_phi_rank_q05"]
            ),
            float(
                entry_two["mean_entry_phi_rank_mean"]
                - entry_two["mean_entry_phi_rank_q05"]
            ),
            float(
                entry_split["mean_entry_phi_rank_mean"]
                - entry_split["mean_entry_phi_rank_q05"]
            ),
        ],
        [
            float(
                entry_one["mean_entry_phi_rank_q95"]
                - entry_one["mean_entry_phi_rank_mean"]
            ),
            float(
                entry_two["mean_entry_phi_rank_q95"]
                - entry_two["mean_entry_phi_rank_mean"]
            ),
            float(
                entry_split["mean_entry_phi_rank_q95"]
                - entry_split["mean_entry_phi_rank_mean"]
            ),
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
        title="",
        ylabel="Mean rank of entered topics",
        ylim=(0.45, 0.76),
        grid="y",
    )
    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=2.2),
        Line2D([0], [0], color=COLORS["one_stage"], lw=1.6, linestyle="--"),
        Line2D([0], [0], color=COLORS["two_stage"], lw=1.6, linestyle=":"),
        Line2D([0], [0], color=COLORS["split_support"], lw=2.0),
        Line2D([0], [0], color=COLORS["baseline"], lw=1.2, linestyle="--"),
    ]
    labels = [
        "Observed",
        MODEL_DISPLAY["one_stage"],
        MODEL_DISPLAY["two_stage"],
        MODEL_DISPLAY["split_support"],
        "Random baseline (panel D)",
    ]
    fig.legend(handles, labels, loc="b", ncols=5, frame=False)
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
        slide_fig = _build_model_slide(
            summary, process_history, process_entry, model_key
        )
        pdf_path = SLIDE_DIR / f"{config['stem']}.pdf"
        png_path = SLIDE_DIR / f"{config['stem']}.png"
        slide_fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        slide_fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(slide_fig)
        print("Wrote", pdf_path)
        print("Wrote", png_path)
        for metric in ("breadth", "popularity", "entry"):
            metric_fig = _build_metric_slide(
                summary, process_history, process_entry, model_key, metric
            )
            metric_pdf = SLIDE_DIR / f"{config['stem']}_{metric}.pdf"
            metric_png = SLIDE_DIR / f"{config['stem']}_{metric}.png"
            metric_fig.savefig(metric_pdf, dpi=300, bbox_inches="tight")
            metric_fig.savefig(
                metric_png, dpi=300, bbox_inches="tight", transparent=True
            )
            plt.close(metric_fig)
            print("Wrote", metric_pdf)
            print("Wrote", metric_png)


if __name__ == "__main__":
    main()
