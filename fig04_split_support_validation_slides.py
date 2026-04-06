"""Slide frames for Figure 4: one PNG per subplot panel.

Outputs (in figures/):
  fig04_slide_A_schematic.png      — model schematic (Panel A)
  fig04_slide_B_breadth.png        — actor breadth over time (Panel B)
  fig04_slide_C_popularity.png     — topic popularity over time (Panel C)
  fig04_slide_D_entry.png          — local phi-entry comparison (Panel D)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import ultraplot as uplt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fig04_split_support_validation import (
    COLORS,
    FIG_DIR,
    MODEL_DISPLAY,
    OUTPUT_DIR,
    PROCESS_ENTRY,
    PROCESS_HISTORY,
    SUMMARY_PATH,
    _draw_network_state,
    _draw_portfolio_schematic,
    _draw_reach_hull,
    _draw_stage_label,
    _load_history,
    _load_summary,
    _time_axis_label,
    build_figure,
    SCHEMATIC_POS,
    SCHEMATIC_EDGES,
    CURRENT_INSET_BOUNDS,
    FINAL_INSET_BOUNDS,
    ROW_Y,
    ROW_INSET_Y,
    ROW_LABEL_X,
    RIGHT_STATE_SCALE,
    STAGE_BOX_HALF_WIDTHS,
    STAGE_LABEL_Y_OFFSET,
    DIRECT_STAGE_X,
    SUPPORT_STAGE_X,
    ALLOCATE_STAGE_X,
    _SCHEMATIC_X_CENTER,
    _SCHEMATIC_Y_CENTER,
)
import fig04_split_support_validation as _f04

SLIDE_FONT = 13
TICK_FONT = 11
INSET_FONT = 11
DPI = 180


def _legend_handles() -> tuple[list, list]:
    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=2.5),
        Line2D([0], [0], color=COLORS["one_stage"], lw=2.0, linestyle="--"),
        Line2D([0], [0], color=COLORS["two_stage"], lw=2.0, linestyle=":"),
        Line2D([0], [0], color=COLORS["split_support"], lw=2.5),
        Line2D([0], [0], color=COLORS["baseline"], lw=1.5, linestyle="--"),
    ]
    labels = [
        "Observed",
        "Direct allocation",
        "Single-rule support",
        "Retain-and-adopt",
        "Random baseline",
    ]
    return handles, labels


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in [(".pdf", {"transparent": True}), (".png", {"dpi": DPI, "transparent": True})]:
        path = FIG_DIR / f"{stem}{ext}"
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"Wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel A — schematic (redrawn standalone)
# ---------------------------------------------------------------------------
def build_panel_a(summary) -> plt.Figure:
    fig, ax = uplt.subplots(refwidth=9.0, refaspect=1.9)
    ax.format(xlim=(0, 1), ylim=(0, 1), xlocator=[], ylocator=[])
    ax.set_axis_off()

    ax.text(
        0.5, 0.99,
        "Three alternative updates from the same portfolio at $t$",
        ha="center", va="top", fontsize=SLIDE_FONT + 1, fontweight="bold",
        color="#111111",
    )

    current_portfolio = {"Medici": 0.38, "Ridolfi": 0.30, "Tornabuoni": 0.26}
    direct_portfolio = {
        "Medici": 0.15, "Ridolfi": 0.13, "Tornabuoni": 0.12,
        "Albizzi": 0.11, "Barbadori": 0.09, "Guadagni": 0.12,
        "Salviati": 0.09, "Strozzi": 0.10,
    }
    pooled_portfolio = {
        "Medici": 0.28, "Ridolfi": 0.21, "Tornabuoni": 0.17,
        "Guadagni": 0.20, "Salviati": 0.14,
    }
    retained_portfolio = {"Medici": 0.36, "Ridolfi": 0.29, "Tornabuoni": 0.24}
    split_portfolio = {
        "Medici": 0.31, "Ridolfi": 0.25, "Tornabuoni": 0.19, "Guadagni": 0.13,
    }

    current_inset = ax.inset_axes(list(CURRENT_INSET_BOUNDS), zoom=0)
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

    from matplotlib.patches import Rectangle, FancyArrowPatch

    # Per-row network insets: each centered exactly on its arrow's y-position.
    # Using one inset per row avoids the shared-inset aspect-ratio misalignment.
    NET_X0 = 0.73   # left edge of each network inset in ax coords
    NET_W  = 0.27   # width  (ax fraction)
    NET_H  = 0.23   # height (ax fraction) — row spacing is 0.24 so this leaves a 0.01 gap

    # t+1 header band — placed above the top network
    band_x0, band_x1 = NET_X0 - 0.02, 0.98
    band_y0, band_h = ROW_Y[0] + NET_H / 2 + 0.03, 0.07
    ax.add_patch(Rectangle(
        (band_x0, band_y0), band_x1 - band_x0, band_h,
        facecolor="#ececec", edgecolor="none", zorder=2,
    ))
    ax.text(
        (band_x0 + band_x1) / 2, band_y0 + band_h / 2, r"$t+1$",
        ha="center", va="center", fontsize=SLIDE_FONT, color="#444444", zorder=3,
    )

    rows = [
        {
            "y": ROW_Y[0],
            "color": COLORS["one_stage"], "label": "Direct allocation",
            "stages": [("Allocate", DIRECT_STAGE_X), ("t+1", NET_X0 + NET_W / 2)],
            "portfolio": direct_portfolio,
            "added": {"Albizzi", "Barbadori", "Guadagni", "Salviati", "Strozzi"},
        },
        {
            "y": ROW_Y[1],
            "color": COLORS["two_stage"], "label": "Single support",
            "stages": [
                ("Choose support", SUPPORT_STAGE_X),
                ("Allocate", ALLOCATE_STAGE_X),
                ("t+1", NET_X0 + NET_W / 2),
            ],
            "portfolio": pooled_portfolio,
            "added": {"Guadagni", "Salviati"},
        },
        {
            "y": ROW_Y[2],
            "color": COLORS["split_support"], "label": "Retain-and-adopt",
            "stages": [
                ("Retain + adopt", SUPPORT_STAGE_X),
                ("Allocate", ALLOCATE_STAGE_X),
                ("t+1", NET_X0 + NET_W / 2),
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
        ax.text(
            ROW_LABEL_X, y + 0.09, row["label"],
            ha="left", va="center", fontsize=SLIDE_FONT, fontweight="bold", color=color,
        )
        prev = (branch_start_x, y)
        for stage, x_stage in row["stages"]:
            stage_half_width = STAGE_BOX_HALF_WIDTHS.get(stage, 0.06)
            target_x = x_stage - stage_half_width if stage != "t+1" else NET_X0 - 0.02
            arrow = FancyArrowPatch(
                prev, (target_x, y),
                arrowstyle="-|>", mutation_scale=16, linewidth=1.5,
                color="#444444", zorder=10, connectionstyle="arc3,rad=0.0",
            )
            ax.add_patch(arrow)
            if stage != "t+1":
                _draw_stage_label(ax, x_stage, y + STAGE_LABEL_Y_OFFSET, stage)
                prev = (x_stage + stage_half_width, y)
            else:
                # Each row gets its own inset, centered exactly on the arrow y
                net_inset = ax.inset_axes(
                    [NET_X0, y - NET_H / 2, NET_W, NET_H], zoom=0
                )
                net_inset.set_axis_off()
                net_inset.set_xlim(0, 1)
                net_inset.set_ylim(0, 1)
                net_inset.set_aspect("equal", adjustable="box")
                _draw_network_state(
                    net_inset,
                    active_nodes=row["portfolio"],
                    accent=color,
                    added_nodes=row["added"],
                    x_center=0.5,
                    y_center=0.5,
                    scale=0.82,
                )
                prev = (NET_X0 + NET_W, y)

    ax.text(
        0.02, 0.03,
        "Observed actors, budgets $K_{at}$, and the pooled concern space $\\Phi$ are fixed.\n"
        "The three rules differ only in how they update support from $t$ to $t+1$.",
        fontsize=SLIDE_FONT - 2, ha="left", va="bottom", color="#333333",
    )
    return fig


# ---------------------------------------------------------------------------
# Panel B — actor breadth (full + sequential reveal frames)
# ---------------------------------------------------------------------------
_B_STAGES = [
    # (model_key, linestyle, label)  — cumulative: each stage adds the next model
    ("one_stage",    "--", "Direct allocation"),
    ("two_stage",    ":",  "Single-rule support"),
    ("split_support", "-", "Retain-and-adopt"),
]


def _breadth_ylim(process_history) -> tuple[float, float]:
    """Compute shared y-range across all models' bands and observed series."""
    import numpy as np
    lo, hi = np.inf, -np.inf
    for key in ("one_stage", "two_stage", "split_support"):
        h = process_history[process_history["model"] == key]
        lo = min(lo, float(h["mean_active_topics_q05"].min()),
                 float(h["mean_active_topics_obs"].min()))
        hi = max(hi, float(h["mean_active_topics_q95"].max()),
                 float(h["mean_active_topics_obs"].max()))
    pad = (hi - lo) * 0.08
    return (max(0.0, lo - pad), hi + pad)


def _build_breadth_ax(ax, summary, process_history, active_keys: list[str], ylim) -> None:
    hist_split = process_history[process_history["model"] == "split_support"].sort_values("window_end")
    years = hist_split["window_end"]
    ax.plot(years, hist_split["mean_active_topics_obs"], color=COLORS["observed"], lw=2.5)
    for key, ls, _ in _B_STAGES:
        if key not in active_keys:
            continue
        hist = process_history[process_history["model"] == key].sort_values("window_end")
        ax.fill_between(
            years, hist["mean_active_topics_q05"], hist["mean_active_topics_q95"],
            color=COLORS[key], alpha=0.30, lw=0,
        )
        ax.plot(years, hist["mean_active_topics_mean"], color=COLORS[key], lw=2.0, linestyle=ls)

    ax.format(
        title="Mean Active Topics Per Actor",
        xlabel=_time_axis_label(summary), ylabel="Topics per actor",
        ylim=ylim, grid=True,
        fontsize=SLIDE_FONT, ticklabelsize=TICK_FONT, labelsize=SLIDE_FONT,
    )


def build_panel_b(summary, process_history) -> plt.Figure:
    """Full Panel B (all models)."""
    ylim = _breadth_ylim(process_history)
    fig, ax = uplt.subplots(refwidth=7.0, refaspect=1.6)
    _build_breadth_ax(ax, summary, process_history, [k for k, _, _ in _B_STAGES], ylim)
    handles, labels = _legend_handles()
    fig.legend(handles, labels, loc="b", ncols=5, frame=False, fontsize=TICK_FONT)
    return fig


def build_panel_b_frames(summary, process_history) -> list[tuple[str, plt.Figure]]:
    """Sequential reveal frames: observed → +direct → +single → +retain-and-adopt."""
    ylim = _breadth_ylim(process_history)
    frames = []

    # Frame 0: observed only
    fig, ax = uplt.subplots(refwidth=7.0, refaspect=1.6)
    _build_breadth_ax(ax, summary, process_history, [], ylim)
    handles = [Line2D([0], [0], color=COLORS["observed"], lw=2.5)]
    fig.legend(handles, ["Observed"], loc="b", ncols=1, frame=False, fontsize=TICK_FONT)
    frames.append(("fig04_slide_B_breadth_f0_observed", fig))

    # Frames 1–3: cumulative model addition
    for i, (key, _, _label) in enumerate(_B_STAGES, start=1):
        active = [k for k, _, _ in _B_STAGES[:i]]
        fig, ax = uplt.subplots(refwidth=7.0, refaspect=1.6)
        _build_breadth_ax(ax, summary, process_history, active, ylim)
        h = [Line2D([0], [0], color=COLORS["observed"], lw=2.5)]
        lb = ["Observed"]
        for k, ls, lbl in _B_STAGES[:i]:
            h.append(Line2D([0], [0], color=COLORS[k], lw=2.0, linestyle=ls))
            lb.append(lbl)
        fig.legend(h, lb, loc="b", ncols=len(h), frame=False, fontsize=TICK_FONT)
        frames.append((f"fig04_slide_B_breadth_f{i}_{key}", fig))

    return frames


# ---------------------------------------------------------------------------
# Panel C — topic popularity
# ---------------------------------------------------------------------------
def build_panel_c(summary, process_history) -> plt.Figure:
    hist_one = process_history[process_history["model"] == "one_stage"].sort_values("window_end")
    hist_two = process_history[process_history["model"] == "two_stage"].sort_values("window_end")
    hist_split = process_history[process_history["model"] == "split_support"].sort_values("window_end")
    years = hist_split["window_end"]

    fig, ax = uplt.subplots(refwidth=7.0, refaspect=1.6)
    ax.plot(years, hist_split["mean_topic_popularity_obs"], color=COLORS["observed"], lw=2.5)
    for hist, key, ls in [
        (hist_one, "one_stage", "--"),
        (hist_two, "two_stage", ":"),
        (hist_split, "split_support", "-"),
    ]:
        ax.fill_between(
            years, hist["mean_topic_popularity_q05"], hist["mean_topic_popularity_q95"],
            color=COLORS[key], alpha=0.30, lw=0,
        )
        ax.plot(years, hist["mean_topic_popularity_mean"], color=COLORS[key], lw=2.0, linestyle=ls)

    ax.text(
        0.98, 0.97,
        (
            f"Observed mean = {summary['mean_topic_popularity_obs_avg']:.2f}\n"
            f"Retain-and-adopt = {summary['mean_topic_popularity_split_avg']:.2f}\n"
            f"$r$ = {summary['corr_mean_topic_popularity_split']:.3f}\n"
            f"Band = 5–95% over {int(summary['process_uncertainty_reps'])} runs"
        ),
        transform=ax.transAxes, va="top", ha="right", fontsize=INSET_FONT,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    ax.format(
        title="Mean Topic Popularity",
        xlabel=_time_axis_label(summary), ylabel="Active actors per topic",
        grid=True, fontsize=SLIDE_FONT, ticklabelsize=TICK_FONT, labelsize=SLIDE_FONT,
    )
    handles, labels = _legend_handles()
    fig.legend(handles, labels, loc="b", ncols=5, frame=False, fontsize=TICK_FONT)
    return fig


# ---------------------------------------------------------------------------
# Panel D — local phi-entry
# ---------------------------------------------------------------------------
def build_panel_d(summary, process_entry) -> plt.Figure:
    entry_obs = process_entry[process_entry["model"] == "observed"].iloc[0]
    entry_one = process_entry[process_entry["model"] == "one_stage"].iloc[0]
    entry_two = process_entry[process_entry["model"] == "two_stage"].iloc[0]
    entry_split = process_entry[process_entry["model"] == "split_support"].iloc[0]

    models_labels = [MODEL_DISPLAY[k] for k in ("one_stage", "two_stage", "split_support")]
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

    fig, ax = uplt.subplots(refwidth=5.5, refaspect=1.2)
    ax.bar(
        models_labels, values, yerr=yerr, color=colors,
        width=0.62, edgecolor="#333333", linewidth=0.6,
        capsize=4, error_kw=dict(ecolor="#333333", lw=1.0),
    )
    ax.axhline(float(entry_obs["mean_entry_phi_rank_mean"]), color=COLORS["observed"], lw=2.0)
    ax.axhline(0.5, color=COLORS["baseline"], lw=1.5, linestyle="--")

    ax.text(
        0.98, 0.97,
        (
            f"Observed = {entry_obs['mean_entry_phi_rank_mean']:.3f}\n"
            f"Retain-and-adopt = {entry_split['mean_entry_phi_rank_mean']:.3f}\n"
            f"5–95% = [{entry_split['mean_entry_phi_rank_q05']:.3f}, "
            f"{entry_split['mean_entry_phi_rank_q95']:.3f}]"
        ),
        transform=ax.transAxes, va="top", ha="right", fontsize=INSET_FONT,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    ax.format(
        title=r"Local Entry In $\Phi$-Space",
        ylabel="Mean rank of entered topics",
        ylim=(0.45, 0.76), grid="y",
        fontsize=SLIDE_FONT, ticklabelsize=TICK_FONT, labelsize=SLIDE_FONT,
    )
    ax.tick_params(axis="x", labelsize=TICK_FONT)

    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=2.0),
        Line2D([0], [0], color=COLORS["baseline"], lw=1.5, linestyle="--"),
    ]
    fig.legend(handles, ["Observed", "Random baseline"], loc="b", ncols=2, frame=False, fontsize=TICK_FONT)
    return fig


def main() -> None:
    summary = _load_summary()
    process_history = _load_history(PROCESS_HISTORY)
    process_entry = _load_history(PROCESS_ENTRY)

    save(build_panel_a(summary), "fig04_slide_A_schematic")
    for stem, fig in build_panel_b_frames(summary, process_history):
        save(fig, stem)
    save(build_panel_b(summary, process_history), "fig04_slide_B_breadth")
    save(build_panel_c(summary, process_history), "fig04_slide_C_popularity")
    save(build_panel_d(summary, process_entry), "fig04_slide_D_entry")


if __name__ == "__main__":
    main()
