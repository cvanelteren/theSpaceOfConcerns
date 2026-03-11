# %%
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import NullLocator

from utils import get_rca, load_data, load_flag, standardize_index_labels

# %%
DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
TOPIC_FP = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
ACTOR_FP = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")
REGION_FP = Path("output/fig45_portfolio_space_ridgelines_region_summary.csv")

OUT_A = Path("output/fig02_regime_exemplars_ridgelines.png")
OUT_A_ALL = Path("output/fig02_regime_allmembers_ridgelines.png")
SLIDES_DIR = Path("slides")
OUT_A_SLIDE = SLIDES_DIR / "fig02_regime_exemplars_ridgelines_slide_clean.png"
OUT_A_ALL_SLIDE = SLIDES_DIR / "fig02_regime_allmembers_ridgelines_slide_clean.png"
OUT_B = Path("output/fig02_centroid_separation.png")
OUT_C = Path("output/fig02_regime_share_fractions.png")
OUT_JOINT = Path("output/fig02_regime_engagement_panels.png")
OUT_JOINT_ALL = Path("figures/fig02_regime_engagement_panels_allmembers.png")
OUT_SUBSET = Path("output/fig02_regime_exemplar_actors.csv")

RIDGE_HEIGHT = 0.78
SMOOTH_RADIUS = 2
SMOOTH_SIGMA = 1.05
N_EXEMPLARS_PER_REGIME = 6
MIN_DOMINANT_SHARE = 0.60
FS_TITLE = 16
FS_LABEL = 13
FS_TICK = 11
FS_SMALL = 10
FS_ABC = 14
FS_ACTOR_YTICK = 13
FLAG_ZOOM = 0.029
SIDE_MARKER_FRACS = (0.2, 0.4, 0.6, 0.8)
PEAK_LABEL_N = 8
PEAK_LABEL_MIN_SEP = 2
PEAK_LABEL_MAX_LEN = 20
TOP_LABEL_PEAK_N = 16
TOP_LABEL_PEAK_MIN_SEP = 1
TOP_LABEL_TARGET_N = 16
TOP_LABEL_CANDIDATE_N = 36
TOP_LABEL_BIN_N = 14
TOP_LABEL_MIN_PROFILE_FRAC = 0.20
TOP_LABEL_SELECTION_MIN_DX = 0.028
TOP_LABEL_MERGE_DX = 0.010
RIDGE_TOP_PAD = 1.32
ALL_INTEREST_DOT_COLOR = "#9ca3af"
ACTIVE_INTEREST_DOT_COLOR = "black"
ALL_INTEREST_DOT_SIZE = 7
ACTIVE_INTEREST_DOT_SIZE = 13
TOP_LABEL_FONTSIZE = 7.6
TOP_LABEL_BASE_PT = 7.0
TOP_LABEL_ROW_STEP_PT = 22.0
TOP_LABEL_PAD_PX = 16.0
TOP_LABEL_GAP_PX = 18.0
TOP_LABEL_MAX_SHIFT_PX = 3.0
TOP_LABEL_EXTRA_SHIFT_PER_ROW_PX = 10.0
TOP_LABEL_ROW_PENALTY_PX = 1.0
TOP_LABEL_STAGGER_CYCLE = 5
TOP_LABEL_STAGGER_STEP_PT = 7.0
TOP_LABEL_STAGGER_ROW_PREFERENCE_PX = 6.0
TOP_LABEL_WRAP_CHARS = 11
TOP_LABEL_MAX_LINES = 2
TOP_LABEL_SPECIAL_Y_NUDGE_PT = {
    "educ": 22.0,
    "opening": 28.0,
    "env. domains": 28.0,
    "env domains": 28.0,
}
REGIME_LABEL_Y_DELTA = -0.22
FORCED_TOP_LABEL_KEYWORDS = (
    "mineral resources",
    "drilling",
    "marine living resources",
)

REGION_COLORS = {
    1: "#e41a1c",  # red
    2: "#377eb8",  # blue
    3: "#4daf4a",  # green
}
RIDGE_COLOR = "#4b5563"


def _smooth_1d(
    values: np.ndarray, radius: int = SMOOTH_RADIUS, sigma: float = SMOOTH_SIGMA
):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _jitter(n: int, width: float = 0.12):
    if n <= 1:
        return np.zeros(1, dtype=float)
    return np.linspace(-width, width, n)


def _short_topic_label(label: str, max_len: int = PEAK_LABEL_MAX_LEN) -> str:
    text = str(label).replace("_", " ").strip()
    replacements = {
        "International": "Intl.",
        "Information": "Info.",
        "Management": "Mgmt.",
        "Environmental": "Env.",
        "Environment": "Env.",
        "Organizations": "Orgs",
        "Organization": "Org.",
        "Operations": "Ops.",
        "Operation": "Op.",
        "Activities": "Acts.",
        "Activity": "Act.",
        "Resources": "Res.",
        "Resource": "Res.",
        "Protected": "Prot.",
        "Conservation": "Conserv.",
        "Statements": "Stmts.",
        "Statement": "Stmt.",
        "Exchange": "Exch.",
        "Monitoring": "Monit.",
        "Research": "Res.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return _wrap_label_lines(
            text, max_line_chars=TOP_LABEL_WRAP_CHARS, max_lines=TOP_LABEL_MAX_LINES
        )

    words = text.split(" ")
    if len(words) > 1:
        compact_words = []
        for w in words:
            if len(w) > 7 and not w.endswith("."):
                compact_words.append(w[:4] + ".")
            else:
                compact_words.append(w)
        compact_text = " ".join(compact_words)
        if len(compact_text) <= max_len:
            return _wrap_label_lines(
                compact_text,
                max_line_chars=TOP_LABEL_WRAP_CHARS,
                max_lines=TOP_LABEL_MAX_LINES,
            )
        text = compact_text

    trimmed = text[: max_len - 3].rstrip() + "..."
    return _wrap_label_lines(
        trimmed, max_line_chars=TOP_LABEL_WRAP_CHARS, max_lines=TOP_LABEL_MAX_LINES
    )


def _wrap_label_lines(text: str, max_line_chars: int, max_lines: int) -> str:
    """Wrap short labels into at most `max_lines` lines."""
    if max_line_chars <= 0 or max_lines <= 1 or len(text) <= max_line_chars:
        return text

    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        candidate = f"{current} {w}"
        if len(candidate) <= max_line_chars:
            current = candidate
        else:
            lines.append(current)
            current = w
            if len(lines) >= max_lines - 1:
                break
    lines.append(current)

    used_word_count = sum(len(line.split()) for line in lines)
    if used_word_count < len(words):
        tail = lines[-1]
        if len(tail) >= max_line_chars - 1:
            tail = tail[: max(1, max_line_chars - 2)].rstrip()
        lines[-1] = tail.rstrip(".") + "..."

    return "\n".join(lines[:max_lines])


def _mean_normalized_profile(rca_df: pd.DataFrame, actors: list[str]) -> np.ndarray:
    curves = []
    for actor in actors:
        vals = np.clip(rca_df[actor].to_numpy(dtype=float), 0.0, None)
        ridge = _smooth_1d(vals)
        m = float(np.nanmax(ridge))
        if m > 0:
            ridge = ridge / m
        curves.append(ridge)
    if not curves:
        return np.zeros(rca_df.shape[0], dtype=float)
    return np.nanmean(np.vstack(curves), axis=0)


def _find_peak_indices(
    profile: np.ndarray, top_n: int = PEAK_LABEL_N, min_sep: int = PEAK_LABEL_MIN_SEP
) -> list[int]:
    y = np.asarray(profile, dtype=float)
    if y.size < 3:
        return list(range(min(y.size, top_n)))
    cand = []
    for idx in range(1, y.size - 1):
        if y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1]:
            cand.append(idx)
    if not cand:
        cand = list(np.argsort(y)[::-1][: top_n * 2])
    cand_sorted = sorted(cand, key=lambda i: y[i], reverse=True)
    keep: list[int] = []
    for idx in cand_sorted:
        if all(abs(idx - j) >= min_sep for j in keep):
            keep.append(int(idx))
        if len(keep) >= top_n:
            break
    return sorted(keep)


def _annotate_profile_peaks(
    ax,
    x_plot: np.ndarray,
    topic_labels: list[str],
    profile: np.ndarray,
    *,
    y_anchor: float,
    y_text: float,
):
    peak_idx = _find_peak_indices(profile)
    for idx in peak_idx:
        x = float(x_plot[idx])
        label = _short_topic_label(topic_labels[idx])
        ax.plot([x, x], [y_anchor, y_text + 0.01], color="#6b7280", lw=0.65, zorder=6)
        ax.text(
            x,
            y_text,
            label,
            rotation=55,
            ha="left",
            va="bottom",
            fontsize=FS_SMALL - 1,
            color="#374151",
            zorder=7,
        )


def _plot_interest_dots(ax, x_plot: np.ndarray, vals: np.ndarray, baseline: float):
    vals = np.asarray(vals, dtype=float)
    all_mask = np.isfinite(vals) & (vals > 0.0)
    if all_mask.any():
        ax.scatter(
            x_plot[all_mask],
            np.full(np.count_nonzero(all_mask), baseline, dtype=float),
            s=ALL_INTEREST_DOT_SIZE,
            color=ALL_INTEREST_DOT_COLOR,
            alpha=0.95,
            linewidth=0,
            zorder=4,
        )
    active_mask = np.isfinite(vals) & (vals > 1.0)
    if active_mask.any():
        ax.scatter(
            x_plot[active_mask],
            np.full(np.count_nonzero(active_mask), baseline, dtype=float),
            s=ACTIVE_INTEREST_DOT_SIZE,
            color=ACTIVE_INTEREST_DOT_COLOR,
            edgecolor="white",
            linewidth=0.25,
            zorder=5,
        )


def _add_non_overlapping_top_labels(ax, x_positions: np.ndarray, labels: list[str]):
    if len(x_positions) == 0 or len(labels) == 0:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    x_display = ax.transData.transform(
        np.column_stack([x_positions, np.zeros_like(x_positions)])
    )[:, 0]
    bbox_style = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "#6b7280",
        "linewidth": 0.72,
        "alpha": 0.98,
    }

    widths = []
    heights = []
    for label in labels:
        tmp = fig.text(
            0,
            0,
            label,
            fontsize=TOP_LABEL_FONTSIZE,
            fontweight="semibold",
            visible=False,
            bbox=bbox_style,
        )
        bbox = tmp.get_window_extent(renderer=renderer)
        widths.append(float(bbox.width) + TOP_LABEL_PAD_PX)
        heights.append(float(bbox.height))
        tmp.remove()

    max_h_px = max(heights) if heights else 0.0
    row_step_pt = max(TOP_LABEL_ROW_STEP_PT, (max_h_px + 16.0) * 72.0 / fig.dpi)

    order = np.argsort(x_display)
    row_indices = np.zeros(len(labels), dtype=int)
    x_offsets_pt = np.zeros(len(labels), dtype=float)
    stagger_offsets_pt = np.zeros(len(labels), dtype=float)
    preferred_rows = np.zeros(len(labels), dtype=int)
    row_right_edges: list[float] = []

    def _label_y_nudge_pt(label_text: str) -> float:
        label_key = str(label_text).replace("\n", " ").strip().lower()
        for key, delta in TOP_LABEL_SPECIAL_Y_NUDGE_PT.items():
            if key in label_key:
                return float(delta)
        return 0.0

    for rank, idx in enumerate(order):
        target_x_px = float(x_display[idx])
        width_px = float(widths[idx])
        half_width_px = width_px / 2.0
        min_center_px = float(axes_bbox.x0 + TOP_LABEL_GAP_PX + half_width_px)
        max_center_px = float(axes_bbox.x1 - TOP_LABEL_GAP_PX - half_width_px)
        target_center_px = min(max(target_x_px, min_center_px), max_center_px)
        stagger_phase = rank % TOP_LABEL_STAGGER_CYCLE
        preferred_rows[idx] = int(stagger_phase)
        stagger_offsets_pt[idx] = float(
            (TOP_LABEL_STAGGER_CYCLE - 1 - stagger_phase) * TOP_LABEL_STAGGER_STEP_PT
        )

        best_row = None
        best_center_px = None
        best_cost = None
        for row_idx, last_right in enumerate(row_right_edges):
            center_px = max(
                target_center_px,
                float(last_right) + TOP_LABEL_GAP_PX + half_width_px,
            )
            if center_px > max_center_px:
                continue
            shift_px = abs(center_px - target_x_px)
            allowed_shift_px = TOP_LABEL_MAX_SHIFT_PX + (
                row_idx * TOP_LABEL_EXTRA_SHIFT_PER_ROW_PX
            )
            if shift_px > allowed_shift_px:
                continue
            cost = (
                shift_px
                + row_idx * TOP_LABEL_ROW_PENALTY_PX
                + abs(row_idx - preferred_rows[idx])
                * TOP_LABEL_STAGGER_ROW_PREFERENCE_PX
            )
            if best_cost is None or cost < best_cost:
                best_row = row_idx
                best_center_px = center_px
                best_cost = cost

        if best_row is None:
            best_row = len(row_right_edges)
            best_center_px = target_center_px
            row_right_edges.append(best_center_px + half_width_px)
        else:
            row_right_edges[best_row] = best_center_px + half_width_px

        row_indices[idx] = best_row
        x_offsets_pt[idx] = (best_center_px - target_x_px) * 72.0 / fig.dpi

    for xpos, label, row_idx, x_offset_pt, stagger_offset_pt in zip(
        x_positions, labels, row_indices, x_offsets_pt, stagger_offsets_pt
    ):
        y_offset_pt = (
            TOP_LABEL_BASE_PT
            + row_idx * row_step_pt
            + stagger_offset_pt
            + _label_y_nudge_pt(label)
        )
        ax.annotate(
            label,
            xy=(float(xpos), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(float(x_offset_pt), y_offset_pt),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=TOP_LABEL_FONTSIZE,
            fontweight="semibold",
            color="#374151",
            zorder=8,
            clip_on=False,
            bbox=bbox_style,
        )
        ax.annotate(
            "",
            xy=(float(xpos), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(float(x_offset_pt), y_offset_pt - 1.2),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "-",
                "color": "#9ca3af",
                "lw": 0.7,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=7,
            clip_on=False,
        )


def _unique_peak_topic_labels(
    x_positions: list[float] | np.ndarray,
    labels: list[str],
    *,
    min_dx: float = 0.018,
) -> tuple[np.ndarray, list[str]]:
    """Collapse repeated peak-topic labels and keep one top annotation per topic."""
    if len(labels) == 0:
        return np.array([], dtype=float), []

    accum: dict[str, list[float]] = {}
    for x, label in zip(x_positions, labels):
        key = str(label).strip()
        if not key:
            continue
        if key not in accum:
            accum[key] = [0.0, 0.0]  # sum_x, count
        accum[key][0] += float(x)
        accum[key][1] += 1.0

    if not accum:
        return np.array([], dtype=float), []

    uniq_labels = list(accum.keys())
    uniq_x = np.array([accum[k][0] / accum[k][1] for k in uniq_labels], dtype=float)
    uniq_n = np.array([accum[k][1] for k in uniq_labels], dtype=float)
    order = np.argsort(uniq_x)
    uniq_x = uniq_x[order]
    uniq_n = uniq_n[order]
    uniq_labels = [uniq_labels[i] for i in order]

    # Merge nearby peaks and keep the most frequent label in each x-neighborhood.
    merged_x = []
    merged_labels = []
    start = 0
    n = len(uniq_x)
    while start < n:
        stop = start + 1
        while stop < n and float(uniq_x[stop] - uniq_x[stop - 1]) <= float(min_dx):
            stop += 1
        cluster_x = uniq_x[start:stop]
        cluster_n = uniq_n[start:stop]
        cluster_labels = uniq_labels[start:stop]
        rep_idx = int(np.argmax(cluster_n))
        rep_label = cluster_labels[rep_idx]
        rep_x = float(np.average(cluster_x, weights=cluster_n))
        merged_x.append(rep_x)
        merged_labels.append(rep_label)
        start = stop

    return np.asarray(merged_x, dtype=float), merged_labels


def _append_forced_topic_labels(
    x_positions: list[float],
    labels: list[str],
    topics: list[str],
    x_plot_vals: np.ndarray,
):
    """Always include selected strategic topic labels in the top annotation list."""
    for idx, topic in enumerate(topics):
        topic_l = str(topic).strip().lower()
        if not any(key in topic_l for key in FORCED_TOP_LABEL_KEYWORDS):
            continue
        x_positions.append(float(x_plot_vals[idx]))
        labels.append(_short_topic_label(topic))


def _top_topic_indices_by_bin(
    profile: np.ndarray, x_plot_vals: np.ndarray, *, bin_n: int = TOP_LABEL_BIN_N
) -> list[int]:
    """Pick one strong label candidate from each x-range bin."""
    if bin_n <= 0:
        return []

    x_vals = np.asarray(x_plot_vals, dtype=float)
    y_vals = np.asarray(profile, dtype=float)
    if x_vals.size == 0 or y_vals.size == 0:
        return []

    max_score = float(np.nanmax(y_vals))
    if not np.isfinite(max_score) or max_score <= 0.0:
        return []
    min_score = TOP_LABEL_MIN_PROFILE_FRAC * max_score

    edges = np.linspace(float(np.nanmin(x_vals)), float(np.nanmax(x_vals)), bin_n + 1)
    keep: list[int] = []
    for bin_idx in range(bin_n):
        if bin_idx == bin_n - 1:
            mask = (x_vals >= edges[bin_idx]) & (x_vals <= edges[bin_idx + 1])
        else:
            mask = (x_vals >= edges[bin_idx]) & (x_vals < edges[bin_idx + 1])
        if not np.any(mask):
            continue
        cand_idx = np.where(mask)[0]
        best_local = int(np.nanargmax(y_vals[cand_idx]))
        best_idx = int(cand_idx[best_local])
        if float(y_vals[best_idx]) >= min_score:
            keep.append(best_idx)
    return keep


def _select_spread_topic_indices(
    profile: np.ndarray,
    x_plot_vals: np.ndarray,
    peak_idx: list[int],
    *,
    target_n: int = TOP_LABEL_TARGET_N,
) -> list[int]:
    """Blend local peaks with strong labels spread across topic space."""
    x_vals = np.asarray(x_plot_vals, dtype=float)
    y_vals = np.asarray(profile, dtype=float)
    if x_vals.size == 0 or y_vals.size == 0:
        return []

    peak_idx = [int(i) for i in peak_idx]
    peak_set = set(peak_idx)
    bin_idx = _top_topic_indices_by_bin(y_vals, x_vals)
    bin_set = set(bin_idx)
    score_idx = [
        int(i)
        for i in np.argsort(y_vals)[::-1][: min(len(y_vals), TOP_LABEL_CANDIDATE_N)]
    ]
    candidate_idx = list(dict.fromkeys(peak_idx + bin_idx + score_idx))
    if not candidate_idx:
        return []

    candidate_idx.sort(
        key=lambda i: (
            float(y_vals[i])
            + (0.14 if i in peak_set else 0.0)
            + (0.08 if i in bin_set else 0.0)
        ),
        reverse=True,
    )

    selected: list[int] = []
    min_dx_schedule = (
        TOP_LABEL_SELECTION_MIN_DX,
        max(TOP_LABEL_SELECTION_MIN_DX * 0.82, TOP_LABEL_MERGE_DX),
        max(TOP_LABEL_SELECTION_MIN_DX * 0.66, TOP_LABEL_MERGE_DX),
    )
    for min_dx in min_dx_schedule:
        for idx in candidate_idx:
            if idx in selected:
                continue
            x_val = float(x_vals[idx])
            if all(abs(x_val - float(x_vals[j])) >= float(min_dx) for j in selected):
                selected.append(int(idx))
            if len(selected) >= target_n:
                break
        if len(selected) >= target_n:
            break

    if not selected:
        return []
    return sorted(selected, key=lambda i: float(x_vals[i]))


def _aggregate_top_topic_labels(
    rca_df: pd.DataFrame,
    actors: list[str],
    topics: list[str],
    x_plot_vals: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Select readable top labels from the aggregate portfolio profile."""
    profile = _mean_normalized_profile(rca_df, actors)
    peak_idx = _find_peak_indices(
        profile, top_n=TOP_LABEL_PEAK_N, min_sep=TOP_LABEL_PEAK_MIN_SEP
    )
    selected_idx = _select_spread_topic_indices(profile, x_plot_vals, peak_idx)
    if not selected_idx:
        selected_idx = peak_idx
    x_positions = [float(x_plot_vals[idx]) for idx in selected_idx]
    labels = [_short_topic_label(topics[idx]) for idx in selected_idx]
    _append_forced_topic_labels(x_positions, labels, topics, x_plot_vals)
    return _unique_peak_topic_labels(x_positions, labels, min_dx=TOP_LABEL_MERGE_DX)


def _local_flag_name(actor: str) -> str:
    lower = actor.lower()
    if "korea (rok)" in lower:
        return "Korea"
    if "korea (dprk)" in lower:
        return "Korea, Democratic People's Republic of"
    return actor


def _load_local_flags(
    actors: list[str], base: str = "./assets/flags"
) -> dict[str, np.ndarray]:
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    flags: dict[str, np.ndarray] = {}
    for actor in actors:
        key = _local_flag_name(actor)
        has_country_flag = (base_path / f"{key}_flag.png").exists()
        has_logo = (base_path / f"{key.lower()}_logo.png").exists()
        if has_country_flag or has_logo:
            img = load_flag(actor, save=False, base=base)
            if img is not None:
                flags[actor] = img
    return flags


def _ternary_xy(shares: np.ndarray) -> np.ndarray:
    vertices = np.array(
        [
            [0.08, 0.07],  # Regime 1
            [0.92, 0.07],  # Regime 2
            [0.50, 0.86],  # Regime 3
        ],
        dtype=float,
    )
    return shares @ vertices


def _plot_regime_ternary(
    ax,
    actor_df: pd.DataFrame,
    flags: dict[str, np.ndarray],
    title: str | None,
):
    def _blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return (1.0 - t) * a + t * b

    v1 = np.array([0.08, 0.07], dtype=float)
    v2 = np.array([0.92, 0.07], dtype=float)
    v3 = np.array([0.50, 0.86], dtype=float)
    tri = np.vstack([v1, v2, v3, v1])

    ax.plot(tri[:, 0], tri[:, 1], color="#4b5563", lw=1.2, zorder=1)
    edge_markers = []
    for frac in SIDE_MARKER_FRACS:
        edge_markers.append(_blend(v1, v2, frac))
        edge_markers.append(_blend(v2, v3, frac))
        edge_markers.append(_blend(v3, v1, frac))
    edge_markers = np.asarray(edge_markers, dtype=float)
    ax.scatter(
        edge_markers[:, 0],
        edge_markers[:, 1],
        s=18,
        marker="o",
        facecolor="white",
        edgecolor="#6b7280",
        linewidth=0.7,
        zorder=1.6,
    )

    ax.scatter(
        [v1[0], v2[0], v3[0]],
        [v1[1], v2[1], v3[1]],
        s=30,
        marker="o",
        facecolor=[REGION_COLORS[1], REGION_COLORS[2], REGION_COLORS[3]],
        edgecolor="white",
        linewidth=0.7,
        zorder=1.8,
    )

    for frac in SIDE_MARKER_FRACS:
        label = f"{frac:.1f}"
        p_bottom = _blend(v1, v2, frac)
        p_right = _blend(v2, v3, frac)
        p_left = _blend(v3, v1, frac)
        ax.text(
            p_bottom[0],
            p_bottom[1] - 0.028,
            label,
            ha="center",
            va="top",
            fontsize=FS_SMALL - 1,
            color="#6b7280",
            zorder=2.1,
        )
        ax.text(
            p_right[0] + 0.022,
            p_right[1] + 0.003,
            label,
            ha="left",
            va="center",
            fontsize=FS_SMALL - 1,
            color="#6b7280",
            rotation=-60,
            zorder=2.1,
        )
        ax.text(
            p_left[0] - 0.022,
            p_left[1] + 0.003,
            label,
            ha="right",
            va="center",
            fontsize=FS_SMALL - 1,
            color="#6b7280",
            rotation=60,
            zorder=2.1,
        )

    for frac in SIDE_MARKER_FRACS:
        p12_a = frac * v1 + (1.0 - frac) * v3
        p12_b = frac * v2 + (1.0 - frac) * v3
        p23_a = frac * v2 + (1.0 - frac) * v1
        p23_b = frac * v3 + (1.0 - frac) * v1
        p13_a = frac * v1 + (1.0 - frac) * v2
        p13_b = frac * v3 + (1.0 - frac) * v2
        ax.plot(
            [p12_a[0], p12_b[0]],
            [p12_a[1], p12_b[1]],
            color="#9ca3af",
            lw=0.55,
            alpha=0.35,
            zorder=0,
        )
        ax.plot(
            [p23_a[0], p23_b[0]],
            [p23_a[1], p23_b[1]],
            color="#9ca3af",
            lw=0.55,
            alpha=0.35,
            zorder=0,
        )
        ax.plot(
            [p13_a[0], p13_b[0]],
            [p13_a[1], p13_b[1]],
            color="#9ca3af",
            lw=0.55,
            alpha=0.35,
            zorder=0,
        )

    shares = actor_df[["region_1_share", "region_2_share", "region_3_share"]].to_numpy(
        dtype=float
    )
    valid = np.isfinite(shares).all(axis=1)
    shares = shares[valid]
    actors = actor_df.loc[valid, "actor"].tolist()
    dominant = actor_df.loc[valid, "dominant_region"].to_numpy(dtype=int)
    row_sum = shares.sum(axis=1, keepdims=True)
    nonzero = row_sum[:, 0] > 0
    shares = shares[nonzero] / row_sum[nonzero]
    actors = [a for a, ok in zip(actors, nonzero) if ok]
    dominant = dominant[nonzero]
    xy = _ternary_xy(shares)

    for (x, y), actor, rid in zip(xy, actors, dominant):
        img = flags.get(actor)
        if img is not None:
            ab = AnnotationBbox(
                OffsetImage(img, zoom=FLAG_ZOOM),
                (x, y),
                frameon=True,
                bboxprops={
                    "boxstyle": "round,pad=0.06",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 1.15,
                    "alpha": 1.0,
                },
                box_alignment=(0.5, 0.5),
                zorder=3.5,
            )
            ax.add_artist(ab)
        else:
            ax.scatter(
                [x],
                [y],
                s=22,
                color=REGION_COLORS.get(int(rid), "#6b7280"),
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )

    ax.text(
        v1[0] - 0.03,
        v1[1] - 0.015,
        "Regime 1",
        ha="right",
        va="top",
        color=REGION_COLORS[1],
        fontsize=FS_SMALL + 1,
    )
    ax.text(
        v1[0] - 0.03,
        v1[1] - 0.075,
        "Coordination / Information",
        ha="right",
        va="top",
        color=REGION_COLORS[1],
        fontsize=FS_SMALL - 1,
    )
    ax.text(
        v2[0] + 0.03,
        v2[1] - 0.015,
        "Regime 2",
        ha="left",
        va="top",
        color=REGION_COLORS[2],
        fontsize=FS_SMALL + 1,
    )
    ax.text(
        v2[0] + 0.03,
        v2[1] - 0.075,
        "Compliance / Environment",
        ha="left",
        va="top",
        color=REGION_COLORS[2],
        fontsize=FS_SMALL - 1,
    )
    ax.text(
        v3[0],
        v3[1] + 0.05,
        "Regime 3",
        ha="center",
        va="bottom",
        color=REGION_COLORS[3],
        fontsize=FS_SMALL + 1,
    )
    ax.text(
        v3[0],
        v3[1] + 0.004,
        "Frontier / Resources / Strategy",
        ha="center",
        va="bottom",
        color=REGION_COLORS[3],
        fontsize=FS_SMALL - 1,
    )
    format_kwargs = dict(
        xlim=(0.0, 1.0),
        ylim=(0.0, 0.94),
        xticks=[],
        yticks=[],
        xlabel="",
        ylabel="",
    )
    if title is not None:
        format_kwargs["title"] = title
    ax.format(**format_kwargs)
    ax.set_aspect("equal")
    ax.axis("off")


def _select_exemplars(actor_df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for rid in [1, 2, 3]:
        cand = actor_df[actor_df["dominant_region"] == rid].sort_values(
            ["dominant_region_share", "k_active", "actor"],
            ascending=[False, False, True],
        )
        strong = cand[cand["dominant_region_share"] >= MIN_DOMINANT_SHARE]
        pick = strong.head(N_EXEMPLARS_PER_REGIME)
        if len(pick) < N_EXEMPLARS_PER_REGIME:
            remaining = cand[~cand["actor"].isin(pick["actor"])].head(
                N_EXEMPLARS_PER_REGIME - len(pick)
            )
            pick = pd.concat([pick, remaining], axis=0)
        chunks.append(pick)
    out = pd.concat(chunks, axis=0).drop_duplicates(subset=["actor"])
    out = out.sort_values(
        ["centroid_xplot_raw_rca", "dominant_region", "actor"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return out


# %%
topic_df = pd.read_csv(TOPIC_FP)
actor_df = pd.read_csv(ACTOR_FP)
region_df = pd.read_csv(REGION_FP)

required_topic = {"topic", "x_plot"}
required_actor = {
    "actor",
    "k_active",
    "dominant_region",
    "dominant_region_share",
    "centroid_xplot_raw_rca",
    "region_1_share",
    "region_2_share",
    "region_3_share",
}
required_region = {"region_id", "boundary_left", "boundary_right"}

if not required_topic.issubset(topic_df.columns):
    raise ValueError(f"Missing required topic columns in {TOPIC_FP}")
if not required_actor.issubset(actor_df.columns):
    raise ValueError(f"Missing required actor columns in {ACTOR_FP}")
if not required_region.issubset(region_df.columns):
    raise ValueError(f"Missing required region columns in {REGION_FP}")

actor_df["dominant_region"] = actor_df["dominant_region"].astype(int)
actor_df = actor_df.sort_values(
    ["centroid_xplot_raw_rca", "dominant_region", "actor"],
    ascending=[True, True, True],
).reset_index(drop=True)

ordered_topics = topic_df["topic"].tolist()
x_plot = topic_df["x_plot"].to_numpy(dtype=float)

counts, submitted, countries, topics = load_data(str(DATA_FP))
counts = standardize_index_labels(counts)
if counts.index.has_duplicates:
    counts = counts.groupby(level=0).sum()
rca = get_rca(counts).reindex(index=ordered_topics)

actor_df = actor_df[actor_df["actor"].isin(rca.columns)].copy()
subset_df = _select_exemplars(actor_df)
flag_images = _load_local_flags(actor_df["actor"].tolist(), base="./assets/flags")

OUT_A.parent.mkdir(parents=True, exist_ok=True)
OUT_A_ALL.parent.mkdir(parents=True, exist_ok=True)
OUT_A_ALL_SLIDE.parent.mkdir(parents=True, exist_ok=True)
OUT_B.parent.mkdir(parents=True, exist_ok=True)
OUT_C.parent.mkdir(parents=True, exist_ok=True)
OUT_JOINT.parent.mkdir(parents=True, exist_ok=True)
OUT_JOINT_ALL.parent.mkdir(parents=True, exist_ok=True)
OUT_SUBSET.parent.mkdir(parents=True, exist_ok=True)
subset_df.to_csv(OUT_SUBSET, index=False)
print(f"Wrote {OUT_SUBSET}")

# %%
# A) Regime exemplar ridgelines (separate figure).
n_subset = len(subset_df)
fig_a_height = max(5.8, 0.36 * n_subset + 1.2)
fig_a, ax_a = uplt.subplots(figsize=(8.8, fig_a_height), share=0, hspace=2)

for row_idx, row in subset_df.iterrows():
    actor = row["actor"]
    vals = np.clip(rca[actor].to_numpy(dtype=float), 0.0, None)
    ridge = _smooth_1d(vals)
    m = float(np.nanmax(ridge))
    if m > 0:
        ridge = ridge / m
    baseline = float(row_idx)
    color = RIDGE_COLOR
    ax_a.fill_between(
        x_plot,
        baseline,
        baseline - RIDGE_HEIGHT * ridge,
        color=color,
        alpha=0.34,
        lw=0,
        zorder=2,
    )
    ax_a.plot(
        x_plot,
        baseline - RIDGE_HEIGHT * ridge,
        color=color,
        lw=1.0,
        alpha=0.95,
        zorder=3,
    )
    _plot_interest_dots(ax_a, x_plot, vals, baseline)

y0 = -RIDGE_HEIGHT - 0.20
yh = n_subset - 0.5
for _, row in region_df.sort_values("region_id").iterrows():
    rid = int(row["region_id"])
    left = float(row["boundary_left"])
    right = float(row["boundary_right"])
    color = REGION_COLORS.get(rid, "#777777")
    ax_a.axvspan(left, right, color=color, alpha=0.14, lw=0, zorder=0)
    ax_a.text(
        0.5 * (left + right),
        0.985,
        f"Regime {rid}",
        transform=ax_a.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=FS_SMALL,
        color=color,
        zorder=5,
    )

ax_a.set_xlim(0.0, 1.0)
ax_a.set_ylim(n_subset - 0.5 + 0.18, -RIDGE_HEIGHT - 0.22)
ax_a.set_yticks(np.arange(n_subset, dtype=float))
ax_a.set_yticklabels(subset_df["actor"].tolist(), fontsize=FS_ACTOR_YTICK)
for tick, reg in zip(
    ax_a.get_yticklabels(), subset_df["dominant_region"].to_numpy(dtype=int)
):
    tick.set_color(REGION_COLORS.get(int(reg), "black"))
ax_a.format(
    abc=["[A]"],
    xlabel="MDS-scaled position in topic space (0-1)",
    ylabel="",
    title="Regime Exemplars in the Space of Concerns",
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
    abcsize=FS_ABC,
)
ax_a.grid(axis="x", alpha=0.22, linewidth=0.7)
ax_a.grid(axis="y", visible=False)
ax_a.yaxis.set_minor_locator(NullLocator())
fig_a.savefig(OUT_A, dpi=230, bbox_inches="tight", pad_inches=0.16)
print(f"Wrote {OUT_A}")

# %%
# A-slide) Reduced-set ridgelines for slide use, with no regime indicators.
fig_a_subset_slide_height = max(5.8, 0.36 * n_subset + 1.2)
fig_a_subset_slide, ax_a_subset_slide = uplt.subplots(
    figsize=(8.8, fig_a_subset_slide_height), share=0, hspace=2
)

for row_idx, row in subset_df.iterrows():
    actor = row["actor"]
    vals = np.clip(rca[actor].to_numpy(dtype=float), 0.0, None)
    ridge = _smooth_1d(vals)
    m = float(np.nanmax(ridge))
    if m > 0:
        ridge = ridge / m
    baseline = float(row_idx)
    ax_a_subset_slide.fill_between(
        x_plot,
        baseline,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        alpha=0.34,
        lw=0,
        zorder=2,
    )
    ax_a_subset_slide.plot(
        x_plot,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        lw=1.0,
        alpha=0.95,
        zorder=3,
    )
    _plot_interest_dots(ax_a_subset_slide, x_plot, vals, baseline)

ax_a_subset_slide.set_xlim(0.0, 1.0)
ax_a_subset_slide.set_ylim(n_subset - 0.5 + 0.18, -RIDGE_HEIGHT - 0.22)
ax_a_subset_slide.set_yticks(np.arange(n_subset, dtype=float))
ax_a_subset_slide.set_yticklabels(subset_df["actor"].tolist(), fontsize=FS_ACTOR_YTICK)
ax_a_subset_slide.format(
    xlabel="MDS-scaled position in topic space (0-1)",
    ylabel="",
    title="Selected Members in the Space of Concerns",
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
)
ax_a_subset_slide.grid(axis="x", alpha=0.22, linewidth=0.7)
ax_a_subset_slide.grid(axis="y", visible=False)
ax_a_subset_slide.yaxis.set_minor_locator(NullLocator())
fig_a_subset_slide.savefig(
    OUT_A_SLIDE,
    dpi=230,
    bbox_inches="tight",
    pad_inches=0.16,
    transparent=True,
)
print(f"Wrote {OUT_A_SLIDE}")

# %%
# A-all) All-member ridgelines for paper use.
n_all = len(actor_df)
fig_a_all_height = max(8.4, 0.22 * n_all + 1.5)
fig_a_all, ax_a_all = uplt.subplots(figsize=(9.2, fig_a_all_height), share=0)

for row_idx, row in actor_df.iterrows():
    actor = row["actor"]
    vals = np.clip(rca[actor].to_numpy(dtype=float), 0.0, None)
    ridge = _smooth_1d(vals)
    m = float(np.nanmax(ridge))
    if m > 0:
        ridge = ridge / m
    baseline = float(row_idx)
    ax_a_all.fill_between(
        x_plot,
        baseline,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        alpha=0.32,
        lw=0,
        zorder=2,
    )
    ax_a_all.plot(
        x_plot,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        lw=0.95,
        alpha=0.92,
        zorder=3,
    )
    _plot_interest_dots(ax_a_all, x_plot, vals, baseline)

for _, row in region_df.sort_values("region_id").iterrows():
    rid = int(row["region_id"])
    left = float(row["boundary_left"])
    right = float(row["boundary_right"])
    color = REGION_COLORS.get(rid, "#777777")
    ax_a_all.axvspan(left, right, color=color, alpha=0.13, lw=0, zorder=0)
    ax_a_all.text(
        0.5 * (left + right),
        y0 + REGIME_LABEL_Y_DELTA,
        f"Regime {rid}",
        ha="center",
        va="top",
        fontsize=FS_SMALL,
        color=color,
        zorder=5,
    )

ax_a_all.set_xlim(0.0, 1.0)
ax_a_all.set_ylim(n_all - 0.5 + 0.18, -RIDGE_HEIGHT - RIDGE_TOP_PAD)
ax_a_all.set_yticks(np.arange(n_all, dtype=float))
ax_a_all.set_yticklabels(actor_df["actor"].tolist(), fontsize=FS_ACTOR_YTICK)
for tick, reg in zip(
    ax_a_all.get_yticklabels(), actor_df["dominant_region"].to_numpy(dtype=int)
):
    tick.set_color(REGION_COLORS.get(int(reg), "black"))
ax_a_all.format(
    abc=["[A]"],
    xlabel="MDS-scaled position in topic space (0-1)",
    ylabel="",
    title="",
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
    abcsize=FS_ABC,
)
ax_a_all.grid(axis="x", alpha=0.22, linewidth=0.7)
ax_a_all.grid(axis="y", visible=False)
ax_a_all.yaxis.set_minor_locator(NullLocator())
peak_x_all, peak_labels_all = _aggregate_top_topic_labels(
    rca, actor_df["actor"].tolist(), ordered_topics, x_plot
)
_add_non_overlapping_top_labels(ax_a_all, peak_x_all, peak_labels_all)
fig_a_all.savefig(OUT_A_ALL, dpi=230, bbox_inches="tight", pad_inches=0.16)
print(f"Wrote {OUT_A_ALL}")

# %%
# A-all-slide) All-member ridgelines with no regime indicators.
fig_a_slide_height = max(8.4, 0.22 * n_all + 1.5)
fig_a_slide, ax_a_slide = uplt.subplots(figsize=(9.2, fig_a_slide_height), share=0)

for row_idx, row in actor_df.iterrows():
    actor = row["actor"]
    vals = np.clip(rca[actor].to_numpy(dtype=float), 0.0, None)
    ridge = _smooth_1d(vals)
    m = float(np.nanmax(ridge))
    if m > 0:
        ridge = ridge / m
    baseline = float(row_idx)
    ax_a_slide.fill_between(
        x_plot,
        baseline,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        alpha=0.32,
        lw=0,
        zorder=2,
    )
    ax_a_slide.plot(
        x_plot,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        lw=0.95,
        alpha=0.92,
        zorder=3,
    )
    _plot_interest_dots(ax_a_slide, x_plot, vals, baseline)

ax_a_slide.set_xlim(0.0, 1.0)
ax_a_slide.set_ylim(n_all - 0.5 + 0.18, -RIDGE_HEIGHT - 0.22)
ax_a_slide.set_yticks(np.arange(n_all, dtype=float))
ax_a_slide.set_yticklabels(actor_df["actor"].tolist(), fontsize=FS_ACTOR_YTICK)
ax_a_slide.format(
    xlabel="MDS-scaled position in topic space (0-1)",
    ylabel="",
    title="All Members in the Space of Concerns",
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
)
ax_a_slide.grid(axis="x", alpha=0.22, linewidth=0.7)
ax_a_slide.grid(axis="y", visible=False)
ax_a_slide.yaxis.set_minor_locator(NullLocator())
fig_a_slide.savefig(
    OUT_A_ALL_SLIDE,
    dpi=230,
    bbox_inches="tight",
    pad_inches=0.16,
    transparent=True,
)
print(f"Wrote {OUT_A_ALL_SLIDE}")

# %%
# B) Centroid separation by dominant regime (separate figure).
fig_b, ax_b = uplt.subplots(figsize=(5.6, 5.2), share=0)
for rid in [1, 2, 3]:
    vals = actor_df.loc[
        actor_df["dominant_region"] == rid, "centroid_xplot_raw_rca"
    ].to_numpy(dtype=float)
    if vals.size == 0:
        continue
    xj = np.full(vals.size, float(rid)) + _jitter(vals.size, width=0.14)
    color = REGION_COLORS[rid]
    ax_b.scatter(
        xj,
        vals,
        s=40,
        color=color,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    med = float(np.median(vals))
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    ax_b.vlines(rid, q1, q3, color=color, lw=3.0, zorder=2)
    ax_b.hlines(med, rid - 0.16, rid + 0.16, color="black", lw=1.2, zorder=5)

ax_b.set_xlim(0.5, 3.5)
ax_b.set_xticks([1, 2, 3])
ax_b.set_xticklabels(["Regime 1", "Regime 2", "Regime 3"])
ax_b.format(
    abc=["[B]"],
    xlabel="Dominant regime",
    ylabel="Portfolio centroid (MDS 0-1)",
    title="Centroid Separation by Regime",
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
    abcsize=FS_ABC,
)
ax_b.grid(axis="y", alpha=0.22, linewidth=0.7)
ax_b.grid(axis="x", visible=False)
fig_b.savefig(
    OUT_B,
    dpi=230,
    bbox_inches="tight",
    pad_inches=0.16,
    transparent=True,
)
print(f"Wrote {OUT_B}")

# %%
# C) Portfolio regime share fractions (separate figure).
fig_c, ax_c = uplt.subplots(figsize=(5.6, 5.2), share=0)
_plot_regime_ternary(
    ax_c,
    actor_df=actor_df,
    flags=flag_images,
    title="Regime Share Fractions Across Actors",
)
ax_c.format(
    abc=["[C]"],
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
    abcsize=FS_ABC,
)
fig_c.savefig(
    OUT_C,
    dpi=230,
    bbox_inches="tight",
    pad_inches=0.16,
    transparent=True,
)
print(f"Wrote {OUT_C}")

# %%
# Joint figure with A/B/C in one layout.
layout = [[1, 2], [1, 3]]
joint_height = max(8.6, 0.33 * n_subset + 1.4)
fig_j, axs = uplt.subplots(
    layout,
    figsize=(13.6, joint_height),
    share=0,
    wspace="10em",
    hspace="30em",
)
ax_ja, ax_jb, ax_jc = axs

# A (left): exemplar ridgelines.
for row_idx, row in subset_df.iterrows():
    actor = row["actor"]
    vals = np.clip(rca[actor].to_numpy(dtype=float), 0.0, None)
    ridge = _smooth_1d(vals)
    m = float(np.nanmax(ridge))
    if m > 0:
        ridge = ridge / m
    baseline = float(row_idx)
    ax_ja.fill_between(
        x_plot,
        baseline,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        alpha=0.34,
        lw=0,
        zorder=2,
    )
    ax_ja.plot(
        x_plot,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        lw=1.0,
        alpha=0.95,
        zorder=3,
    )
    _plot_interest_dots(ax_ja, x_plot, vals, baseline)

for _, row in region_df.sort_values("region_id").iterrows():
    rid = int(row["region_id"])
    left = float(row["boundary_left"])
    right = float(row["boundary_right"])
    color = REGION_COLORS.get(rid, "#777777")
    ax_ja.axvspan(left, right, color=color, alpha=0.14, lw=0, zorder=0)
    ax_ja.text(
        0.5 * (left + right),
        0.985,
        f"Regime {rid}",
        transform=ax_ja.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=FS_SMALL,
        color=color,
        zorder=5,
    )

ax_ja.set_xlim(0.0, 1.0)
ax_ja.set_ylim(n_subset - 0.5 + 0.18, -RIDGE_HEIGHT - 0.22)
ax_ja.set_yticks(np.arange(n_subset, dtype=float))
ax_ja.set_yticklabels(subset_df["actor"].tolist(), fontsize=FS_ACTOR_YTICK)
for tick, reg in zip(
    ax_ja.get_yticklabels(), subset_df["dominant_region"].to_numpy(dtype=int)
):
    tick.set_color(REGION_COLORS.get(int(reg), "black"))
ax_ja.format(
    xlabel="MDS-scaled position in topic space (0-1)",
    ylabel="",
    title="Regime Exemplars in the Space of Concerns",
)
ax_ja.grid(axis="x", alpha=0.22, linewidth=0.7)
ax_ja.grid(axis="y", visible=False)
ax_ja.yaxis.set_minor_locator(NullLocator())

# B (top-right): centroid separation.
for rid in [1, 2, 3]:
    vals = actor_df.loc[
        actor_df["dominant_region"] == rid, "centroid_xplot_raw_rca"
    ].to_numpy(dtype=float)
    if vals.size == 0:
        continue
    xj = np.full(vals.size, float(rid)) + _jitter(vals.size, width=0.14)
    color = REGION_COLORS[rid]
    ax_jb.scatter(
        xj,
        vals,
        s=40,
        color=color,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    med = float(np.median(vals))
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    ax_jb.vlines(rid, q1, q3, color=color, lw=3.0, zorder=2)
    ax_jb.hlines(med, rid - 0.16, rid + 0.16, color="black", lw=1.2, zorder=5)

ax_jb.set_xlim(0.5, 3.5)
ax_jb.set_xticks([1, 2, 3])
ax_jb.set_xticklabels(["Regime 1", "Regime 2", "Regime 3"])
ax_jb.format(
    xlabel="Dominant regime",
    ylabel="Portfolio centroid (MDS 0-1)",
    title="Centroid Separation by Regime",
)
ax_jb.grid(axis="y", alpha=0.22, linewidth=0.7)
ax_jb.grid(axis="x", visible=False)

# C (bottom-right): regime-share ternary with flags.
_plot_regime_ternary(
    ax_jc,
    actor_df=actor_df,
    flags=flag_images,
    title=None,
)
axs.format(
    abc="[A]",
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
    abcsize=FS_ABC,
)

fig_j.savefig(OUT_JOINT, dpi=230, bbox_inches="tight", pad_inches=0.16)
print(f"Wrote {OUT_JOINT}")

# %%
# Joint all-members figure (paper-ready left panel).
layout = [[1, 2], [1, 3]]
joint_all_height = max(10.0, 0.22 * n_all + 1.8)
fig_ja, axs_all = uplt.subplots(
    layout,
    width_ratios=(1.22, 1.0),
    figsize=(14.2, joint_all_height),
    share=0,
    wspace="10em",
    hspace="6em",
)
ax_la, ax_tb, ax_cb = axs_all

for row_idx, row in actor_df.iterrows():
    actor = row["actor"]
    vals = np.clip(rca[actor].to_numpy(dtype=float), 0.0, None)
    ridge = _smooth_1d(vals)
    m = float(np.nanmax(ridge))
    if m > 0:
        ridge = ridge / m
    baseline = float(row_idx)
    ax_la.fill_between(
        x_plot,
        baseline,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        alpha=0.32,
        lw=0,
        zorder=2,
    )
    ax_la.plot(
        x_plot,
        baseline - RIDGE_HEIGHT * ridge,
        color=RIDGE_COLOR,
        lw=0.95,
        alpha=0.92,
        zorder=3,
    )
    _plot_interest_dots(ax_la, x_plot, vals, baseline)

for _, row in region_df.sort_values("region_id").iterrows():
    rid = int(row["region_id"])
    left = float(row["boundary_left"])
    right = float(row["boundary_right"])
    color = REGION_COLORS.get(rid, "#777777")
    ax_la.axvspan(left, right, color=color, alpha=0.13, lw=0, zorder=0)
    ax_la.text(
        0.5 * (left + right),
        y0 + REGIME_LABEL_Y_DELTA,
        f"Regime {rid}",
        ha="center",
        va="top",
        fontsize=FS_SMALL,
        color=color,
        zorder=5,
    )

ax_la.set_xlim(0.0, 1.0)
ax_la.set_ylim(n_all - 0.5 + 0.18, -RIDGE_HEIGHT - RIDGE_TOP_PAD)
ax_la.set_yticks(np.arange(n_all, dtype=float))
ax_la.set_yticklabels(actor_df["actor"].tolist(), fontsize=FS_ACTOR_YTICK)
for tick, reg in zip(
    ax_la.get_yticklabels(), actor_df["dominant_region"].to_numpy(dtype=int)
):
    tick.set_color(REGION_COLORS.get(int(reg), "black"))
ax_la.grid(axis="x", alpha=0.22, linewidth=0.7)
ax_la.grid(axis="y", visible=False)
ax_la.yaxis.set_minor_locator(NullLocator())

for rid in [1, 2, 3]:
    vals = actor_df.loc[
        actor_df["dominant_region"] == rid, "centroid_xplot_raw_rca"
    ].to_numpy(dtype=float)
    if vals.size:
        xj = np.full(vals.size, float(rid)) + _jitter(vals.size, width=0.14)
        color = REGION_COLORS[rid]
        ax_tb.scatter(
            xj,
            vals,
            s=40,
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        med = float(np.median(vals))
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        ax_tb.vlines(rid, q1, q3, color=color, lw=3.0, zorder=2)
        ax_tb.hlines(med, rid - 0.16, rid + 0.16, color="black", lw=1.2, zorder=5)

ax_tb.set_xlim(0.5, 3.5)
ax_tb.set_xticks([1, 2, 3])
ax_tb.set_xticklabels(["Regime 1", "Regime 2", "Regime 3"])
ax_tb.grid(axis="y", alpha=0.22, linewidth=0.7)
ax_tb.grid(axis="x", visible=False)

_plot_regime_ternary(
    ax_cb,
    actor_df=actor_df,
    flags=flag_images,
    title="Regime Share Fractions Across Actors",
)

ax_la.format(
    xlabel="MDS-scaled position in topic space (0-1)",
    ylabel="",
    title="",
)
peak_x_joint_all, peak_labels_joint_all = _aggregate_top_topic_labels(
    rca, actor_df["actor"].tolist(), ordered_topics, x_plot
)
_add_non_overlapping_top_labels(ax_la, peak_x_joint_all, peak_labels_joint_all)
ax_tb.format(
    xlabel="Dominant regime",
    ylabel="Portfolio centroid (MDS 0-1)",
    title="Centroid Separation by Regime",
)
ax_cb.format(
    xlabel="",
    ylabel="",
    title="Regime Share Fractions Across Actors",
)
axs_all.format(
    abc="[A]",
    labelsize=FS_LABEL,
    ticklabelsize=FS_TICK,
    titlesize=FS_TITLE,
    abcsize=FS_ABC,
)
fig_ja.savefig(OUT_JOINT_ALL, dpi=230, bbox_inches="tight", pad_inches=0.16)
print(f"Wrote {OUT_JOINT_ALL}")
