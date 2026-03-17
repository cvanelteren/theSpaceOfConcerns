# %%
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hazard panel export (at-risk rows) + hero-ready summaries
#
# Goal: create a single dataframe with one row per *(member, topic, transition)* for topics that are **at risk**
# (i.e., not specialized in prior window), with:
#
# - `adopted` = 1 if member adopts topic in next window (RPA/RCA >= threshold)
# - `distance` = min distance from the candidate topic to the member's prior active portfolio
# - controls: `prev_diversity`, `topic_popularity`
#
# This panel is the source for:
# - `P(adopt at t | at risk at t-1, distance in bin)`  (empirical binned mean)
# - pooled / controlled logistic curves (optional)
#
# Outputs (default):
# - `output/hazard_panel_at_risk.parquet`
# - `output/hazard_panel_at_risk.csv` (optional; bigger/slower)
# - a couple of quick plots under `figures/`
#
# Notes:
# - This notebook follows the same logic as existing scripts such as `fig3_fig_space_of_concerns_hazard_timeseries.py`
#   and `fig26_hazard_space_sensitivity.py`.
# - Distances are computed from the *aggregate* (full-history) concern space by default.

# %%
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

# --- Make repo-root imports work when running from notebooks/ ---
# When executing `python notebooks/hazard_panel_export.py`, sys.path[0] points to
# `.../notebooks`, so `import utils` fails unless we add the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional plotting (kept lightweight; safe to skip if running headless)
try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception:  # pragma: no cover
    plt = None
    Line2D = None

# Reuse your project utilities (these exist in the repo root)
from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the hazard panels and, optionally, an overlaid comparison "
            "graph that contrasts lagged year windows with lagged sequential meetings."
        )
    )
    parser.add_argument(
        "--lookback-comparison",
        action="store_true",
        help=(
            "Export the combined lookback panel and a single comparison graph "
            "that overlays year-window and meeting-window decay curves."
        ),
    )
    parser.add_argument(
        "--max-lookback",
        type=int,
        default=6,
        help="Maximum lag n to include in the comparison graph (default: 6).",
    )
    parser.add_argument(
        "--lookback-bins",
        type=int,
        default=18,
        help="Number of log-phi bins for the comparison curves (default: 18).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix for comparison outputs.",
    )
    args, _ = parser.parse_known_args()
    return args


CLI_ARGS = _parse_cli_args()
if CLI_ARGS.lookback_comparison:
    os.environ["HAZARD_EXPORT_LOOKBACK_COMPARISON"] = "1"
    os.environ["HAZARD_EXPORT_MAX_LOOKBACK"] = str(int(CLI_ARGS.max_lookback))
    os.environ["HAZARD_EXPORT_LOOKBACK_BINS"] = str(int(CLI_ARGS.lookback_bins))
    if CLI_ARGS.output_tag.strip():
        os.environ["HAZARD_EXPORT_OUTPUT_TAG"] = CLI_ARGS.output_tag.strip()
    else:
        os.environ.pop("HAZARD_EXPORT_OUTPUT_TAG", None)

# %% [markdown]
# ## Configuration


# %%
@dataclass(frozen=True)
class HazardPanelConfig:
    # Rolling window size, interpreted in CFG.time_unit.
    window_years: int = 1
    time_unit: str = "year"  # "meeting" | "year"
    rca_threshold: float = 1.0  # specialization threshold used for "active topics"
    distance_mode: str = (
        "aggregate"  # "aggregate" | "instantaneous" | "cumulative_lagged"
    )
    # How to compute topic-topic distance from the window interaction matrix.
    # - "proximity": shortest-path cost on edge weights -log(phi), where phi is
    #   product-space proximity (RCA-based)
    # - "cosine": 1 - cosine_similarity on binary active-topic vectors
    # - "jaccard": 1 - jaccard_similarity on binary active-topic vectors
    distance_metric: str = "proximity"
    max_transitions: Optional[int] = None  # set to int for fast dev runs

    # Where to load data from (fallbacks)
    data_paths: tuple[Path, ...] = (
        Path("antarctic-database-go/data/processed/document-summary.parquet"),
        Path(
            "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"
        ),
        Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
        Path("document-summary.csv"),
    )

    # Outputs
    out_dir: Path = Path("output")
    fig_dir: Path = Path("figures")

    # Adoption (at-risk) panel
    out_panel_parquet: Path = Path("output/hazard_panel_at_risk.parquet")
    out_panel_csv: Optional[Path] = None  # e.g. Path("output/hazard_panel_at_risk.csv")
    out_meta_json: Path = Path("output/hazard_panel_at_risk_meta.json")

    # Persistence (previously-active) panel
    out_persist_panel_parquet: Path = Path("output/hazard_panel_prev_active.parquet")
    out_persist_panel_csv: Optional[Path] = (
        None  # e.g. Path("output/hazard_panel_prev_active.csv")
    )


CFG = HazardPanelConfig()
LOOKBACK_COMPARISON_ENABLED = (
    os.environ.get("HAZARD_EXPORT_LOOKBACK_COMPARISON", "").strip().lower()
    in {"1", "true", "yes", "on"}
)
LOOKBACK_COMPARISON_MAX_LAG = max(
    1, int(os.environ.get("HAZARD_EXPORT_MAX_LOOKBACK", "6"))
)
LOOKBACK_COMPARISON_N_BINS = max(
    4, int(os.environ.get("HAZARD_EXPORT_LOOKBACK_BINS", "18"))
)
LOOKBACK_OUTPUT_TAG = os.environ.get("HAZARD_EXPORT_OUTPUT_TAG", "").strip()

# %% [markdown]
# ## Helpers


# %%
def load_data_with_fallback(paths: Iterable[Path]):
    """Load (counts_df, submitted_df, members_raw, topics_raw) from first existing path."""
    last_err: Optional[Exception] = None
    for p in paths:
        if not p.exists():
            continue
        try:
            return load_data(str(p))
        except Exception as exc:  # pragma: no cover
            last_err = exc
    if last_err is not None:
        raise RuntimeError(
            "Failed to load ATS data from all fallback paths."
        ) from last_err
    raise FileNotFoundError("No known ATS data path exists. Update CFG.data_paths.")


def sanitize_years(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=[year_col]).copy()
    out[year_col] = out[year_col].astype(int)
    return out


def build_periods(year_min: int, year_max: int, window: int) -> list[tuple[int, int]]:
    """Rolling windows as (start, end), with step 1 year."""
    return [(y - window + 1, y) for y in range(year_min + window - 1, year_max + 1)]


def build_periods_for_unit(
    submitted_df: pd.DataFrame,
    *,
    year_col: str,
    meeting_col: str,
    time_unit: str,
    window_size: int,
) -> list[tuple[int, int]]:
    """Return rolling periods on calendar years or sequential meetings."""
    time_unit = str(time_unit).strip().lower()
    window_size = max(1, int(window_size))

    if time_unit == "year":
        year_min = int(submitted_df[year_col].min())
        year_max = int(submitted_df[year_col].max())
        return build_periods(year_min, year_max, window_size)

    if time_unit == "meeting":
        values = sorted(pd.to_numeric(submitted_df[meeting_col], errors="coerce").dropna().astype(int).unique().tolist())
        return [
            (int(values[idx - window_size + 1]), int(values[idx]))
            for idx in range(window_size - 1, len(values))
        ]

    raise ValueError(f"Unknown time_unit={time_unit!r}. Use year|meeting.")


def build_window_interaction(
    submitted_df: pd.DataFrame,
    year_col: str,
    year_start: int,
    year_end: int,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    members_order: list[str],
) -> pd.DataFrame:
    window_df = submitted_df[
        (submitted_df[year_col] >= int(year_start))
        & (submitted_df[year_col] <= int(year_end))
    ]
    interaction = generate_interaction_matrix(
        window_df, all_members_raw, all_topics_raw
    )
    interaction = standardize_index_labels(interaction)
    if interaction.index.has_duplicates:
        interaction = interaction.groupby(level=0).sum()
    return interaction.reindex(index=topics_order, columns=members_order, fill_value=0)


def tagged_output_path(path: Path, tag: str) -> Path:
    """Append a sanitized suffix before the file extension."""
    tag = str(tag).strip()
    if not tag:
        return path
    safe = "_".join(tag.replace("-", "_").split())
    if not safe:
        return path
    return path.with_name(f"{path.stem}_{safe}{path.suffix}")


def _cosine_similarity_from_binary(M: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between rows of a binary matrix M (topics x members).
    Returns a dense (topics x topics) similarity matrix in [0, 1].
    """
    M = M.astype(float, copy=False)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    denom = norms @ norms.T
    sim = (M @ M.T) / np.clip(denom, 1e-12, None)
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def _jaccard_similarity_from_binary(M: np.ndarray) -> np.ndarray:
    """
    Jaccard similarity between rows of a binary matrix M (topics x members).
    Returns a dense (topics x topics) similarity matrix in [0, 1].
    """
    M = (M > 0).astype(int, copy=False)
    inter = M @ M.T
    row_sums = M.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - inter
    sim = inter / np.clip(union, 1, None)
    sim = sim.astype(float)
    np.fill_diagonal(sim, 1.0)
    return sim


def distance_from_interaction(
    interaction: pd.DataFrame, topics_order: list[str], metric: str = "proximity"
) -> np.ndarray:
    """
    Compute symmetric topic-topic distance matrix from an interaction matrix.

    metric:
      - "proximity": product-space proximity phi (RCA-based), where edge cost is
        -log(phi) and pairwise distance is the shortest-path cost in that graph
      - "cosine": cosine similarity on binary active-topic matrix, distance = 1 - cos
      - "jaccard": jaccard similarity on binary active-topic matrix, distance = 1 - jac
    """
    metric = str(metric).strip().lower()

    if metric == "proximity":
        rca = get_rca(interaction)
        phi = compute_product_space(rca).reindex(
            index=topics_order, columns=topics_order, fill_value=0.0
        )
        g = nx.from_pandas_adjacency(phi)
        for _, _, data in g.edges(data=True):
            w = float(data.get("weight", 0.0))
            data["distance"] = -np.log(np.clip(w, 1e-12, 1.0))

        n_topics = len(topics_order)
        dist = np.full((n_topics, n_topics), np.inf, dtype=float)
        np.fill_diagonal(dist, 0.0)
        topic_to_idx = {topic: idx for idx, topic in enumerate(topics_order)}

        for src, lengths in nx.all_pairs_dijkstra_path_length(g, weight="distance"):
            i = topic_to_idx[src]
            for dst, val in lengths.items():
                j = topic_to_idx[dst]
                dist[i, j] = float(val)

        finite = dist[np.isfinite(dist)]
        max_finite = float(np.max(finite)) if finite.size else 1.0
        fill_value = 1.25 * max_finite if max_finite > 0 else 1.0
        dist[~np.isfinite(dist)] = fill_value
        return dist
    else:
        # Binary active-topic matrix (topics x members)
        rca = get_rca(interaction)
        M = (
            (rca > float(CFG.rca_threshold))
            .reindex(
                index=topics_order,
                columns=interaction.columns.tolist(),
                fill_value=False,
            )
            .to_numpy(dtype=int)
        )

        if metric == "cosine":
            sim = _cosine_similarity_from_binary(M)
        elif metric == "jaccard":
            sim = _jaccard_similarity_from_binary(M)
        else:
            raise ValueError(
                f"Unknown distance_metric={metric!r}. Use proximity|cosine|jaccard."
            )

    dist = 1.0 - sim
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_distance_matrix_for_transition(
    *,
    mode: str,
    t: int,
    source_idx: Optional[int],
    periods: list[tuple[int, int]],
    sequence_start: int,
    submitted_df: pd.DataFrame,
    time_col: str,
    interaction_by_period: list[pd.DataFrame],
    aggregate_dist: np.ndarray,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics: list[str],
    members: list[str],
) -> np.ndarray:
    """Return distance matrix used for transition (t-1 -> t)."""
    if source_idx is None:
        source_idx = int(t - 1)

    if mode == "aggregate":
        return aggregate_dist

    if mode == "instantaneous":
        return distance_from_interaction(
            interaction_by_period[int(source_idx)], topics, metric=CFG.distance_metric
        )

    if mode == "cumulative_lagged":
        prev_end = int(periods[int(source_idx)][1])
        cumulative_interaction = build_window_interaction(
            submitted_df=submitted_df,
            year_col=time_col,
            year_start=sequence_start,
            year_end=prev_end,
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics,
            members_order=members,
        )
        return distance_from_interaction(
            cumulative_interaction, topics, metric=CFG.distance_metric
        )

    raise ValueError(
        f"Unknown distance_mode={mode!r}. Use aggregate|instantaneous|cumulative_lagged."
    )


def build_interactions_and_active_by_period(
    *,
    periods: list[tuple[int, int]],
    submitted_df: pd.DataFrame,
    time_col: str,
    topics: list[str],
    members: list[str],
    all_members_raw: set[str],
    all_topics_raw: set[str],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    interaction_by_period: list[pd.DataFrame] = []
    active_by_period: list[pd.DataFrame] = []

    for start, end in periods:
        interaction = build_window_interaction(
            submitted_df=submitted_df,
            year_col=time_col,
            year_start=int(start),
            year_end=int(end),
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics,
            members_order=members,
        )
        interaction_by_period.append(interaction)

        active = get_rca(interaction) > float(CFG.rca_threshold)
        active = active.reindex(index=topics, columns=members, fill_value=False)
        active_by_period.append(active)

    return interaction_by_period, active_by_period


def build_lookback_adoption_panel(
    *,
    time_unit: str,
    periods: list[tuple[int, int]],
    time_col: str,
    submitted_df: pd.DataFrame,
    topics: list[str],
    members: list[str],
    all_members_raw: set[str],
    all_topics_raw: set[str],
    aggregate_dist: np.ndarray,
    max_lookback_steps: int,
) -> pd.DataFrame:
    interaction_by_period, active_by_period = build_interactions_and_active_by_period(
        periods=periods,
        submitted_df=submitted_df,
        time_col=time_col,
        topics=topics,
        members=members,
        all_members_raw=all_members_raw,
        all_topics_raw=all_topics_raw,
    )

    if not periods:
        return pd.DataFrame()

    sequence_start = int(periods[0][0])
    max_lookback_steps = min(max(1, int(max_lookback_steps)), max(len(periods) - 1, 1))
    rows: list[dict] = []

    for lookback in range(1, max_lookback_steps + 1):
        for t in range(lookback, len(periods)):
            period_start, period_end = periods[t]
            source_start, source_end = periods[t - lookback]
            prev_active = active_by_period[t - lookback]
            curr_active = active_by_period[t]
            dist = compute_distance_matrix_for_transition(
                mode=CFG.distance_mode,
                t=t,
                source_idx=t - lookback,
                periods=periods,
                sequence_start=sequence_start,
                submitted_df=submitted_df,
                time_col=time_col,
                interaction_by_period=interaction_by_period,
                aggregate_dist=aggregate_dist,
                all_members_raw=all_members_raw,
                all_topics_raw=all_topics_raw,
                topics=topics,
                members=members,
            )
            prev_topic_popularity = prev_active.sum(axis=1) / max(len(members), 1)

            for member in members:
                prev_mask = prev_active[member].to_numpy(dtype=bool)
                if not prev_mask.any():
                    continue

                curr_mask = curr_active[member].to_numpy(dtype=bool)
                at_risk = ~prev_mask
                if not at_risk.any():
                    continue

                adopted = curr_mask & at_risk
                prev_indices = np.where(prev_mask)[0]
                min_dist = dist[:, prev_indices].min(axis=1)
                prev_diversity = int(len(prev_indices))

                for idx, topic in enumerate(topics):
                    if not at_risk[idx]:
                        continue

                    distance = float(min_dist[idx])
                    log_phi = (
                        float(-distance)
                        if CFG.distance_metric == "proximity" and np.isfinite(distance)
                        else float("nan")
                    )
                    phi_best_path = (
                        float(np.exp(-distance))
                        if CFG.distance_metric == "proximity" and np.isfinite(distance)
                        else float("nan")
                    )
                    rows.append(
                        {
                            "time_unit": str(time_unit),
                            "lookback_steps": int(lookback),
                            "transition_idx": int(t),
                            "source_idx": int(t - lookback),
                            "source_window_start": int(source_start),
                            "source_window_end": int(source_end),
                            "window_start": int(period_start),
                            "window_end": int(period_end),
                            "period_end": int(period_end),
                            "member": str(member),
                            "topic": str(topic),
                            "adopted": int(adopted[idx]),
                            "distance": distance,
                            "log_phi": log_phi,
                            "phi_best_path": phi_best_path,
                            "prev_diversity": int(prev_diversity),
                            "topic_popularity": float(prev_topic_popularity.loc[topic]),
                            "distance_mode": str(CFG.distance_mode),
                            "distance_metric": str(CFG.distance_metric),
                            "window_years": int(CFG.window_years),
                            "rca_threshold": float(CFG.rca_threshold),
                        }
                    )

    return pd.DataFrame(rows)


def binned_grouped_adoption_curve(
    df: pd.DataFrame,
    *,
    value_col: str,
    group_cols: list[str],
    n_bins: int,
) -> pd.DataFrame:
    d = df.copy()
    d = d[np.isfinite(d[value_col])].copy()
    if d.empty:
        return pd.DataFrame(
            columns=[*group_cols, "bin_left", "bin_right", "bin_center", "n", "p_adopt"]
        )

    low = float(d[value_col].min())
    high = float(d[value_col].max())
    if not np.isfinite(low) or not np.isfinite(high):
        return pd.DataFrame(
            columns=[*group_cols, "bin_left", "bin_right", "bin_center", "n", "p_adopt"]
        )
    if abs(high - low) < 1e-12:
        high = low + 1e-6
    bin_edges = np.linspace(low, high, max(2, int(n_bins)) + 1)

    curve_parts: list[pd.DataFrame] = []
    for group_key, group_df in d.groupby(group_cols, observed=True, sort=True):
        part = group_df.copy()
        part["dist_bin"] = pd.cut(
            part[value_col], bins=bin_edges, include_lowest=True, duplicates="drop"
        )
        out = (
            part.groupby("dist_bin", observed=True)["adopted"]
            .agg(n="size", p_adopt="mean")
            .reset_index()
        )
        if out.empty:
            continue
        bin_left = out["dist_bin"].apply(lambda iv: float(iv.left)).to_numpy(dtype=float)
        bin_right = out["dist_bin"].apply(lambda iv: float(iv.right)).to_numpy(dtype=float)
        out["bin_left"] = bin_left
        out["bin_right"] = bin_right
        out["bin_center"] = 0.5 * (bin_left + bin_right)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for col, value in zip(group_cols, group_key):
            out[col] = value
        curve_parts.append(
            out.drop(columns=["dist_bin"]).sort_values("bin_center").reset_index(drop=True)
        )

    if not curve_parts:
        return pd.DataFrame(
            columns=[*group_cols, "bin_left", "bin_right", "bin_center", "n", "p_adopt"]
        )
    return pd.concat(curve_parts, ignore_index=True)


def plot_lookback_log_phi_comparison(curve_df: pd.DataFrame, out_path: Path) -> None:
    if plt is None or curve_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    color_map = plt.cm.viridis(
        np.linspace(0.15, 0.92, curve_df["lookback_steps"].nunique())
    )
    color_lookup = {
        lag: color_map[idx]
        for idx, lag in enumerate(
            sorted(curve_df["lookback_steps"].astype(int).unique().tolist())
        )
    }
    style_lookup = {"year": "-", "meeting": "--"}
    label_lookup = {"year": "n year windows", "meeting": "n meetings"}

    for time_unit in ["year", "meeting"]:
        sub = curve_df[curve_df["time_unit"] == time_unit].copy()
        if sub.empty:
            continue
        for lag in sorted(sub["lookback_steps"].astype(int).unique().tolist()):
            dfi = sub[sub["lookback_steps"] == lag].sort_values("bin_center")
            ax.plot(
                dfi["bin_center"],
                dfi["p_adopt"],
                marker="o",
                ms=3.0,
                lw=1.5,
                color=color_lookup[lag],
                alpha=0.9,
                linestyle=style_lookup.get(time_unit, "-"),
            )

    ax.set_xlabel(r"log $\phi$ to lagged portfolio")
    ax.set_ylabel("P(adopt at t | at risk in lagged window)")
    ax.set_title("Adoption-vs-log-$\\phi$ decay under lagged year and meeting windows")
    ax.set_xlim(float(curve_df["bin_center"].min()), float(curve_df["bin_center"].max()))
    ax.grid(alpha=0.24, linewidth=0.6)

    lookback_handles = [
        Line2D([0], [0], color=color_lookup[lag], lw=1.8)
        for lag in sorted(color_lookup)
    ]
    lookback_labels = [f"n={lag}" for lag in sorted(color_lookup)]
    unit_handles = [
        Line2D([0], [0], color="#374151", lw=1.8, linestyle=style_lookup[unit])
        for unit in ["year", "meeting"]
    ]
    unit_labels = [label_lookup[unit] for unit in ["year", "meeting"]]
    legend_one = ax.legend(
        lookback_handles,
        lookback_labels,
        title="Lookback",
        loc="upper left",
        frameon=False,
        ncols=min(3, len(lookback_handles)),
    )
    ax.add_artist(legend_one)
    ax.legend(
        unit_handles,
        unit_labels,
        title="Definition",
        loc="lower right",
        frameon=False,
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

# %% [markdown]
# ## Load data

# %%
counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback(
    CFG.data_paths
)

year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
if year_col not in submitted_df.columns:
    raise KeyError("No meeting year or year column found in source data.")
meeting_col = (
    "meeting number" if "meeting number" in submitted_df.columns else "meeting_number"
)
submitted_df = sanitize_years(submitted_df, year_col)
if meeting_col in submitted_df.columns:
    submitted_df[meeting_col] = pd.to_numeric(submitted_df[meeting_col], errors="coerce")
    submitted_df = submitted_df.dropna(subset=[meeting_col]).copy()
    submitted_df[meeting_col] = submitted_df[meeting_col].astype(int)
elif str(CFG.time_unit).strip().lower() == "meeting":
    raise KeyError("No meeting-number column found in source data for time_unit='meeting'.")

# Canonical ordering (matches many scripts)
topics: list[str] = counts_df.index.tolist()
members: list[str] = counts_df.columns.tolist()
all_members_raw = set(members_raw)
all_topics_raw = set(topics_raw)

year_min = int(submitted_df[year_col].min())
year_max = int(submitted_df[year_col].max())
time_unit = str(CFG.time_unit).strip().lower()
time_col = meeting_col if time_unit == "meeting" else year_col
periods = build_periods_for_unit(
    submitted_df=submitted_df,
    year_col=year_col,
    meeting_col=meeting_col,
    time_unit=time_unit,
    window_size=CFG.window_years,
)
sequence_start = int(periods[0][0]) if periods else 0
period_min = int(periods[0][0]) if periods else 0
period_max = int(periods[-1][1]) if periods else 0

print(
    f"{time_unit.title()} periods: {period_min}–{period_max} | windows={len(periods)} | transitions={max(len(periods)-1, 0)}"
)
print(f"Topics: {len(topics)} | Members: {len(members)}")

# %% [markdown]
# ## Precompute interactions and active matrices per window

# %%
interaction_by_period, active_by_period = build_interactions_and_active_by_period(
    periods=periods,
    submitted_df=submitted_df,
    time_col=time_col,
    topics=topics,
    members=members,
    all_members_raw=all_members_raw,
    all_topics_raw=all_topics_raw,
)

# Aggregate distance (full history)
aggregate_interaction = counts_df.reindex(index=topics, columns=members, fill_value=0)
aggregate_dist = distance_from_interaction(
    aggregate_interaction, topics, metric=CFG.distance_metric
)

# %% [markdown]
# ## Build the hazard panel dataframes
#
# We export two panels:
# 1) Adoption (at-risk) panel:
#    one row per (member, topic, transition) where the topic is NOT active at t-1.
#    Outcome: `adopted` = 1 if the topic becomes active at t.
#
# 2) Persistence (previously-active) panel:
#    one row per (member, topic, transition) where the topic IS active at t-1.
#    Outcome: `persisted` = 1 if the topic remains active at t.
#
# For the persistence panel we also compute `distance_to_other_prev`, defined as the minimum
# distance from the focal topic to the member's OTHER previously-active topics at t-1
# (excluding itself). If the member had only one active topic at t-1, this value is NaN.

# %%
panel_rows: list[dict] = []
persist_rows: list[dict] = []

n_transitions_total = max(len(periods) - 1, 0)
n_transitions = (
    n_transitions_total
    if CFG.max_transitions is None
    else min(n_transitions_total, CFG.max_transitions)
)

for t in range(1, 1 + n_transitions):
    period_start, period_end = periods[t]
    prev_start, prev_end = periods[t - 1]

    prev_active = active_by_period[t - 1]
    curr_active = active_by_period[t]

    # distance matrix for this transition
    dist = compute_distance_matrix_for_transition(
        mode=CFG.distance_mode,
        t=t,
        source_idx=None,
        periods=periods,
        sequence_start=sequence_start,
        submitted_df=submitted_df,
        time_col=time_col,
        interaction_by_period=interaction_by_period,
        aggregate_dist=aggregate_dist,
        all_members_raw=all_members_raw,
        all_topics_raw=all_topics_raw,
        topics=topics,
        members=members,
    )

    # topic popularity in prev window
    prev_topic_popularity = prev_active.sum(axis=1) / max(len(members), 1)

    for member in members:
        prev_mask = prev_active[member].to_numpy(dtype=bool)
        if not prev_mask.any():
            # No prior portfolio => undefined min distance; skip (consistent with sensitivity script)
            continue

        curr_mask = curr_active[member].to_numpy(dtype=bool)

        at_risk = ~prev_mask
        if not at_risk.any():
            continue

        adopted = curr_mask & at_risk

        prev_indices = np.where(prev_mask)[0]
        min_dist = dist[:, prev_indices].min(axis=1)
        prev_diversity = int(len(prev_indices))

        # --- Adoption (at-risk) rows ---
        for idx, topic in enumerate(topics):
            if not at_risk[idx]:
                continue
            panel_rows.append(
                {
                    "time_unit": str(time_unit),
                    "window_size": int(CFG.window_years),
                    "transition_idx": int(t),
                    "prev_window_start": int(prev_start),
                    "prev_window_end": int(prev_end),
                    "window_start": int(period_start),
                    "window_end": int(period_end),
                    "period_end": int(period_end),
                    "member": str(member),
                    "topic": str(topic),
                    "adopted": int(adopted[idx]),
                    "distance": float(min_dist[idx]),
                    "prev_diversity": int(prev_diversity),
                    "topic_popularity": float(prev_topic_popularity.loc[topic]),
                    "distance_mode": str(CFG.distance_mode),
                    "distance_metric": str(CFG.distance_metric),
                    "window_years": int(CFG.window_years),
                    "rca_threshold": float(CFG.rca_threshold),
                }
            )

        # --- Persistence (previously-active) rows ---
        # Only defined if the member has at least one previously-active topic (guaranteed here),
        # and distance_to_other_prev is only defined if prev_diversity >= 2.
        for idx, topic in enumerate(topics):
            if not prev_mask[idx]:
                continue

            if prev_diversity >= 2:
                other_prev_indices = prev_indices[prev_indices != idx]
                # other_prev_indices must be non-empty
                d_to_other = float(dist[idx, other_prev_indices].min())
            else:
                d_to_other = float("nan")

            persist_rows.append(
                {
                    "time_unit": str(time_unit),
                    "window_size": int(CFG.window_years),
                    "transition_idx": int(t),
                    "prev_window_start": int(prev_start),
                    "prev_window_end": int(prev_end),
                    "window_start": int(period_start),
                    "window_end": int(period_end),
                    "period_end": int(period_end),
                    "member": str(member),
                    "topic": str(topic),
                    "persisted": int(curr_mask[idx]),
                    "distance_to_other_prev": d_to_other,
                    "prev_diversity": int(prev_diversity),
                    "topic_popularity": float(prev_topic_popularity.loc[topic]),
                    "distance_mode": str(CFG.distance_mode),
                    "distance_metric": str(CFG.distance_metric),
                    "window_years": int(CFG.window_years),
                    "rca_threshold": float(CFG.rca_threshold),
                }
            )

panel_df = pd.DataFrame(panel_rows)
if panel_df.empty:
    raise RuntimeError(
        "Adoption (at-risk) panel dataframe is empty. Check inputs / thresholds / year coverage."
    )

persist_df = pd.DataFrame(persist_rows)
if persist_df.empty:
    raise RuntimeError(
        "Persistence (prev-active) panel dataframe is empty. Check inputs / thresholds / year coverage."
    )

print("Adoption panel:", panel_df.shape)
print("Persistence panel:", persist_df.shape)
panel_df.head()

# %% [markdown]
# ## Quick sanity checks

# %%
adoption_rate = float(panel_df["adopted"].mean())
n_adopted = int(panel_df["adopted"].sum())
print(f"Rows (at-risk): {len(panel_df):,}")
print(f"Adoptions: {n_adopted:,} ({adoption_rate:.4%})")
print("Adoption distance summary:")
print(panel_df["distance"].describe())

persist_rate = float(persist_df["persisted"].mean())
n_persisted = int(persist_df["persisted"].sum())
print(f"\nRows (prev-active): {len(persist_df):,}")
print(f"Persisted: {n_persisted:,} ({persist_rate:.4%})")
print("Persistence distance_to_other_prev summary (excluding NaN):")
print(persist_df["distance_to_other_prev"].dropna().describe())

# %% [markdown]
# ## Export dataframe (Parquet recommended)

# %%
CFG.out_dir.mkdir(parents=True, exist_ok=True)
CFG.fig_dir.mkdir(parents=True, exist_ok=True)

# Adoption panel
panel_df.to_parquet(CFG.out_panel_parquet, index=False)
print(f"Wrote: {CFG.out_panel_parquet}")

if CFG.out_panel_csv is not None:
    panel_df.to_csv(CFG.out_panel_csv, index=False)
    print(f"Wrote: {CFG.out_panel_csv}")

# Persistence panel
persist_df.to_parquet(CFG.out_persist_panel_parquet, index=False)
print(f"Wrote: {CFG.out_persist_panel_parquet}")

if CFG.out_persist_panel_csv is not None:
    persist_df.to_csv(CFG.out_persist_panel_csv, index=False)
    print(f"Wrote: {CFG.out_persist_panel_csv}")

meta = {
    "window_size": CFG.window_years,
    "window_years": CFG.window_years,
    "time_unit": time_unit,
    "rca_threshold": CFG.rca_threshold,
    "distance_mode": CFG.distance_mode,
    "distance_metric": CFG.distance_metric,
    "distance_definition": (
        "shortest_path_neg_log_proximity"
        if CFG.distance_metric == "proximity"
        else "one_minus_similarity"
    ),
    "n_rows_at_risk": int(len(panel_df)),
    "n_adopted": int(panel_df["adopted"].sum()),
    "adoption_rate": float(panel_df["adopted"].mean()),
    "n_rows_prev_active": int(len(persist_df)),
    "n_persisted": int(persist_df["persisted"].sum()),
    "persistence_rate": float(persist_df["persisted"].mean()),
    "n_members": int(panel_df["member"].nunique()),
    "n_topics": int(panel_df["topic"].nunique()),
    "year_min": year_min,
    "year_max": year_max,
    "period_min": period_min,
    "period_max": period_max,
    "n_windows": int(len(periods)),
    "n_transitions": int(n_transitions),
}
CFG.out_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"Wrote: {CFG.out_meta_json}")

# %% [markdown]
# ## Hero-ready empirical curve: P(adopt | at risk) vs distance (binned)
#
# This is the direct source for the y-axis you asked about:
# \n
# **P(adopt i at t | at risk at t-1, distance bin)**


# %%
def binned_adoption_curve(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    d = df.copy()
    d = d[np.isfinite(d["distance"])].copy()
    if d.empty:
        return pd.DataFrame(
            columns=["bin_left", "bin_right", "bin_center", "n", "p_adopt"]
        )

    d["dist_bin"] = pd.cut(d["distance"], bins=n_bins, include_lowest=True)
    g = d.groupby("dist_bin", observed=True)["adopted"]
    out = g.agg(n="size", p_adopt="mean").reset_index()

    # Convert Interval edges to plain floats, then do arithmetic on numpy float arrays.
    bin_left = out["dist_bin"].apply(lambda x: float(x.left)).to_numpy(dtype=float)
    bin_right = out["dist_bin"].apply(lambda x: float(x.right)).to_numpy(dtype=float)

    out["bin_left"] = bin_left
    out["bin_right"] = bin_right
    out["bin_center"] = 0.5 * (bin_left + bin_right)

    return (
        out.drop(columns=["dist_bin"]).sort_values("bin_center").reset_index(drop=True)
    )


def binned_persistence_curve(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    d = df.copy()
    d = d[np.isfinite(d["distance_to_other_prev"])].copy()
    if d.empty:
        return pd.DataFrame(
            columns=["bin_left", "bin_right", "bin_center", "n", "p_persist"]
        )

    d["dist_bin"] = pd.cut(
        d["distance_to_other_prev"], bins=n_bins, include_lowest=True
    )
    g = d.groupby("dist_bin", observed=True)["persisted"]
    out = g.agg(n="size", p_persist="mean").reset_index()

    bin_left = out["dist_bin"].apply(lambda x: float(x.left)).to_numpy(dtype=float)
    bin_right = out["dist_bin"].apply(lambda x: float(x.right)).to_numpy(dtype=float)

    out["bin_left"] = bin_left
    out["bin_right"] = bin_right
    out["bin_center"] = 0.5 * (bin_left + bin_right)

    return (
        out.drop(columns=["dist_bin"]).sort_values("bin_center").reset_index(drop=True)
    )


curve_df = binned_adoption_curve(panel_df, n_bins=20)
curve_df.head()

persist_curve_df = binned_persistence_curve(persist_df, n_bins=20)
persist_curve_df.head()

# %%
curve_out = Path("output/hazard_empirical_adoption_curve_by_distance.csv")
curve_df.to_csv(curve_out, index=False)
print(f"Wrote: {curve_out}")

persist_curve_out = Path("output/hazard_empirical_persistence_curve_by_distance.csv")
persist_curve_df.to_csv(persist_curve_out, index=False)
print(f"Wrote: {persist_curve_out}")

# %% [markdown]
# ## Quick plots (optional)
# Produces:
# - `figures/hazard_adoption_curve_by_distance.png`
# - `figures/hazard_adoption_distance_hist.png`

# %%
if plt is not None:
    # Adoption curve
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.plot(curve_df["bin_center"], curve_df["p_adopt"], marker="o", lw=1.6, ms=4)
    ax.set_xlabel("Min log-distance to prior portfolio (t-1)")
    ax.set_ylabel("P(adopt at t | at risk at t-1)")
    ax.set_title("Empirical adoption probability declines with log-distance")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.savefig(
        Path("figures/hazard_adoption_curve_by_distance.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Persistence curve
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.plot(
        persist_curve_df["bin_center"],
        persist_curve_df["p_persist"],
        marker="o",
        lw=1.6,
        ms=4,
    )
    ax.set_xlabel("Min log-distance to other prior topics (t-1)")
    ax.set_ylabel("P(persist at t | active at t-1)")
    ax.set_title("Empirical persistence probability vs embeddedness (log-distance)")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.savefig(
        Path("figures/hazard_persistence_curve_by_distance.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Distance histogram (adoptions vs non-adoptions)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    adopted = panel_df.loc[panel_df["adopted"] == 1, "distance"].to_numpy(dtype=float)
    not_adopted = panel_df.loc[panel_df["adopted"] == 0, "distance"].to_numpy(
        dtype=float
    )
    bins = 30
    ax.hist(not_adopted, bins=bins, alpha=0.35, label="Not adopted", density=True)
    ax.hist(adopted, bins=bins, alpha=0.55, label="Adopted", density=True)
    ax.set_xlabel("Min log-distance to prior portfolio (t-1)")
    ax.set_ylabel("Density")
    ax.set_title("Adoptions concentrate at shorter log-distances")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.savefig(
        Path("figures/hazard_adoption_distance_hist.png"), dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    print("Wrote: figures/hazard_adoption_curve_by_distance.png")
    print("Wrote: figures/hazard_persistence_curve_by_distance.png")
    print("Wrote: figures/hazard_adoption_distance_hist.png")

if LOOKBACK_COMPARISON_ENABLED:
    if CFG.distance_metric != "proximity":
        print(
            "Skipping lookback comparison because log-phi is only defined for distance_metric='proximity'."
        )
    else:
        meeting_col = (
            "meeting number"
            if "meeting number" in submitted_df.columns
            else "meeting_number"
        )
        if meeting_col not in submitted_df.columns:
            print("Skipping lookback comparison because no meeting-number column is available.")
        else:
            submitted_meeting_df = submitted_df.copy()
            submitted_meeting_df[meeting_col] = pd.to_numeric(
                submitted_meeting_df[meeting_col], errors="coerce"
            )
            submitted_meeting_df = submitted_meeting_df.dropna(subset=[meeting_col]).copy()
            submitted_meeting_df[meeting_col] = submitted_meeting_df[meeting_col].astype(int)

            year_panel_compare = build_lookback_adoption_panel(
                time_unit="year",
                periods=build_periods_for_unit(
                    submitted_df=submitted_df,
                    year_col=year_col,
                    meeting_col=meeting_col,
                    time_unit="year",
                    window_size=CFG.window_years,
                ),
                time_col=year_col,
                submitted_df=submitted_df,
                topics=topics,
                members=members,
                all_members_raw=all_members_raw,
                all_topics_raw=all_topics_raw,
                aggregate_dist=aggregate_dist,
                max_lookback_steps=LOOKBACK_COMPARISON_MAX_LAG,
            )
            meeting_periods = build_periods_for_unit(
                submitted_df=submitted_meeting_df,
                year_col=year_col,
                meeting_col=meeting_col,
                time_unit="meeting",
                window_size=CFG.window_years,
            )
            meeting_panel_compare = build_lookback_adoption_panel(
                time_unit="meeting",
                periods=meeting_periods,
                time_col=meeting_col,
                submitted_df=submitted_meeting_df,
                topics=topics,
                members=members,
                all_members_raw=all_members_raw,
                all_topics_raw=all_topics_raw,
                aggregate_dist=aggregate_dist,
                max_lookback_steps=LOOKBACK_COMPARISON_MAX_LAG,
            )

            lookback_panel_df = pd.concat(
                [year_panel_compare, meeting_panel_compare], ignore_index=True
            )
            lookback_curve_df = binned_grouped_adoption_curve(
                lookback_panel_df,
                value_col="log_phi",
                group_cols=["time_unit", "lookback_steps"],
                n_bins=LOOKBACK_COMPARISON_N_BINS,
            )

            lookback_panel_out = tagged_output_path(
                Path("output/hazard_panel_lookback_comparison.parquet"),
                LOOKBACK_OUTPUT_TAG,
            )
            lookback_curve_out = tagged_output_path(
                Path("output/hazard_empirical_adoption_curve_by_log_phi_lookback_comparison.csv"),
                LOOKBACK_OUTPUT_TAG,
            )
            lookback_meta_out = tagged_output_path(
                Path("output/hazard_panel_lookback_comparison_meta.json"),
                LOOKBACK_OUTPUT_TAG,
            )
            lookback_fig_out = tagged_output_path(
                Path("figures/hazard_adoption_log_phi_decay_years_vs_meetings.png"),
                LOOKBACK_OUTPUT_TAG,
            )

            lookback_panel_df.to_parquet(lookback_panel_out, index=False)
            lookback_curve_df.to_csv(lookback_curve_out, index=False)
            lookback_meta = {
                "distance_mode": CFG.distance_mode,
                "distance_metric": CFG.distance_metric,
                "distance_definition": "shortest_path_neg_log_proximity",
                "max_lookback_steps": int(LOOKBACK_COMPARISON_MAX_LAG),
                "lookback_bins": int(LOOKBACK_COMPARISON_N_BINS),
                "window_years": int(CFG.window_years),
                "time_units": ["year", "meeting"],
                "n_rows": int(len(lookback_panel_df)),
                "n_topics": int(lookback_panel_df["topic"].nunique()),
                "n_members": int(lookback_panel_df["member"].nunique()),
                "rows_by_time_unit": {
                    str(k): int(v)
                    for k, v in lookback_panel_df.groupby("time_unit").size().to_dict().items()
                },
            }
            lookback_meta_out.write_text(
                json.dumps(lookback_meta, indent=2), encoding="utf-8"
            )
            plot_lookback_log_phi_comparison(lookback_curve_df, lookback_fig_out)

            print(f"Wrote: {lookback_panel_out}")
            print(f"Wrote: {lookback_curve_out}")
            print(f"Wrote: {lookback_meta_out}")
            if plt is not None:
                print(f"Wrote: {lookback_fig_out}")

# %% [markdown]
# ## How to load the exported dataframe elsewhere
#
# ```python
# import pandas as pd
# df = pd.read_parquet("output/hazard_panel_at_risk.parquet")
# ```
#
# Then compute your y-axis:
# - `df.groupby(pd.cut(df["distance"], 20))["adopted"].mean()`
#
# Or fit controlled models off this panel.
