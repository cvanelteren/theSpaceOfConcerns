"""Supplementary Figure S18. Compares collaboration frequency with portfolio overlap for country pairs. Shows that co-submission tracks shared concerns but does not fully explain the concern space."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy.stats import pearsonr, spearmanr

from utils import generate_interaction_matrix, get_rca, load_data, standardize_index_labels

DATA_PATH = Path("antarctic-database-go/data/processed/document-summary.parquet")
PERIOD_YEARS = 10
RPA_THRESHOLD = 1.0

OUT_DATA = Path("output/fig18_collaboration_portfolio_overlap_pair_period.csv")
OUT_SUMMARY = Path("output/fig18_collaboration_portfolio_overlap_summary.json")
OUT_PNG = Path("figures/figS18_collaboration_portfolio_overlap.png")
OUT_PDF = Path("figures/figS18_collaboration_portfolio_overlap.pdf")


def build_periods(year_min: int, year_max: int, step: int) -> list[tuple[int, int, str, int]]:
    periods: list[tuple[int, int, str, int]] = []
    period_idx = 0
    year = year_min
    while year <= year_max:
        end = min(year + step - 1, year_max)
        periods.append((year, end, f"{year}-{end}", period_idx))
        year = end + 1
        period_idx += 1
    return periods


def to_party_list(value) -> list[str]:
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        out = [str(v).strip() for v in value]
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            return []
        out = [p.strip() for p in text.split(",")]
    return [v for v in out if v and v.lower() not in {"nan", "none", "<na>"}]


def binned_positive_curve(
    x: np.ndarray, y: np.ndarray, n_bins: int = 7, min_count: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < max(min_count, 5):
        return np.array([]), np.array([])

    logx = np.log1p(x)
    edges = np.quantile(logx, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return np.array([]), np.array([])

    mids = []
    meds = []
    for idx in range(edges.size - 1):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == edges.size - 2:
            sel = (logx >= lo) & (logx <= hi)
        else:
            sel = (logx >= lo) & (logx < hi)
        if sel.sum() < min_count:
            continue
        mids.append(float(np.median(logx[sel])))
        meds.append(float(np.median(y[sel])))
    return np.asarray(mids), np.asarray(meds)


def log_tick_positions(values: list[int]) -> tuple[list[float], list[str]]:
    vals = [v for v in values if v >= 0]
    return [float(np.log1p(v)) for v in vals], [str(v) for v in vals]


def main() -> None:
    df, submitted, countries, topics = load_data(str(DATA_PATH))
    countries = sorted(countries)
    topics = sorted(topics)

    submitted = submitted.copy()
    submitted["meeting year"] = pd.to_numeric(
        submitted["meeting year"], errors="coerce"
    )
    submitted = submitted.dropna(subset=["meeting year"]).copy()
    submitted["meeting year"] = submitted["meeting year"].astype(int)
    submitted["parties_list"] = submitted["parties"].apply(to_party_list)
    submitted["n_parties"] = submitted["parties_list"].apply(len)
    submitted = submitted[submitted["n_parties"] > 0].copy()

    periods = build_periods(
        int(submitted["meeting year"].min()),
        int(submitted["meeting year"].max()),
        PERIOD_YEARS,
    )

    pair_rows: list[dict[str, object]] = []

    for period_start, period_end, period_label, period_idx in periods:
        period_submitted = submitted[
            submitted["meeting year"].between(period_start, period_end)
        ].copy()
        if period_submitted.empty:
            continue

        counts_df = generate_interaction_matrix(period_submitted, countries, topics)
        counts_df = standardize_index_labels(counts_df)
        counts_df = counts_df.reindex(index=sorted(counts_df.index), columns=sorted(counts_df.columns), fill_value=0)
        presence = counts_df.gt(0).astype(int).T
        specialized = get_rca(counts_df).gt(RPA_THRESHOLD).astype(int).T

        collab_counts: dict[tuple[str, str], int] = {}
        for parties in period_submitted["parties_list"]:
            uniq = sorted(set(parties))
            for actor_a, actor_b in combinations(uniq, 2):
                key = (actor_a, actor_b)
                collab_counts[key] = collab_counts.get(key, 0) + 1

        actors = list(presence.index)
        active_presence = presence.sum(axis=1).to_dict()
        active_specialized = specialized.sum(axis=1).to_dict()
        for idx, actor_a in enumerate(actors):
            vec_presence_a = presence.loc[actor_a].to_numpy(dtype=int)
            vec_specialized_a = specialized.loc[actor_a].to_numpy(dtype=int)
            for actor_b in actors[idx + 1 :]:
                vec_presence_b = presence.loc[actor_b].to_numpy(dtype=int)
                vec_specialized_b = specialized.loc[actor_b].to_numpy(dtype=int)

                inter_presence = int(np.logical_and(vec_presence_a, vec_presence_b).sum())
                union_presence = int(np.logical_or(vec_presence_a, vec_presence_b).sum())
                inter_specialized = int(np.logical_and(vec_specialized_a, vec_specialized_b).sum())
                union_specialized = int(np.logical_or(vec_specialized_a, vec_specialized_b).sum())

                pair_rows.append(
                    {
                        "period_idx": period_idx,
                        "period_start": period_start,
                        "period_end": period_end,
                        "period_label": period_label,
                        "period_mid": 0.5 * (period_start + period_end),
                        "actor_a": actor_a,
                        "actor_b": actor_b,
                        "collaboration_count": int(collab_counts.get((actor_a, actor_b), 0)),
                        "presence_jaccard": (inter_presence / union_presence) if union_presence > 0 else np.nan,
                        "specialized_jaccard": (inter_specialized / union_specialized) if union_specialized > 0 else np.nan,
                        "presence_active_a": int(active_presence.get(actor_a, 0)),
                        "presence_active_b": int(active_presence.get(actor_b, 0)),
                        "specialized_active_a": int(active_specialized.get(actor_a, 0)),
                        "specialized_active_b": int(active_specialized.get(actor_b, 0)),
                    }
                )

    pair_df = pd.DataFrame(pair_rows).sort_values(
        ["period_idx", "actor_a", "actor_b"]
    ).reset_index(drop=True)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(OUT_DATA, index=False)

    panels = [
        (
            "specialized_jaccard",
            "Portfolio overlap (RPA > 1)",
            "blue7",
            "specialized_active_a",
            "specialized_active_b",
        ),
        (
            "presence_jaccard",
            "Portfolio overlap (any activity)",
            "orange7",
            "presence_active_a",
            "presence_active_b",
        ),
    ]

    fig, axs = uplt.subplots(
        ncols=3,
        share=False,
        refwidth=2.6,
        refaspect=1.0,
        left="8em",
        right="3em",
        bottom="7em",
        top="5em",
        wspace="3em",
    )
    axs.format(abc="[A]", grid=False)

    collab_pair_df = pair_df[pair_df["collaboration_count"] > 0].copy()
    summary: dict[str, object] = {
        "period_years": PERIOD_YEARS,
        "n_pair_periods_all": int(len(pair_df)),
        "n_pair_periods_collaborating": int(len(collab_pair_df)),
        "share_zero_collaboration_all_pairs": float((pair_df["collaboration_count"] == 0).mean()),
        "max_collaboration_count": int(pair_df["collaboration_count"].max()),
    }

    tick_values = [0, 1, 3, 10, 30, 100]
    tick_positions, tick_labels = log_tick_positions(tick_values)

    for ax, (ycol, ylabel, line_color, active_a_col, active_b_col) in zip(axs[:2], panels):
        panel_df = collab_pair_df[
            (collab_pair_df[active_a_col] > 0) & (collab_pair_df[active_b_col] > 0)
        ].dropna(subset=[ycol]).copy()
        x_raw = panel_df["collaboration_count"].to_numpy(dtype=float)
        x = np.log1p(x_raw)
        y = panel_df[ycol].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=9,
            c="black",
            alpha=0.16,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        curve_x, curve_y = binned_positive_curve(
            panel_df["collaboration_count"].to_numpy(dtype=float), y, n_bins=7, min_count=20
        )
        if curve_x.size:
            ax.plot(curve_x, curve_y, color=line_color, lw=2.2, marker="o", ms=3.8, zorder=4)

        sp = spearmanr(panel_df["collaboration_count"], y)
        pr = pearsonr(np.log1p(panel_df["collaboration_count"]), y)
        summary[ycol] = {
            "n_collaborating_pair_periods_active": int(len(panel_df)),
            "spearman_rho": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "pearson_r_log1p": float(pr.statistic),
            "pearson_p_log1p": float(pr.pvalue),
            "median_overlap": float(np.median(y)),
        }

        ax.text(
            0.03,
            0.97,
            (
                f"n pair-periods = {len(panel_df)}\n"
                f"Spearman $\\rho$ = {sp.statistic:.2f}\n"
                f"Pearson $r$ on log(1+x) = {pr.statistic:.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
            zorder=5,
        )

        ax.format(
            xlabel="Co-submissions in period",
            ylabel=ylabel,
            xlim=(0, float(np.log1p(max(100, collab_pair_df["collaboration_count"].max())))),
            ylim=(0, 1),
            xticks=tick_positions,
            xticklabels=tick_labels,
            ylocator=np.linspace(0, 1, 6),
        )
        ax.xaxis.labelpad = 10
        ax.grid(alpha=0.18, color="black")

    overall_counts_df = generate_interaction_matrix(submitted, countries, topics)
    overall_counts_df = standardize_index_labels(overall_counts_df)
    overall_counts_df = overall_counts_df.reindex(
        index=sorted(overall_counts_df.index), columns=sorted(overall_counts_df.columns), fill_value=0
    )
    overall_presence = overall_counts_df.gt(0).astype(int).T
    overall_specialized = get_rca(overall_counts_df).gt(RPA_THRESHOLD).astype(int).T

    overall_collab_counts: dict[tuple[str, str], int] = {}
    for parties in submitted["parties_list"]:
        uniq = sorted(set(parties))
        for actor_a, actor_b in combinations(uniq, 2):
            key = (actor_a, actor_b)
            overall_collab_counts[key] = overall_collab_counts.get(key, 0) + 1

    overall_rows: list[dict[str, object]] = []
    actors = list(overall_presence.index)
    for idx, actor_a in enumerate(actors):
        vec_presence_a = overall_presence.loc[actor_a].to_numpy(dtype=int)
        vec_specialized_a = overall_specialized.loc[actor_a].to_numpy(dtype=int)
        for actor_b in actors[idx + 1 :]:
            collab = int(overall_collab_counts.get((actor_a, actor_b), 0))
            if collab <= 0:
                continue
            vec_presence_b = overall_presence.loc[actor_b].to_numpy(dtype=int)
            vec_specialized_b = overall_specialized.loc[actor_b].to_numpy(dtype=int)
            inter_presence = int(np.logical_and(vec_presence_a, vec_presence_b).sum())
            union_presence = int(np.logical_or(vec_presence_a, vec_presence_b).sum())
            inter_specialized = int(np.logical_and(vec_specialized_a, vec_specialized_b).sum())
            union_specialized = int(np.logical_or(vec_specialized_a, vec_specialized_b).sum())
            overall_rows.append(
                {
                    "actor_a": actor_a,
                    "actor_b": actor_b,
                    "collaboration_count": collab,
                    "presence_jaccard": (inter_presence / union_presence) if union_presence > 0 else np.nan,
                    "specialized_jaccard": (inter_specialized / union_specialized) if union_specialized > 0 else np.nan,
                    "presence_active_a": int(overall_presence.loc[actor_a].sum()),
                    "presence_active_b": int(overall_presence.loc[actor_b].sum()),
                    "specialized_active_a": int(overall_specialized.loc[actor_a].sum()),
                    "specialized_active_b": int(overall_specialized.loc[actor_b].sum()),
                }
            )
    overall_df = pd.DataFrame(overall_rows)

    ax = axs[2]
    overall_stats_lines = []
    legend_handles = []
    legend_labels = []
    for ycol, _ylabel, line_color, active_a_col, active_b_col in panels:
        panel_df = overall_df[
            (overall_df[active_a_col] > 0) & (overall_df[active_b_col] > 0)
        ].dropna(subset=[ycol]).copy()
        if panel_df.empty:
            continue
        x = np.log1p(panel_df["collaboration_count"].to_numpy(dtype=float))
        y = panel_df[ycol].to_numpy(dtype=float)
        scatter = ax.scatter(
            x,
            y,
            s=10,
            color=line_color,
            alpha=0.16,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        curve_x, curve_y = binned_positive_curve(
            panel_df["collaboration_count"].to_numpy(dtype=float), y, n_bins=7, min_count=12
        )
        if curve_x.size:
            ax.plot(curve_x, curve_y, color=line_color, lw=2.2, marker="o", ms=3.4, zorder=3)
        sp = spearmanr(panel_df["collaboration_count"], y)
        summary[f"overall_{ycol}"] = {
            "n_collaborating_pairs_active": int(len(panel_df)),
            "spearman_rho": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "median_overlap": float(np.median(y)),
        }
        label = "RPA > 1" if ycol == "specialized_jaccard" else "Any activity"
        overall_stats_lines.append(f"{label}: $\\rho$ = {sp.statistic:.2f}")
        legend_handles.append(scatter)
        legend_labels.append(label)

    if overall_stats_lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(overall_stats_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
            zorder=4,
        )
    ax.format(
        xlabel="Total co-submissions",
        ylabel="Portfolio overlap",
        xlim=(0, float(np.log1p(max(100, overall_df["collaboration_count"].max() if len(overall_df) else 100)))),
        ylim=(0, 1),
        xticks=tick_positions,
        xticklabels=tick_labels,
        ylocator=np.linspace(0, 1, 6),
    )
    ax.xaxis.labelpad = 10
    ax.grid(alpha=0.18, color="black")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="lr", frame=False, fontsize=7, borderaxespad=0.6)

    fig.format(
        suptitle="Countries that collaborate more often also hold more similar portfolios",
        toplabels=("RPA > 1", "Any observed activity", "Overall"),
    )

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
