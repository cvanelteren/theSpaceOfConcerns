from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import generate_interaction_matrix, get_rca, load_data, standardize_index_labels

DATA_PATH = Path("antarctic-database-go/data/processed/document-summary.parquet")
PERIOD_YEARS = 10
AGGREGATE_ONLY = os.getenv("FIG19_AGGREGATE_ONLY", "0") == "1"
RPA_THRESHOLD = 1.0
N_DRAWS = int(os.getenv("FIG19_N_DRAWS", "400"))
NULL_KIND = os.getenv("FIG19_NULL_KIND", "exact")
NULL_BURN_IN_FACTOR = int(os.getenv("FIG19_NULL_BURN_IN_FACTOR", "20"))
NULL_SWAPS_PER_DRAW_FACTOR = int(os.getenv("FIG19_NULL_SWAPS_PER_DRAW_FACTOR", "8"))
PROGRESS_EVERY = int(os.getenv("FIG19_PROGRESS_EVERY", "25"))
SEED = 7

OUT_SUMMARY = Path("output/fig19_division_of_labor_summary.csv")
OUT_DRAWS = Path("output/fig19_division_of_labor_null_draws.csv")
OUT_META = Path("output/fig19_division_of_labor_meta.json")
OUT_PDF = Path("figures/fig19_division_of_labor_null.pdf")
OUT_PNG = Path("figures/fig19_division_of_labor_null.png")


def build_periods(year_min: int, year_max: int, step: int) -> list[tuple[int, int, str, int]]:
    if AGGREGATE_ONLY:
        return [(year_min, year_max, f"{year_min}-{year_max}", 0)]
    periods: list[tuple[int, int, str, int]] = []
    period_idx = 0
    year = year_min
    while year <= year_max:
        end = min(year + step - 1, year_max)
        periods.append((year, end, f"{year}-{end}", period_idx))
        year = end + 1
        period_idx += 1
    return periods


def mean_pairwise_jaccard_from_matrix(mat: np.ndarray) -> float:
    active = mat[mat.sum(axis=1) > 0]
    if active.shape[0] < 2:
        return np.nan
    active = active.astype(bool, copy=False)
    inter = (active[:, None, :] & active[None, :, :]).sum(axis=2)
    union = (active[:, None, :] | active[None, :, :]).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = inter / union
    tri = np.triu_indices_from(jac, k=1)
    vals = jac[tri]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan


def mean_pairwise_jaccard(binary_actor_topic: pd.DataFrame) -> float:
    return mean_pairwise_jaccard_from_matrix(binary_actor_topic.to_numpy(dtype=np.uint8, copy=False))


def weighted_row_constrained_null(
    binary_actor_topic: pd.DataFrame,
    rng: np.random.Generator,
    n_draws: int,
) -> np.ndarray:
    active = binary_actor_topic.loc[binary_actor_topic.sum(axis=1) > 0]
    mat = active.to_numpy(dtype=np.uint8, copy=False)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return np.full(n_draws, np.nan, dtype=float)
    row_sums = mat.sum(axis=1).astype(int)
    col_sums = mat.sum(axis=0).astype(float)
    if row_sums.sum() == 0 or np.all(col_sums == 0):
        return np.full(n_draws, np.nan, dtype=float)
    probs = col_sums / col_sums.sum()
    topic_idx = np.arange(mat.shape[1])

    draws = np.empty(n_draws, dtype=float)
    for draw_idx in range(n_draws):
        sim = np.zeros_like(mat, dtype=np.uint8)
        for row_idx, k in enumerate(row_sums):
            if k <= 0:
                continue
            choose = rng.choice(topic_idx, size=int(k), replace=False, p=probs)
            sim[row_idx, choose] = 1
        draws[draw_idx] = mean_pairwise_jaccard(pd.DataFrame(sim))
    return draws


def accepted_row_pair_swap(mat: np.ndarray, rng: np.random.Generator) -> bool:
    n_rows, n_cols = mat.shape
    if n_rows < 2 or n_cols < 2:
        return False
    r1, r2 = rng.choice(n_rows, size=2, replace=False)
    row1 = mat[r1]
    row2 = mat[r2]
    only1 = np.flatnonzero((row1 == 1) & (row2 == 0))
    only2 = np.flatnonzero((row1 == 0) & (row2 == 1))
    if only1.size == 0 or only2.size == 0:
        return False
    c1 = int(rng.choice(only1))
    c2 = int(rng.choice(only2))
    row1[c1] = 0
    row2[c1] = 1
    row1[c2] = 1
    row2[c2] = 0
    return True


def exact_row_col_swap_null(
    binary_actor_topic: pd.DataFrame,
    rng: np.random.Generator,
    n_draws: int,
    *,
    burn_in_factor: int,
    swaps_per_draw_factor: int,
    progress_every: int,
    progress_label: str,
) -> np.ndarray:
    active = binary_actor_topic.loc[binary_actor_topic.sum(axis=1) > 0]
    mat = active.to_numpy(dtype=np.uint8, copy=True)
    n_rows, n_cols = mat.shape
    n_edges = int(mat.sum())
    if n_rows < 2 or n_cols < 2 or n_edges == 0:
        return np.full(n_draws, np.nan, dtype=float)

    burn_in_swaps = max(n_edges, burn_in_factor * n_edges)
    swaps_per_draw = max(n_edges // 2, swaps_per_draw_factor * n_edges)

    def run_swaps(target_accepts: int) -> None:
        accepted = 0
        attempts = 0
        max_attempts = max(target_accepts * 100, 1000)
        while accepted < target_accepts and attempts < max_attempts:
            attempts += 1
            if accepted_row_pair_swap(mat, rng):
                accepted += 1
        if accepted < target_accepts:
            print(
                f"{progress_label} warning: accepted {accepted}/{target_accepts} swaps after {attempts} attempts",
                flush=True,
            )

    print(
        f"{progress_label} exact null: burn-in {burn_in_swaps} accepted swaps, "
        f"then {n_draws} draws every {swaps_per_draw} accepted swaps",
        flush=True,
    )
    run_swaps(burn_in_swaps)

    draws = np.empty(n_draws, dtype=float)
    for draw_idx in range(n_draws):
        run_swaps(swaps_per_draw)
        draws[draw_idx] = mean_pairwise_jaccard_from_matrix(mat)
        if (draw_idx + 1) % max(progress_every, 1) == 0 or draw_idx + 1 == n_draws:
            print(
                f"{progress_label} draw {draw_idx + 1}/{n_draws}: "
                f"null mean overlap so far = {np.nanmean(draws[: draw_idx + 1]):.3f}",
                flush=True,
            )
    return draws


def main() -> None:
    _df, submitted, countries, topics = load_data(str(DATA_PATH))
    submitted = submitted.copy()
    submitted["meeting year"] = pd.to_numeric(
        submitted["meeting year"], errors="coerce"
    )
    submitted = submitted.dropna(subset=["meeting year"]).copy()
    submitted["meeting year"] = submitted["meeting year"].astype(int)

    countries = sorted(countries)
    topics = sorted(topics)
    periods = build_periods(
        int(submitted["meeting year"].min()),
        int(submitted["meeting year"].max()),
        PERIOD_YEARS,
    )
    rng = np.random.default_rng(SEED)

    summary_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []

    bases = [
        ("rpa", "RPA > 1"),
        ("presence", "Any activity"),
    ]

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
        rpa = get_rca(counts_df).gt(RPA_THRESHOLD).astype(int).T

        for basis_key, basis_label in bases:
            active = rpa if basis_key == "rpa" else presence
            active = active.loc[active.sum(axis=1) > 0]
            if len(active) < 2:
                continue
            observed = mean_pairwise_jaccard(active)
            progress_label = f"[{basis_label} {period_label}]"
            print(
                f"{progress_label} actors={len(active)}, topics={active.shape[1]}, "
                f"observed mean overlap={observed:.3f}; running {NULL_KIND} null with {N_DRAWS} draws...",
                flush=True,
            )
            if NULL_KIND == "exact":
                null_draws = exact_row_col_swap_null(
                    active,
                    rng=rng,
                    n_draws=N_DRAWS,
                    burn_in_factor=NULL_BURN_IN_FACTOR,
                    swaps_per_draw_factor=NULL_SWAPS_PER_DRAW_FACTOR,
                    progress_every=PROGRESS_EVERY,
                    progress_label=progress_label,
                )
            elif NULL_KIND == "approx":
                null_draws = weighted_row_constrained_null(active, rng=rng, n_draws=N_DRAWS)
                print(f"{progress_label} approximate null completed", flush=True)
            else:
                raise ValueError(f"Unsupported FIG19_NULL_KIND={NULL_KIND!r}; use 'exact' or 'approx'.")
            valid = null_draws[np.isfinite(null_draws)]
            if valid.size == 0:
                continue
            null_mean = float(valid.mean())
            null_lo = float(np.quantile(valid, 0.05))
            null_hi = float(np.quantile(valid, 0.95))
            null_sd = float(valid.std(ddof=1)) if valid.size > 1 else np.nan
            p_lower = float((np.sum(valid <= observed) + 1) / (valid.size + 1))
            summary_rows.append(
                {
                    "period_idx": period_idx,
                    "period_start": period_start,
                    "period_end": period_end,
                    "period_label": period_label,
                    "basis": basis_key,
                    "basis_label": basis_label,
                    "n_active_actors": int(len(active)),
                    "n_topics_active": int((active.sum(axis=0) > 0).sum()),
                    "observed_mean_pairwise_jaccard": observed,
                    "null_mean_pairwise_jaccard": null_mean,
                    "null_q05": null_lo,
                    "null_q95": null_hi,
                    "null_sd": null_sd,
                    "gap_observed_minus_null": float(observed - null_mean),
                    "z_observed_minus_null": float((observed - null_mean) / null_sd) if np.isfinite(null_sd) and null_sd > 0 else np.nan,
                    "p_lower_than_null": p_lower,
                }
            )
            draw_rows.extend(
                {
                    "period_idx": period_idx,
                    "period_label": period_label,
                    "basis": basis_key,
                    "draw": int(draw_idx),
                    "null_mean_pairwise_jaccard": float(val),
                }
                for draw_idx, val in enumerate(valid)
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["basis", "period_idx"]).reset_index(drop=True)
    draws_df = pd.DataFrame(draw_rows).sort_values(["basis", "period_idx", "draw"]).reset_index(drop=True)

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    draws_df.to_csv(OUT_DRAWS, index=False)
    OUT_META.write_text(
        json.dumps(
            {
                "aggregate_only": AGGREGATE_ONLY,
                "period_years": PERIOD_YEARS,
                "rpa_threshold": RPA_THRESHOLD,
                "n_draws": N_DRAWS,
                "seed": SEED,
                "null_kind": NULL_KIND,
                "null_burn_in_factor": NULL_BURN_IN_FACTOR,
                "null_swaps_per_draw_factor": NULL_SWAPS_PER_DRAW_FACTOR,
                "progress_every": PROGRESS_EVERY,
                "null_description": (
                    "Exact row+column-preserving bipartite swap null within each period."
                    if NULL_KIND == "exact"
                    else "Row-constrained null preserving actor breadth exactly and topic popularity in expectation within each period."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    colors = {"rpa": "blue7", "presence": "orange7"}
    fig, axs = uplt.subplots(ncols=2, share=False, refwidth=3.2, refaspect=0.95)
    axs.format(abc="[A]", grid=False)

    for ax, (basis_key, basis_label) in zip(axs, bases):
        dfi = summary_df[summary_df["basis"] == basis_key].copy()
        x = np.arange(len(dfi))
        ax.fill_between(
            x,
            dfi["null_q05"].to_numpy(dtype=float),
            dfi["null_q95"].to_numpy(dtype=float),
            color=colors[basis_key],
            alpha=0.18,
            zorder=1,
        )
        ax.plot(
            x,
            dfi["null_mean_pairwise_jaccard"].to_numpy(dtype=float),
            color=colors[basis_key],
            lw=2.0,
            label="Null mean",
            zorder=2,
        )
        ax.plot(
            x,
            dfi["observed_mean_pairwise_jaccard"].to_numpy(dtype=float),
            color="black",
            lw=2.2,
            marker="o",
            ms=4,
            label="Observed",
            zorder=3,
        )
        ax.axhline(0, color="black", alpha=0.15, lw=0.8, zorder=0)
        mean_gap = float(dfi["gap_observed_minus_null"].mean())
        mean_p = float(dfi["p_lower_than_null"].mean())
        ax.text(
            0.03,
            0.97,
            f"mean gap = {mean_gap:+.03f}\nmean $p_{{low}}$ = {mean_p:.02f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
            zorder=4,
        )
        ax.format(
            title=basis_label,
            xlabel="Archive span" if AGGREGATE_ONLY else "Period",
            ylabel="Mean pairwise overlap",
            xticks=x,
            xticklabels=dfi["period_label"].tolist(),
            xrotation=35,
            ylim=(0, max(0.45, float(np.nanmax(dfi[["null_q95", "observed_mean_pairwise_jaccard"]].to_numpy(dtype=float))) * 1.08)),
        )
        ax.grid(alpha=0.18, color="black")

    fig.legend(loc="b", ncols=2, frame=False)
    scope_label = "aggregate archive" if AGGREGATE_ONLY else "periodized archive"
    fig.format(
        suptitle=(
            f"Observed portfolio overlap versus an exact row+column-preserving null ({scope_label})"
            if NULL_KIND == "exact"
            else f"Observed portfolio overlap versus a breadth-and-topic-preserving null ({scope_label})"
        )
    )
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
