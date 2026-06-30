from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

from fig19_division_of_labor_null import accepted_row_pair_swap
from utils import get_rca, load_data, standardize_index_labels

DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
ACTOR_SUMMARY_FP = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")

RPA_THRESHOLD = float(os.getenv("FIG23_RPA_THRESHOLD", "1.0"))
N_DRAWS = int(os.getenv("FIG23_N_DRAWS", "200"))
BURN_IN_FACTOR = int(os.getenv("FIG23_BURN_IN_FACTOR", "20"))
SWAPS_PER_DRAW_FACTOR = int(os.getenv("FIG23_SWAPS_PER_DRAW_FACTOR", "8"))
PROGRESS_EVERY = int(os.getenv("FIG23_PROGRESS_EVERY", "25"))
SEED = 11

OUT_SUMMARY = Path("output/fig23_division_of_labor_regime_pairs_summary.csv")
OUT_DRAWS = Path("output/fig23_division_of_labor_regime_pairs_draws.csv")
OUT_META = Path("output/fig23_division_of_labor_regime_pairs_meta.json")
OUT_PDF = Path("figures/fig23_division_of_labor_regime_pairs.pdf")
OUT_PNG = Path("figures/fig23_division_of_labor_regime_pairs.png")


PAIR_ORDER = ["1-1", "1-2", "1-3", "2-2", "2-3", "3-3"]
PAIR_LABELS = {
    "1-1": "R1-R1",
    "1-2": "R1-R2",
    "1-3": "R1-R3",
    "2-2": "R2-R2",
    "2-3": "R2-R3",
    "3-3": "R3-R3",
}


def load_support_and_regimes() -> tuple[np.ndarray, np.ndarray, list[str]]:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()
    support = get_rca(counts).gt(RPA_THRESHOLD).astype(int).T
    support = support.loc[support.sum(axis=1) > 0]

    actor_summary = pd.read_csv(ACTOR_SUMMARY_FP)
    actor_regime = (
        actor_summary[["actor", "dominant_region"]]
        .rename(columns={"actor": "country"})
        .drop_duplicates("country")
        .set_index("country")
    )
    support = support.join(actor_regime, how="inner")
    support = support.dropna(subset=["dominant_region"]).copy()
    support["dominant_region"] = support["dominant_region"].astype(int)

    actor_names = support.index.tolist()
    regimes = support["dominant_region"].to_numpy(dtype=int)
    mat = support.drop(columns="dominant_region").to_numpy(dtype=np.uint8, copy=True)
    return mat, regimes, actor_names


def pair_category_stats(mat: np.ndarray, regimes: np.ndarray) -> dict[str, float]:
    n = mat.shape[0]
    stats: dict[str, list[float]] = {key: [] for key in PAIR_ORDER}
    for i in range(n):
        row_i = mat[i].astype(bool, copy=False)
        reg_i = int(regimes[i])
        for j in range(i + 1, n):
            row_j = mat[j].astype(bool, copy=False)
            reg_j = int(regimes[j])
            key = f"{min(reg_i, reg_j)}-{max(reg_i, reg_j)}"
            inter = np.logical_and(row_i, row_j).sum()
            union = np.logical_or(row_i, row_j).sum()
            if union == 0:
                continue
            stats[key].append(float(inter / union))
    return {
        key: (float(np.mean(vals)) if vals else np.nan)
        for key, vals in stats.items()
    }


def exact_swap_draw_matrices(
    mat: np.ndarray,
    rng: np.random.Generator,
    *,
    n_draws: int,
    burn_in_factor: int,
    swaps_per_draw_factor: int,
    progress_every: int,
) -> list[np.ndarray]:
    mat = mat.copy()
    n_edges = int(mat.sum())
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
                f"[fig23] warning: accepted {accepted}/{target_accepts} swaps after {attempts} attempts",
                flush=True,
            )

    print(
        f"[fig23] exact regime-pair null: burn-in {burn_in_swaps} accepted swaps, "
        f"then {n_draws} draws every {swaps_per_draw} accepted swaps",
        flush=True,
    )
    run_swaps(burn_in_swaps)

    draws: list[np.ndarray] = []
    for draw_idx in range(n_draws):
        run_swaps(swaps_per_draw)
        draws.append(mat.copy())
        if (draw_idx + 1) % max(progress_every, 1) == 0 or draw_idx + 1 == n_draws:
            print(f"[fig23] draw {draw_idx + 1}/{n_draws}", flush=True)
    return draws


def main() -> None:
    mat, regimes, actor_names = load_support_and_regimes()
    rng = np.random.default_rng(SEED)
    print(
        f"[fig23] actors={len(actor_names)}, topics={mat.shape[1]}, threshold={RPA_THRESHOLD:.2f}",
        flush=True,
    )

    observed_stats = pair_category_stats(mat, regimes)
    draw_mats = exact_swap_draw_matrices(
        mat,
        rng=rng,
        n_draws=N_DRAWS,
        burn_in_factor=BURN_IN_FACTOR,
        swaps_per_draw_factor=SWAPS_PER_DRAW_FACTOR,
        progress_every=PROGRESS_EVERY,
    )

    draw_rows: list[dict[str, object]] = []
    for draw_idx, draw_mat in enumerate(draw_mats):
        stats = pair_category_stats(draw_mat, regimes)
        for pair_key, value in stats.items():
            draw_rows.append(
                {
                    "draw": int(draw_idx),
                    "pair_key": pair_key,
                    "pair_label": PAIR_LABELS[pair_key],
                    "null_mean_jaccard": float(value),
                }
            )

    draws_df = pd.DataFrame(draw_rows)
    summary_rows: list[dict[str, object]] = []
    for pair_key in PAIR_ORDER:
        dfi = draws_df[draws_df["pair_key"] == pair_key]
        valid = dfi["null_mean_jaccard"].dropna().to_numpy(dtype=float)
        obs = float(observed_stats[pair_key])
        null_mean = float(np.mean(valid)) if valid.size else np.nan
        summary_rows.append(
            {
                "pair_key": pair_key,
                "pair_label": PAIR_LABELS[pair_key],
                "observed_mean_jaccard": obs,
                "null_mean_jaccard": null_mean,
                "null_q05": float(np.quantile(valid, 0.05)) if valid.size else np.nan,
                "null_q95": float(np.quantile(valid, 0.95)) if valid.size else np.nan,
                "gap_observed_minus_null": float(obs - null_mean) if np.isfinite(obs) and np.isfinite(null_mean) else np.nan,
                "n_draws": int(valid.size),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    draws_df.to_csv(OUT_DRAWS, index=False)
    OUT_META.write_text(
        json.dumps(
            {
                "rpa_threshold": RPA_THRESHOLD,
                "n_draws": N_DRAWS,
                "seed": SEED,
                "burn_in_factor": BURN_IN_FACTOR,
                "swaps_per_draw_factor": SWAPS_PER_DRAW_FACTOR,
                "actor_count": int(mat.shape[0]),
                "topic_count": int(mat.shape[1]),
                "pair_order": PAIR_ORDER,
                "note": "Aggregate-archive exact swap null by dominant-regime pair category.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    x = np.arange(len(PAIR_ORDER))
    colors = ["#6baed6", "#9ecae1", "#c6dbef", "#74c476", "#a1d99b", "#31a354"]

    fig, axs = uplt.subplots(ncols=2, refwidth=3.1, refaspect=0.95, share=False)
    axs.format(abc="[A]", grid=False)

    ax = axs[0]
    ax.fill_between(
        x,
        summary_df["null_q05"].to_numpy(dtype=float),
        summary_df["null_q95"].to_numpy(dtype=float),
        color="gray6",
        alpha=0.25,
        zorder=1,
    )
    ax.plot(
        x,
        summary_df["null_mean_jaccard"].to_numpy(dtype=float),
        color="gray8",
        lw=2.0,
        marker="o",
        ms=4,
        label="Exact null",
        zorder=2,
    )
    ax.plot(
        x,
        summary_df["observed_mean_jaccard"].to_numpy(dtype=float),
        color="black",
        lw=2.2,
        marker="o",
        ms=4,
        label="Observed",
        zorder=3,
    )
    ax.format(
        title="Observed overlap by regime pair",
        xlabel="Regime pair",
        ylabel="Mean pairwise Jaccard",
        xticks=x,
        xticklabels=[PAIR_LABELS[k] for k in PAIR_ORDER],
        ylim=(0, max(0.28, float(np.nanmax(summary_df[["observed_mean_jaccard", "null_q95"]].to_numpy(dtype=float))) * 1.12)),
    )
    ax.grid(alpha=0.18, color="black")

    ax = axs[1]
    gaps = summary_df["gap_observed_minus_null"].to_numpy(dtype=float)
    ax.bar(
        x,
        gaps,
        color=colors,
        edgecolor="black",
        lw=0.6,
    )
    ax.axhline(0, color="black", lw=1.0, alpha=0.6)
    ax.format(
        title="Observed minus exact null",
        xlabel="Regime pair",
        ylabel="Overlap gap",
        xticks=x,
        xticklabels=[PAIR_LABELS[k] for k in PAIR_ORDER],
    )
    ax.grid(alpha=0.18, color="black")

    fig.legend(loc="b", ncols=2, frame=False)
    fig.format(
        suptitle="Does division of labor appear only within selected regime pairs?"
    )
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
