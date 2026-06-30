from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy.spatial import cKDTree

from utils import get_rca, load_data, standardize_index_labels

DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
SOVEREIGN_PAIRS_FP = Path("output/step6_geo_portfolio_pairs.csv")
ACTOR_SUMMARY_FP = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")

OUT_DATA = Path("output/fig29_regime_triplet_coverage_anchored_triplets.csv")
OUT_SUMMARY = Path("output/fig29_regime_triplet_coverage_anchored_summary.csv")
OUT_META = Path("output/fig29_regime_triplet_coverage_anchored_meta.json")
OUT_PDF = Path("figures/fig29_regime_triplet_coverage_anchored.pdf")
OUT_PNG = Path("figures/fig29_regime_triplet_coverage_anchored.png")

RPA_THRESHOLD = 1.0
MIN_SPECIALIZED_TOPICS = 5
ANCHOR_THRESHOLD = float(os.environ.get("FIG29_ANCHOR_THRESHOLD", "0.7"))
MATCH_NEIGHBORS = 250
N_PERMUTATIONS = int(os.environ.get("FIG29_N_PERMUTATIONS", "2000"))
RNG_SEED = int(os.environ.get("FIG29_RNG_SEED", "42"))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else np.nan


def main() -> None:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()
    rpa = get_rca(counts)

    sovereign_pairs = pd.read_csv(SOVEREIGN_PAIRS_FP)
    sovereign_states = sorted(
        set(sovereign_pairs["actor_i"]).union(set(sovereign_pairs["actor_j"]))
    )

    actor_summary = pd.read_csv(ACTOR_SUMMARY_FP)
    actor_summary = actor_summary[actor_summary["actor"].isin(sovereign_states)].copy()
    actor_summary = actor_summary[
        actor_summary["support_raw_rca"] >= MIN_SPECIALIZED_TOPICS
    ].copy()
    actor_summary = actor_summary[
        actor_summary["dominant_region_share"] >= ANCHOR_THRESHOLD
    ].copy()
    keep_states = actor_summary["actor"].tolist()
    dominant_region = actor_summary.set_index("actor")["dominant_region"].to_dict()

    support = rpa.T.reindex(keep_states).fillna(0).gt(RPA_THRESHOLD).astype(int)

    rows: list[dict[str, object]] = []
    actor_index = {actor: idx for idx, actor in enumerate(keep_states)}
    for trio in combinations(keep_states, 3):
        mats = support.loc[list(trio)].to_numpy(dtype=bool)
        ks = mats.sum(axis=1)
        k_sorted = sorted(int(k) for k in ks)
        union_topics = int(np.logical_or.reduce(mats).sum())
        sum_support = int(ks.sum())
        coverage_efficiency = (
            float(union_topics / sum_support) if sum_support else np.nan
        )
        unique_topics = int((mats.sum(axis=0) == 1).sum())

        overlaps = [jaccard(mats[i], mats[j]) for i, j in combinations(range(3), 2)]
        mean_overlap = float(np.nanmean(overlaps))

        comp = "-".join(str(int(r)) for r in sorted(dominant_region[a] for a in trio))
        rows.append(
            {
                "actor_a": trio[0],
                "actor_b": trio[1],
                "actor_c": trio[2],
                "actor_a_idx": actor_index[trio[0]],
                "actor_b_idx": actor_index[trio[1]],
                "actor_c_idx": actor_index[trio[2]],
                "regime_triplet": comp,
                "k_small": k_sorted[0],
                "k_mid": k_sorted[1],
                "k_large": k_sorted[2],
                "union_topics": union_topics,
                "sum_support": sum_support,
                "coverage_efficiency": coverage_efficiency,
                "unique_topics": unique_topics,
                "mean_pairwise_overlap": mean_overlap,
            }
        )

    triplet_df = pd.DataFrame(rows)
    if triplet_df.empty:
        raise RuntimeError("No anchored triplets available at the current threshold.")

    size_features = triplet_df[["k_small", "k_mid", "k_large"]].to_numpy(dtype=float)
    size_features = (
        size_features - size_features.mean(axis=0, keepdims=True)
    ) / size_features.std(axis=0, keepdims=True)
    tree = cKDTree(size_features)
    k_query = min(MATCH_NEIGHBORS + 1, len(triplet_df))
    _, neighbor_idx = tree.query(size_features, k=k_query)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]
    matched_neighbors = neighbor_idx[:, 1:]

    cov_vals = triplet_df["coverage_efficiency"].to_numpy(dtype=float)
    ov_vals = triplet_df["mean_pairwise_overlap"].to_numpy(dtype=float)
    triplet_df["coverage_efficiency_null"] = [
        float(np.mean(cov_vals[idx])) for idx in matched_neighbors
    ]
    triplet_df["mean_pairwise_overlap_null"] = [
        float(np.mean(ov_vals[idx])) for idx in matched_neighbors
    ]
    triplet_df["coverage_efficiency_resid"] = (
        triplet_df["coverage_efficiency"] - triplet_df["coverage_efficiency_null"]
    )
    triplet_df["mean_pairwise_overlap_resid"] = (
        triplet_df["mean_pairwise_overlap"] - triplet_df["mean_pairwise_overlap_null"]
    )

    triplet_codes_obs = triplet_df["regime_triplet"].to_numpy(dtype=object)
    unique_codes = sorted(pd.unique(triplet_codes_obs).tolist())
    resid_cov = triplet_df["coverage_efficiency_resid"].to_numpy(dtype=float)
    resid_ov = triplet_df["mean_pairwise_overlap_resid"].to_numpy(dtype=float)
    triplet_actor_idx = triplet_df[
        ["actor_a_idx", "actor_b_idx", "actor_c_idx"]
    ].to_numpy(dtype=int)
    observed_regimes = (
        actor_summary.set_index("actor")
        .loc[keep_states, "dominant_region"]
        .to_numpy(dtype=int)
    )
    rng = np.random.default_rng(RNG_SEED)
    perm_cov = {code: [] for code in unique_codes}
    perm_ov = {code: [] for code in unique_codes}
    for _ in range(N_PERMUTATIONS):
        perm_reg = rng.permutation(observed_regimes)
        trip_regs = np.sort(perm_reg[triplet_actor_idx], axis=1)
        perm_codes = np.char.add(
            np.char.add(trip_regs[:, 0].astype(str), "-"),
            np.char.add(
                trip_regs[:, 1].astype(str),
                np.char.add("-", trip_regs[:, 2].astype(str)),
            ),
        )
        for code in unique_codes:
            mask = perm_codes == code
            if np.any(mask):
                perm_cov[code].append(float(np.mean(resid_cov[mask])))
                perm_ov[code].append(float(np.mean(resid_ov[mask])))
            else:
                perm_cov[code].append(np.nan)
                perm_ov[code].append(np.nan)

    summary_df = (
        triplet_df.groupby("regime_triplet")
        .agg(
            n_triplets=("coverage_efficiency", "size"),
            coverage_efficiency_mean=("coverage_efficiency", "mean"),
            coverage_efficiency_resid_mean=("coverage_efficiency_resid", "mean"),
            mean_pairwise_overlap_mean=("mean_pairwise_overlap", "mean"),
            mean_pairwise_overlap_resid_mean=("mean_pairwise_overlap_resid", "mean"),
            union_topics_mean=("union_topics", "mean"),
            unique_topics_mean=("unique_topics", "mean"),
        )
        .reset_index()
        .sort_values("coverage_efficiency_resid_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_df["coverage_perm_p_two_sided"] = np.nan
    summary_df["overlap_perm_p_two_sided"] = np.nan
    summary_df["coverage_perm_q05"] = np.nan
    summary_df["coverage_perm_q95"] = np.nan
    summary_df["overlap_perm_q05"] = np.nan
    summary_df["overlap_perm_q95"] = np.nan
    for idx, row in summary_df.iterrows():
        code = row["regime_triplet"]
        pc = np.array(perm_cov[code], dtype=float)
        po = np.array(perm_ov[code], dtype=float)
        pc = pc[np.isfinite(pc)]
        po = po[np.isfinite(po)]
        obs_c = float(row["coverage_efficiency_resid_mean"])
        obs_o = float(row["mean_pairwise_overlap_resid_mean"])
        summary_df.loc[idx, "coverage_perm_p_two_sided"] = float(
            (np.count_nonzero(np.abs(pc) >= abs(obs_c)) + 1) / (len(pc) + 1)
        )
        summary_df.loc[idx, "overlap_perm_p_two_sided"] = float(
            (np.count_nonzero(np.abs(po) >= abs(obs_o)) + 1) / (len(po) + 1)
        )
        summary_df.loc[idx, "coverage_perm_q05"] = float(np.quantile(pc, 0.05))
        summary_df.loc[idx, "coverage_perm_q95"] = float(np.quantile(pc, 0.95))
        summary_df.loc[idx, "overlap_perm_q05"] = float(np.quantile(po, 0.05))
        summary_df.loc[idx, "overlap_perm_q95"] = float(np.quantile(po, 0.95))

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    triplet_df.to_csv(OUT_DATA, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    OUT_META.write_text(
        json.dumps(
            {
                "rpa_threshold": RPA_THRESHOLD,
                "min_specialized_topics": MIN_SPECIALIZED_TOPICS,
                "anchor_threshold": ANCHOR_THRESHOLD,
                "n_states": int(len(keep_states)),
                "match_neighbors": MATCH_NEIGHBORS,
                "n_permutations": N_PERMUTATIONS,
                "rng_seed": RNG_SEED,
                "note": (
                    "Anchored-triplet rerun of fig28. Actors are restricted to dominant_region_share >= anchor_threshold. "
                    "Coverage efficiency is union_topics / total specialized-topic memberships across the triplet. "
                    "Residuals subtract the mean of size-matched triplets using nearest neighbors in sorted support-size space. "
                    "Permutation inference shuffles dominant-regime labels across the anchored actor set while preserving regime counts."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    order = summary_df["regime_triplet"].tolist()
    triplet_df["regime_triplet"] = pd.Categorical(
        triplet_df["regime_triplet"], categories=order, ordered=True
    )
    color_map = {
        "1-1-1": "#7f7f7f",
        "1-1-2": "#4c78a8",
        "1-1-3": "#2a9d8f",
        "1-2-2": "#3a86c8",
        "1-2-3": "#1b9e77",
        "1-3-3": "#55a630",
        "2-2-2": "#f28e2b",
        "2-2-3": "#c9a227",
        "2-3-3": "#e15759",
        "3-3-3": "#8f63b8",
    }
    colors = [color_map.get(comp, "gray6") for comp in order]
    xpos = np.arange(len(order))

    fig, axs = uplt.subplots(ncols=2, refwidth=3.4, refaspect=1.05, share=False)
    axs.format(abc="[A]", grid=True)

    def sig_label(p: float, resid: float) -> str:
        if p >= 0.05:
            return ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        else:
            stars = "*"
        direction = "†" if resid > 0 else "‡"
        return f"{stars}\n{direction}"

    for ax, metric, ylabel, pcol, rcol in [
        (
            axs[0],
            "coverage_efficiency",
            r"Distinct-topic share ($|\cup S_i| / \sum |S_i|$)",
            "coverage_perm_p_two_sided",
            "coverage_efficiency_resid_mean",
        ),
        (
            axs[1],
            "mean_pairwise_overlap",
            "Mean pairwise overlap",
            "overlap_perm_p_two_sided",
            "mean_pairwise_overlap_resid_mean",
        ),
    ]:
        ymax = -np.inf
        for xi, comp in enumerate(order):
            vals = triplet_df.loc[
                triplet_df["regime_triplet"] == comp, metric
            ].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            local_rng = np.random.default_rng(2000 + xi)
            sample_n = min(120, vals.size)
            sample = (
                vals
                if vals.size <= sample_n
                else local_rng.choice(vals, size=sample_n, replace=False)
            )
            jitter = local_rng.uniform(-0.18, 0.18, size=sample.size)
            ax.scatter(
                np.full(sample.size, xi) + jitter,
                sample,
                s=7,
                c=colors[xi],
                alpha=0.42,
                edgecolor="none",
                zorder=2,
            )
            q10, q25, q50, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.5, 0.75, 0.90])
            ymax = max(ymax, float(q90))
            ax.vlines(xi, q10, q90, color="black", lw=0.9, alpha=0.8, zorder=3)
            ax.vlines(xi, q25, q75, color="black", lw=3.0, zorder=4)
            ax.scatter([xi], [q50], s=20, c="black", zorder=5)
            row = summary_df.loc[summary_df["regime_triplet"] == comp].iloc[0]
            pval = float(row[pcol])
            resid = float(row[rcol])
            label = sig_label(pval, resid)
            if label:
                ax.text(
                    xi,
                    q90 + 0.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    zorder=6,
                )
        ax.format(
            xlabel="Anchored triplet composition",
            ylabel=ylabel,
            xticks=xpos,
            xticklabels=order,
            xrotation=90,
        )
        if np.isfinite(ymax):
            ymin, ymax_old = ax.get_ylim()
            ax.set_ylim(ymin, max(ymax_old, ymax + 0.08))

    axs[0].format(title="Distinct Topic Coverage")
    axs[1].format(title="Internal Redundancy")

    legend_handles = [
        uplt.pyplot.Line2D(
            [0],
            [0],
            color="none",
            ls="none",
            label="* p < 0.05, ** p < 0.01, *** p < 0.001",
        ),
        uplt.pyplot.Line2D(
            [0], [0], color="none", ls="none", label="† : above size-matched null"
        ),
        uplt.pyplot.Line2D(
            [0], [0], color="none", ls="none", label="‡ : below size-matched null"
        ),
    ]
    axs[1].legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lr",
        ncols=1,
        frame=False,
        fontsize=8,
        handlelength=0,
        handletextpad=0.25,
        borderpad=0.2,
        labelspacing=0.2,
        columnspacing=0.5,
        zorder=7,
    )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=220)


if __name__ == "__main__":
    main()
