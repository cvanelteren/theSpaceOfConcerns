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

OUT_DATA = Path("output/fig31_regime_pair_coverage_anchored_pairs.csv")
OUT_SUMMARY = Path("output/fig31_regime_pair_coverage_anchored_summary.csv")
OUT_META = Path("output/fig31_regime_pair_coverage_anchored_meta.json")
OUT_PDF = Path("figures/fig31_regime_pair_coverage_anchored.pdf")
OUT_PNG = Path("figures/fig31_regime_pair_coverage_anchored.png")

RPA_THRESHOLD = 1.0
MIN_SPECIALIZED_TOPICS = 5
ANCHOR_THRESHOLD = float(os.environ.get("FIG31_ANCHOR_THRESHOLD", "0.7"))
MATCH_NEIGHBORS = 250
N_PERMUTATIONS = int(os.environ.get("FIG31_N_PERMUTATIONS", "2000"))
RNG_SEED = int(os.environ.get("FIG31_RNG_SEED", "42"))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else np.nan


def marker_label(p: float, resid: float) -> str:
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
    for pair in combinations(keep_states, 2):
        mats = support.loc[list(pair)].to_numpy(dtype=bool)
        ks = mats.sum(axis=1)
        k_sorted = sorted(int(k) for k in ks)
        union_topics = int(np.logical_or.reduce(mats).sum())
        sum_support = int(ks.sum())
        distinct_share = float(union_topics / sum_support) if sum_support else np.nan
        overlap = jaccard(mats[0], mats[1])
        comp = "-".join(str(int(r)) for r in sorted(dominant_region[a] for a in pair))
        rows.append(
            {
                "actor_a": pair[0],
                "actor_b": pair[1],
                "actor_a_idx": actor_index[pair[0]],
                "actor_b_idx": actor_index[pair[1]],
                "regime_pair": comp,
                "k_small": k_sorted[0],
                "k_large": k_sorted[1],
                "union_topics": union_topics,
                "sum_support": sum_support,
                "distinct_share": distinct_share,
                "pairwise_overlap": overlap,
            }
        )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise RuntimeError("No anchored pairs available at the current threshold.")

    size_features = pair_df[["k_small", "k_large"]].to_numpy(dtype=float)
    size_features = (
        size_features - size_features.mean(axis=0, keepdims=True)
    ) / size_features.std(axis=0, keepdims=True)
    tree = cKDTree(size_features)
    k_query = min(MATCH_NEIGHBORS + 1, len(pair_df))
    _, neighbor_idx = tree.query(size_features, k=k_query)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]
    matched_neighbors = neighbor_idx[:, 1:]

    distinct_vals = pair_df["distinct_share"].to_numpy(dtype=float)
    overlap_vals = pair_df["pairwise_overlap"].to_numpy(dtype=float)
    pair_df["distinct_share_null"] = [
        float(np.mean(distinct_vals[idx])) for idx in matched_neighbors
    ]
    pair_df["pairwise_overlap_null"] = [
        float(np.mean(overlap_vals[idx])) for idx in matched_neighbors
    ]
    pair_df["distinct_share_resid"] = (
        pair_df["distinct_share"] - pair_df["distinct_share_null"]
    )
    pair_df["pairwise_overlap_resid"] = (
        pair_df["pairwise_overlap"] - pair_df["pairwise_overlap_null"]
    )

    pair_codes_obs = pair_df["regime_pair"].to_numpy(dtype=object)
    unique_codes = sorted(pd.unique(pair_codes_obs).tolist())
    resid_distinct = pair_df["distinct_share_resid"].to_numpy(dtype=float)
    resid_overlap = pair_df["pairwise_overlap_resid"].to_numpy(dtype=float)
    pair_actor_idx = pair_df[["actor_a_idx", "actor_b_idx"]].to_numpy(dtype=int)
    observed_regimes = (
        actor_summary.set_index("actor")
        .loc[keep_states, "dominant_region"]
        .to_numpy(dtype=int)
    )
    rng = np.random.default_rng(RNG_SEED)
    perm_distinct = {code: [] for code in unique_codes}
    perm_overlap = {code: [] for code in unique_codes}
    for _ in range(N_PERMUTATIONS):
        perm_reg = rng.permutation(observed_regimes)
        pair_regs = np.sort(perm_reg[pair_actor_idx], axis=1)
        perm_codes = np.char.add(
            pair_regs[:, 0].astype(str), np.char.add("-", pair_regs[:, 1].astype(str))
        )
        for code in unique_codes:
            mask = perm_codes == code
            if np.any(mask):
                perm_distinct[code].append(float(np.mean(resid_distinct[mask])))
                perm_overlap[code].append(float(np.mean(resid_overlap[mask])))
            else:
                perm_distinct[code].append(np.nan)
                perm_overlap[code].append(np.nan)

    summary_df = (
        pair_df.groupby("regime_pair")
        .agg(
            n_pairs=("distinct_share", "size"),
            distinct_share_mean=("distinct_share", "mean"),
            distinct_share_resid_mean=("distinct_share_resid", "mean"),
            pairwise_overlap_mean=("pairwise_overlap", "mean"),
            pairwise_overlap_resid_mean=("pairwise_overlap_resid", "mean"),
            union_topics_mean=("union_topics", "mean"),
        )
        .reset_index()
        .sort_values("distinct_share_resid_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_df["distinct_perm_p_two_sided"] = np.nan
    summary_df["overlap_perm_p_two_sided"] = np.nan
    for idx, row in summary_df.iterrows():
        code = row["regime_pair"]
        pdist = np.array(perm_distinct[code], dtype=float)
        pov = np.array(perm_overlap[code], dtype=float)
        pdist = pdist[np.isfinite(pdist)]
        pov = pov[np.isfinite(pov)]
        obs_d = float(row["distinct_share_resid_mean"])
        obs_o = float(row["pairwise_overlap_resid_mean"])
        summary_df.loc[idx, "distinct_perm_p_two_sided"] = float(
            (np.count_nonzero(np.abs(pdist) >= abs(obs_d)) + 1) / (len(pdist) + 1)
        )
        summary_df.loc[idx, "overlap_perm_p_two_sided"] = float(
            (np.count_nonzero(np.abs(pov) >= abs(obs_o)) + 1) / (len(pov) + 1)
        )

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(OUT_DATA, index=False)
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
                    "Anchored-pair analogue of fig29. Actors are restricted to dominant_region_share >= anchor_threshold. "
                    "Distinct-topic share is union_topics / total specialized-topic claims across the pair. "
                    "Residuals subtract the mean of size-matched pairs using nearest neighbors in sorted support-size space. "
                    "Permutation inference shuffles dominant-regime labels across the anchored actor set while preserving regime counts."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    order = summary_df["regime_pair"].tolist()
    pair_df["regime_pair"] = pd.Categorical(
        pair_df["regime_pair"], categories=order, ordered=True
    )
    color_map = {
        "1-1": "#4c78a8",
        "1-2": "#1b9e77",
        "1-3": "#2a9d8f",
        "2-2": "#f28e2b",
        "2-3": "#e15759",
        "3-3": "#8f63b8",
    }
    colors = [color_map.get(comp, "gray6") for comp in order]
    xpos = np.arange(len(order))

    fig, axs = uplt.subplots(ncols=2, refwidth=3.25, refaspect=1.05, share=False)
    axs.format(abc="[A]", grid=True)

    panel_specs = [
        (
            axs[0],
            "distinct_share",
            r"Distinct-topic share ($|\cup S_i| / \sum |S_i|$)",
            "distinct_perm_p_two_sided",
            "distinct_share_resid_mean",
        ),
        (
            axs[1],
            "pairwise_overlap",
            "Pairwise overlap",
            "overlap_perm_p_two_sided",
            "pairwise_overlap_resid_mean",
        ),
    ]
    for ax, metric, ylabel, pcol, rcol in panel_specs:
        ymax = -np.inf
        for xi, comp in enumerate(order):
            vals = pair_df.loc[pair_df["regime_pair"] == comp, metric].to_numpy(
                dtype=float
            )
            if vals.size == 0:
                continue
            local_rng = np.random.default_rng(3000 + xi)
            sample_n = min(150, vals.size)
            sample = (
                vals
                if vals.size <= sample_n
                else local_rng.choice(vals, size=sample_n, replace=False)
            )
            jitter = local_rng.uniform(-0.18, 0.18, size=sample.size)
            ax.scatter(
                np.full(sample.size, xi) + jitter,
                sample,
                s=8,
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
            row = summary_df.loc[summary_df["regime_pair"] == comp].iloc[0]
            label = marker_label(float(row[pcol]), float(row[rcol]))
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
            xlabel="Anchored pair composition",
            ylabel=ylabel,
            xticks=xpos,
            xticklabels=order,
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
        loc="ll",
        ncols=1,
        frame=False,
        fontsize=8,
        handlelength=0,
        handletextpad=0.25,
        borderpad=0.2,
        labelspacing=0.2,
        columnspacing=0.5,
        bbox_to_anchor=(1.0, 1.08),
    )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=220)


if __name__ == "__main__":
    main()
