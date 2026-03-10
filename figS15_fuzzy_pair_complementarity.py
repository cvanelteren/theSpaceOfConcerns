"""Supplementary Figure S15. Tests complementary versus redundant fuzzy regime pairings among strongly anchored actors. Shows that complementarity concentrates on the 1–3 boundary and adjacent bridge pairings."""

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

OUT_DATA = Path("output/fig34_fuzzy_pair_coverage_anchored_pairs.csv")
OUT_SUMMARY = Path("output/fig34_fuzzy_pair_coverage_anchored_summary.csv")
OUT_META = Path("output/fig34_fuzzy_pair_coverage_anchored_meta.json")
OUT_PDF = Path("figures/figS15_fuzzy_pair_complementarity.pdf")
OUT_PNG = Path("figures/figS15_fuzzy_pair_complementarity.png")

RPA_THRESHOLD = 1.0
MIN_SPECIALIZED_TOPICS = 5
ANCHOR_THRESHOLD = float(os.environ.get("FIG34_ANCHOR_THRESHOLD", "0.7"))
FUZZY_TAU = float(os.environ.get("FIG34_FUZZY_TAU", "0.25"))
MATCH_NEIGHBORS = 250
N_PERMUTATIONS = int(os.environ.get("FIG34_N_PERMUTATIONS", "2000"))
RNG_SEED = int(os.environ.get("FIG34_RNG_SEED", "42"))
MIN_PLOT_PAIRS = int(os.environ.get("FIG34_MIN_PLOT_PAIRS", "4"))

COLOR_MAP = {
    "1-1+2": "#4c78a8",
    "1+2-1+2": "#2a9d8f",
    "1+2-2": "#6f9f44",
    "1+2-2+3": "#8ec07c",
    "1+2-3": "#b07aa1",
    "1-2": "#72b7b2",
    "1-2+3": "#d98c3f",
    "1-3": "#e15759",
    "2-2": "#f28e2b",
    "2-2+3": "#ffbe7d",
    "2-3": "#af7aa1",
    "2+3-3": "#c85252",
    "3-3": "#8f63b8",
}


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


def assign_fuzzy_class(row: pd.Series, tau: float = FUZZY_TAU) -> str:
    active = [str(i) for i in (1, 2, 3) if row[f"region_{i}_share"] >= tau]
    if not active:
        active = [str(int(row["dominant_region"]))]
    return "+".join(active)


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
    actor_summary["fuzzy_class"] = actor_summary.apply(assign_fuzzy_class, axis=1)
    keep_states = actor_summary["actor"].tolist()
    fuzzy_class = actor_summary.set_index("actor")["fuzzy_class"].to_dict()

    support = rpa.T.reindex(keep_states).fillna(0).gt(RPA_THRESHOLD).astype(int)

    rows = []
    actor_index = {actor: idx for idx, actor in enumerate(keep_states)}
    for pair in combinations(keep_states, 2):
        mats = support.loc[list(pair)].to_numpy(dtype=bool)
        ks = mats.sum(axis=1)
        k_sorted = sorted(int(k) for k in ks)
        union_topics = int(np.logical_or.reduce(mats).sum())
        sum_support = int(ks.sum())
        distinct_share = float(union_topics / sum_support) if sum_support else np.nan
        overlap = jaccard(mats[0], mats[1])
        comp = "-".join(sorted([fuzzy_class[pair[0]], fuzzy_class[pair[1]]]))
        rows.append(
            {
                "actor_a": pair[0],
                "actor_b": pair[1],
                "actor_a_idx": actor_index[pair[0]],
                "actor_b_idx": actor_index[pair[1]],
                "fuzzy_pair": comp,
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
        raise RuntimeError(
            "No anchored fuzzy pairs available at the current threshold."
        )

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

    pair_codes_obs = pair_df["fuzzy_pair"].to_numpy(dtype=object)
    unique_codes = sorted(pd.unique(pair_codes_obs).tolist())
    resid_distinct = pair_df["distinct_share_resid"].to_numpy(dtype=float)
    resid_overlap = pair_df["pairwise_overlap_resid"].to_numpy(dtype=float)
    pair_actor_idx = pair_df[["actor_a_idx", "actor_b_idx"]].to_numpy(dtype=int)
    observed_labels = (
        actor_summary.set_index("actor")
        .loc[keep_states, "fuzzy_class"]
        .to_numpy(dtype=object)
    )
    rng = np.random.default_rng(RNG_SEED)
    perm_distinct = {code: [] for code in unique_codes}
    perm_overlap = {code: [] for code in unique_codes}
    for _ in range(N_PERMUTATIONS):
        perm_labels = rng.permutation(observed_labels)
        perm_codes = np.array(
            [
                "-".join(sorted((perm_labels[i], perm_labels[j])))
                for i, j in pair_actor_idx
            ],
            dtype=object,
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
        pair_df.groupby("fuzzy_pair")
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
        code = row["fuzzy_pair"]
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
                "fuzzy_tau": FUZZY_TAU,
                "n_states": int(len(keep_states)),
                "match_neighbors": MATCH_NEIGHBORS,
                "n_permutations": N_PERMUTATIONS,
                "rng_seed": RNG_SEED,
                "min_plot_pairs": MIN_PLOT_PAIRS,
                "note": "Anchored-pair analogue of fig31 using fuzzy regime classes from fig33. Actors are restricted to dominant_region_share >= anchor_threshold; fuzzy class membership requires region share >= fuzzy_tau. Residuals subtract the mean of size-matched pairs. Permutation inference shuffles fuzzy-class labels across the anchored actor set while preserving class counts.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_summary = summary_df[summary_df["n_pairs"] >= MIN_PLOT_PAIRS].copy()
    order = plot_summary["fuzzy_pair"].tolist()
    plot_pair_df = pair_df[pair_df["fuzzy_pair"].isin(order)].copy()
    plot_pair_df["fuzzy_pair"] = pd.Categorical(
        plot_pair_df["fuzzy_pair"], categories=order, ordered=True
    )
    colors = [COLOR_MAP.get(comp, "gray6") for comp in order]
    xpos = np.arange(len(order))

    fig, axs = uplt.subplots(
        ncols=2,
        refwidth=3.75,
        refaspect=1.05,
        share=False,
    )
    axs.format(abc="[A]", grid=True)

    panel_specs = [
        (
            axs[0],
            "distinct_share",
            r"Distinct-topic share ($|\cup S_i| / \sum |S_i|$)",
            "distinct_perm_p_two_sided",
            "distinct_share_resid_mean",
            "Fuzzy-pair complementarity",
        ),
        (
            axs[1],
            "pairwise_overlap",
            "Pairwise overlap",
            "overlap_perm_p_two_sided",
            "pairwise_overlap_resid_mean",
            "Fuzzy-pair redundancy",
        ),
    ]
    for ax, metric, ylabel, pcol, rcol, title in panel_specs:
        ymax = -np.inf
        for xi, comp in enumerate(order):
            vals = plot_pair_df.loc[
                plot_pair_df["fuzzy_pair"] == comp, metric
            ].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            local_rng = np.random.default_rng(5000 + xi)
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
            ax.hlines(q50, xi - 0.18, xi + 0.18, color="black", lw=1.2, zorder=5)
            row = plot_summary.loc[plot_summary["fuzzy_pair"] == comp].iloc[0]
            label = marker_label(float(row[pcol]), float(row[rcol]))
            if label:
                ytext = min(
                    float(q90) + 0.03, 0.98 if metric == "distinct_share" else 0.98
                )
                ax.text(xi, ytext, label, ha="center", va="bottom", fontsize=9)
        if metric == "distinct_share":
            ylim = (0.5, min(1.02, ymax + 0.24))
        else:
            ylim = (0.0, min(0.98, ymax + 0.20))
        ax.format(
            title=title,
            ylabel=ylabel,
            xlabel="Fuzzy pair class",
            xticks=xpos,
            xticklabels=order,
            ylim=ylim,
        )
    axs[0].yaxis.labelpad = 10

    import matplotlib.lines as mlines

    legend_handles = [
        mlines.Line2D([], [], linestyle="none", label="* p < 0.05"),
        mlines.Line2D([], [], linestyle="none", label="** p < 0.01"),
        mlines.Line2D([], [], linestyle="none", label="*** p < 0.001"),
        mlines.Line2D([], [], linestyle="none", label="† above size-matched null"),
        mlines.Line2D([], [], linestyle="none", label="‡ below size-matched null"),
    ]
    axs[1].legend(
        handles=legend_handles,
        loc="lr",
        ncols=1,
        frame=False,
        title="Permutation significance",
    )

    fig.save(OUT_PDF)
    fig.save(OUT_PNG, transparent=False)


if __name__ == "__main__":
    main()
