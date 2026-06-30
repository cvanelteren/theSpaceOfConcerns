from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import compute_product_space, get_rca, load_data, standardize_index_labels

DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
SOVEREIGN_PAIRS_FP = Path("output/step6_geo_portfolio_pairs.csv")
ACTOR_SUMMARY_FP = Path("output/fig45_portfolio_space_ridgelines_actor_summary.csv")

OUT_DATA = Path("output/fig27_pairwise_portfolio_complementarity_pairs.csv")
OUT_SUMMARY = Path("output/fig27_pairwise_portfolio_complementarity_summary.json")
OUT_PDF = Path("figures/fig27_pairwise_portfolio_complementarity.pdf")
OUT_PNG = Path("figures/fig27_pairwise_portfolio_complementarity.png")

RPA_THRESHOLD = 1.0
MIN_SPECIALIZED_TOPICS = 5
N_LABELS = 10


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else np.nan


def exclusive_support_proximity(
    support_a: np.ndarray,
    support_b: np.ndarray,
    phi: np.ndarray,
) -> float:
    a_only = np.flatnonzero(np.logical_and(support_a, ~support_b))
    b_only = np.flatnonzero(np.logical_and(support_b, ~support_a))
    if len(a_only) == 0 or len(b_only) == 0:
        return np.nan
    sub = phi[np.ix_(a_only, b_only)]
    return float(0.5 * (sub.max(axis=1).mean() + sub.max(axis=0).mean()))


def main() -> None:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()

    rpa = get_rca(counts)
    phi = compute_product_space(rpa).to_numpy(dtype=float).copy()

    sovereign_pairs = pd.read_csv(SOVEREIGN_PAIRS_FP)
    sovereign_states = sorted(
        set(sovereign_pairs["actor_i"]).union(set(sovereign_pairs["actor_j"]))
    )

    actor_summary = pd.read_csv(ACTOR_SUMMARY_FP)
    actor_summary = actor_summary[actor_summary["actor"].isin(sovereign_states)].copy()
    dominant_region = actor_summary.set_index("actor")["dominant_region"].to_dict()

    support = (
        rpa.T.reindex(sovereign_states).fillna(0).gt(RPA_THRESHOLD).astype(int)
    )
    specialized_topics = support.sum(axis=1)
    keep_states = specialized_topics[specialized_topics >= MIN_SPECIALIZED_TOPICS].index.tolist()
    support = support.reindex(keep_states)

    rows: list[dict[str, object]] = []
    for i, actor_i in enumerate(keep_states):
        sup_i = support.loc[actor_i].to_numpy(dtype=bool)
        for j in range(i + 1, len(keep_states)):
            actor_j = keep_states[j]
            sup_j = support.loc[actor_j].to_numpy(dtype=bool)
            overlap = jaccard(sup_i, sup_j)
            proximity = exclusive_support_proximity(sup_i, sup_j, phi)
            if not np.isfinite(overlap) or not np.isfinite(proximity):
                continue
            rows.append(
                {
                    "actor_i": actor_i,
                    "actor_j": actor_j,
                    "overlap_jaccard_rpa": overlap,
                    "exclusive_phi_proximity": proximity,
                    "complementarity_score": float(proximity - overlap),
                    "k_i": int(support.loc[actor_i].sum()),
                    "k_j": int(support.loc[actor_j].sum()),
                    "same_regime": int(dominant_region.get(actor_i, -1) == dominant_region.get(actor_j, -2)),
                    "regime_pair": "-".join(
                        sorted(
                            [
                                str(int(dominant_region.get(actor_i, -1))),
                                str(int(dominant_region.get(actor_j, -1))),
                            ]
                        )
                    ),
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise RuntimeError("No eligible sovereign-state pairs for complementarity plot.")

    x_med = float(pair_df["overlap_jaccard_rpa"].median())
    y_med = float(pair_df["exclusive_phi_proximity"].median())

    def classify(row: pd.Series) -> str:
        if row["overlap_jaccard_rpa"] < x_med and row["exclusive_phi_proximity"] >= y_med:
            return "Complementary"
        if row["overlap_jaccard_rpa"] >= x_med and row["exclusive_phi_proximity"] >= y_med:
            return "Aligned"
        return "Separate / weak"

    pair_df["pair_type"] = pair_df.apply(classify, axis=1)
    pair_df["pair_label"] = pair_df["actor_i"] + " - " + pair_df["actor_j"]
    pair_df.to_csv(OUT_DATA, index=False)

    summary = {
        "rpa_threshold": RPA_THRESHOLD,
        "min_specialized_topics": MIN_SPECIALIZED_TOPICS,
        "n_states": int(len(keep_states)),
        "n_pairs": int(len(pair_df)),
        "median_overlap_jaccard_rpa": x_med,
        "median_exclusive_phi_proximity": y_med,
        "share_complementary": float((pair_df["pair_type"] == "Complementary").mean()),
        "share_aligned": float((pair_df["pair_type"] == "Aligned").mean()),
        "share_same_regime": float(pair_df["same_regime"].mean()),
        "corr_overlap_vs_exclusive_phi": float(
            pair_df[["overlap_jaccard_rpa", "exclusive_phi_proximity"]].corr().iloc[0, 1]
        ),
        "top_complementary_pairs": pair_df.loc[
            pair_df["pair_type"] == "Complementary",
            ["pair_label", "overlap_jaccard_rpa", "exclusive_phi_proximity", "complementarity_score"],
        ]
        .sort_values(
            ["exclusive_phi_proximity", "complementarity_score"],
            ascending=[False, False],
        )
        .head(N_LABELS)
        .to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = {
        "Complementary": "teal7",
        "Aligned": "orange7",
        "Separate / weak": "gray6",
    }

    fig, ax = uplt.subplots(refwidth=5.8, refaspect=0.78)
    ax.format(abc="[A]", xlabel=r"Overlap (Jaccard on $RPA>1$ topics)", ylabel=r"Exclusive support proximity in $\phi$")

    for label, group in pair_df.groupby("pair_type", sort=False):
        ax.scatter(
            group["overlap_jaccard_rpa"],
            group["exclusive_phi_proximity"],
            s=18,
            alpha=0.72 if label != "Separate / weak" else 0.42,
            c=colors[label],
            edgecolor="none",
            label=label,
            zorder=3 if label != "Separate / weak" else 2,
        )

    ax.axvline(x_med, color="black", lw=1.0, ls="--", alpha=0.65, zorder=1)
    ax.axhline(y_med, color="black", lw=1.0, ls="--", alpha=0.65, zorder=1)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), y_med + 0.01 * (ylim[1] - ylim[0]), "Complementary", color="teal8", fontsize=9, weight="bold")
    ax.text(x_med + 0.01 * (xlim[1] - xlim[0]), y_med + 0.01 * (ylim[1] - ylim[0]), "Aligned", color="orange8", fontsize=9, weight="bold")
    ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), ylim[0] + 0.02 * (ylim[1] - ylim[0]), "Separate / weak", color="gray8", fontsize=9, weight="bold")

    top_labels = (
        pair_df.loc[pair_df["pair_type"] == "Complementary"]
        .sort_values(["exclusive_phi_proximity", "complementarity_score"], ascending=[False, False])
        .head(N_LABELS)
        .copy()
    )
    for idx, (_, row) in enumerate(top_labels.iterrows()):
        dx = 0.006 + (idx % 2) * 0.002
        dy = 0.003 + (idx % 3) * 0.004
        ax.text(
            row["overlap_jaccard_rpa"] + dx,
            row["exclusive_phi_proximity"] + dy,
            row["pair_label"],
            fontsize=7,
            color="black",
            alpha=0.92,
        )

    ax.legend(loc="lower right", ncols=1, frame=False)
    ax.grid(alpha=0.15, color="black")
    fig.format(suptitle="Low overlap does not imply distance: some ATS state pairs are complementary in the concern space")

    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
