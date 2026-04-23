"""Supplementary Figure S01. Validates why a three-regime summary is the minimum adequate partition of the concern space. Supports the main regime construction before actor-position and transition results are interpreted."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy.stats import spearmanr

from utils import compute_product_space, get_rca, load_data, standardize_index_labels

DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
TOPIC_FP = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
OUT_CSV = Path("output/fig21_regime_portfolio_validation_summary.csv")
OUT_ASSIGNMENTS_CSV = Path("output/fig21_regime_topic_assignments_by_k.csv")
OUT_JSON = Path("output/fig21_regime_portfolio_validation_meta.json")
OUT_PDF = Path("figures/figS01_regime_k_validation.pdf")
OUT_PNG = Path("figures/figS01_regime_k_validation.png")

RCA_THRESHOLD = 1.0
GAUSS_SIGMA = 1.15
GAUSS_RADIUS = 3
K_MIN = 2
K_MAX = 6
N_ITER = 50


def _prepare_counts() -> pd.DataFrame:
    data_paths = [
        DATA_FP,
        Path("../antarctic-database-go/data/processed/document-summary.parquet"),
    ]
    for path in data_paths:
        if path.exists():
            counts, _, _, _ = load_data(path)
            break
    else:
        raise FileNotFoundError(
            "Could not find document-summary.parquet in expected data paths."
        )
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()
    return counts


def _gaussian_smooth(values: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _select_region_centers(
    x_positions: np.ndarray,
    aggregate_signal: np.ndarray,
    n_regions: int,
    n_iter: int = N_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    smooth = _gaussian_smooth(aggregate_signal, sigma=GAUSS_SIGMA, radius=GAUSS_RADIUS)
    if smooth.size == 0:
        return np.array([], dtype=int), smooth

    weights = np.asarray(smooth, dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)

    lo = float(np.min(x_positions))
    hi = float(np.max(x_positions))
    centers = np.linspace(lo, hi, n_regions)
    x = np.asarray(x_positions, dtype=float)

    for _ in range(n_iter):
        assign = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for idx in range(n_regions):
            mask = assign == idx
            if not np.any(mask):
                continue
            new_centers[idx] = float(np.average(x[mask], weights=weights[mask]))
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    anchor_idx: list[int] = []
    for center in centers:
        anchor_idx.append(int(np.argmin(np.abs(x - center))))
    anchor_idx = sorted(dict.fromkeys(anchor_idx), key=lambda idx: x[idx])
    return np.asarray(anchor_idx, dtype=int), smooth


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    return unit @ unit.T


def main() -> None:
    topic_df = pd.read_csv(TOPIC_FP).sort_values("topic_order").reset_index(drop=True)
    ordered_topics = topic_df["topic"].tolist()
    x_plot = topic_df["x_plot"].to_numpy(dtype=float)

    counts = _prepare_counts()
    rca = get_rca(counts).reindex(index=ordered_topics)
    excess = np.clip(rca.to_numpy(dtype=float) - RCA_THRESHOLD, 0.0, None)

    aggregate_signal = excess.sum(axis=1)
    active_actor_mask = excess.sum(axis=0) > 0
    actor_names = rca.columns[active_actor_mask].tolist()
    full_actor_profiles = excess[:, active_actor_mask].T
    full_actor_profiles = full_actor_profiles / full_actor_profiles.sum(
        axis=1, keepdims=True
    )
    full_similarity = _cosine_similarity_matrix(full_actor_profiles)
    tri_upper = np.triu_indices(full_similarity.shape[0], k=1)

    rows: list[dict[str, float | int]] = []
    assignment_rows: list[dict[str, float | int | str]] = []
    centers_meta: dict[str, list[float]] = {}
    anchors_meta: dict[str, list[str]] = {}
    for k in range(K_MIN, K_MAX + 1):
        anchor_idx, smooth = _select_region_centers(
            x_positions=x_plot,
            aggregate_signal=aggregate_signal,
            n_regions=k,
        )
        anchor_x = x_plot[anchor_idx]
        boundaries = (
            0.5 * (anchor_x[:-1] + anchor_x[1:]) if len(anchor_x) > 1 else np.array([])
        )
        topic_region_idx = np.digitize(x_plot, boundaries).astype(int)
        for topic_idx, topic in enumerate(ordered_topics):
            region_idx = int(topic_region_idx[topic_idx])
            anchor_topic = ordered_topics[anchor_idx[region_idx]]
            assignment_rows.append(
                {
                    "k": int(k),
                    "topic_order": int(topic_idx),
                    "topic": topic,
                    "x_plot": float(x_plot[topic_idx]),
                    "region": int(region_idx + 1),
                    "region_anchor": anchor_topic,
                    "region_anchor_x": float(x_plot[anchor_idx[region_idx]]),
                }
            )

        region_centers = x_plot[anchor_idx]
        weighted_inertia = float(
            np.sum(smooth * (x_plot - region_centers[topic_region_idx]) ** 2)
        )

        regime_weights = np.zeros((len(actor_names), len(anchor_idx)), dtype=float)
        for ridx in range(len(anchor_idx)):
            region_mass = excess[topic_region_idx == ridx][:, active_actor_mask].sum(
                axis=0
            )
            regime_weights[:, ridx] = region_mass
        regime_weights = regime_weights / regime_weights.sum(axis=1, keepdims=True)

        regime_similarity = _cosine_similarity_matrix(regime_weights)
        portfolio_rho = float(
            spearmanr(
                full_similarity[tri_upper], regime_similarity[tri_upper]
            ).statistic
        )

        dominant_share = np.max(regime_weights, axis=1)
        topic_counts = np.bincount(topic_region_idx, minlength=len(anchor_idx)).astype(
            int
        )
        signal_shares = (
            np.bincount(
                topic_region_idx, weights=aggregate_signal, minlength=len(anchor_idx)
            )
            / aggregate_signal.sum()
        )

        rows.append(
            {
                "k": k,
                "weighted_inertia_actual_rule": weighted_inertia,
                "portfolio_similarity_spearman": portfolio_rho,
                "mean_dominant_regime_share": float(np.mean(dominant_share)),
                "median_dominant_regime_share": float(np.median(dominant_share)),
                "share_strongly_anchored_ge_0.7": float(np.mean(dominant_share >= 0.7)),
                "smallest_region_topic_count": int(topic_counts.min()),
                "smallest_region_signal_share": float(signal_shares.min()),
                "n_regions_realized": int(len(anchor_idx)),
            }
        )
        centers_meta[str(k)] = [float(v) for v in anchor_x.tolist()]
        anchors_meta[str(k)] = [ordered_topics[idx] for idx in anchor_idx]

    summary = pd.DataFrame(rows)
    summary["inertia_gain"] = (
        summary["weighted_inertia_actual_rule"].shift(1)
        - summary["weighted_inertia_actual_rule"]
    )
    summary["portfolio_rho_gain"] = summary["portfolio_similarity_spearman"].diff()
    summary["dominant_share_drop"] = -summary["mean_dominant_regime_share"].diff()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    pd.DataFrame(assignment_rows).to_csv(OUT_ASSIGNMENTS_CSV, index=False)
    OUT_JSON.write_text(
        json.dumps(
            {
                "k_min": K_MIN,
                "k_max": K_MAX,
                "anchors_by_k": anchors_meta,
                "centers_by_k": centers_meta,
                "topic_assignments_csv": str(OUT_ASSIGNMENTS_CSV),
                "note": (
                    "Validation uses the same snapped weighted-center regime rule as the "
                    "main paper, then asks how well the resulting regime shares preserve "
                    "pairwise actor similarity in full excess-specialization portfolios."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ks = summary["k"].to_numpy(dtype=int)
    fig, axs = uplt.subplots(
        ncols=3,
        refwidth=2.7,
        refaspect=1.0,
        share=False,
    )
    axs.format(abc="[A]", grid=False)

    ax = axs[0]
    ax.plot(
        ks, summary["weighted_inertia_actual_rule"], marker="o", color="blue7", lw=2.0
    )
    ax.axvline(3, color="black", lw=1.0, ls="--", alpha=0.45)
    ax.scatter(
        [3],
        summary.loc[summary["k"] == 3, "weighted_inertia_actual_rule"],
        color="black",
        s=28,
        zorder=4,
    )
    ax.format(
        xlabel="Regimes $k$",
        ylabel="Weighted within-region inertia",
        xticks=ks,
    )
    ax.grid(alpha=0.18, color="black")

    ax = axs[1]
    ax.plot(
        ks, summary["portfolio_similarity_spearman"], marker="o", color="green7", lw=2.0
    )
    ax.axvline(3, color="black", lw=1.0, ls="--", alpha=0.45)
    ax.scatter(
        [3],
        summary.loc[summary["k"] == 3, "portfolio_similarity_spearman"],
        color="black",
        s=28,
        zorder=4,
    )
    ax.format(
        xlabel="Regimes $k$",
        ylabel="Portfolio-similarity preservation\n(Spearman $\\rho$)",
        xticks=ks,
        ylim=(0.0, max(0.5, summary["portfolio_similarity_spearman"].max() + 0.03)),
    )
    ax.grid(alpha=0.18, color="black")

    ax = axs[2]
    ax.plot(
        ks,
        summary["mean_dominant_regime_share"],
        marker="o",
        color="orange7",
        lw=2.0,
        label="Mean dominant share",
    )
    ax.bar(
        ks,
        summary["smallest_region_topic_count"]
        / summary["smallest_region_topic_count"].max(),
        color="gray6",
        alpha=0.35,
        width=0.65,
        label="Smallest regime size (scaled)",
    )
    ax.axvline(3, color="black", lw=1.0, ls="--", alpha=0.45)
    ax.format(
        xlabel="Regimes $k$",
        ylabel="Anchoring / fragmentation",
        xticks=ks,
        ylim=(0.0, 1.05),
    )
    ax.legend(loc="lower left", ncols=1, fontsize=7, frame=False)
    ax.grid(alpha=0.18, color="black")

    fig.format(
        suptitle="Regime-count validation against actor portfolios",
        toplabels=(
            "Fit of the weighted contiguous partition",
            "How well regime shares preserve\nfull portfolio structure",
            "Anchoring stays interpretable before regimes fragment",
        ),
    )
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
