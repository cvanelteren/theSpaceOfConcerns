from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from sklearn.cluster import KMeans

TOPIC_FP = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
OUT_CSV = Path("output/fig20_regime_k_selection_summary.csv")
OUT_JSON = Path("output/fig20_regime_k_selection_meta.json")
OUT_PDF = Path("figures/fig20_regime_k_selection.pdf")
OUT_PNG = Path("figures/fig20_regime_k_selection.png")

K_MIN = 1
K_MAX = 7
N_INIT = 100
SEED = 7


def _fit_weighted_kmeans(x: np.ndarray, w: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
    labels = km.fit_predict(x, sample_weight=w)
    centers = np.sort(km.cluster_centers_.ravel())
    inertia = float(km.inertia_)
    return labels, centers, inertia


def main() -> None:
    topic_df = pd.read_csv(TOPIC_FP).sort_values("topic_order").reset_index(drop=True)
    x = topic_df[["x_plot"]].to_numpy(dtype=float)
    w = np.clip(topic_df["aggregate_region_signal_smooth"].to_numpy(dtype=float), 1e-9, None)

    rows: list[dict[str, float | int]] = []
    centers_by_k: dict[int, np.ndarray] = {}
    prev_inertia = None
    for k in range(K_MIN, K_MAX + 1):
        _labels, centers, inertia = _fit_weighted_kmeans(x, w, k)
        gain = np.nan if prev_inertia is None else float(prev_inertia - inertia)
        gain_share = np.nan if prev_inertia is None else float((prev_inertia - inertia) / prev_inertia)
        rows.append(
            {
                "k": k,
                "weighted_inertia": inertia,
                "absolute_gain": gain,
                "relative_gain": gain_share,
            }
        )
        centers_by_k[k] = centers
        prev_inertia = inertia

    summary_df = pd.DataFrame(rows)
    gains = summary_df["absolute_gain"].to_numpy(dtype=float)
    elbow_scores = np.full_like(gains, np.nan)
    for idx in range(2, len(gains)):
        if np.isfinite(gains[idx - 1]) and np.isfinite(gains[idx]):
            elbow_scores[idx] = gains[idx - 1] - gains[idx]
    summary_df["elbow_score"] = elbow_scores
    elbow_idx = int(np.nanargmax(elbow_scores))
    elbow_k = int(summary_df.loc[elbow_idx, "k"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(
        json.dumps(
            {
                "k_min": K_MIN,
                "k_max": K_MAX,
                "n_init": N_INIT,
                "seed": SEED,
                "elbow_k": elbow_k,
                "selection_note": "Elbow defined as the largest drop in absolute gain between successive k values on the weighted 1D inertia curve.",
                "centers_by_k": {str(k): centers.tolist() for k, centers in centers_by_k.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axs = uplt.subplots(ncols=2, share=False, refwidth=3.2, refaspect=1.0)
    axs.format(abc="[A]", grid=False)

    ks = summary_df["k"].to_numpy(dtype=int)
    inertia = summary_df["weighted_inertia"].to_numpy(dtype=float)
    gain = summary_df["absolute_gain"].to_numpy(dtype=float)

    ax = axs[0]
    ax.plot(ks, inertia, color="blue7", marker="o", lw=2.2, ms=5)
    ax.axvline(elbow_k, color="black", lw=1.0, ls="--", alpha=0.45)
    ax.scatter([elbow_k], [inertia[elbow_idx]], color="black", s=30, zorder=4)
    ax.text(
        elbow_k + 0.1,
        inertia[elbow_idx],
        f"elbow at $k={elbow_k}$",
        fontsize=8,
        ha="left",
        va="bottom",
    )
    ax.format(
        xlabel="Number of regimes $k$",
        ylabel="Weighted within-region inertia",
        xticks=ks,
    )
    ax.grid(alpha=0.18, color="black")

    ax = axs[1]
    valid = np.isfinite(gain)
    ax.bar(ks[valid], gain[valid], color="orange7", alpha=0.9, width=0.72)
    ax.axvline(elbow_k, color="black", lw=1.0, ls="--", alpha=0.45)
    ax.format(
        xlabel="Additional regime added",
        ylabel="Absolute inertia reduction",
        xticks=ks[1:],
    )
    ax.grid(axis="y", alpha=0.18, color="black")

    fig.format(
        suptitle="Weighted 1D regime-selection check on the ordered concern space",
        toplabels=(
            "Elbow in weighted partition fit",
            "Marginal improvement from adding another regime",
        ),
    )
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
