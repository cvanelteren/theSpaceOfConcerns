from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import get_rca, load_data, standardize_index_labels

DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
TOPIC_FP = Path("output/fig45_portfolio_space_ridgelines_topic_order.csv")
OUT_PDF = Path("figures/fig22_regime_weighting_sensitivity.pdf")
OUT_PNG = Path("figures/fig22_regime_weighting_sensitivity.png")
OUT_TOPICS = Path("output/fig22_regime_weighting_sensitivity_topics.csv")
OUT_SUMMARY = Path("output/fig22_regime_weighting_sensitivity_summary.json")

RCA_THRESHOLD = 1.0
GAUSS_SIGMA = 1.15
GAUSS_RADIUS = 3


def _gaussian_smooth(values: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _select_partition(
    x_positions: np.ndarray,
    signal: np.ndarray,
    n_regions: int,
    n_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = _gaussian_smooth(signal, sigma=GAUSS_SIGMA, radius=GAUSS_RADIUS)
    weights = np.asarray(smooth, dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)

    x = np.asarray(x_positions, dtype=float)
    centers = np.linspace(float(x.min()), float(x.max()), n_regions)
    for _ in range(n_iter):
        assign = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for idx in range(n_regions):
            mask = assign == idx
            if np.any(mask):
                new_centers[idx] = float(np.average(x[mask], weights=weights[mask]))
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    anchor_idx = sorted(
        dict.fromkeys(int(np.argmin(np.abs(x - c))) for c in centers),
        key=lambda idx: x[idx],
    )
    anchor_idx = np.asarray(anchor_idx, dtype=int)
    boundaries = (
        0.5 * (x[anchor_idx[:-1]] + x[anchor_idx[1:]])
        if len(anchor_idx) > 1
        else np.array([])
    )
    region_idx = np.digitize(x, boundaries).astype(int) + 1
    return anchor_idx, boundaries, region_idx


def _draw_partition_bar(ax, topic_df: pd.DataFrame, region_col: str, title: str) -> None:
    palette = ["#5bb6a6", "#d7a54d", "#a96ec5", "#7f8da6"]
    x = topic_df["x_plot"].to_numpy(dtype=float)
    topics = topic_df["topic"].tolist()
    regions = topic_df[region_col].to_numpy(dtype=int)

    widths = np.empty_like(x)
    widths[1:-1] = 0.5 * (x[2:] - x[:-2])
    widths[0] = 0.5 * (x[1] - x[0])
    widths[-1] = 0.5 * (x[-1] - x[-2])

    for xpos, width, region in zip(x, widths, regions):
        ax.axvspan(
            xpos - width,
            xpos + width,
            ymin=0.15,
            ymax=0.85,
            color=palette[region - 1],
            alpha=0.9,
            lw=0,
        )

    ax.format(
        xlim=(0, 1),
        ylim=(0, 1),
        ylocator=[],
        xlabel="Ordered concern-space coordinate",
        title=title,
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    changed = topic_df["changed"].to_numpy(dtype=bool)
    if changed.any():
        ax.scatter(
            x[changed],
            np.full(changed.sum(), 0.93),
            marker="|",
            s=120,
            color="black",
            linewidths=0.9,
            zorder=5,
        )


def main() -> None:
    topic_df = pd.read_csv(TOPIC_FP).sort_values("topic_order").reset_index(drop=True)
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()
    rca = get_rca(counts).reindex(index=topic_df["topic"])
    excess = np.clip(rca.to_numpy(dtype=float) - RCA_THRESHOLD, 0.0, None)
    aggregate_signal = excess.sum(axis=1)
    x_plot = topic_df["x_plot"].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for k in (3, 4):
        w_anchor, _, w_region = _select_partition(x_plot, aggregate_signal, n_regions=k)
        u_anchor, _, u_region = _select_partition(x_plot, np.ones_like(aggregate_signal), n_regions=k)
        changed = w_region != u_region

        actor_weights_w = np.zeros((rca.shape[1], k), dtype=float)
        actor_weights_u = np.zeros((rca.shape[1], k), dtype=float)
        for ridx in range(k):
            actor_weights_w[:, ridx] = excess[w_region == (ridx + 1)].sum(axis=0)
            actor_weights_u[:, ridx] = excess[u_region == (ridx + 1)].sum(axis=0)
        mask = actor_weights_w.sum(axis=1) > 0
        actor_weights_w[mask] = actor_weights_w[mask] / actor_weights_w[mask].sum(axis=1, keepdims=True)
        actor_weights_u[mask] = actor_weights_u[mask] / actor_weights_u[mask].sum(axis=1, keepdims=True)
        dom_w = np.argmax(actor_weights_w, axis=1) + 1
        dom_u = np.argmax(actor_weights_u, axis=1) + 1

        summary[str(k)] = {
            "weighted_anchors": topic_df.loc[w_anchor, "topic"].tolist(),
            "unweighted_anchors": topic_df.loc[u_anchor, "topic"].tolist(),
            "n_topics_changed": int(changed.sum()),
            "changed_topics": topic_df.loc[changed, "topic"].tolist(),
            "dominant_regime_agreement": float(np.mean(dom_w == dom_u)),
        }

        for _, row in topic_df.assign(
            k=k,
            weighted_region=w_region,
            unweighted_region=u_region,
            changed=changed,
        ).iterrows():
            rows.append(row.to_dict())

    compare_df = pd.DataFrame(rows)
    OUT_TOPICS.parent.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(OUT_TOPICS, index=False)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_df = compare_df[compare_df["k"] == 3].copy()
    agreement = summary["3"]["dominant_regime_agreement"]

    fig, axs = uplt.subplots(nrows=2, figsize=(8.4, 3.2), sharex=True)
    axs.format(abc="[A]", leftlabels=("Weighted", "Unweighted"))
    _draw_partition_bar(axs[0], plot_df, "weighted_region", "Current weighted partition")
    _draw_partition_bar(axs[1], plot_df, "unweighted_region", "Same geometry, uniform weights")
    fig.format(
        suptitle="How much do regime cuts depend on weighting the concern space?",
    )
    axs[1].text(
        1.0,
        -0.32,
        (
            f"$k=3$: {int(summary['3']['n_topics_changed'])} topics move under uniform weights; "
            f"dominant-regime agreement = {agreement:.2f}. "
            "Black ticks mark moved topics."
        ),
        transform=axs[1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
