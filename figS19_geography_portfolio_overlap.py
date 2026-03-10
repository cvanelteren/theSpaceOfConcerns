"""Supplementary Figure S19. Compares portfolio overlap with geographic distance. Shows that simple geography is not the main organizer of ATS concern-space alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy.stats import spearmanr

from utils import get_rca, load_data, standardize_index_labels

DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
GEO_PAIRS_FP = Path("output/step6_geo_portfolio_pairs.csv")

OUT_DATA = Path("output/fig24_geography_portfolio_overlap_pairs.csv")
OUT_SUMMARY = Path("output/fig24_geography_portfolio_overlap_summary.json")
OUT_PDF = Path("figures/figS19_geography_portfolio_overlap.pdf")
OUT_PNG = Path("figures/figS19_geography_portfolio_overlap.png")

RPA_THRESHOLD = 1.0
N_BINS = 10


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else np.nan


def binned_median(df: pd.DataFrame, x_col: str, y_col: str, n_bins: int) -> pd.DataFrame:
    x = df[x_col].to_numpy(dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["x_mid", "y_med", "count"])
    bins = pd.cut(df[x_col], bins=edges, include_lowest=True, duplicates="drop")
    grouped = (
        df.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .agg(
            x_mid=(x_col, "median"),
            y_med=(y_col, "median"),
            count=(y_col, "size"),
        )
        .reset_index(drop=True)
    )
    return grouped[grouped["count"] > 0].copy()


def main() -> None:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()

    presence = counts.gt(0).astype(int).T
    rpa = get_rca(counts).gt(RPA_THRESHOLD).astype(int).T

    geo_pairs = pd.read_csv(GEO_PAIRS_FP)
    valid_actors = sorted(set(geo_pairs["actor_i"]).union(set(geo_pairs["actor_j"])))
    presence = presence.reindex(index=valid_actors).dropna(how="all")
    rpa = rpa.reindex(index=valid_actors).dropna(how="all")

    rows: list[dict[str, object]] = []
    for _, pair in geo_pairs.iterrows():
        ai = pair["actor_i"]
        aj = pair["actor_j"]
        if ai not in presence.index or aj not in presence.index:
            continue
        rows.append(
            {
                "actor_i": ai,
                "actor_j": aj,
                "geo_km": float(pair["geo_km"]),
                "log_geo_km": float(pair["log_geo_km"]),
                "jaccard_presence": jaccard(
                    presence.loc[ai].to_numpy(dtype=bool),
                    presence.loc[aj].to_numpy(dtype=bool),
                ),
                "jaccard_rpa": jaccard(
                    rpa.loc[ai].to_numpy(dtype=bool),
                    rpa.loc[aj].to_numpy(dtype=bool),
                ),
            }
        )

    df = pd.DataFrame(rows).dropna().copy()
    df["geo_thousand_km"] = df["geo_km"] / 1000.0
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DATA, index=False)

    summary = {}
    for label, y_col in [("rpa", "jaccard_rpa"), ("presence", "jaccard_presence")]:
        rho = spearmanr(df["geo_km"], df[y_col], nan_policy="omit")
        summary[label] = {
            "n_pairs": int(df[y_col].notna().sum()),
            "mean_overlap": float(df[y_col].mean()),
            "spearman_rho_geo_km": float(rho.statistic),
            "spearman_p_geo_km": float(rho.pvalue),
        }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axs = uplt.subplots(ncols=2, refwidth=3.1, refaspect=1.0, share=False)
    axs.format(abc="[A]", grid=False)

    panel_specs = [
        ("jaccard_rpa", "RPA > 1", "blue7"),
        ("jaccard_presence", "Any activity", "orange7"),
    ]

    for ax, (y_col, title, color) in zip(axs, panel_specs):
        ax.scatter(
            df["geo_thousand_km"],
            df[y_col],
            s=12,
            alpha=0.35,
            color=color,
            edgecolor="none",
            zorder=2,
        )
        med = binned_median(df, "geo_thousand_km", y_col, N_BINS)
        if not med.empty:
            ax.plot(
                med["x_mid"],
                med["y_med"],
                color="black",
                lw=2.0,
                marker="o",
                ms=4,
                zorder=3,
            )
        rho = summary["rpa" if y_col == "jaccard_rpa" else "presence"]["spearman_rho_geo_km"]
        pval = summary["rpa" if y_col == "jaccard_rpa" else "presence"]["spearman_p_geo_km"]
        ax.text(
            0.97,
            0.97,
            f"$\\rho$ = {rho:.2f}\n$p$ = {pval:.2g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "boxstyle": "round,pad=0.2"},
            zorder=4,
        )
        ax.format(
            title=title,
            xlabel="Geographic distance (1000 km)",
            ylabel="Portfolio overlap",
            ylim=(0, max(0.75, float(df[y_col].max()) * 1.05)),
        )
        ax.grid(alpha=0.18, color="black")

    fig.format(
        suptitle="Do geographically closer countries hold more similar portfolios?"
    )
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
