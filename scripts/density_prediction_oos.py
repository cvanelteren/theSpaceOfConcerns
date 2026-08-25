#!/usr/bin/env python3
"""Expanding-window test of whether the space adds to a popularity baseline.

Reads the prior-information panel from
``scripts/density_prediction_validation.py`` and fits pairwise ranking models
on earlier choice sets before evaluating the fitted linear score on the next
disjoint five-year outcome block:

- popularity only (the counting baseline)
- max_phi only / density only (the space alone)
- max_phi + popularity / density + popularity (does the space add?)

Outputs:
- output/density_prediction_oos_summary.csv
- output/density_prediction_oos_meta.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

warnings.filterwarnings("ignore")

PANEL = Path("output/density_prediction_panel.parquet")
OUT_SUMMARY = Path("output/density_prediction_oos_summary.csv")
OUT_META = Path("output/density_prediction_oos_meta.json")

SPECS = {
    "popularity_only": ["popularity"],
    "max_phi_only": ["max_phi"],
    "density_only": ["density"],
    "max_phi_plus_popularity": ["max_phi", "popularity"],
    "density_plus_popularity": ["density", "popularity"],
}


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int, rng) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0 or k <= 0:
        return np.nan
    k = min(k, len(scores))
    tiebreak = rng.random(len(scores))
    order = np.lexsort((tiebreak, -scores))
    return float(labels[order[:k]].sum() / n_pos)


def evaluate_spec(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str], seed: int
) -> dict:
    means = train_df[cols].mean().to_numpy(dtype=float)
    scales = train_df[cols].std(ddof=0).to_numpy(dtype=float)
    scales = np.where(scales > 0, scales, 1.0)

    pair_differences = []
    for _, gdf in train_df.groupby("group", sort=False):
        x = (gdf[cols].to_numpy(dtype=float) - means) / scales
        y = gdf["adopted"].to_numpy(dtype=bool)
        if not y.any() or y.all():
            continue
        pair_differences.append((x[y, None, :] - x[None, ~y, :]).reshape(-1, len(cols)))
    if not pair_differences:
        return {"auc": np.nan, "recall": np.nan, "top_decile": np.nan,
                "n_sets": 0, "set_auc": {}}
    differences = np.vstack(pair_differences)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        margin = differences @ beta
        loss = np.logaddexp(0.0, -margin).sum() + 1e-4 * np.dot(beta, beta)
        grad = -(differences.T @ expit(-margin)) + 2e-4 * beta
        return float(loss), grad

    fit = minimize(
        objective,
        np.zeros(len(cols), dtype=float),
        jac=True,
        method="BFGS",
        options={"maxiter": 300},
    )
    if not fit.success and not np.isfinite(fit.fun):
        return {"auc": np.nan, "recall": np.nan, "top_decile": np.nan,
                "n_sets": 0, "set_auc": {}}
    betas = fit.x

    aucs, recalls, tops = [], [], []
    set_auc: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    for group, gdf in test_df.groupby("group", sort=False):
        x = (gdf[cols].to_numpy(dtype=float) - means) / scales
        eta = x @ betas
        labels = gdf["adopted"].to_numpy(dtype=int)
        if labels.sum() == 0:
            continue
        a = auc_from_scores(eta, labels)
        if not np.isnan(a):
            aucs.append(a)
            set_auc[group] = a
            recalls.append(recall_at_k(eta, labels, int(labels.sum()), rng))
            n = len(gdf)
            if n >= 10:
                order = np.argsort(-eta, kind="mergesort")
                tops.append(labels[order[: max(1, n // 10)]].mean())
    return {
        "auc": float(np.mean(aucs)) if aucs else np.nan,
        "recall": float(np.mean(recalls)) if recalls else np.nan,
        "top_decile": float(np.mean(tops)) if tops else np.nan,
        "n_sets": int(len(aucs)),
        "set_auc": set_auc,
    }


def main() -> None:
    panel_df = pd.read_parquet(PANEL)
    period_ends = sorted(panel_df["period_end"].unique())

    rows = []
    per_set: dict[str, dict[str, float]] = {name: {} for name in SPECS}
    for test_end in period_ends[1:]:
        train_df = panel_df[panel_df["period_end"] < test_end]
        test_df = panel_df[panel_df["period_end"] == test_end]
        if train_df["group"].nunique() < 50 or test_df["group"].nunique() < 5:
            continue
        for name, cols in SPECS.items():
            out = evaluate_spec(train_df, test_df, cols, seed=int(test_end))
            per_set[name].update(out["set_auc"])
            rows.append(
                {
                    "train_through": int(test_end - 5),
                    "test_end": int(test_end),
                    "spec": name,
                    "n_train_sets": int(train_df["group"].nunique()),
                    "n_test_sets": out["n_sets"],
                    "auc": out["auc"],
                    "recall_at_k": out["recall"],
                    "top_decile_rate": out["top_decile"],
                }
            )
        print(f"test block ending {test_end}: done")

    paired_comparisons = []
    common_all = set(per_set["popularity_only"])
    for name in SPECS:
        common_all &= set(per_set[name])
    common = sorted(common_all)
    pop = np.array([per_set["popularity_only"][g] for g in common])
    for name in ["density_plus_popularity", "max_phi_plus_popularity", "density_only", "max_phi_only"]:
        other = np.array([per_set[name][g] for g in common])
        diff = other - pop
        paired_comparisons.append(
            {
                "comparison": f"{name} vs popularity_only",
                "n_sets": int(len(common)),
                "mean_auc_diff": float(diff.mean()),
                "share_sets_better": float((diff > 0).mean()),
            }
        )
    paired_df = pd.DataFrame(paired_comparisons)

    df = pd.DataFrame(rows)
    pooled = (
        df.groupby("spec")[["auc", "recall_at_k", "top_decile_rate"]]
        .mean()
        .reset_index()
    )
    pooled.columns = ["spec", "mean_oos_auc", "mean_oos_recall_at_k", "mean_oos_top_decile_rate"]

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_SUMMARY, index=False)
    meta = {
        "test_blocks": sorted(df["test_end"].unique().astype(int).tolist()),
        "window_design": "disjoint five-year outcome blocks; expanding training history",
        "estimator": (
            "pairwise logistic ranker trained on adopted-minus-nonadopted "
            "covariate differences within earlier choice sets"
        ),
        "panel": str(PANEL),
        "pooled": pooled.to_dict(orient="records"),
        "paired_comparisons": paired_df.to_dict(orient="records"),
        "inference_note": (
            "Descriptive held-out performance; no independence test is reported because "
            "actors recur across blocks and each block becomes history for the next."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 220):
        print("\nPer-split AUC:")
        print(
            df.pivot(index="test_end", columns="spec", values="auc")
            .round(3)
            .to_string()
        )
        print("\nPooled out-of-sample performance:")
        print(pooled.to_string(index=False))
        print("\nPaired held-out AUC comparisons across choice sets:")
        print(paired_df.to_string(index=False))
    print(f"\nWrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_META}")


if __name__ == "__main__":
    main()
