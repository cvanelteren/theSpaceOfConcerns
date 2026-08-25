"""Test whether marginal concern counts reconstruct relational proximity.

The unit of the primary analysis is a pair of concerns.  The outcome is pooled
proximity, phi_ij.  Symmetric predictors use only the two concerns' marginal
paper volumes or numbers of specializing actors.  Actor bootstraps rebuild the
entire count matrix, RPA assignments, and proximity matrix.  A node-label QAP
tests whether the observed alignment exceeds arbitrary assignments of the
same marginal values to concerns.

This is a reconstruction diagnostic, not a causal or prospective model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from fig01_space_of_concerns_topology import build_graphs
from utils import compute_product_space, get_rca


OUT_DIR = Path("output/counts_vs_relations")
SEED = 20260817
N_ACTOR_BOOTSTRAPS = 1000
N_QAP_PERMUTATIONS = 9999


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def symmetric_features(values: np.ndarray, *, log_transform: bool) -> np.ndarray:
    x = np.log1p(values) if log_transform else np.asarray(values, dtype=float)
    i, j = np.triu_indices(len(x), k=1)
    return np.column_stack(
        [
            x[i] + x[j],
            np.abs(x[i] - x[j]),
            np.minimum(x[i], x[j]),
        ]
    )


def fit_r2(y: np.ndarray, X: np.ndarray) -> float:
    return float(LinearRegression().fit(X, y).score(X, y))


def leave_endpoint_out_r2(y: np.ndarray, X: np.ndarray, n_nodes: int) -> float:
    """Predict every dyad from fits excluding each of its endpoints.

    Each pair receives two predictions: one from a model trained without its
    first endpoint and one from a model trained without its second endpoint.
    Their mean is scored once, avoiding duplicate dyads in the final R2.
    """
    i, j = np.triu_indices(n_nodes, k=1)
    predictions = np.zeros((len(y), 2), dtype=float)
    filled = np.zeros((len(y), 2), dtype=bool)
    for node in range(n_nodes):
        test = (i == node) | (j == node)
        train = ~test
        model = LinearRegression().fit(X[train], y[train])
        node_predictions = model.predict(X[test])
        rows = np.flatnonzero(test)
        for row, value in zip(rows, node_predictions, strict=True):
            slot = 0 if i[row] == node else 1
            predictions[row, slot] = value
            filled[row, slot] = True
    if not filled.all():
        raise AssertionError("Every dyad should receive one prediction per endpoint")
    return float(r2_score(y, predictions.mean(axis=1)))


def construct_metrics(counts: pd.DataFrame) -> dict:
    rpa = get_rca(counts)
    phi_df = compute_product_space(rpa)
    phi = phi_df.to_numpy(dtype=float)
    volume = counts.sum(axis=1).to_numpy(dtype=float)
    specializing = (rpa >= 1.0).sum(axis=1).to_numpy(dtype=float)
    strength = phi.sum(axis=1)
    positive_degree = (phi > 0).sum(axis=1).astype(float)
    y = upper_values(phi)
    X_volume = symmetric_features(volume, log_transform=True)
    X_actor = symmetric_features(specializing, log_transform=False)
    X_both = np.column_stack([X_volume, X_actor])
    return {
        "phi": phi,
        "y": y,
        "volume": volume,
        "specializing": specializing,
        "strength": strength,
        "positive_degree": positive_degree,
        "X_volume": X_volume,
        "X_actor": X_actor,
        "X_both": X_both,
        "r2_volume": fit_r2(y, X_volume),
        "r2_actor": fit_r2(y, X_actor),
        "r2_both": fit_r2(y, X_both),
        "rho_volume_strength": float(spearmanr(volume, strength).statistic),
        "r_volume_strength": float(pearsonr(volume, strength).statistic),
        "rho_actor_strength": float(spearmanr(specializing, strength).statistic),
        "r_actor_strength": float(pearsonr(specializing, strength).statistic),
    }


def actor_bootstrap(counts: pd.DataFrame, n_draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n_actors = counts.shape[1]
    rows = []
    for draw in range(n_draws):
        selected = rng.integers(0, n_actors, size=n_actors)
        sampled = counts.iloc[:, selected].copy()
        # Duplicate labels would be collapsed by pandas alignment in later
        # operations, so give sampled actor histories unique bootstrap IDs.
        sampled.columns = [f"bootstrap_actor_{idx}" for idx in range(n_actors)]
        result = construct_metrics(sampled)
        rows.append(
            {
                "draw": draw,
                "r2_volume": result["r2_volume"],
                "r2_actor": result["r2_actor"],
                "r2_both": result["r2_both"],
                "delta_r2_volume_given_actor": result["r2_both"]
                - result["r2_actor"],
                "rho_volume_strength": result["rho_volume_strength"],
                "r_volume_strength": result["r_volume_strength"],
                "rho_actor_strength": result["rho_actor_strength"],
                "r_actor_strength": result["r_actor_strength"],
            }
        )
    return pd.DataFrame(rows)


def qap_permutations(observed: dict, n_draws: int) -> pd.DataFrame:
    """Shuffle concern labels on marginal attributes relative to fixed phi."""
    rng = np.random.default_rng(SEED + 1)
    y = observed["y"]
    volume = observed["volume"]
    specializing = observed["specializing"]
    rows = []
    for draw in range(n_draws):
        order = rng.permutation(len(volume))
        X_volume = symmetric_features(volume[order], log_transform=True)
        X_actor = symmetric_features(specializing[order], log_transform=False)
        X_both = np.column_stack([X_volume, X_actor])
        rows.append(
            {
                "draw": draw,
                "r2_volume": fit_r2(y, X_volume),
                "r2_actor": fit_r2(y, X_actor),
                "r2_both": fit_r2(y, X_both),
            }
        )
    return pd.DataFrame(rows)


def interval(series: pd.Series) -> list[float]:
    return [float(value) for value in series.quantile([0.025, 0.5, 0.975])]


def qap_p(observed: float, null: pd.Series) -> float:
    return float((1 + np.sum(null.to_numpy() >= observed)) / (len(null) + 1))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, _, counts, _ = build_graphs()
    counts = counts.sort_index().copy()
    observed = construct_metrics(counts)

    observed_rows = pd.DataFrame(
        {
            "concern": counts.index,
            "paper_volume": observed["volume"],
            "specializing_actors": observed["specializing"],
            "weighted_strength": observed["strength"],
            "positive_degree": observed["positive_degree"],
        }
    )
    observed_rows.to_csv(OUT_DIR / "concern_marginals.csv", index=False)

    bootstrap = actor_bootstrap(counts, N_ACTOR_BOOTSTRAPS)
    bootstrap.to_csv(OUT_DIR / "actor_bootstrap.csv", index=False)
    qap = qap_permutations(observed, N_QAP_PERMUTATIONS)
    qap.to_csv(OUT_DIR / "qap_null.csv", index=False)

    n_nodes = counts.shape[0]
    loo = {
        "paper_volume": leave_endpoint_out_r2(
            observed["y"], observed["X_volume"], n_nodes
        ),
        "specializing_actors": leave_endpoint_out_r2(
            observed["y"], observed["X_actor"], n_nodes
        ),
        "both": leave_endpoint_out_r2(
            observed["y"], observed["X_both"], n_nodes
        ),
    }

    summary = {
        "design": {
            "category_treatment": "fractional multi-label",
            "n_concerns": int(counts.shape[0]),
            "n_actors": int(counts.shape[1]),
            "n_pairs": int(len(observed["y"])),
            "actor_bootstrap_draws": N_ACTOR_BOOTSTRAPS,
            "qap_permutations": N_QAP_PERMUTATIONS,
            "outcome": "pooled pairwise proximity phi",
            "paper_predictors": "sum, absolute difference, and minimum of log1p marginal paper volume",
            "actor_predictors": "sum, absolute difference, and minimum of numbers of specializing actors",
        },
        "observed": {
            "r2_paper_volume": observed["r2_volume"],
            "r2_specializing_actors": observed["r2_actor"],
            "r2_both": observed["r2_both"],
            "delta_r2_paper_given_actor": observed["r2_both"]
            - observed["r2_actor"],
            "rho_paper_volume_vs_strength": observed["rho_volume_strength"],
            "r_paper_volume_vs_strength": observed["r_volume_strength"],
            "rho_specializing_actors_vs_strength": observed["rho_actor_strength"],
            "r_specializing_actors_vs_strength": observed["r_actor_strength"],
        },
        "leave_endpoint_out_r2": loo,
        "actor_bootstrap_central_95": {
            column: interval(bootstrap[column])
            for column in bootstrap.columns
            if column != "draw"
        },
        "qap_upper_tail_p": {
            "paper_volume": qap_p(observed["r2_volume"], qap["r2_volume"]),
            "specializing_actors": qap_p(observed["r2_actor"], qap["r2_actor"]),
            "both": qap_p(observed["r2_both"], qap["r2_both"]),
        },
        "interpretation": (
            "Marginal prevalence is associated with proximity but does not "
            "reconstruct most pairwise variation. The actor-count model is "
            "substantially stronger than paper volume; paper volume adds "
            "little once specialization prevalence is known."
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
