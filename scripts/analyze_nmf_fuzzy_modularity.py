#!/usr/bin/env python3
"""Explore overlapping concern regions with symmetric non-negative factorization.

The pooled proximity matrix is symmetric and non-negative, so the natural NMF
model is

    Phi_ij ~= sum_k H_ik H_jk,   H_ik >= 0.

Rows of H are normalized to sum to one only *after* fitting.  The normalized
rows are fuzzy component memberships; the unnormalized row magnitudes retain
how strongly a concern participates in the low-rank structure.  The diagonal
is excluded from the loss because the empirical proximity matrix defines only
relations between different concerns.

This is an exploratory companion to the CPM analysis.  It evaluates ranks
2--10 with pair-held-out reconstruction, random-start stability, and actor-
resampled stability.  It does not edit the manuscript or main figures.

Run with

    micromamba run -n ultraplot-dev python -m scripts.analyze_nmf_fuzzy_modularity
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from fig01_space_of_concerns_topology import build_graphs, normalize_topic_key


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "nmf_fuzzy_modularity"
SEED = 20260818
RANKS = tuple(range(2, 11))
CV_FOLDS = 10
OBSERVED_STARTS = 32
CV_STARTS = 6
BOOTSTRAP_STARTS = 3
N_ACTOR_BOOTSTRAP = 120
# L-BFGS invokes threaded linear algebra internally. Two outer workers avoid
# oversubscribing memory while still parallelizing the rank/bootstrap loops.
N_JOBS = 2
RIDGE = 0.01
EPS = 1e-12


@dataclass
class Fit:
    loadings: np.ndarray
    loss: float
    converged: bool
    iterations: int


def _validate_similarity(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("proximity matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("proximity matrix contains non-finite values")
    if np.min(matrix) < -1e-10:
        raise ValueError("proximity matrix must be non-negative")
    matrix = np.clip(0.5 * (matrix + matrix.T), 0.0, None)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _objective_and_gradient(
    flat: np.ndarray,
    proximity: np.ndarray,
    mask: np.ndarray,
    rank: int,
) -> tuple[float, np.ndarray]:
    loadings = flat.reshape(proximity.shape[0], rank)
    residual = mask * (loadings @ loadings.T - proximity)
    # The residual is symmetric, hence d(1/2 ||R||^2)/dH = 2 R H.
    # A small fixed ridge penalty prevents a factor that is weakly identified
    # under a particular held-out mask from taking an arbitrarily large value.
    # It is applied identically at every rank and in every bootstrap draw.
    loss = (
        0.5 * float(np.sum(residual * residual))
        + 0.5 * RIDGE * float(np.sum(loadings * loadings))
    )
    gradient = 2.0 * residual @ loadings + RIDGE * loadings
    return loss, gradient.ravel()


def fit_symmetric_nmf(
    proximity: np.ndarray,
    rank: int,
    *,
    mask: np.ndarray | None = None,
    n_starts: int = 8,
    seed: int = SEED,
    return_all: bool = False,
) -> Fit | tuple[Fit, list[Fit]]:
    proximity = _validate_similarity(proximity)
    if mask is None:
        mask = np.ones_like(proximity, dtype=float)
        np.fill_diagonal(mask, 0.0)
    else:
        mask = np.asarray(mask, dtype=float)
        if mask.shape != proximity.shape or not np.allclose(mask, mask.T):
            raise ValueError("mask must be symmetric and match proximity")
        mask = mask.copy()
        np.fill_diagonal(mask, 0.0)

    observed = proximity[mask.astype(bool)]
    scale = np.sqrt(max(float(observed.mean()), 1e-4) / rank)
    rng = np.random.default_rng(seed)
    fits: list[Fit] = []
    for _ in range(n_starts):
        initial = rng.gamma(shape=2.0, scale=scale / 2.0,
                            size=(proximity.shape[0], rank))
        result = minimize(
            _objective_and_gradient,
            initial.ravel(),
            args=(proximity, mask, rank),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * initial.size,
            options={"maxiter": 2500, "ftol": 1e-12, "gtol": 1e-8},
        )
        fits.append(
            Fit(
                loadings=result.x.reshape(proximity.shape[0], rank),
                loss=float(result.fun),
                converged=bool(result.success),
                iterations=int(result.nit),
            )
        )
    best = min(fits, key=lambda item: item.loss)
    if return_all:
        return best, fits
    return best


def memberships(loadings: np.ndarray) -> np.ndarray:
    totals = loadings.sum(axis=1, keepdims=True)
    return np.divide(
        loadings,
        totals,
        out=np.full_like(loadings, 1.0 / loadings.shape[1]),
        where=totals > EPS,
    )


def membership_entropy(theta: np.ndarray) -> np.ndarray:
    rank = theta.shape[1]
    values = np.zeros_like(theta)
    positive = theta > 0
    values[positive] = theta[positive] * np.log(theta[positive])
    return -values.sum(axis=1) / np.log(rank)


def upper_values(matrix: np.ndarray) -> np.ndarray:
    upper = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[upper], dtype=float)


def comembership(theta: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(theta, axis=1, keepdims=True)
    normalized = np.divide(theta, norms, out=np.zeros_like(theta), where=norms > EPS)
    return normalized @ normalized.T


def matrix_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x, y = upper_values(first), upper_values(second)
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def align_columns(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align candidate NMF columns to reference columns by cosine similarity."""
    ref_norm = reference / np.maximum(np.linalg.norm(reference, axis=0), EPS)
    cand_norm = candidate / np.maximum(np.linalg.norm(candidate, axis=0), EPS)
    similarity = ref_norm.T @ cand_norm
    rows, cols = linear_sum_assignment(-similarity)
    order = cols[np.argsort(rows)]
    return candidate[:, order]


def observed_rank_fit(
    proximity: np.ndarray,
    rank: int,
) -> tuple[Fit, dict[str, float]]:
    best, fits = fit_symmetric_nmf(
        proximity, rank, n_starts=OBSERVED_STARTS,
        seed=SEED + 1000 * rank, return_all=True,
    )
    best_theta = memberships(best.loadings)
    best_comembership = comembership(best_theta)
    start_correlations = []
    for fit in fits:
        theta = memberships(align_columns(best.loadings, fit.loadings))
        start_correlations.append(
            matrix_correlation(best_comembership, comembership(theta))
        )
    mask = ~np.eye(len(proximity), dtype=bool)
    prediction = best.loadings @ best.loadings.T
    baseline = float(proximity[mask].mean())
    sse = float(np.sum((proximity[mask] - prediction[mask]) ** 2))
    sst = float(np.sum((proximity[mask] - baseline) ** 2))
    metrics = {
        "rank": rank,
        "reconstruction_r2": 1.0 - sse / sst,
        "normalized_rmse": float(
            np.sqrt(np.mean((proximity[mask] - prediction[mask]) ** 2))
            / max(np.mean(proximity[mask]), EPS)
        ),
        "median_start_comembership_correlation": float(
            np.nanmedian(start_correlations)
        ),
        "min_start_comembership_correlation": float(
            np.nanmin(start_correlations)
        ),
        "mean_membership_entropy": float(
            membership_entropy(best_theta).mean()
        ),
        "median_max_membership": float(np.median(best_theta.max(axis=1))),
        "converged_starts": int(sum(fit.converged for fit in fits)),
    }
    return best, metrics


def cross_validate_rank(
    proximity: np.ndarray,
    rank: int,
    folds: list[np.ndarray],
) -> list[dict[str, float]]:
    n = len(proximity)
    pairs = np.column_stack(np.triu_indices(n, k=1))
    rows = []
    for fold_id, test_indices in enumerate(folds):
        mask = np.ones_like(proximity, dtype=float)
        np.fill_diagonal(mask, 0.0)
        test_pairs = pairs[test_indices]
        mask[test_pairs[:, 0], test_pairs[:, 1]] = 0.0
        mask[test_pairs[:, 1], test_pairs[:, 0]] = 0.0
        fit = fit_symmetric_nmf(
            proximity,
            rank,
            mask=mask,
            n_starts=CV_STARTS,
            seed=SEED + 100_000 * rank + fold_id,
        )
        prediction = fit.loadings @ fit.loadings.T
        observed = proximity[test_pairs[:, 0], test_pairs[:, 1]]
        predicted = prediction[test_pairs[:, 0], test_pairs[:, 1]]
        train_values = proximity[mask.astype(bool)]
        baseline = float(train_values.mean())
        sse = float(np.sum((observed - predicted) ** 2))
        sst = float(np.sum((observed - baseline) ** 2))
        rows.append(
            {
                "rank": rank,
                "fold": fold_id,
                "heldout_r2": 1.0 - sse / sst,
                "heldout_rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
                "converged": bool(fit.converged),
            }
        )
    return rows


def phi_from_actor_sample(values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    sampled = values[:, sample]
    actor_totals = sampled.sum(axis=0)
    concern_totals = sampled.sum(axis=1)
    grand_total = float(sampled.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rpa = (sampled / actor_totals[None, :]) / (
            concern_totals[:, None] / grand_total
        )
    active = (
        np.nan_to_num(rpa, nan=0.0, posinf=0.0, neginf=0.0) >= 1.0
    ).astype(np.int16)
    cooccurrence = active @ active.T
    prevalence = active.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        proximity = np.minimum(
            cooccurrence / prevalence[:, None],
            cooccurrence / prevalence[None, :],
        )
    proximity = np.nan_to_num(proximity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(proximity, 0.0)
    return proximity


def bootstrap_one(
    draw: int,
    values: np.ndarray,
    reference_loadings: dict[int, np.ndarray],
) -> list[dict[str, float]]:
    rng = np.random.default_rng(SEED + 1_000_000 + draw)
    sample = rng.integers(0, values.shape[1], size=values.shape[1])
    proximity = phi_from_actor_sample(values, sample)
    rows = []
    for rank, reference in reference_loadings.items():
        fit = fit_symmetric_nmf(
            proximity,
            rank,
            n_starts=BOOTSTRAP_STARTS,
            seed=SEED + 2_000_000 + draw * 100 + rank,
        )
        aligned = align_columns(reference, fit.loadings)
        reference_theta = memberships(reference)
        theta = memberships(aligned)
        rows.append(
            {
                "draw": draw,
                "rank": rank,
                "comembership_correlation": matrix_correlation(
                    comembership(reference_theta), comembership(theta)
                ),
                "hard_ari": float(
                    adjusted_rand_score(
                        reference_theta.argmax(axis=1), theta.argmax(axis=1)
                    )
                ),
                "mean_absolute_membership_change": float(
                    np.mean(np.abs(reference_theta - theta))
                ),
                "mean_membership_entropy": float(membership_entropy(theta).mean()),
                "converged": bool(fit.converged),
            }
        )
    return rows


def cpm_labels(topics: list[str]) -> np.ndarray | None:
    path = ROOT / "cpm_representative_partitions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    gamma = min(frame["gamma"].unique(), key=lambda value: abs(value - 0.273))
    frame = frame[frame["gamma"].eq(gamma)]
    mapping = {
        normalize_topic_key(row["concern"]): int(row["community"])
        for _, row in frame.iterrows()
    }
    if any(normalize_topic_key(topic) not in mapping for topic in topics):
        return None
    return np.array([mapping[normalize_topic_key(topic)] for topic in topics])


def order_components(loadings: np.ndarray) -> np.ndarray:
    theta = memberships(loadings)
    hard_sizes = np.bincount(theta.argmax(axis=1), minlength=theta.shape[1])
    strength = loadings.sum(axis=0)
    return np.lexsort((-strength, -hard_sizes))


def select_rank(cv_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame) -> int:
    """Use the standard one-SE rule to retain the smallest predictive rank."""
    best_row = cv_summary.loc[cv_summary["heldout_r2_mean"].idxmax()]
    cutoff = float(best_row["heldout_r2_mean"] - best_row["heldout_r2_se"])
    eligible = cv_summary[cv_summary["heldout_r2_mean"].ge(cutoff)]
    if eligible.empty:
        return int(best_row["rank"])
    return int(eligible["rank"].min())


def write_selected_solution(
    selected_rank: int,
    selected_loadings: np.ndarray,
    topics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_order = order_components(selected_loadings)
    selected_loadings = selected_loadings[:, component_order]
    selected_theta = memberships(selected_loadings)
    selected_entropy = membership_entropy(selected_theta)

    membership_rows = []
    for index, topic in enumerate(topics):
        row = {
            "concern": topic,
            "dominant_component": int(np.argmax(selected_theta[index]) + 1),
            "max_membership": float(np.max(selected_theta[index])),
            "normalized_entropy": float(selected_entropy[index]),
            "loading_strength": float(selected_loadings[index].sum()),
        }
        for component in range(selected_rank):
            row[f"component_{component + 1}"] = float(selected_theta[index, component])
        membership_rows.append(row)
    membership_frame = pd.DataFrame(membership_rows)

    component_rows = []
    for component in range(selected_rank):
        order = np.argsort(selected_theta[:, component])[::-1]
        component_rows.append(
            {
                "component": component + 1,
                "hard_members": int(np.sum(selected_theta.argmax(axis=1) == component)),
                "effective_size": float(selected_theta[:, component].sum()),
                "top_concerns": " | ".join(topics[index] for index in order[:7]),
            }
        )
    components = pd.DataFrame(component_rows)

    membership_frame.to_csv(OUT / "selected_memberships.csv", index=False)
    components.to_csv(OUT / "selected_components.csv", index=False)
    np.save(OUT / "selected_loadings.npy", selected_loadings)
    return membership_frame, components


def finalize_cached() -> None:
    """Reapply the rank rule to completed fits without rerunning bootstraps."""
    cv_summary = pd.read_csv(OUT / "pair_heldout_summary.csv")
    bootstrap_summary = pd.read_csv(OUT / "actor_bootstrap_summary.csv")
    topics = json.loads((OUT / "topics.json").read_text())
    selected_rank = select_rank(cv_summary, bootstrap_summary)
    _, components = write_selected_solution(
        selected_rank,
        np.load(OUT / f"loadings_rank_{selected_rank}.npy"),
        topics,
    )
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["selection_rule"] = (
        "smallest rank within one standard error of the best pair-held-out R2"
    )
    summary["selected_rank"] = selected_rank
    summary["ridge_penalty"] = RIDGE
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Selected cached rank: {selected_rank}")
    print(components.to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, graph, counts, _ = build_graphs()
    topics = sorted(graph.nodes())
    proximity = nx.to_numpy_array(graph, nodelist=topics, weight="weight")
    proximity = _validate_similarity(proximity)
    counts = counts.reindex(index=topics)

    observed_fits: dict[int, Fit] = {}
    observed_rows = []
    for rank in RANKS:
        fit, metrics = observed_rank_fit(proximity, rank)
        observed_fits[rank] = fit
        observed_rows.append(metrics)
        np.save(OUT / f"loadings_rank_{rank}.npy", fit.loadings)
    observed = pd.DataFrame(observed_rows)

    rng = np.random.default_rng(SEED)
    pair_count = len(topics) * (len(topics) - 1) // 2
    pair_order = rng.permutation(pair_count)
    folds = [np.asarray(indices, dtype=int) for indices in np.array_split(pair_order, CV_FOLDS)]
    cv_nested = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(cross_validate_rank)(proximity, rank, folds) for rank in RANKS
    )
    cv = pd.DataFrame([row for rows in cv_nested for row in rows])
    cv_summary = (
        cv.groupby("rank", as_index=False)
        .agg(
            heldout_r2_mean=("heldout_r2", "mean"),
            heldout_r2_sd=("heldout_r2", "std"),
            heldout_rmse_mean=("heldout_rmse", "mean"),
            converged_folds=("converged", "sum"),
        )
    )
    cv_summary["heldout_r2_se"] = cv_summary["heldout_r2_sd"] / np.sqrt(CV_FOLDS)

    reference_loadings = {rank: fit.loadings for rank, fit in observed_fits.items()}
    bootstrap_nested = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(bootstrap_one)(draw, counts.to_numpy(float), reference_loadings)
        for draw in range(N_ACTOR_BOOTSTRAP)
    )
    bootstrap = pd.DataFrame([row for rows in bootstrap_nested for row in rows])
    bootstrap_summary = (
        bootstrap.groupby("rank", as_index=False)
        .agg(
            bootstrap_comembership_median=("comembership_correlation", "median"),
            bootstrap_comembership_low=("comembership_correlation", lambda x: np.quantile(x, 0.025)),
            bootstrap_comembership_high=("comembership_correlation", lambda x: np.quantile(x, 0.975)),
            bootstrap_hard_ari_median=("hard_ari", "median"),
            bootstrap_membership_change_median=("mean_absolute_membership_change", "median"),
            converged_bootstraps=("converged", "sum"),
        )
    )

    selected_rank = select_rank(cv_summary, bootstrap_summary)
    selected_fit = observed_fits[selected_rank]

    cpm = cpm_labels(topics)
    if cpm is not None:
        for rank, fit in observed_fits.items():
            theta = memberships(fit.loadings)
            observed.loc[observed["rank"].eq(rank), "cpm_ari"] = adjusted_rand_score(
                cpm, theta.argmax(axis=1)
            )
            observed.loc[observed["rank"].eq(rank), "cpm_nmi"] = normalized_mutual_info_score(
                cpm, theta.argmax(axis=1)
            )

    membership_frame, components = write_selected_solution(
        selected_rank, selected_fit.loadings, topics
    )

    observed.to_csv(OUT / "observed_rank_diagnostics.csv", index=False)
    cv.to_csv(OUT / "pair_heldout_folds.csv", index=False)
    cv_summary.to_csv(OUT / "pair_heldout_summary.csv", index=False)
    bootstrap.to_csv(OUT / "actor_bootstrap_stability.csv", index=False)
    bootstrap_summary.to_csv(OUT / "actor_bootstrap_summary.csv", index=False)
    np.save(OUT / "proximity.npy", proximity)
    (OUT / "topics.json").write_text(json.dumps(topics, indent=2))

    summary = {
        "method": "masked symmetric non-negative matrix factorization",
        "model": "Phi_ij approximately equals sum_k H_ik H_jk",
        "diagonal_in_loss": False,
        "rank_candidates": list(RANKS),
        "selection_rule": "smallest rank within one standard error of the best pair-held-out R2",
        "selected_rank": selected_rank,
        "n_concerns": len(topics),
        "n_actors": counts.shape[1],
        "cv_folds": CV_FOLDS,
        "actor_bootstrap_draws": N_ACTOR_BOOTSTRAP,
        "random_starts_observed": OBSERVED_STARTS,
        "random_starts_cv": CV_STARTS,
        "random_starts_bootstrap": BOOTSTRAP_STARTS,
        "ridge_penalty": RIDGE,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nObserved fit")
    print(observed.to_string(index=False))
    print("\nPair-held-out reconstruction")
    print(cv_summary.to_string(index=False))
    print("\nActor-bootstrap stability")
    print(bootstrap_summary.to_string(index=False))
    print(f"\nSelected rank: {selected_rank}")
    print(components.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finalize-cached",
        action="store_true",
        help="apply the current rank-selection rule to existing fit outputs",
    )
    arguments = parser.parse_args()
    if arguments.finalize_cached:
        finalize_cached()
    else:
        main()
