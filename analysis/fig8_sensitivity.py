from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from utils import (
    compute_product_space,
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
)


def build_country_topic_matrix(submitted: pd.DataFrame) -> pd.DataFrame:
    """Build topic x country count matrix from the canonical submitted dataframe."""
    countries = extract_unique_countries(submitted)
    topics = extract_unique_topics(submitted)
    return generate_interaction_matrix(submitted, countries, topics)


def compute_graph_laplacian(proximity_matrix: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Compute graph Laplacian from a proximity matrix."""
    adjacency = (proximity_matrix + proximity_matrix.T) / 2
    degrees = np.sum(adjacency, axis=1)
    degree_mat = np.diag(degrees)

    if normalized:
        degrees_sqrt_inv = np.zeros_like(degrees)
        nonzero = degrees > 1e-10
        degrees_sqrt_inv[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        degree_sqrt_inv_mat = np.diag(degrees_sqrt_inv)
        return np.eye(len(degrees)) - degree_sqrt_inv_mat @ adjacency @ degree_sqrt_inv_mat
    return degree_mat - adjacency


def compute_eigenvalues(matrix: np.ndarray, k: int | None = None, method: str = "full") -> np.ndarray:
    """Compute sorted eigenvalues from dense/sparse solver."""
    if method == "sparse" and k is not None:
        try:
            eigv = eigsh(
                matrix,
                k=min(k, matrix.shape[0] - 1),
                which="LA",
                return_eigenvectors=False,
            )
            return np.sort(eigv)
        except Exception:
            eigv = eigvals(matrix)
            return np.sort(np.real(eigv))[-k:]
    eigv = eigvals(matrix)
    return np.sort(np.real(eigv))


def spectral_distance(
    matrix1: np.ndarray, matrix2: np.ndarray, method: str = "eigenvalue_l2", k: int | None = None
) -> float:
    """Compute spectral distance between two Laplacians."""
    if method == "frobenius":
        return float(np.linalg.norm(matrix1 - matrix2, "fro"))
    if method not in {"eigenvalue_l2", "eigenvalue_max"}:
        raise ValueError("method must be 'frobenius', 'eigenvalue_l2', or 'eigenvalue_max'")

    eigs1 = compute_eigenvalues(matrix1, k=k)
    eigs2 = compute_eigenvalues(matrix2, k=k)
    max_len = max(len(eigs1), len(eigs2))
    eigs1 = np.pad(eigs1, (0, max_len - len(eigs1)))
    eigs2 = np.pad(eigs2, (0, max_len - len(eigs2)))

    if method == "eigenvalue_l2":
        return float(np.linalg.norm(eigs1 - eigs2))
    return float(np.max(np.abs(eigs1 - eigs2)))


def sensitivity_analysis(export_data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Eliminate countries greedily by largest spectral perturbation to the baseline space.
    Returns rows with columns: country, rca, phi, spectral_distance, n.
    """
    countries = export_data.columns.tolist()
    baseline_rca = get_rca(export_data)
    baseline_proximity = compute_product_space(baseline_rca, threshold)
    base_laplacian = compute_graph_laplacian(baseline_proximity.values, normalized=True)

    def eliminate_once(active_countries: list[str], active_data: pd.DataFrame) -> dict:
        largest = -np.inf
        selected = {}
        for country in active_countries:
            reduced_data = active_data.drop(columns=[country])
            reduced_rca = get_rca(reduced_data)
            reduced_proximity = compute_product_space(reduced_rca, threshold)
            reduced_laplacian = compute_graph_laplacian(reduced_proximity.values, normalized=True)
            dist = spectral_distance(base_laplacian, reduced_laplacian, method="eigenvalue_l2")
            if dist > largest:
                largest = dist
                selected = {
                    "country": country,
                    "rca": reduced_rca,
                    "phi": reduced_proximity,
                    "spectral_distance": dist,
                }
        return selected

    total = len(countries)
    rows = []
    active_data = export_data.copy()
    with tqdm(total=len(countries)) as pbar:
        while len(countries) > 1:
            row = eliminate_once(countries, active_data)
            dropped = row["country"]
            countries.remove(dropped)
            active_data.drop(columns=[dropped], inplace=True)
            row["n"] = total - len(countries)
            rows.append(row)
            pbar.update(1)
    return pd.DataFrame(rows)

