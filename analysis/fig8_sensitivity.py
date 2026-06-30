from __future__ import annotations

import numpy as np
import pandas as pd
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


def proximity_rmse(full_proximity: np.ndarray, reduced_proximity: np.ndarray) -> float:
    """Compute RMSE over unique topic-topic pairs (upper triangle, excluding diagonal)."""
    diff = np.asarray(reduced_proximity, dtype=float) - np.asarray(full_proximity, dtype=float)
    tri = np.triu_indices_from(diff, k=1)
    vals = diff[tri]
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals**2)))


def sensitivity_analysis(export_data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Eliminate countries greedily by largest RMSE perturbation to the baseline proximity matrix.
    Returns rows with columns: country, rca, phi, rmse, n.
    """
    countries = export_data.columns.tolist()
    baseline_rca = get_rca(export_data)
    baseline_proximity = compute_product_space(baseline_rca, threshold)

    def eliminate_once(active_countries: list[str], active_data: pd.DataFrame) -> dict:
        largest = -np.inf
        selected = {}
        for country in active_countries:
            reduced_data = active_data.drop(columns=[country])
            reduced_rca = get_rca(reduced_data)
            reduced_proximity = compute_product_space(reduced_rca, threshold)
            dist = proximity_rmse(baseline_proximity.values, reduced_proximity.values)
            if dist > largest:
                largest = dist
                selected = {
                    "country": country,
                    "rca": reduced_rca,
                    "phi": reduced_proximity,
                    "rmse": dist,
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
