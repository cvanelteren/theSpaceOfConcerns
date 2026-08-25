#!/usr/bin/env python3
"""Compare the inferred primary concern with three archive-category treatments.

The main analysis assigns one primary concern to each paper using conditional
information in the archive-category bundle. This script reruns the pooled
concern map and the prospective five-meeting locality model under:

1. the inferred primary concern (main analysis);
2. fractional credit across every archive category;
3. papers carrying exactly one archive category; and
4. the collector's historical first-match category.

The comparison is end to end: category treatment can alter the actor--concern
matrix, the map, rolling specialization, adoption events, and the risk set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hazard_conditional_logit import (  # noqa: E402
    RCA_THRESHOLD,
    WINDOW_MEETINGS,
    build_periods,
    build_window_interaction,
    choose_period_col,
    phi_from_interaction,
    sanitize_periods,
    topic_first_appearance,
)
from utils import (  # noqa: E402
    _clean_category_cell,
    _deduplicate_document_rows,
    _normalize_column_label,
    _split_multi_value,
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
    standardize_index_labels,
)

PRIMARY = ROOT / "data/document-summary-primary-concern.parquet"
MULTI = ROOT / "data/document-summary-multilabel.parquet"
LEGACY = ROOT / "antarctic-database-go/data/processed/document-summary.parquet"
OUT_CSV = ROOT / "output/primary_concern_measurement_sensitivity.csv"
OUT_JSON = ROOT / "output/primary_concern_measurement_sensitivity.json"


def prepare(path: Path) -> pd.DataFrame:
    """Standardize a parquet without the canonical-path redirect in utils."""
    df = pd.read_parquet(path).convert_dtypes()
    df = df.rename(columns=_normalize_column_label)
    if "year" not in df.columns and "meeting year" in df.columns:
        df["year"] = df["meeting year"]
    if "meeting year" not in df.columns and "year" in df.columns:
        df["meeting year"] = df["year"]
    if "submitted by" not in df.columns:
        if "parties" in df.columns:
            df["submitted by"] = df["parties"].apply(
                lambda values: ", ".join(_split_multi_value(values))
            )
        else:
            df["submitted by"] = df["party"].astype(str)
    df["category"] = df["category"].apply(_clean_category_cell)
    df = df.dropna(subset=["category"]).copy()
    return _deduplicate_document_rows(df)


def variants() -> dict[str, pd.DataFrame]:
    primary = prepare(PRIMARY)
    multi = prepare(MULTI)
    single = multi[
        multi["category"].fillna("").str.count("\t").add(1).eq(1)
    ].copy()
    legacy = prepare(LEGACY)
    return {
        "inferred_primary": primary,
        "fractional_multilabel": multi,
        "single_category_papers": single,
        "collector_first_match": legacy,
    }


def corpus_objects(df: pd.DataFrame, canonical_topics: list[str]):
    actors_raw = extract_unique_countries(df)
    topics_raw = extract_unique_topics(df)
    counts = standardize_index_labels(
        generate_interaction_matrix(df, actors_raw, topics_raw)
    )
    actors = sorted(counts.columns)
    counts = counts.reindex(index=canonical_topics, columns=actors, fill_value=0.0)
    phi = phi_from_interaction(counts, canonical_topics)
    rca = get_rca(counts).reindex(index=canonical_topics, columns=actors, fill_value=0.0)
    active = rca.ge(RCA_THRESHOLD)
    return counts, phi, active, actors_raw, topics_raw


def fit_prospective_locality(
    df: pd.DataFrame,
    canonical_topics: list[str],
) -> dict[str, float | int]:
    period_col = choose_period_col(df)
    clean = sanitize_periods(df, period_col)
    actors_raw = extract_unique_countries(clean)
    topics_raw = extract_unique_topics(clean)
    counts = standardize_index_labels(
        generate_interaction_matrix(clean, actors_raw, topics_raw)
    )
    actors = sorted(counts.columns)
    topics = canonical_topics
    period_min = int(clean[period_col].min())
    period_max = int(clean[period_col].max())
    first = topic_first_appearance(clean, period_col)
    periods = build_periods(period_min, period_max, WINDOW_MEETINGS)

    interactions: list[pd.DataFrame] = []
    active: list[pd.DataFrame] = []
    for start, end in periods:
        interaction = build_window_interaction(
            clean,
            period_col,
            start,
            end,
            set(actors_raw),
            set(topics_raw),
            topics,
            actors,
        )
        interactions.append(interaction)
        active.append(get_rca(interaction).ge(RCA_THRESHOLD))

    relative_rows: list[dict] = []
    document_rows: list[dict] = []
    for t in range(1, len(periods)):
        prev_end = int(periods[t - 1][1])
        period_end = int(periods[t][1])
        cumulative = build_window_interaction(
            clean,
            period_col,
            period_min,
            prev_end,
            set(actors_raw),
            set(topics_raw),
            topics,
            actors,
        )
        phi = phi_from_interaction(cumulative, topics)
        prev_active = active[t - 1]
        curr_active = active[t]
        prev_counts = interactions[t - 1]
        curr_counts = interactions[t]
        popularity = prev_active.sum(axis=1) / max(len(actors), 1)
        for actor in actors:
            held = prev_active[actor].to_numpy(dtype=bool)
            current = curr_active[actor].to_numpy(dtype=bool)
            if not held.any():
                continue
            at_risk = ~held
            adopted = current & at_risk
            distances = 1.0 - phi[:, np.where(held)[0]].max(axis=1)
            group = f"{actor}::{period_end}"
            for idx, topic in enumerate(topics):
                if first.get(topic, period_end + 1) > prev_end:
                    continue
                base = {
                    "group": group,
                    "distance": float(distances[idx]),
                    "topic_popularity": float(popularity.loc[topic]),
                }
                if at_risk[idx]:
                    relative_rows.append(dict(base, adopted=int(adopted[idx])))
                if float(prev_counts.iloc[idx][actor]) <= 0:
                    document_rows.append(
                        dict(
                            base,
                            adopted=int(float(curr_counts.iloc[idx][actor]) > 0),
                        )
                    )

    def _fit(rows: list[dict]) -> tuple[pd.DataFrame, object]:
        panel = pd.DataFrame(rows)
        group_counts = panel.groupby("group")["adopted"].agg(["sum", "count"])
        valid = group_counts[
            group_counts["sum"].gt(0) & group_counts["sum"].lt(group_counts["count"])
        ].index
        panel = panel[panel["group"].isin(valid)].copy()
        result = ConditionalLogit(
            panel["adopted"].astype(int),
            panel[["distance", "topic_popularity"]],
            groups=panel["group"],
        ).fit(disp=False, maxiter=200)
        return panel, result

    panel, result = _fit(relative_rows)
    document_panel, document_result = _fit(document_rows)
    beta = float(result.params["distance"])
    se = float(result.bse["distance"])
    document_beta = float(document_result.params["distance"])
    document_se = float(document_result.bse["distance"])
    return {
        "papers": int(clean["paper id"].nunique()),
        "actor_periods": int(panel["group"].nunique()),
        "risk_rows": int(len(panel)),
        "distance_beta": beta,
        "distance_se": se,
        "odds_ratio_per_0_1": float(np.exp(0.1 * beta)),
        "odds_ratio_per_0_1_ci_low": float(np.exp(0.1 * (beta - 1.96 * se))),
        "odds_ratio_per_0_1_ci_high": float(np.exp(0.1 * (beta + 1.96 * se))),
        "new_document_events": int(document_panel["adopted"].sum()),
        "new_document_actor_periods": int(document_panel["group"].nunique()),
        "new_document_risk_rows": int(len(document_panel)),
        "new_document_distance_beta": document_beta,
        "new_document_distance_se": document_se,
        "new_document_odds_ratio_per_0_1": float(np.exp(0.1 * document_beta)),
        "new_document_odds_ratio_per_0_1_ci_low": float(
            np.exp(0.1 * (document_beta - 1.96 * document_se))
        ),
        "new_document_odds_ratio_per_0_1_ci_high": float(
            np.exp(0.1 * (document_beta + 1.96 * document_se))
        ),
    }


def upper_triangle(phi: np.ndarray) -> np.ndarray:
    return phi[np.triu_indices_from(phi, k=1)]


def top_pairs(phi: np.ndarray, share: float = 0.10) -> set[tuple[int, int]]:
    idx = np.triu_indices_from(phi, k=1)
    values = phi[idx]
    n = max(1, int(np.ceil(len(values) * share)))
    chosen = np.argpartition(values, -n)[-n:]
    return {(int(idx[0][i]), int(idx[1][i])) for i in chosen}


def strongest_neighbors(phi: np.ndarray) -> np.ndarray:
    arr = phi.copy()
    np.fill_diagonal(arr, -np.inf)
    return np.argmax(arr, axis=1)


def modularity_summary(phi: np.ndarray, n_seeds: int = 100) -> dict[str, float | int]:
    """Summarize weighted Louvain modularity on the unfiltered full graph."""
    graph = nx.from_numpy_array(phi)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    runs = []
    for seed in range(n_seeds):
        communities = nx.community.louvain_communities(
            graph, weight="weight", resolution=1.0, seed=seed
        )
        runs.append(
            (
                float(nx.community.modularity(graph, communities, weight="weight")),
                int(len(communities)),
            )
        )
    best = max(runs, key=lambda item: item[0])
    upper = upper_triangle(phi)
    return {
        "positive_pair_share": float(np.mean(upper > 0)),
        "mean_positive_proximity": float(upper[upper > 0].mean()),
        "louvain_modularity_mean_100_seeds": float(np.mean([item[0] for item in runs])),
        "louvain_modularity_max_100_seeds": float(best[0]),
        "louvain_communities_at_max": int(best[1]),
    }


def main() -> None:
    data = variants()
    canonical_topics = sorted(
        standardize_index_labels(
            pd.DataFrame(index=sorted(extract_unique_topics(data["inferred_primary"])))
        ).index
    )
    objects = {
        name: corpus_objects(df, canonical_topics) for name, df in data.items()
    }
    primary_phi = objects["inferred_primary"][1]
    primary_active = objects["inferred_primary"][2]
    primary_top = top_pairs(primary_phi)
    primary_strongest = strongest_neighbors(primary_phi)

    rows = []
    for name, df in data.items():
        _, phi, active, _, _ = objects[name]
        pooled = upper_triangle(phi)
        reference = upper_triangle(primary_phi)
        candidate_top = top_pairs(phi)
        union = primary_top | candidate_top
        common_actors = primary_active.columns.intersection(active.columns)
        agreement = primary_active[common_actors].to_numpy().ravel() == active[
            common_actors
        ].to_numpy().ravel()
        locality = fit_prospective_locality(df, canonical_topics)
        network_summary = modularity_summary(phi)
        rows.append(
            {
                "category_treatment": name,
                **locality,
                **network_summary,
                "phi_pearson_vs_primary": float(pearsonr(reference, pooled).statistic),
                "phi_spearman_vs_primary": float(spearmanr(reference, pooled).statistic),
                "top_10pct_edge_jaccard_vs_primary": float(
                    len(primary_top & candidate_top) / len(union)
                ),
                "strongest_neighbor_agreement_vs_primary": float(
                    np.mean(primary_strongest == strongest_neighbors(phi))
                ),
                "actor_topic_specialization_agreement_vs_primary": float(
                    agreement.mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)
    payload = {
        "main_treatment": "inferred_primary",
        "primary_assignment": "maximum -log P(label | remaining bundle), with marginal IDF tie-break",
        "locality_model": "five-meeting cumulative-lagged map; topics observed before outcome window; actor-period conditional logit controlling previous holder share",
        "map_comparison": "45 common concerns; upper-triangle proximity values",
        "rows": summary.to_dict(orient="records"),
    }
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
