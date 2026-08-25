#!/usr/bin/env python3
"""Compare detailed concern-space geometry across category treatments.

Modularity can remain nearly unchanged even when the map's individual edges,
nearest neighbours, backbone, and centrality ordering change.  This script
measures both levels.  The inferred-primary treatment is the reference and is
compared with fractional use of all official categories, singleton papers only,
and the collector's historical first-match category.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from primary_concern_sensitivity import corpus_objects, variants  # noqa: E402
from utils import extract_unique_topics, standardize_index_labels  # noqa: E402

OUT_SUMMARY = ROOT / "output/concern_geometry_comparison.csv"
OUT_NODES = ROOT / "output/concern_geometry_node_changes.csv"
OUT_EDGES = ROOT / "output/concern_geometry_edge_changes.csv"
OUT_JSON = ROOT / "output/concern_geometry_comparison.json"


def edge_set(graph: nx.Graph) -> set[tuple[str, str]]:
    return {tuple(sorted((str(a), str(b)))) for a, b in graph.edges()}


def jaccard(left: set, right: set) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def graph_from_phi(phi: np.ndarray, topics: list[str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(topics)
    for i, source in enumerate(topics):
        for j in range(i + 1, len(topics)):
            weight = float(phi[i, j])
            if weight <= 0:
                continue
            graph.add_edge(
                source,
                topics[j],
                weight=weight,
                # High proximity means close.  -log(phi) is therefore the
                # additive distance used throughout the project.
                distance=float(-np.log(np.clip(weight, 1e-12, 1.0))),
            )
    return graph


def display_backbone(graph: nx.Graph) -> tuple[nx.Graph, nx.Graph]:
    mst = nx.maximum_spanning_tree(graph, weight="weight")
    backbone = mst.copy()
    weights = np.asarray(
        [float(data["weight"]) for _, _, data in graph.edges(data=True)]
    )
    if weights.size:
        cutoff = float(np.percentile(weights, 95))
        for source, target, data in graph.edges(data=True):
            if float(data["weight"]) >= cutoff:
                backbone.add_edge(source, target, **data)
    return mst, backbone


def best_louvain(graph: nx.Graph, seeds: int = 100) -> tuple[float, list[set[str]]]:
    best_modularity = -np.inf
    best_partition: list[set[str]] = []
    for seed in range(seeds):
        partition = nx.community.louvain_communities(
            graph, weight="weight", resolution=1.0, seed=seed
        )
        modularity = float(nx.community.modularity(graph, partition, weight="weight"))
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = [set(group) for group in partition]
    return best_modularity, best_partition


def partition_labels(partition: list[set[str]], topics: list[str]) -> np.ndarray:
    membership = {
        topic: group_index
        for group_index, group in enumerate(partition)
        for topic in group
    }
    return np.asarray([membership[topic] for topic in topics], dtype=int)


def rank_series(values: dict[str, float], topics: list[str]) -> pd.Series:
    return pd.Series(values, index=topics, dtype=float).rank(
        method="average", ascending=False
    )


def centralities(graph: nx.Graph, topics: list[str]) -> dict[str, pd.Series]:
    strength = {
        topic: float(sum(data["weight"] for _, _, data in graph.edges(topic, data=True)))
        for topic in topics
    }
    positive_degree = {topic: float(graph.degree(topic)) for topic in topics}
    closeness = nx.closeness_centrality(graph, distance="distance")
    betweenness = nx.betweenness_centrality(
        graph, weight="distance", normalized=True
    )
    eigenvector = nx.eigenvector_centrality_numpy(graph, weight="weight")
    return {
        "strength": pd.Series(strength, index=topics, dtype=float),
        "positive_degree": pd.Series(positive_degree, index=topics, dtype=float),
        "closeness": pd.Series(closeness, index=topics, dtype=float),
        "betweenness": pd.Series(betweenness, index=topics, dtype=float),
        "eigenvector": pd.Series(eigenvector, index=topics, dtype=float),
    }


def top_neighbors(phi: np.ndarray, topics: list[str], k: int = 5) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for i, topic in enumerate(topics):
        order = np.argsort(-phi[i], kind="stable")
        selected = [
            topics[j]
            for j in order
            if j != i and float(phi[i, j]) > 0
        ][:k]
        output[topic] = set(selected)
    return output


def strongest_neighbor(phi: np.ndarray, topics: list[str]) -> dict[str, str]:
    copy = phi.copy()
    np.fill_diagonal(copy, -np.inf)
    return {topic: topics[int(np.argmax(copy[i]))] for i, topic in enumerate(topics)}


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    return float(spearmanr(left.to_numpy(), right.to_numpy()).statistic)


def main() -> None:
    data = variants()
    topics = sorted(
        standardize_index_labels(
            pd.DataFrame(index=sorted(extract_unique_topics(data["inferred_primary"])))
        ).index
    )
    objects = {name: corpus_objects(frame, topics) for name, frame in data.items()}

    geometries = {}
    for name, (_, phi, _, _, _) in objects.items():
        graph = graph_from_phi(phi, topics)
        mst, backbone = display_backbone(graph)
        modularity, partition = best_louvain(graph)
        geometries[name] = {
            "phi": phi,
            "graph": graph,
            "mst": mst,
            "backbone": backbone,
            "modularity": modularity,
            "partition": partition,
            "partition_labels": partition_labels(partition, topics),
            "centralities": centralities(graph, topics),
            "top_neighbors": top_neighbors(phi, topics, k=5),
            "strongest_neighbor": strongest_neighbor(phi, topics),
        }

    reference = geometries["inferred_primary"]
    upper = np.triu_indices(len(topics), k=1)
    reference_tree_steps = np.asarray(
        nx.floyd_warshall_numpy(
            reference["mst"], nodelist=topics, weight=None
        )
    )[upper]
    summary_rows = []
    node_rows = []
    edge_rows = []

    for name, geometry in geometries.items():
        reference_mst_edges = edge_set(reference["mst"])
        candidate_mst_edges = edge_set(geometry["mst"])
        reference_backbone_edges = edge_set(reference["backbone"])
        candidate_backbone_edges = edge_set(geometry["backbone"])
        neighbour_jaccards = {
            topic: jaccard(
                reference["top_neighbors"][topic], geometry["top_neighbors"][topic]
            )
            for topic in topics
        }
        centrality_correlations = {
            metric: safe_spearman(
                reference["centralities"][metric], geometry["centralities"][metric]
            )
            for metric in reference["centralities"]
        }
        reference_strength_rank = rank_series(
            reference["centralities"]["strength"].to_dict(), topics
        )
        candidate_strength_rank = rank_series(
            geometry["centralities"]["strength"].to_dict(), topics
        )
        candidate_tree_steps = np.asarray(
            nx.floyd_warshall_numpy(
                geometry["mst"], nodelist=topics, weight=None
            )
        )[upper]
        summary_rows.append(
            {
                "category_treatment": name,
                "modularity": geometry["modularity"],
                "community_count": len(geometry["partition"]),
                "partition_ari_vs_primary": float(
                    adjusted_rand_score(
                        reference["partition_labels"], geometry["partition_labels"]
                    )
                ),
                "mst_edge_jaccard_vs_primary": jaccard(
                    reference_mst_edges, candidate_mst_edges
                ),
                "primary_mst_edges_retained": float(
                    len(reference_mst_edges & candidate_mst_edges)
                    / len(reference_mst_edges)
                ),
                "display_backbone_edge_jaccard_vs_primary": jaccard(
                    reference_backbone_edges, candidate_backbone_edges
                ),
                "primary_display_backbone_edges_retained": float(
                    len(reference_backbone_edges & candidate_backbone_edges)
                    / len(reference_backbone_edges)
                ),
                # Figure 1's layout preserves shortest-path steps through the
                # maximum spanning tree.  This therefore measures change in
                # the structural scaffold behind the visual coordinates.
                "mst_pairwise_step_spearman_vs_primary": float(
                    spearmanr(reference_tree_steps, candidate_tree_steps).statistic
                ),
                "mean_top5_neighbor_jaccard_vs_primary": float(
                    np.mean(list(neighbour_jaccards.values()))
                ),
                "median_top5_neighbor_jaccard_vs_primary": float(
                    np.median(list(neighbour_jaccards.values()))
                ),
                "strongest_neighbor_agreement_vs_primary": float(
                    np.mean(
                        [
                            reference["strongest_neighbor"][topic]
                            == geometry["strongest_neighbor"][topic]
                            for topic in topics
                        ]
                    )
                ),
                **{
                    f"{metric}_rank_spearman_vs_primary": value
                    for metric, value in centrality_correlations.items()
                },
            }
        )

        for topic in topics:
            node_rows.append(
                {
                    "category_treatment": name,
                    "topic": topic,
                    "top5_neighbor_jaccard_vs_primary": neighbour_jaccards[topic],
                    "primary_strongest_neighbor": reference["strongest_neighbor"][topic],
                    "candidate_strongest_neighbor": geometry["strongest_neighbor"][topic],
                    "strongest_neighbor_same": (
                        reference["strongest_neighbor"][topic]
                        == geometry["strongest_neighbor"][topic]
                    ),
                    "primary_strength": float(
                        reference["centralities"]["strength"].loc[topic]
                    ),
                    "candidate_strength": float(
                        geometry["centralities"]["strength"].loc[topic]
                    ),
                    "primary_strength_rank": float(reference_strength_rank.loc[topic]),
                    "candidate_strength_rank": float(candidate_strength_rank.loc[topic]),
                    "absolute_strength_rank_change": float(
                        abs(reference_strength_rank.loc[topic] - candidate_strength_rank.loc[topic])
                    ),
                }
            )

        reference_phi = reference["phi"]
        candidate_phi = geometry["phi"]
        for i, source in enumerate(topics):
            for j in range(i + 1, len(topics)):
                primary_weight = float(reference_phi[i, j])
                candidate_weight = float(candidate_phi[i, j])
                edge_rows.append(
                    {
                        "category_treatment": name,
                        "source": source,
                        "target": topics[j],
                        "primary_proximity": primary_weight,
                        "candidate_proximity": candidate_weight,
                        "proximity_change": candidate_weight - primary_weight,
                        "absolute_proximity_change": abs(candidate_weight - primary_weight),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows)
    payload = {
        "reference": "inferred_primary",
        "distance_definition": "-log(proximity); high proximity is close",
        "display_backbone": "maximum spanning tree plus full-graph edges at or above the 95th weight percentile",
        "summary": summary.to_dict(orient="records"),
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    nodes.to_csv(OUT_NODES, index=False)
    edges.to_csv(OUT_EDGES, index=False)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_NODES}")
    print(f"Wrote {OUT_EDGES}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
