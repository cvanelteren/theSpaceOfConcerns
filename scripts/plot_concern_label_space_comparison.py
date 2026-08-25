#!/usr/bin/env python3
"""Compare co-specialization spaces at the paper and register-label resolutions.

The left panel uses the 45 Secretariat concern labels attached to submitted
papers. The right panel uses the 15 official instrument-register categories
after paper labels are mapped to those categories. Both networks use the same
actor co-specialization proximity and are computed directly from paper data.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle  # noqa: E402
from scripts.analyze_output_category_families import paper_family_relations  # noqa: E402
from scripts.official_regular_atcm_outputs import (  # noqa: E402
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
)
from scripts.primary_concern_sensitivity import corpus_objects, variants  # noqa: E402
from utils import (  # noqa: E402
    compute_product_space,
    extract_unique_topics,
    get_rca,
    standardize_index_labels,
)

OUT_PDF = ROOT / "figures" / "exploratory_concern_vs_secretariat_label_space.pdf"
OUT_PNG = ROOT / "figures" / "exploratory_concern_vs_secretariat_label_space.png"
OUT_SUMMARY = ROOT / "output" / "concern_vs_secretariat_label_space_summary.json"


def full_concern_space() -> tuple[list[str], pd.DataFrame, pd.Series]:
    submitted = variants()["fractional_multilabel"]
    topics = sorted(
        standardize_index_labels(
            pd.DataFrame(index=sorted(extract_unique_topics(submitted)))
        ).index
    )
    topics = [topic for topic in topics if topic.strip().lower() not in {"all", "other"}]
    counts, phi_values, _, _, _ = corpus_objects(submitted, topics)
    phi = pd.DataFrame(phi_values, index=topics, columns=topics)
    return topics, phi, counts.sum(axis=1).reindex(topics)


def register_label_space() -> tuple[list[str], pd.DataFrame, pd.Series]:
    relations = paper_family_relations(variants()["fractional_multilabel"])
    labels = sorted(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values())
    labels = list(dict.fromkeys(labels))
    actors = sorted(relations["actor"].unique())
    counts = (
        relations.groupby(["family", "actor"])["paper_weight"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=labels, columns=actors, fill_value=0.0)
    )
    phi = compute_product_space(get_rca(counts)).reindex(
        index=labels, columns=labels, fill_value=0.0
    )
    np.fill_diagonal(phi.values, 1.0)
    paper_mass = (
        relations.drop_duplicates(["paper_id", "family"])
        .groupby("family")["paper_weight"]
        .sum()
        .reindex(labels, fill_value=0.0)
    )
    return labels, phi, paper_mass


def graph_from_phi(labels: list[str], phi: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            weight = float(phi.loc[left, right])
            if weight > 0:
                graph.add_edge(
                    left,
                    right,
                    weight=weight,
                    distance=float(-np.log(np.clip(weight, 1e-12, 1.0))),
                )
    return graph


def display_backbone(graph: nx.Graph, neighbours: int) -> nx.Graph:
    """Retain a common, sparse display rule while preserving connectivity."""
    backbone = nx.maximum_spanning_tree(graph, weight="weight")
    for node in graph.nodes:
        ranked = sorted(
            graph.edges(node, data=True), key=lambda edge: edge[2]["weight"], reverse=True
        )
        for left, right, data in ranked[:neighbours]:
            backbone.add_edge(left, right, **data)
    return backbone


def stable_layout(graph: nx.Graph, labels: list[str]) -> dict[str, np.ndarray]:
    positions = nx.spring_layout(graph, weight="weight", seed=17, k=1.25, iterations=800)
    coordinates = np.asarray([positions[label] for label in labels])
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vectors.T
    # Fix reflections so repeated runs place alphabetically early labels in a
    # stable direction without assigning meaning to either axis.
    if rotated[0, 0] > rotated[-1, 0]:
        rotated[:, 0] *= -1
    if rotated[0, 1] > rotated[-1, 1]:
        rotated[:, 1] *= -1
    return {label: rotated[index] for index, label in enumerate(labels)}


def label_text(label: str, width: int) -> str:
    replacements = {
        "Operation of the Antarctic Treaty system": "Treaty-system",
        "Environmental impact assessment": "Environmental impact\nassessment",
        "Tourism and Non-Governmental Activities": "Tourism and\nnon-governmental activity",
        "Area protection and management": "Area protection\nand management",
        "Institutional & legal matters": "Institutional and\nlegal matters",
        "Waste disposal and management": "Waste disposal\nand management",
        "Marine living resources": "Marine living\nresources",
    }
    if label in replacements:
        return replacements[label]
    return "\n".join(textwrap.wrap(label.replace("_", " "), width=width))


def draw_space(
    ax,
    *,
    labels: list[str],
    graph: nx.Graph,
    backbone: nx.Graph,
    mass: pd.Series,
    title: str,
    node_color: str,
) -> dict[str, float]:
    positions = stable_layout(graph, labels)
    all_weights = np.asarray([data["weight"] for _, _, data in graph.edges(data=True)])
    max_weight = float(all_weights.max())

    for left, right, data in backbone.edges(data=True):
        line_width = 0.35 + 2.2 * float(data["weight"]) / max_weight
        ax.plot(
            [positions[left][0], positions[right][0]],
            [positions[left][1], positions[right][1]],
            color="#AAB5C1",
            lw=line_width,
            alpha=0.72,
            zorder=1,
        )

    values = mass.reindex(labels).to_numpy(dtype=float)
    node_sizes = 60 + 570 * np.sqrt(values / values.max())
    ax.scatter(
        [positions[label][0] for label in labels],
        [positions[label][1] for label in labels],
        s=node_sizes,
        color=node_color,
        edgecolor="white",
        linewidth=0.9,
        zorder=2,
    )

    centre = np.mean(np.asarray(list(positions.values())), axis=0)
    width = 15 if len(labels) > 20 else 19
    fontsize = 5.8 if len(labels) > 20 else 8.0
    offset = 0.045 if len(labels) > 20 else 0.055
    for label in labels:
        point = positions[label]
        direction = point - centre
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm
        horizontal = "left" if direction[0] >= 0 else "right"
        ax.text(
            point[0] + offset * direction[0],
            point[1] + offset * direction[1],
            label_text(label, width),
            fontsize=fontsize,
            color=figstyle.TEXT,
            ha=horizontal,
            va="center",
            linespacing=0.92,
            zorder=3,
        )

    coordinates = np.asarray([positions[label] for label in labels])
    x_padding = 0.52 if len(labels) > 20 else 0.68
    y_padding = 0.34 if len(labels) > 20 else 0.48
    ax.set_xlim(coordinates[:, 0].min() - x_padding, coordinates[:, 0].max() + x_padding)
    ax.set_ylim(coordinates[:, 1].min() - y_padding, coordinates[:, 1].max() + y_padding)
    ax.format(title=title, titleweight="bold", titlecolor=figstyle.TEXT)
    ax.set_axis_off()
    return {
        "nodes": len(labels),
        "positive_edges": graph.number_of_edges(),
        "display_edges": backbone.number_of_edges(),
        "median_proximity": float(np.median(all_weights)),
        "modularity": float(
            nx.community.modularity(
                graph,
                nx.community.louvain_communities(graph, weight="weight", seed=17),
                weight="weight",
            )
        ),
    }


def main() -> None:
    concern_labels, concern_phi, concern_mass = full_concern_space()
    register_labels, register_phi, register_mass = register_label_space()
    concern_graph = graph_from_phi(concern_labels, concern_phi)
    register_graph = graph_from_phi(register_labels, register_phi)

    fig, axes = uplt.subplots(
        ncols=2, refwidth=5.35, refaspect=1.0, share=False, wspace=0.7
    )
    concern_summary = draw_space(
        axes[0],
        labels=concern_labels,
        graph=concern_graph,
        backbone=display_backbone(concern_graph, neighbours=3),
        mass=concern_mass,
        title="45 Secretariat paper concerns",
        node_color="#2C6E9C",
    )
    register_summary = draw_space(
        axes[1],
        labels=register_labels,
        graph=register_graph,
        backbone=display_backbone(register_graph, neighbours=3),
        mass=register_mass,
        title="15 official instrument-register categories",
        node_color="#2A9D8F",
    )
    fig.format(
        abc="a.",
        abcloc="ul",
        abcsize=figstyle.FS_PANEL,
        facecolor="white",
        suptitle="Actor co-specialization at two institutional label resolutions",
        suptitleweight="bold",
        suptitlesize=13,
    )
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    uplt.close(fig)

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "construction": "Actor co-specialization proximity from the same paper archive.",
                "paper_concerns": concern_summary,
                "official_categories": register_summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
