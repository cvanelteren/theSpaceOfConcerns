#!/usr/bin/env python3
"""Test how strongly Figure 1's manual section guides partition the network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import (  # noqa: E402
    REGION_SPECS,
    build_graphs,
    load_topic_meta,
    normalize_topic_key,
)

OUT = ROOT / "output/section_label_diagnostic.json"
N_PERMUTATIONS = 10_000
SEED = 20260814


def main() -> None:
    _, _, graph, counts, _ = build_graphs()
    topics = list(counts.index)
    phi = nx.to_pandas_adjacency(
        graph, nodelist=topics, weight="weight"
    ).to_numpy(float)
    lookup = {normalize_topic_key(topic): topic for topic in topics}
    section_of = {}
    for section, spec in enumerate(REGION_SPECS):
        for raw in spec["nodes"]:
            key = normalize_topic_key(raw)
            if key in lookup:
                section_of[lookup[key]] = section
    for topic in topics:
        section_of.setdefault(topic, len(REGION_SPECS) - 1)

    partitions = [
        {topic for topic in topics if section_of[topic] == section}
        for section in range(len(REGION_SPECS))
    ]
    within, between = [], []
    section_values = [[] for _ in partitions]
    for left in range(len(topics)):
        for right in range(left + 1, len(topics)):
            value = float(phi[left, right])
            if section_of[topics[left]] == section_of[topics[right]]:
                within.append(value)
                section_values[section_of[topics[left]]].append(value)
            else:
                between.append(value)
    observed = float(np.mean(within) - np.mean(between))

    labels = np.asarray([section_of[topic] for topic in topics])
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERMUTATIONS)
    upper = [(left, right) for left in range(len(topics)) for right in range(left + 1, len(topics))]
    for draw in range(N_PERMUTATIONS):
        shuffled = rng.permutation(labels)
        within_draw, between_draw = [], []
        for left, right in upper:
            target = within_draw if shuffled[left] == shuffled[right] else between_draw
            target.append(phi[left, right])
        null[draw] = np.mean(within_draw) - np.mean(between_draw)

    _, old_region, _ = load_topic_meta()
    old_partition = [
        {
            topic
            for topic in topics
            if int(old_region[normalize_topic_key(topic)]) == region
        }
        for region in (1, 2, 3)
    ]
    payload = {
        "interpretation": (
            "The manual sections collect nearer-than-random topics, but their low "
            "weighted modularity does not support drawing them as discrete compartments."
        ),
        "section_sizes": [len(partition) for partition in partitions],
        "mean_within_section_proximity": float(np.mean(within)),
        "mean_between_section_proximity": float(np.mean(between)),
        "within_to_between_ratio": float(np.mean(within) / np.mean(between)),
        "within_minus_between": observed,
        "permutation_upper_tail_p": float(
            (1 + np.count_nonzero(null >= observed)) / (N_PERMUTATIONS + 1)
        ),
        "weighted_modularity_manual_four_sections": float(
            nx.community.modularity(graph, partitions, weight="weight")
        ),
        "weighted_modularity_old_three_regions": float(
            nx.community.modularity(graph, old_partition, weight="weight")
        ),
        "sections": [
            {
                "label": spec["label"].replace("\n", " "),
                "n_topics": len(partitions[index]),
                "mean_internal_proximity": float(np.mean(section_values[index])),
                "median_internal_proximity": float(np.median(section_values[index])),
            }
            for index, spec in enumerate(REGION_SPECS)
        ],
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
