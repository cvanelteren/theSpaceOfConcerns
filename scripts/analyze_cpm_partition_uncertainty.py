"""Actor-bootstrap uncertainty for the seven-community CPM reading."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt

from fig01_space_of_concerns_topology import build_graphs
from fig_cpm_partition_comparison import FINE_COLORS, FINE_NAMES


PARTITIONS = ROOT / "cpm_representative_partitions.csv"
N_BOOTSTRAP = 300
N_RESTARTS = 4
SEED = 20260817


def phi_from_sample(counts: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    sample = counts.to_numpy(dtype=float)[:, indices]
    actor_totals = sample.sum(axis=0)
    concern_totals = sample.sum(axis=1)
    grand_total = sample.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rpa = (sample / actor_totals[None, :]) / (
            concern_totals[:, None] / grand_total
        )
    active = (
        np.nan_to_num(rpa, nan=0.0, posinf=0.0, neginf=0.0) >= 1
    ).astype(np.int16)
    cooccurrence = active @ active.T
    ubiquity = active.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        proximity = np.minimum(
            cooccurrence / ubiquity[:, None],
            cooccurrence / ubiquity[None, :],
        )
    proximity = np.nan_to_num(proximity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(proximity, 0.0)
    return proximity


def best_cpm(adjacency: np.ndarray, gamma: float) -> np.ndarray:
    graph = ig.Graph.Weighted_Adjacency(
        adjacency.tolist(), mode="undirected", attr="weight", loops=False
    )
    candidates = []
    for seed in range(N_RESTARTS):
        partition = leidenalg.find_partition(
            graph,
            leidenalg.CPMVertexPartition,
            weights="weight",
            resolution_parameter=float(gamma),
            n_iterations=-1,
            seed=seed,
        )
        candidates.append((float(partition.quality()), np.asarray(partition.membership)))
    return max(candidates, key=lambda item: item[0])[1]


def bootstrap_consensus(counts, gamma, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n_nodes, n_actors = counts.shape
    consensus = np.zeros((n_nodes, n_nodes), dtype=float)
    labels_by_bootstrap = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_actors, size=n_actors)
        labels = best_cpm(phi_from_sample(counts, indices), gamma)
        consensus += labels[:, None].eq(labels[None, :]) if hasattr(labels, "eq") else (
            labels[:, None] == labels[None, :]
        )
        labels_by_bootstrap.append(labels)

    consensus /= n_bootstrap
    return consensus, np.asarray(labels_by_bootstrap)


def group_consensus(consensus, reference_labels, groups):
    matrix = np.zeros((len(groups), len(groups)), dtype=float)
    for row, group_a in enumerate(groups):
        a = np.flatnonzero(reference_labels == group_a)
        for col, group_b in enumerate(groups):
            b = np.flatnonzero(reference_labels == group_b)
            block = consensus[np.ix_(a, b)]
            if group_a == group_b:
                block = block[~np.eye(len(a), dtype=bool)]
            matrix[row, col] = float(block.mean()) if block.size else 1.0
    return matrix


def node_uncertainty(consensus, reference_labels, groups, nodes):
    rows = []
    for index, node in enumerate(nodes):
        own_group = reference_labels[index]
        affinities = {}
        for group in groups:
            peers = np.flatnonzero(reference_labels == group)
            if group == own_group:
                peers = peers[peers != index]
            affinities[group] = (
                float(consensus[index, peers].mean()) if len(peers) else 1.0
            )
        alternatives = {group: value for group, value in affinities.items() if group != own_group}
        best_alternative = max(alternatives, key=alternatives.get)
        rows.append(
            {
                "concern": node,
                "reference_community": own_group,
                "own_community_coassignment": affinities[own_group],
                "best_alternative_community": best_alternative,
                "best_alternative_coassignment": alternatives[best_alternative],
                "stability_margin": affinities[own_group] - alternatives[best_alternative],
            }
        )
    return pd.DataFrame(rows).sort_values("stability_margin")


def group_fragmentation(labels_by_bootstrap, reference_labels, groups):
    rows = []
    for group in groups:
        members = np.flatnonzero(reference_labels == group)
        fragment_counts = np.array(
            [len(np.unique(labels[members])) for labels in labels_by_bootstrap]
        )
        rows.append(
            {
                "community": group,
                "size": len(members),
                "intact_probability": float(np.mean(fragment_counts == 1)),
                "median_fragments": float(np.median(fragment_counts)),
                "fragment_p95": float(np.quantile(fragment_counts, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def plot_uncertainty(group_matrix, fragmentation, nodes):
    fig, axs = uplt.subplots(ncols=3, figsize=(12.2, 4.35), share=False)
    groups = fragmentation["community"].to_numpy()

    mesh = axs[0].pcolormesh(
        np.arange(len(groups) + 1),
        np.arange(len(groups) + 1),
        group_matrix,
        cmap="batlow",
        vmin=0,
        vmax=1,
    )
    for row in range(len(groups)):
        for col in range(len(groups)):
            value = group_matrix[row, col]
            axs[0].text(
                col + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color="white" if value < 0.42 else "#1E252B",
            )
    axs[0].format(
        title="Regions mostly split, rather than merge",
        xlabel="Reference community",
        ylabel="Reference community",
        xlim=(0, len(groups)),
        ylim=(len(groups), 0),
        xticks=np.arange(len(groups)) + 0.5,
        yticks=np.arange(len(groups)) + 0.5,
        xticklabels=[str(group) for group in groups],
        yticklabels=[str(group) for group in groups],
        grid=False,
    )
    fig.colorbar(mesh, loc="b", label="Bootstrap co-assignment", span=1)

    colors = [FINE_COLORS[group] for group in groups]
    axs[1].bar(
        groups,
        fragmentation["intact_probability"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    axs[1].format(
        title="Probability a whole region stays intact",
        xlabel="Reference community",
        ylabel="Proportion of actor bootstraps",
        ylim=(0, 1),
        xlocator=1,
        grid=False,
    )

    least_stable = nodes.head(12).sort_values("stability_margin", ascending=False)
    y = np.arange(len(least_stable))
    for position, (_, row) in zip(y, least_stable.iterrows()):
        axs[2].plot(
            [row["best_alternative_coassignment"], row["own_community_coassignment"]],
            [position, position],
            color="#AAB1B8",
            lw=1.2,
            zorder=1,
        )
        axs[2].scatter(
            row["best_alternative_coassignment"],
            position,
            s=31,
            color=FINE_COLORS[row["best_alternative_community"]],
            marker="x",
            zorder=2,
        )
        axs[2].scatter(
            row["own_community_coassignment"],
            position,
            s=42,
            color=FINE_COLORS[row["reference_community"]],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    axs[2].format(
        title="Least stable concerns",
        xlabel="Mean bootstrap co-assignment",
        xlim=(0, 1),
        yticks=y,
        yticklabels=least_stable["concern"].str.replace("_", " ").tolist(),
        grid=False,
    )
    axs[2].text(
        0.02,
        0.02,
        "● assigned region     × strongest alternative",
        transform=axs[2].transAxes,
        fontsize=7.4,
        color="#4B535A",
    )

    for ax in axs:
        ax.format(abc=True, abcloc="ul")
    return fig


def main():
    _, _, graph, counts, _ = build_graphs()
    nodes = list(graph.nodes())
    partitions = pd.read_csv(PARTITIONS)
    gamma = min(partitions["gamma"].unique(), key=lambda value: abs(value - 0.273))
    reference = (
        partitions[partitions["gamma"].eq(gamma)]
        .set_index("concern")
        .reindex(nodes)
    )
    reference_labels = reference["community"].to_numpy(dtype=int)
    groups = np.sort(np.unique(reference_labels))
    counts = counts.reindex(nodes)

    consensus, bootstrap_labels = bootstrap_consensus(counts, gamma)
    group_matrix = group_consensus(consensus, reference_labels, groups)
    nodes_frame = node_uncertainty(consensus, reference_labels, groups, nodes)
    fragmentation = group_fragmentation(bootstrap_labels, reference_labels, groups)

    pd.DataFrame(consensus, index=nodes, columns=nodes).to_csv(
        ROOT / "cpm_k7_bootstrap_coassignment.csv"
    )
    pd.DataFrame(group_matrix, index=groups, columns=groups).to_csv(
        ROOT / "cpm_k7_group_coassignment.csv"
    )
    nodes_frame.to_csv(ROOT / "cpm_k7_node_uncertainty.csv", index=False)
    fragmentation.to_csv(ROOT / "cpm_k7_group_fragmentation.csv", index=False)

    figure = plot_uncertainty(group_matrix, fragmentation, nodes_frame)
    figure.savefig(ROOT / "cpm_k7_uncertainty.pdf", bbox_inches="tight")
    figure.savefig(
        ROOT / "figures" / "figS09b_cpm_k7_uncertainty.pdf",
        bbox_inches="tight",
    )

    print("Group fragmentation")
    print(fragmentation.round(3).to_string(index=False))
    print("\nLeast stable concerns")
    print(nodes_frame.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
