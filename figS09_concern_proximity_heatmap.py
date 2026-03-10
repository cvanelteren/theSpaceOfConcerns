"""Supplementary Figure S09. Provides the full topic-topic proximity matrix of the concern space. Gives a direct matrix view of the joint-interest structure summarized by the network layout."""

# %%
from pathlib import Path

import numpy as np
import ultraplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from utils import compute_product_space, get_rca, load_data


def load_data_with_fallback():
    paths = [
        Path("antarctic-database-go/data/processed/document-summary.parquet"),
        Path(
            "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"
        ),
        Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
        Path("document-summary.csv"),
    ]
    last_err = None
    for path in paths:
        if not path.exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_err = exc
            print(f"Failed to load {path}: {exc}")
    if last_err:
        raise RuntimeError("No usable data file found for fig6.") from last_err
    raise FileNotFoundError("No data file found for fig6.")


counts_df, submitted_df, countries, topics = load_data_with_fallback()
rcaf = get_rca(counts_df)
phi = compute_product_space(rcaf)

# Compute Fiedler vector for ordering
D = squareform(pdist(phi.to_numpy(), metric="euclidean"))
L, _ = laplacian(D, normed=True, return_diag=True)
_, eigvecs = eigh(L)
fiedler = eigvecs[:, 1]

# Cluster and reorder into visual blocks
k = 3
phi_np = phi.to_numpy()
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
labels_raw = kmeans.fit_predict(fiedler.reshape(-1, 1))

# KMeans labels are arbitrary; remap them by cluster center on the Fiedler axis.
cluster_order = np.argsort(kmeans.cluster_centers_.ravel())
label_map = {old_label: new_label for new_label, old_label in enumerate(cluster_order)}
labels = np.array([label_map[label] for label in labels_raw])

dist_from_phi = np.clip(1 - phi_np, 0, None)
np.fill_diagonal(dist_from_phi, 0)


def _order_within_cluster(cluster_idx):
    if cluster_idx.size <= 2:
        return cluster_idx[np.argsort(fiedler[cluster_idx])]
    cluster_dist = dist_from_phi[np.ix_(cluster_idx, cluster_idx)]
    condensed = squareform(cluster_dist, checks=False)
    if np.allclose(condensed, 0):
        return cluster_idx[np.argsort(fiedler[cluster_idx])]
    tree = linkage(condensed, method="average")
    tree = optimal_leaf_ordering(tree, condensed)
    return cluster_idx[leaves_list(tree)]


ordered_blocks = []
block_sizes = []
for cluster_id in range(k):
    cluster_idx = np.where(labels == cluster_id)[0]
    ordered_cluster = _order_within_cluster(cluster_idx)
    ordered_blocks.append(ordered_cluster)
    block_sizes.append(ordered_cluster.size)

sort_idx = np.concatenate(ordered_blocks)
block_boundaries = np.cumsum(block_sizes)[:-1]

phi_clustered = phi_np[sort_idx][:, sort_idx]
fig, ax = plt.subplots(
    width="15cm",
    height="16.8cm",
    left="22em",
    right="5em",
    bottom="20em",
    top="2em",
)
ax.heatmap(
    phi_clustered,
    vmin=0,
    vmax=1,
    colorbar="r",
    colorbar_kw=dict(title=r"Concern proximity $\phi(i, j)$"),
    cmap="oranges",
)

sorted_concerns = [str(i) for i in phi.columns[sort_idx]]
ticks = np.arange(len(sorted_concerns))

fs = 5.5
ax.format(
    xticks=ticks,
    xticklabels=sorted_concerns,
    yticks=ticks,
    yticklabels=sorted_concerns,
    xrotation=90,
    xticklabelsize=fs,
    yticklabelsize=fs,
    xlabel="Ordered topics",
)
# for tick in ax[0].get_yticklabels():
# tick.set_fontsize(fs)

# for tick in ax[0].get_xticklabels():
#     tick.set_fontsize(fs)
#     tick.set_rotation(45)
#     tick.set_rotation_mode("anchor")
#     tick.set_ha("right")

# Draw separators to make cluster blocks explicit.
for boundary in block_boundaries:
    ax.axhline(boundary - 0.5, color="black", lw=0.3, alpha=0.6, ls="--")
    ax.axvline(boundary - 0.5, color="black", lw=0.3, alpha=0.6, ls="--")

ax.xaxis.labelpad = 16
fig.savefig("./figures/figS09_concern_proximity_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("./figures/figS09_concern_proximity_heatmap.png", dpi=320, bbox_inches="tight", pad_inches=0.02)
