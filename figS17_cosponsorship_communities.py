"""Supplementary Figure S17. Projects Leiden co-sponsorship communities into the concern space. Shows that coordination clusters overlap with, but do not replace, the regime structure."""

# %%
import networkx as nx
import numpy as np
import pandas as pd
import squarify
import ultraplot as uplt
from colorspacious.transform_graph import Edge
from matplotlib.patches import Rectangle
from PIL import Image

from utils import load_data


def add_flags_to_nx(flags, g, pos, ax, output_size=11):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    for node, loc in pos.items():
        img_array = flags.get(node)
        if img_array is not None:
            height, width = img_array.shape[:2]
            zoom = output_size / max(height, width)
            box = OffsetImage(img_array, zoom=zoom, cmap=None)
            ab = AnnotationBbox(
                box,
                loc,
                xybox=(0, 0),
                frameon=True,
                boxcoords="offset points",
                pad=0.0,
                bboxprops=dict(lw=1),
            )
            ax.add_artist(ab)
        else:
            ax.text(
                *loc,
                node,
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transData,
            )


def deterministic_layout(g):
    """Match the base layout used in the main concern-space figure."""
    init = nx.circular_layout(g, scale=0.5)
    return nx.kamada_kawai_layout(g, pos=init, weight="weight")


def tessellate_mask(png_path, stride=6):
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = alpha > 10

    m = mask.astype(np.uint8)
    up = np.pad(m[1:, :], ((0, 1), (0, 0)), mode="constant")
    down = np.pad(m[:-1, :], ((1, 0), (0, 0)), mode="constant")
    left = np.pad(m[:, 1:], ((0, 0), (0, 1)), mode="constant")
    right = np.pad(m[:, :-1], ((0, 0), (1, 0)), mode="constant")
    inner = (m == 1) & (up == 1) & (down == 1) & (left == 1) & (right == 1)

    ys, xs = np.where(inner)
    coords = np.vstack([xs, ys]).T
    coords = coords[(coords[:, 0] % stride == 0) & (coords[:, 1] % stride == 0)]
    if len(coords) == 0:
        raise ValueError("No interior points found for tessellation.")

    coords = coords.astype(float)
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (
        coords[:, 0].max() - coords[:, 0].min()
    )
    coords[:, 1] = (coords[:, 1].max() - coords[:, 1]) / (
        coords[:, 1].max() - coords[:, 1].min()
    )

    aspect = arr.shape[0] / arr.shape[1]
    target_width = 7.0
    target_height = target_width * aspect
    coords[:, 0] = (coords[:, 0] - 0.5) * target_width
    coords[:, 1] = (coords[:, 1] - 0.5) * target_height

    extent = [
        -target_width / 2,
        target_width / 2,
        -target_height / 2,
        target_height / 2,
    ]
    return coords, extent


def snap_to_tessellation(pos, tess_points):
    nodes = list(pos.keys())
    pos_arr = np.array([pos[n] for n in nodes])
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(tess_points)
        _, idx = tree.query(pos_arr, k=1)
    except Exception:
        d = ((pos_arr[:, None, :] - tess_points[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d, axis=1)
    return {n: tess_points[i] for n, i in zip(nodes, idx)}


def main_space_positions(graph):
    """Reproduce the transformed topic coordinates used in the main figure."""
    pos = deterministic_layout(graph)
    fp = "1024px-AntarcticaContour.svg.png"
    try:
        tess_points, mask_extent = tessellate_mask(fp)
        snapped = snap_to_tessellation(pos, tess_points)
        x_min, x_max, y_min, y_max = mask_extent
        snapped = {
            n: np.array([x_min + x_max - p[0], y_min + y_max - p[1]])
            for n, p in snapped.items()
        }
        theta = np.deg2rad(45.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        snapped = {
            n: np.array(
                [
                    cx + (p[0] - cx) * cos_t - (p[1] - cy) * sin_t,
                    cy + (p[0] - cx) * sin_t + (p[1] - cy) * cos_t,
                ]
            )
            for n, p in snapped.items()
        }
        scale = 4
        return {
            n: np.array([cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale])
            for n, p in snapped.items()
        }
    except Exception:
        return {n: np.array(p, dtype=float) * 4.0 for n, p in pos.items()}


def plot_nested_treemap(
    ax,
    topic_fracs,
    communities,
    colors,
    min_threshold=0.1,
    label_threshold=0.001,
    exclude_topics=None,
    add_labels=True,
    add_cluster_labels=True,
):
    """
    Plot a nested treemap showing hierarchical cluster → topic relationships.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    topic_fracs : pd.DataFrame
        DataFrame with topics as index and clusters as columns, containing
        fraction of submissions per topic-cluster combination
    communities : list
        List of community/cluster identifiers (as strings)
    colors : array-like
        Color array for clusters, indexed by cluster ID
    min_threshold : float, optional
        Minimum fraction threshold for including topics in a cluster (default: 0.1)
    label_threshold : float, optional
        Minimum area (dx*dy) threshold for adding topic labels (default: 0.003)
    exclude_topics : list, optional
        List of topic names to exclude from the treemap (default: None)
    add_labels : bool, optional
        Whether to add topic labels (default: True)
    add_cluster_labels : bool, optional
        Whether to add cluster labels (default: True)

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axes object
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Filter out excluded topics if specified
    if exclude_topics is not None:
        topic_fracs = topic_fracs.drop(index=exclude_topics, errors="ignore")

    # Calculate cluster sizes (total submissions per cluster)
    topic_counts = (
        topic_fracs * topic_fracs.sum().sum()
    )  # Convert fractions back to counts
    cluster_sizes = topic_counts.sum(axis=0).to_dict()
    total_submissions = sum(cluster_sizes.values())

    # Normalize cluster sizes
    cluster_sizes_norm = {k: v / total_submissions for k, v in cluster_sizes.items()}

    # Create layout using squarify for clusters
    cluster_list = sorted(
        communities, key=lambda x: cluster_sizes_norm[x], reverse=True
    )
    cluster_rects = squarify.normalize_sizes(
        [cluster_sizes_norm[c] for c in cluster_list], 1, 1
    )
    cluster_positions = squarify.squarify(cluster_rects, 0, 0, 1, 1)

    # Draw each cluster and its topics
    for cluster, rect in zip(cluster_list, cluster_positions):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        cluster_id = int(cluster)
        cluster_color = colors[cluster_id]

        # Draw cluster boundary
        cluster_rect = Rectangle(
            (x, y), dx, dy, linewidth=3, edgecolor="white", facecolor="none"
        )
        ax.add_patch(cluster_rect)

        # Get topics for this cluster (where contribution exceeds threshold)
        cluster_topics = topic_fracs[topic_fracs[cluster] > min_threshold][
            cluster
        ].sort_values(ascending=False)

        if len(cluster_topics) > 0:
            # Normalize topic sizes within cluster
            topic_sizes = cluster_topics.values
            topic_sizes_norm = topic_sizes / topic_sizes.sum()

            # Create sub-treemap for topics within this cluster
            topic_rects = squarify.normalize_sizes(topic_sizes_norm.tolist(), dx, dy)
            topic_positions = squarify.squarify(topic_rects, x, y, dx, dy)

            # Draw each topic
            for topic_name, topic_rect in zip(cluster_topics.index, topic_positions):
                tx, ty, tdx, tdy = (
                    topic_rect["x"],
                    topic_rect["y"],
                    topic_rect["dx"],
                    topic_rect["dy"],
                )

                # Vary shade for topics within cluster based on contribution strength
                topic_alpha = 0.4 + 0.6 * (
                    cluster_topics[topic_name] / cluster_topics.max()
                )

                topic_box = Rectangle(
                    (tx, ty),
                    tdx,
                    tdy,
                    linewidth=0.5,
                    edgecolor="white",
                    facecolor=cluster_color,
                    alpha=topic_alpha,
                )
                ax.add_patch(topic_box)

                # Add label if rectangle is large enough
                if tdx * tdy > label_threshold and add_labels:
                    # Dynamic font sizing based on rectangle size
                    area = tdx * tdy
                    if area > 0.02:
                        fontsize = 7
                        max_chars = 30
                    elif area > 0.01:
                        fontsize = 6
                        max_chars = 25
                    elif area > 0.005:
                        fontsize = 5
                        max_chars = 20
                    else:
                        fontsize = 4
                        max_chars = 15

                    # Wrap text to fit in rectangle
                    import textwrap

                    # Estimate characters per line based on width
                    chars_per_line = int(tdx * 100)  # Rough estimate
                    chars_per_line = max(5, min(chars_per_line, max_chars))

                    wrapped_lines = textwrap.wrap(topic_name, width=chars_per_line)

                    # Limit number of lines based on height
                    max_lines = max(1, int(tdy * 50))  # Rough estimate
                    if len(wrapped_lines) > max_lines:
                        wrapped_lines = wrapped_lines[:max_lines]
                        if len(wrapped_lines) > 0:
                            wrapped_lines[-1] = (
                                wrapped_lines[-1][: chars_per_line - 3] + "..."
                            )

                    label = "\n".join(wrapped_lines)

                    ax.text(
                        tx + tdx / 2,
                        ty + tdy / 2,
                        label,
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        weight="bold",
                        color="white",
                        wrap=False,  # We already wrapped it manually
                    )

        # Add cluster label at the top of each cluster rectangle
        if add_cluster_labels:
            ax.text(
                x + dx / 2,
                y + dy - 0.01,
                f"Cluster {cluster}",
                ha="center",
                va="top",
                fontsize=10,
                weight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    return ax


fp = "./antarctic-database-go/data/processed/document-summary.parquet"
df, submitted, countries, topics = load_data(fp)
# Note: ATS "ALL" category is normalized to "Other" in `utils.load_data`.
countries = sorted(countries)
cluster_topics = {}
c_map = {c: idx for idx, c in enumerate(countries)}
# %%
from utils import load_flag

flags = {}
for country in countries:
    flags[country] = load_flag(country, base="./assets/flags")

# %%

# Get co-occurrence network of countries based on co-submissions
from itertools import combinations

A = np.zeros((len(countries), len(countries)))
for _, row in submitted.dropna(subset=["parties"]).iterrows():

    cleaned = sorted(set(str(p).strip() for p in row["parties"] if str(p).strip()))
    for i, j in combinations(cleaned, 2):
        idx = c_map[i]
        jdx = c_map[j]
        A[idx, jdx] += 1
A += A.T
A_country = pd.DataFrame(A, index=countries, columns=countries)
g = nx.from_pandas_adjacency(A_country)
gc = max(nx.connected_components(g), key=len)
g = g.subgraph(gc)
A_country = nx.to_pandas_adjacency(g)

fig, ax = uplt.subplots(ncols=2, share=0)
ax[0].pcolormesh(np.log1p(A_country.values), colorbar="r")
ax[1].graph(nx.from_pandas_adjacency(A_country), node_kw=dict(node_size=16))
fig.show()
print(A.max(), A.min())


# %%  Run Leiden algorithm
from scipy import sparse
from sknetwork.clustering import Leiden

X = A_country.values
norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
S = (X / norm) @ (X / norm).T
np.fill_diagonal(S, 0)
A_sim = pd.DataFrame(S, index=A_country.index, columns=A_country.columns)
fits = Leiden(random_state=0).fit_predict(sparse.csr_matrix(A_sim.values))

n_clusters = np.unique(fits).size
print(fits, np.unique(fits).size)
# %%


# %% Setup first ax positions
g_country = nx.from_pandas_adjacency(A_country)
node2community = {node: fits[idx] for idx, node in enumerate(g_country.nodes())}
print(node2community)
n_communities = np.unique(fits).size
thetas = np.linspace(0, 2 * np.pi, n_communities, endpoint=False) + 1 / 2 * np.pi
radius = 2.5
sub_radius = 1.25
country_pos, subplot_loc = {}, {}

community_ids = np.unique(fits)
for idx, community in enumerate(community_ids):
    members = [i for i, v in node2community.items() if v == community]
    n_members = len(members)
    sub_angles = np.linspace(0, 2 * np.pi, n_members, endpoint=False)
    center = radius * np.array([np.cos(thetas[idx]), np.sin(thetas[idx])])
    subplot_loc[idx] = (radius) * np.array([np.cos(thetas[idx]), np.sin(thetas[idx])])
    for idx, member in enumerate(members):
        country_pos[member] = center + sub_radius * np.array(
            [np.cos(sub_angles[idx]), np.sin(sub_angles[idx])]
        )
# %% Recode submitted based on the new community ids
tmp = []
for _, row in submitted.iterrows():
    communities = set()
    parties = row.get("parties", [])
    if not isinstance(parties, (list, tuple, set, np.ndarray)):
        parties = []
    for party in parties:
        if party in node2community:
            # Use string labels to match generate_interaction_matrix casting.
            communities.add(str(int(node2community[party])))
    tmp.append(sorted(communities))
submitted_recoded = submitted.copy()
submitted_recoded["parties"] = tmp
submitted_recoded["submitted by"] = submitted_recoded["parties"]
submitted_recoded = submitted_recoded[submitted_recoded["parties"].map(len) > 0]

print(np.unique(submitted_recoded["parties"]))
# %%
from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    standardize_index_labels,
)

communities = [str(i) for i in range(n_communities)]
counts_df = generate_interaction_matrix(submitted_recoded, communities, topics)
final_df = standardize_index_labels(counts_df)
cluster_rca = get_rca(final_df)

counts_df = generate_interaction_matrix(submitted, countries, topics)
final_df = standardize_index_labels(counts_df)
rca_countries = get_rca(final_df)
phi = compute_product_space(rca_countries)
G = nx.from_pandas_adjacency(phi)

mst = nx.maximum_spanning_tree(G)
W = np.array([d["weight"] for u, v, d in G.edges(data=True)])
threshold = np.percentile(W, 95)

POS = main_space_positions(mst)
mst.add_weighted_edges_from(
    [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if d["weight"] >= threshold]
)
print(mst)
edge_widths_panel_c = np.array(
    [float(mst[u][v].get("weight", 1.0)) * 8 for u, v in mst.edges()]
)


# %% Compute cluster interests
cluster_interests = {}
cluster_top_interest = {}
rca_max = float(cluster_rca.max().max())
for cluster in range(n_clusters):
    tmp = cluster_rca[str(cluster)]
    interest = tmp[tmp >= 1].sort_values(ascending=False)
    cluster_interests[cluster] = list(interest.index)
    cluster_top_interest[cluster] = interest.index[0] if not interest.empty else None
    tmp_filled = tmp.fillna(0)
    print(sorted(tmp_filled)[-3:])
    print(cluster_top_interest[cluster])
# %%
# Define thematic color mapping for nodes
theme_colors = {
    "Environmental Protection": "#2E7D32",  # Green
    "Marine & Wildlife": "#0277BD",  # Blue
    "Operations & Safety": "#F57C00",  # Orange
    "Governance & Legal": "#6A1B9A",  # Purple
    "Science & Research": "#C62828",  # Red
    "Tourism & Human Activity": "#D84315",  # Deep Orange
    "Infrastructure & Planning": "#5D4037",  # Brown
    "Resource Extraction": "#00838F",  # Teal
}

# Map each topic to a theme
topic_to_theme = {
    "State of the Antarctic Environment Report SAER": "Environmental Protection",
    "Management Plans": "Governance & Legal",
    "Biological Prospecting": "Resource Extraction",
    "Climate Change": "Environmental Protection",
    "Environmental Domains Analysis": "Environmental Protection",
    "Educational issues": "Governance & Legal",
    "Comprehensive Environmental Evaluations": "Environmental Protection",
    "Site Guidelines for Visitors": "Tourism & Human Activity",
    "Repair and remediation of environmental damage": "Environmental Protection",
    "Multiyear strategic workplan": "Governance & Legal",
    "Inspections": "Operations & Safety",
    "Sub glacial Lakes": "Science & Research",
    "Drilling": "Resource Extraction",
    "Opening statements": "Governance & Legal",
    "Operation of the Antarctic Treaty system General": "Governance & Legal",
    "CEP Strategy Discussions": "Governance & Legal",
    "Fauna and Flora General": "Marine & Wildlife",
    "Historic Sites and Monuments": "Infrastructure & Planning",
    "Operational issues": "Operations & Safety",
    "Operation of the Antarctic Treaty system Reports": "Governance & Legal",
    "Science issues": "Science & Research",
    "International Polar Year": "Science & Research",
    "Liability": "Governance & Legal",
    "Prevention of marine pollution": "Environmental Protection",
    "Safety and Operations in Antarctica": "Operations & Safety",
    "Marine living resources": "Resource Extraction",
    "Institutional and legal matters": "Governance & Legal",
    "Nonnative Species and Quarantine": "Environmental Protection",
    "Marine Protected Areas": "Marine & Wildlife",
    "Tourism and NG Activities": "Tourism & Human Activity",
    "Cooperation with Other Organisations": "Governance & Legal",
    "Environmental Impact Assessment EIA Other EIA Matters": "Environmental Protection",
    "Specially Protected Species": "Marine & Wildlife",
    "Marine Acoustics": "Science & Research",
    "Mineral resources": "Resource Extraction",
    "Environmental Monitoring and Reporting": "Environmental Protection",
    "Exchange of Information": "Governance & Legal",
    "Area Protection and Management Plans General": "Infrastructure & Planning",
    "Operation of the CEP": "Governance & Legal",
    "Waste management and disposal": "Operations & Safety",
    "Human Footprint and wilderness values": "Environmental Protection",
    "Search and Rescue": "Operations & Safety",
    "Environmental Protection General": "Environmental Protection",
    "Emergency report and contingency planning": "Operations & Safety",
    "Operation of the Antarctic Treaty system The Secretariat": "Governance & Legal",
}

# Create color list for nodes
node_colors = [
    theme_colors[topic_to_theme.get(node, "Governance & Legal")] for node in mst.nodes()
]


# %%
fig, ax = uplt.subplots()
ax.pcolormesh(phi, colorbar="r")
fig.show()


# %%

edge_counts_country = np.array([d["weight"] for u, v, d in g_country.edges(data=True)], dtype=float)
edge_widths_country = 0.25 + 2.6 * (
    np.log1p(edge_counts_country) / np.log1p(edge_counts_country.max())
)
import textwrap

from matplotlib import patches
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

# %%

layout = [
    [1, 3, 4],
    [2, 3, 4],
]
fig, ax = uplt.subplots(layout, share=0, refnum=1)
ax.format(abc="[A]")

ax[0].graph(
    g_country,
    country_pos,
    nodes=True,
    node_kw=dict(node_size=14, node_color="0.15", alpha=0.9, linewidths=0),
    edge_kw=dict(alpha=0.38, width=edge_widths_country, edge_color="0.15"),
    rescale=0,
)
add_flags_to_nx(flags, g_country, country_pos, ax[0], output_size=8)
colors = uplt.Colormap("bmh")(np.arange(len(community_ids)))

for clus, loc in subplot_loc.items():
    circle = patches.Circle(
        loc,
        radius=sub_radius * 1.33,
        fill=False,
        color=colors[clus],
        ls="--",
    )
    ax[0].add_artist(circle)

topic_counts = standardize_index_labels(
    generate_interaction_matrix(submitted_recoded, communities, topics)
)
topic_specialized = cluster_rca.where(cluster_rca >= 1.0, 0.0)
topic_specialized = topic_specialized.loc[topic_specialized.sum(axis=1) > 0]
topic_fracs = topic_counts.reindex(topic_specialized.index).fillna(0)
topic_fracs = topic_fracs.div(
    topic_fracs.sum(axis=1).replace(0, np.nan), axis=0
).fillna(0)
topic_order = np.array(list(topic_fracs.index))

ax[2].graph(
    mst,
    POS,
    node_kw=dict(node_size=32, node_color=node_colors),
    edge_kw=dict(width=edge_widths_panel_c),
    rescale=0,
)
xlim = ax[2].get_xlim()
ylim = ax[2].get_ylim()
pad = -0.25
ring_base = 100
ring_step = 110
ring_counts = {}
annotations = {"left": [], "right": [], "bottom": [], "top": []}
for cluster, interests in cluster_interests.items():
    for interest in interests:
        xy = POS[interest]
        ring_idx = ring_counts.get(interest, 0)
        ring_counts[interest] = ring_idx + 1
        rca_val = float(cluster_rca.at[interest, str(cluster)])
        rca_min = 1.0
        lw_min, lw_max = 0.6, 2.6
        if rca_max <= rca_min:
            lw = lw_max
        else:
            lw = lw_min + (rca_val - rca_min) / (rca_max - rca_min) * (lw_max - lw_min)
            lw = max(lw_min, min(lw, lw_max))
        ax[2].scatter(
            *xy,
            edgecolor=colors[cluster],
            linewidths=lw,
            s=ring_base + ring_idx * ring_step,
            color="none",
            ls="--",
        )
        if interest == cluster_top_interest.get(cluster):
            dist_left = abs(xy[0] - xlim[0])
            dist_right = abs(xlim[1] - xy[0])
            dist_bottom = abs(xy[1] - ylim[0])
            dist_top = abs(ylim[1] - xy[1])
            dists = {
                "left": dist_left,
                "right": dist_right,
                "bottom": dist_bottom,
                "top": dist_top,
            }
            side = min(dists, key=dists.get)
            annotations[side].append((xy, interest, cluster))

for side, items in annotations.items():
    if not items:
        continue
    if side in {"left", "right"}:
        items.sort(key=lambda item: item[0][1])
    else:
        items.sort(key=lambda item: item[0][0])
    positions = np.linspace(0.2, 0.8, len(items)) if len(items) > 1 else [0.5]
    for (xy, interest, cluster), pos in zip(items, positions):
        if side == "left":
            label_xy = (-pad, pos)
            ha, va = "right", "center"
        elif side == "right":
            label_xy = (1 + pad, pos)
            ha, va = "left", "center"
        elif side == "bottom":
            label_xy = (pos, -pad)
            ha, va = "center", "top"
        else:
            label_xy = (pos, 1 + pad)
            ha, va = "center", "bottom"
        ax[2].annotate(
            "\n".join(textwrap.wrap(interest, 20)),
            xy=xy,
            xytext=label_xy,
            textcoords="axes fraction",
            fontsize=7,
            color=colors[cluster],
            ha=ha,
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec=colors[cluster],
                lw=0.8,
                alpha=0.9,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=colors[cluster],
                lw=0.8,
                shrinkA=2,
                shrinkB=2,
            ),
            annotation_clip=False,
        )

years = sorted(submitted_recoded["meeting_year"].dropna().unique())
cluster_ids = list(range(len(community_ids)))
cluster_colors = {cid: colors[cid][:3] for cid in cluster_ids}
topic_count = len(topic_order)
year_count = len(years)
max_cluster = np.full((topic_count, year_count), -1, dtype=int)
max_value = np.zeros((topic_count, year_count), dtype=float)

for j, year in enumerate(years):
    df_year = submitted_recoded[submitted_recoded["meeting_year"] == year]
    counts_year = generate_interaction_matrix(df_year, communities, topics)
    final_year = standardize_index_labels(counts_year)
    rca_year = get_rca(final_year)
    rca_mat = np.column_stack(
        [
            rca_year[str(cid)].reindex(topic_order).fillna(0).values
            for cid in cluster_ids
        ]
    )
    rca_mat = np.where(rca_mat >= 1.0, rca_mat, 0.0)
    max_cluster[:, j] = np.argmax(rca_mat, axis=1)
    max_value[:, j] = np.max(rca_mat, axis=1)


def longest_run(labels):
    best_cluster = labels[0]
    best_len = 1
    current_cluster = labels[0]
    current_len = 1
    for value in labels[1:]:
        if value == current_cluster:
            current_len += 1
        else:
            if current_len > best_len:
                best_len = current_len
                best_cluster = current_cluster
            current_cluster = value
            current_len = 1
    if current_len > best_len:
        best_len = current_len
        best_cluster = current_cluster
    return best_cluster, best_len


dominant_runs = [longest_run(max_cluster[i, :]) for i in range(topic_count)]
avg_max_rca = max_value.mean(axis=1)
sort_idx = sorted(
    range(topic_count),
    key=lambda i: (
        dominant_runs[i][0],
        -dominant_runs[i][1],
        -avg_max_rca[i],
        topic_order[i],
    ),
)
max_cluster = max_cluster[sort_idx, :]
max_value = max_value[sort_idx, :]
topic_order = [topic_order[i] for i in sort_idx]

# Now plot ax[1] bar chart with the sorted topic order to match ax[3] heatmap
topic_fracs = topic_fracs.reindex(topic_order)
y = np.arange(len(topic_order))
stack_values = [topic_fracs[cluster].values for cluster in communities]
stack_colors = [colors[int(cluster)] for cluster in communities]
left = np.zeros(len(topic_order), dtype=float)
for vals, color in zip(stack_values, stack_colors):
    right = left + vals
    ax[1].fill_betweenx(y, left, right, color=color, alpha=0.95, step="mid")
    left = right
ax[1].format(
    xlabel="Fraction of submissions",
    yticks=y,
    yticklabels=topic_order,
    xlim=(0, 1),
    ylim=(-0.5, len(topic_order) - 0.5),
)
ax[1].invert_yaxis()
ax[1].tick_params(axis="y", labelsize=5.8)
for label in ax[1].get_yticklabels():
    label.set_verticalalignment("center")

rca_floor = 1.0
global_max = max_value.max()
if global_max <= rca_floor:
    intensity = np.zeros_like(max_value)
else:
    intensity = (max_value - rca_floor) / (global_max - rca_floor)
    intensity = np.clip(intensity, 0, 1)
intensity = np.power(intensity, 0.3)
rgb = np.ones((topic_count, year_count, 3), dtype=float)
for i in range(topic_count):
    for j in range(year_count):
        cid = cluster_ids[max_cluster[i, j]]
        color = cluster_colors[cid]
        rgb[i, j, :] = 1 - intensity[i, j] + intensity[i, j] * color
mesh = ax[3].imshow(
    rgb,
    rasterized=True,
    extent=[-0.5, year_count - 0.5, topic_count - 0.5, -0.5],
)

ax[3].xaxis.set_major_locator(uplt.ticker.mticker.MaxNLocator(nbins=6, integer=True))
ax[3].format(
    xlabel="Year",
    yticks=np.arange(topic_count),
    yticklabels=topic_order,
    title="Dominant cluster by topic over time (RPA > 1 only)",
)
ax[3].invert_yaxis()
ax[3].tick_params(axis="y", labelsize=5.8)
# Explicitly set ytick label alignment to center
for label in ax[3].get_yticklabels():
    label.set_verticalalignment("center")
ax[3].tick_params(axis="x", labelsize=8, rotation=0)
ax[3].set_xticks(np.arange(year_count + 1), minor=True)
ax[3].set_yticks(np.arange(topic_count + 1), minor=True)
ax[3].grid(which="minor", color="white", linewidth=0.3)
ax[3].tick_params(which="minor", length=0)

ax[3].xaxis.set_major_formatter(
    uplt.ticker.mticker.FuncFormatter(
        lambda x, pos: str(years[int(x)]) if 0 <= int(x) < year_count else ""
    )
)

grad_steps = 256
grad = np.linspace(0, 1, grad_steps)
colorbar_img = np.zeros((len(cluster_ids), grad_steps, 3))
for row, cid in enumerate(cluster_ids):
    color = cluster_colors[cid]
    colorbar_img[row, :, :] = (1 - grad)[:, None] + grad[:, None] * color

cb_ax = ax[3].inset_axes([1.07, 0.2, 0.08, 0.6], zoom=0)
cb_ax.imshow(colorbar_img, aspect="auto", origin="lower")
cb_ax.set_yticks(np.arange(len(cluster_ids)))
cb_ax.set_yticklabels([f"Cluster {cid}" for cid in cluster_ids], fontsize=8)
cb_ax.yaxis.tick_right()
cb_ax.set_xticks([0, grad_steps - 1])
cb_ax.set_xticklabels(["low", "high"], fontsize=7, rotation=90)
cb_ax.tick_params(length=0)
cb_ax.set_title("Intensity", fontsize=8.5, pad=2)

legend_elements = [
    Patch(facecolor=color, label=theme) for theme, color in theme_colors.items()
]
ax[2].legend(
    handles=legend_elements,
    loc="lower center",
    fontsize=8,
    framealpha=0.0,
    title="Themes",
    title_fontsize=9,
    ncol=2,  # Use 2 columns to keep legend compact within axis
    bbox_to_anchor=(0.5, 0.0),  # Center at bottom of axis
    borderaxespad=0.5,
)
ax[2].axis("equal")
fig.savefig("./figures/figS17_cosponsorship_communities.pdf")
fig.savefig("./figures/figS17_cosponsorship_communities.png", dpi=320, transparent=True)
fig.show()

# %%
fig, ax = uplt.subplots(journal="nat2")
plot_nested_treemap(
    ax,
    topic_fracs,
    communities,
    colors,
    min_threshold=0.1,
    label_threshold=0.003,
    exclude_topics=["Other"],  # Optionally exclude "Other" (formerly "ALL") if desired
)

# %%

# %%  Run Leiden algorithm
from scipy import sparse
from sknetwork.clustering import Leiden

X = A_country.values
norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
S = (X / norm) @ (X / norm).T
np.fill_diagonal(S, 0)
A_sim = pd.DataFrame(S, index=A_country.index, columns=A_country.columns)
fits = Leiden(random_state=0).fit_predict(sparse.csr_matrix(A_sim.values))

n_clusters = np.unique(fits).size
print(fits, np.unique(fits).size)

# %%
fig, ax = uplt.subplots(ncols=2, nrows=2)
for cluster in range(n_clusters):
    subset = np.where(fits == cluster)[0]
    print(A_sim.shape)
    A_small = A_sim.values[subset][:, subset]

    small_fits = Leiden(random_state=0).fit_predict(sparse.csr_matrix(A_small))

    h = nx.from_numpy_array(A_small)
    nc = []

    l = {idx: A_country.index[c] for idx, c in enumerate(subset)}
    h = nx.relabel_nodes(h, l)
    colors = uplt.Colormap("spectral_r")(np.linspace(0, 1, np.unique(small_fits).size))
    print(h.nodes())
    for idx, node in enumerate(h.nodes()):
        color = colors[small_fits[idx]]
        nc.append(color)
    assert len(h) == len(nc)
    w = np.array([d["weight"] for i, j, d in h.edges(data=True)])
    w /= w.sum()
    w *= 10
    # w = np.log1p(w) * 0.25
    if cluster == 0:
        print(w)

    ax[cluster].graph(
        h, "circular", node_kw=dict(node_color=nc), labels=True, edge_kw=dict(width=w)
    )
fig.savefig("./output/fig16_hierarchical_leiden.png", transparent=True)
