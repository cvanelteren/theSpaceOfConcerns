from io import BytesIO
from pathlib import Path

import countryflag
import numpy as np
import pandas as pd
import pycountry
import requests
import ultraplot as plt
from imojify import imojify
from PIL import Image
from scipy import stats


def get_country_code(country_name):
    """Convert country name to 2-letter country code"""
    country_name = country_name.strip()
    country = pycountry.countries.get(name=country_name)
    if country is None:
        return None
    return country.alpha_2.lower()


def load_flag(name, save=True, base="./figures/flags"):
    """
    Load flag from url (and download), or load from disk
    """
    dir = Path(base)
    # Check if country flag exists
    if "korea (rok)" in name.lower():
        name = "Korea"  # contains (ROK)
    elif "korea (dprk)" in name.lower():
        name = "Korea, Democratic People's Republic of"
    if dir / f"{name}_flag.png" in dir.iterdir():
        return plt.pyplot.imread(dir / f"{name}_flag.png")

    # Check if logo organization exists
    if dir / f"{name.lower()}_logo.png" in dir.iterdir():
        return plt.pyplot.imread(dir / f"{name.lower()}_logo.png")

    # Attempt to load
    country_code = get_country_code(name)
    if country_code is None:
        return None
    # Get flat flag from flagcdn.com
    flag_url = f"https://flagcdn.com/w640/{country_code}.png"
    response = requests.get(flag_url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    # Keep the logo?
    if save:
        fig, ax = plt.subplots()
        ax.imshow(img_array, cmap=None)
        ax.axis(False)
        fig.savefig(dir / f"{name}_flag.png", dpi=800, transparent=True)
        plt.close(fig)
    return img_array


def get_country_code(country_name):
    """Convert country name to 2-letter country code"""
    country_name = country_name.strip()
    country = pycountry.countries.get(name=country_name)
    if country is None:
        return None
    return country.alpha_2.lower()


def load_flag(name, save=True, base="./figures/flags"):
    """
    Load flag from url (and download), or load from disk
    """
    dir = Path(base)
    # Check if country flag exists
    if "korea (rok)" in name.lower():
        name = "Korea"  # contains (ROK)
    elif "korea (dprk)" in name.lower():
        name = "Korea, Democratic People's Republic of"
    if dir / f"{name}_flag.png" in dir.iterdir():
        return plt.pyplot.imread(dir / f"{name}_flag.png")

    # Check if logo organization exists
    if dir / f"{name.lower()}_logo.png" in dir.iterdir():
        return plt.pyplot.imread(dir / f"{name.lower()}_logo.png")

    # Attempt to load
    country_code = get_country_code(name)
    if country_code is None:
        return None
    # Get flat flag from flagcdn.com
    flag_url = f"https://flagcdn.com/w640/{country_code}.png"
    response = requests.get(flag_url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    # Keep the logo?
    if save:
        fig, ax = plt.subplots()
        ax.imshow(img_array, cmap=None)
        ax.axis(False)
        fig.savefig(dir / f"{name}_flag.png", dpi=800, transparent=True)
        plt.close(fig)
    return img_array


def get_rca(df):
    """
    Args:
        df: DataFrame with countries as columns, products as rows

    Returns:
        RCA matrix of same shape as input
    """
    # Total exports by country (column sums)
    country_totals = df.sum(axis=0)  # sum over products → total per country

    # Total exports by product (row sums)
    product_totals = df.sum(axis=1)  # sum over countries → total per product

    # Total exports globally (sum of all values)
    total_exports = df.values.sum()

    # Compute RCA
    rca = (df / country_totals) / (
        product_totals.to_numpy()[:, np.newaxis] / total_exports
    )
    return rca


def compute_product_space(rca: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Computes the product space (product proximity matrix) as defined by Hidalgo et al.

    The proximity between products i and j is defined as:
    φ(i,j) = min(P(Xi|Xj), P(Xj|Xi))

    Where:
    - P(Xi|Xj) = probability a country exports product i given it exports product j
    - Xi = 1 if country exports product i (RCA > threshold), 0 otherwise

    Args:
        rca: RCA matrix (products × countries)
        threshold: RCA threshold for defining comparative advantage (default: 1.0)

    Returns:
        Symmetric product proximity matrix (products × products)
    """
    # Binarize RCA matrix: 1 if RCA > threshold, 0 otherwise
    M = (rca > threshold).astype(int).values  # shape: (products × countries)

    # Compute co-occurrence matrix: how many countries export both products i and j
    cooccurrence = M @ M.T  # shape: (products × products)

    # Compute ubiquity: how many countries export each product
    ubiquity = M.sum(axis=1, keepdims=True)  # shape: (products × 1)

    # Compute conditional probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        # P(Xi|Xj): probability of exporting product i given exporting product j
        P_i_given_j = cooccurrence / ubiquity.T  # divide by row (j's ubiquity)

        # P(Xj|Xi): probability of exporting product j given exporting product i
        P_j_given_i = cooccurrence / ubiquity  # divide by column (i's ubiquity)

    # Product proximity: minimum of the two conditional probabilities
    proximity = np.minimum(P_i_given_j, P_j_given_i)

    # Handle NaN values (products with zero ubiquity)
    proximity = np.nan_to_num(proximity, nan=0.0)

    # Set diagonal to 0 (product's proximity to itself is undefined)
    np.fill_diagonal(proximity, 0)

    return pd.DataFrame(proximity, index=rca.index, columns=rca.index)


def compute_space_of_concerns(rca: pd.DataFrame, threshold=1.0) -> pd.DataFrame:
    return compute_product_space(rca, threshold)


def safe_kl_divergence(p, q, epsilon=1e-12):
    """
    Calculates Kullback-Leibler divergence with added epsilon for numerical stability.
    Ensures p and q are properly normalized after adding epsilon.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Add epsilon to avoid log(0) and ensure positive values
    p = p + epsilon
    q = q + epsilon
    # Normalize to sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    return stats.entropy(p, q)


def diffuse_interest(theta_vector, phi_matrix, alpha=0.85, tol=1e-6, max_iter=100):
    """
    Performs the diffusion process for a single agent's interest profile.

    Args:
        theta_vector (np.array): A 1D array representing one agent's interest profile across topics.
                                 Assumed to be normalized (sums to 1) and its length must match phi_matrix dimensions.
        phi_matrix (np.array): The topic-topic transition matrix (rows sum to 1). Its shape must be (num_topics, num_topics).
        alpha (float): Parameter controlling the balance between revealed signal and diffusion.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations for diffusion.

    Returns:
        np.array: The diffused latent interest profile.
    """
    if not (theta_vector.shape[0] == phi_matrix.shape[0] == phi_matrix.shape[1]):
        raise ValueError(
            f"Dimension mismatch in diffuse_interest: theta_vector length ({theta_vector.shape[0]}) "
            f"must match phi_matrix dimensions ({phi_matrix.shape[0]}x{phi_matrix.shape[1]})."
        )

    f = theta_vector.copy()
    for _ in range(max_iter):
        f_new = alpha * theta_vector + (1 - alpha) * (f @ phi_matrix)
        if (
            np.linalg.norm(f_new - f, ord=1) < tol
        ):  # Using L1 norm for convergence check
            break
        f = f_new
    return f


def diffuse_interests(
    rca_norm_df, phi_matrix, alphas, topic_to_idx_map, max_iter=100, tol=1e-6
):
    """
    Calculates latent interest profiles and related metrics for all agents across different alphas.

    Args:
        rca_norm_df (pd.DataFrame): DataFrame of normalized RCAs (agents as index, topics as columns).
                                   Its columns MUST correspond to the topics in topic_to_idx_map and phi_matrix.
        phi_matrix (np.array): The topic-topic transition matrix (rows sum to 1).
                               Its shape must be (num_all_topics, num_all_topics).
        alphas (list/np.array): List of alpha values for diffusion.
        topic_to_idx_map (dict): A mapping from topic name to its column index in phi_matrix.
                                 Must cover ALL topics (len(topic_to_idx_map) == phi_matrix.shape[0]).
        max_iter (int): Maximum iterations for individual diffusion.
        tol (float): Tolerance for individual diffusion convergence.

    Returns:
        pd.DataFrame: A DataFrame containing calculated metrics for each agent and alpha.
    """
    latent_interest_data = []
    num_all_topics = len(topic_to_idx_map)

    # Ensure phi_matrix dimensions are consistent with num_all_topics
    if not (phi_matrix.shape[0] == phi_matrix.shape[1] == num_all_topics):
        raise ValueError(
            f"phi_matrix dimension mismatch with global topics: "
            f"Expected {num_all_topics}x{num_all_topics}, got {phi_matrix.shape[0]}x{phi_matrix.shape[1]}."
        )

    # Pre-calculate one-hot diffusions for all topics across all alphas
    one_hot_diffusions = {alpha: {} for alpha in alphas}

    # Ensure topics are processed in the consistent order matching phi_matrix and topic_to_idx_map
    topics_in_order = sorted(topic_to_idx_map.keys(), key=topic_to_idx_map.get)

    for alpha in alphas:
        for topic_name in topics_in_order:
            theta_topic_one_hot = np.zeros(num_all_topics)
            idx = topic_to_idx_map[topic_name]
            theta_topic_one_hot[idx] = 1.0

            f_one_hot = diffuse_interest(
                theta_topic_one_hot, phi_matrix, alpha=alpha, max_iter=max_iter, tol=tol
            )
            one_hot_diffusions[alpha][topic_name] = f_one_hot

    # Now calculate actual agent diffusions and KL divergences
    for (
        agent_name
    ) in rca_norm_df.columns:  # Iterate through agents (rows of rca_norm_df)
        assert agent_name in rca_norm_df.columns, rca_norm_df.columns
        theta_agent_profile = np.array(
            [rca_norm_df[agent_name][t] for t in topics_in_order]
        )
        e = stats.entropy(theta_agent_profile + 1e-12)

        if np.isclose(np.sum(theta_agent_profile), 0) or np.isclose(e, 0):
            for alpha in alphas:
                row = dict(
                    agent=agent_name,
                    alpha=alpha,
                    f=np.full(num_all_topics, np.nan),
                    kl=np.nan,
                    kl_norm=np.nan,
                    baseline_kl=np.nan,
                    kl_excess=np.nan,
                    p=np.nan,
                    e=0.0,
                )
                latent_interest_data.append(row)
            continue

        for alpha in alphas:
            f_a = diffuse_interest(
                theta_agent_profile, phi_matrix, alpha=alpha, max_iter=max_iter, tol=tol
            )
            kl = safe_kl_divergence(f_a, theta_agent_profile)

            baseline_kl = 0.0
            tmp = rca_norm_df.loc[:, agent_name]

            for topic_name in topics_in_order:
                rca_weight = tmp[topic_name]
                if rca_weight > 0:

                    f_t = one_hot_diffusions[alpha][topic_name]
                    idx_original_topic = topic_to_idx_map[topic_name]
                    p_t = f_t[idx_original_topic]
                    baseline_kl += rca_weight * (-np.log(p_t + 1e-12))

            kl_norm = kl / e if e > 0 else np.nan
            kl_excess = kl - baseline_kl
            p_val = f_a.T @ phi_matrix @ f_a

            row = dict(
                agent=agent_name,
                alpha=alpha,
                f=f_a,
                kl=kl,
                kl_norm=kl_norm,
                baseline_kl=baseline_kl,
                kl_excess=kl_excess,
                p=p_val,
                e=e,
            )
            latent_interest_data.append(row)

    latent_df = pd.DataFrame(latent_interest_data)
    return latent_df


def compute_rca_for_time_point(df_cumulative, all_countries, all_topics):
    """
    Computes the raw submission counts and then RCA for a given cumulative DataFrame.
    Ensures the output DataFrame has consistent dimensions (all_countries x all_topics).
    This function name is generic, but here it processes the full aggregate data.
    """
    submission_counts = {
        country: {topic: 0 for topic in all_topics} for country in all_countries
    }

    for _, row in df_cumulative.dropna(subset="category").iterrows():
        categories = row["category"].split("\t")
        cleaned_categories = [
            topic.replace("envirom", "environ").strip()
            for topic in categories
            if topic.replace("envirom", "environ").strip() in all_topics
        ]

        submitters = row["submitted by"].split(",")
        cleaned_submitters = [
            country.strip()
            for country in submitters
            if country.strip() in all_countries
        ]

        for topic in cleaned_categories:
            for country in cleaned_submitters:
                submission_counts[country][topic] += 1

    rcaf_df_raw_counts = pd.DataFrame.from_dict(
        submission_counts, orient="index", columns=all_topics
    )
    rcaf_df_raw_counts = rcaf_df_raw_counts.reindex(all_countries, axis=0).fillna(0)

    rcaf = get_rca(rcaf_df_raw_counts)
    rcaf = rcaf.reindex(columns=all_topics, fill_value=0).reindex(
        index=all_countries, fill_value=0
    )

    return rcaf


def disparity_filter(G, alpha=0.05, weight="weight"):
    """
    Apply the disparity filter to extract the backbone of a weighted network.

    Parameters
    ----------
    G : networkx.Graph
        Input weighted graph.
    alpha : float, default=0.05
        Significance level for edge filtering.
    weight : str, default='weight'
        Edge attribute name containing the edge weights.

    Returns
    -------
    networkx.Graph
        Filtered graph containing only statistically significant edges.
    """

    # Copy the graph to preserve the original
    backbone = G.copy()

    # Compute the strength of each node (sum of weights of incident edges)
    node_strengths = dict(G.degree(weight=weight))

    # Function to compute the alpha_ij significance level
    def compute_alpha(p_ij, k):
        if k <= 1:
            return 1.0
        integral, _ = quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)
        return 1 - (k - 1) * integral

    # List of edges to remove
    edges_to_remove = []

    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        s_u = node_strengths.get(u, 0.0)
        s_v = node_strengths.get(v, 0.0)
        k_u = G.degree(u)
        k_v = G.degree(v)

        p_ij_u = w / s_u if s_u > 0 else 0.0
        p_ij_v = w / s_v if s_v > 0 else 0.0

        alpha_u = compute_alpha(p_ij_u, k_u)
        alpha_v = compute_alpha(p_ij_v, k_v)

        # Keep edge only if it is significant from both perspectives
        if max(alpha_u, alpha_v) >= alpha:
            edges_to_remove.append((u, v))

    # Remove edges not passing the significance test
    backbone.remove_edges_from(edges_to_remove)

    print(f"Original edges: {G.number_of_edges()}")
    print(f"Backbone edges: {backbone.number_of_edges()}")
    print(
        f"Edges removed: {len(edges_to_remove)} "
        f"({len(edges_to_remove) / G.number_of_edges() * 100:.1f}%)"
    )

    return backbone


def estimate_surround(pos, offset=1 / 3 * np.pi, padding=100, countries=[]):
    # Center layout
    p = np.asarray(list(pos.values()))
    avg = p.mean(0)
    pos = {k: np.array(v) - avg for k, v in pos.items()}

    # Define quadrant anchor points (ideal directions)
    quad_theta = np.linspace(0, 2 * np.pi, 4, 0)
    quads = np.stack([np.cos(quad_theta), np.sin(quad_theta)], axis=1)

    # Group countries into quadrants based on their neighbor's position
    quad_members = {}
    for country in countries:
        if country not in tree:
            continue  # skip if not yet connected

        neighbor = next(tree.neighbors(country))
        target = np.array(pos[neighbor])
        dists = np.linalg.norm(quads - target, axis=1)
        q = np.argmin(dists)
        quad_members.setdefault(q, []).append(country)
        tree.add_node(country)  # add country to the tree if not yet present

    # Assign countries on semicircles around quadrant anchors
    # Estimate radius dynamically from current layout
    existing_positions = np.array(list(pos.values()))
    max_distance = np.linalg.norm(existing_positions, axis=1).max()
    radius = max_distance + padding

    # Assign countries on semicircles around quadrant anchors
    for q, members in quad_members.items():
        anchor_theta = quad_theta[q]
        anchor_vec = np.array([np.cos(anchor_theta), np.sin(anchor_theta)])

        # Compute angles of neighbors relative to the anchor direction
        neighbor_angles = []
        for m in members:
            neighbor = next(tree.neighbors(m))
            target_vec = pos[neighbor]
            rel_vec = target_vec / np.linalg.norm(target_vec)
            angle = np.arctan2(rel_vec[1], rel_vec[0]) - anchor_theta
            neighbor_angles.append(angle)

        # Sort countries by these relative angles
        sorted_members = [m for _, m in sorted(zip(neighbor_angles, members))]

        # Create arc of candidate positions
        angles = (
            np.linspace(-(np.pi / 2 - offset), np.pi / 2 - offset, len(sorted_members))
            + anchor_theta
        )
        semicircle_positions = radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

        for m, p in zip(sorted_members, semicircle_positions):
            pos[m] = p

    return pos


def smart_wrap(text, width=10, max_lines=3):
    """Wrap text with more control"""
    import textwrap

    wrapped = textwrap.fill(
        str(text), width=width, max_lines=max_lines, placeholder="..."
    )
    return wrapped


def nx_layout(graph, layout):
    data = [[node] + list(layout[node]) for node in graph.nodes]

    nodes = pd.DataFrame(data, columns=["id", "x", "y"])
    nodes.set_index("id", inplace=True)

    edges = pd.DataFrame(list(graph.edges), columns=["source", "target"])
    return nodes, edges
