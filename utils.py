from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pycountry
import requests
import ultraplot as plt
from PIL import Image
from scipy import stats
from scipy.integrate import quad  # Added missing import

# ==========================================
# 1. Image & Flag Utilities
# ==========================================


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
    dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Check if country flag exists
    if "korea (rok)" in name.lower():
        name = "Korea"  # contains (ROK)
    elif "korea (dprk)" in name.lower():
        name = "Korea, Democratic People's Republic of"

    if (dir / f"{name}_flag.png").exists():
        return plt.pyplot.imread(dir / f"{name}_flag.png")

    # Check if logo organization exists
    if (dir / f"{name.lower()}_logo.png").exists():
        return plt.pyplot.imread(dir / f"{name.lower()}_logo.png")

    # Attempt to load
    country_code = get_country_code(name)
    if country_code is None:
        return None

    # Get flat flag from flagcdn.com
    flag_url = f"https://flagcdn.com/w640/{country_code}.png"
    try:
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
    except Exception as e:
        print(f"Could not load flag for {name}: {e}")
        return None


# ==========================================
# 2. Data Processing & ETL (Refactored)
# ==========================================


def preprocess_dataframe(fp: str) -> pd.DataFrame:
    """Loads CSV, converts types, and standardizes column names."""
    df = pd.read_csv(fp)
    df = df.convert_dtypes()
    df.columns = df.columns.str.lower()
    return df


def extract_unique_countries(df: pd.DataFrame) -> Set[str]:
    """Parses 'submitted by' column to find all unique countries."""
    countries = set()
    subset = df.dropna(subset=["submitted by"])
    for _, row in subset.iterrows():
        parties = str(row["submitted by"]).split(",")
        for i in parties:
            countries.add(i.strip())
    return countries


def extract_unique_topics(df: pd.DataFrame) -> Set[str]:
    """Parses 'category' column to find all unique topics, handling typos."""
    topics = set()
    subset = df.dropna(subset=["category"])
    for _, row in subset.iterrows():
        for topic in row["category"].split("\t"):
            topic = topic.replace("envirom", "environ")
            topics.add(topic)
    return topics


def generate_interaction_matrix(
    subset_df: pd.DataFrame, all_countries: Set[str], all_topics: Set[str]
) -> pd.DataFrame:
    """
    Vectorized generation of interaction matrix.
    Much faster than the loop-based approach.
    """
    # 1. Filter NaNs
    df = subset_df.dropna(subset=["category", "submitted by"]).copy()
    if df.empty:
        # Return empty matrix with correct shape if no data
        return pd.DataFrame(0, index=list(all_topics), columns=list(all_countries))

    # 2. Vectorized cleaning and splitting
    # Split "category" string into a list, then 'explode'
    df["category"] = df["category"].str.split("\t")
    df = df.explode("category")
    df["category"] = df["category"].str.replace("envirom", "environ")

    # Split "submitted by" string into a list, then 'explode'
    df["submitted by"] = df["submitted by"].astype(str).str.split(",")
    df = df.explode("submitted by")
    df["submitted by"] = df["submitted by"].str.strip()

    # 3. Create the Matrix (Crosstab)
    matrix = pd.crosstab(df["submitted by"], df["category"])

    # 4. Reindex to ensure ALL countries/topics are present (Global Dimensions)
    matrix = matrix.reindex(
        index=list(all_countries), columns=list(all_topics), fill_value=0
    )

    # 5. Transpose (Rows=Topics, Cols=Countries)
    return matrix.T


def standardize_index_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Renames index labels based on a predefined spelling correction map."""
    spelling_correction_mapping = {
        "Marine acoustics": "Marine acoustics",
        "Fauna and flora general": "Fauna and flora general",
        "Specially protocted species": "Specially protected species",
        "Exchange of information": "Exchange of information",
        "Emergency report and contingency plan": "Emergency report and contingency plan",
        "Operational issues": "Operational issues",
        "Sub glacial lakes": "Sub glacial lakes",
        "Repair and remediation of environental damage": "Repair and remediation of environmental damage",
        "Non-native speci es and quarantine": "Non-native species and quarantine",
        "Operation of antarctic treaty system: Reports": "Operation of Antarctic Treaty System: Reports",
        "Mineral resources": "Mineral resources",
        "Enviromental protection general": "Environmental protection general",
        "Human footprint and wilderness values": "Human footprint and wilderness values",
        "Inspections": "Inspections",
        "Liability": "Liability",
        "Climate change": "Climate change",
        "Drilling": "Drilling",
        "Operation of the antarctic treaty system: General": "Operation of the Antarctic Treaty System: General",
        "Marine living resources": "Marine living resources",
        "Enviromental Domains Analysis": "Environmental Domains Analysis",
        "State of the antarctic environent report": "State of the Antarctic environment report",
        "Site guidelines for visitors": "Site guidelines for visitors",
        "Tourism and NG activities": "Tourism and NG activities",
        "Area Management and protection plans: General": "Area Management and protection plans: General",
        "Management Plans": "Management Plans",
        "Operation of antarctic treaty system: The secretariat": "Operation of Antarctic Treaty System: The secretariat",
        "Prevention of marine pollution": "Prevention of marine pollution",
        "Comprehensive environental evaluation": "Comprehensive environmental evaluation",
        "CEP strategy discussions": "CEP strategy discussions",
        "Marine protected areas": "Marine protected areas",
        "Enviromental impact assessment(EIA): other EIA matters": "Environmental impact assessment (EIA): other EIA matters",
        "Historic sites and monuments": "Historic sites and monuments",
        "Waste management and disposal": "Waste management and disposal",
        "Opening statement": "Opening statement",
        "Instututional and legal matters": "Institutional and legal matters",
        "Operation of CEP": "Operation of CEP",
        "Search and rescue": "Search and rescue",
        "Multi-year strategic workplan": "Multi-year strategic workplan",
        "Science issues": "Science issues",
        "International polar year": "International polar year",
        "Biological Prospecting": "Biological Prospecting",
        "Cooperation with other organization": "Cooperation with other organization",
        "Enviromental monitoring and reporting": "Environmental monitoring and reporting",
        "Safety and Operations in antarctica": "Safety and Operations in Antarctica",
        "Educational issues": "Educational issues",
    }
    return df.rename(index=spelling_correction_mapping)


def load_data(fp) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str], Set[str]]:
    """
    Orchestrator function that pipelines the separate operators.
    Returns: Final_Counts, Raw_DF, Countries_Set, Topics_Set
    """
    submitted_df = preprocess_dataframe(fp)
    countries = extract_unique_countries(submitted_df)
    topics = extract_unique_topics(submitted_df)

    counts_df = generate_interaction_matrix(submitted_df, countries, topics)
    final_df = standardize_index_labels(counts_df)

    # UPDATED: Now returns 'topics' as well, which is crucial for the Time Block analysis
    return final_df, submitted_df, countries, topics


# ==========================================
# 3. RCA & Economic Complexity
# ==========================================


def get_rca(df):
    """
    Args:
        df: DataFrame with countries as columns, products as rows
    Returns:
        RCA matrix of same shape as input
    """
    df_filled = df.fillna(0)  # Safety
    country_totals = df_filled.sum(axis=0)
    product_totals = df_filled.sum(axis=1)
    total_exports = df_filled.values.sum()

    # Safety against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        rca = (df_filled / country_totals) / (
            product_totals.to_numpy()[:, np.newaxis] / total_exports
        )
    return rca.fillna(0)


def compute_product_space(rca: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Computes the product space (product proximity matrix)"""
    M = (rca > threshold).astype(int).values
    cooccurrence = M @ M.T
    ubiquity = M.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        P_i_given_j = cooccurrence / ubiquity.T
        P_j_given_i = cooccurrence / ubiquity

    proximity = np.minimum(P_i_given_j, P_j_given_i)
    proximity = np.nan_to_num(proximity, nan=0.0)
    np.fill_diagonal(proximity, 0)

    return pd.DataFrame(proximity, index=rca.index, columns=rca.index)


def compute_space_of_concerns(rca: pd.DataFrame, threshold=1.0) -> pd.DataFrame:
    return compute_product_space(rca, threshold)


def compute_rca_for_time_point(df_cumulative, all_countries, all_topics):
    """
    Refactored to use the modular utilities instead of duplicated logic.
    """
    # 1. Generate matrix
    interaction = generate_interaction_matrix(df_cumulative, all_countries, all_topics)
    # 2. Get RCA
    return get_rca(interaction)


# ==========================================
# 4. Diffusion & Interest Logic
# ==========================================


def safe_kl_divergence(p, q, epsilon=1e-12):
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return stats.entropy(p, q)


def diffuse_interest(theta_vector, phi_matrix, alpha=0.85, tol=1e-6, max_iter=100):
    if not (theta_vector.shape[0] == phi_matrix.shape[0] == phi_matrix.shape[1]):
        raise ValueError("Dimension mismatch in diffuse_interest")

    f = theta_vector.copy()
    for _ in range(max_iter):
        f_new = alpha * theta_vector + (1 - alpha) * (f @ phi_matrix)
        if np.linalg.norm(f_new - f, ord=1) < tol:
            break
        f = f_new
    return f


def diffuse_interests(
    rca_norm_df, phi_matrix, alphas, topic_to_idx_map, max_iter=100, tol=1e-6
):
    """
    Calculates latent interest profiles.
    """
    latent_interest_data = []
    num_all_topics = len(topic_to_idx_map)
    topics_in_order = sorted(topic_to_idx_map.keys(), key=topic_to_idx_map.get)

    # Pre-calculate one-hot diffusions
    one_hot_diffusions = {alpha: {} for alpha in alphas}
    for alpha in alphas:
        for topic_name in topics_in_order:
            theta_topic_one_hot = np.zeros(num_all_topics)
            idx = topic_to_idx_map[topic_name]
            theta_topic_one_hot[idx] = 1.0

            f_one_hot = diffuse_interest(
                theta_topic_one_hot, phi_matrix, alpha=alpha, max_iter=max_iter, tol=tol
            )
            one_hot_diffusions[alpha][topic_name] = f_one_hot

    # Calculate agent diffusions
    for agent_name in rca_norm_df.columns:
        theta_agent_profile = np.array(
            [rca_norm_df[agent_name][t] for t in topics_in_order]
        )
        e = stats.entropy(theta_agent_profile + 1e-12)

        if np.isclose(np.sum(theta_agent_profile), 0) or np.isclose(e, 0):
            # Handle empty/zero entropy agents
            for alpha in alphas:
                latent_interest_data.append(
                    {
                        "agent": agent_name,
                        "alpha": alpha,
                        "e": 0.0,
                        "f": np.full(num_all_topics, np.nan),
                        "kl": np.nan,
                    }
                )
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

            row = {
                "agent": agent_name,
                "alpha": alpha,
                "f": f_a,
                "kl": kl,
                "kl_norm": kl / e if e > 0 else np.nan,
                "baseline_kl": baseline_kl,
                "kl_excess": kl - baseline_kl,
                "p": f_a.T @ phi_matrix @ f_a,
                "e": e,
            }
            latent_interest_data.append(row)

    return pd.DataFrame(latent_interest_data)


# ==========================================
# 5. Graph & Network Layouts
# ==========================================


def disparity_filter(G, alpha=0.05, weight="weight"):
    """Apply the disparity filter to extract the backbone of a weighted network."""
    backbone = G.copy()
    node_strengths = dict(G.degree(weight=weight))

    def compute_alpha(p_ij, k):
        if k <= 1:
            return 1.0
        integral, _ = quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)
        return 1 - (k - 1) * integral

    edges_to_remove = []
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        s_u = node_strengths.get(u, 0.0)
        s_v = node_strengths.get(v, 0.0)

        p_ij_u = w / s_u if s_u > 0 else 0.0
        p_ij_v = w / s_v if s_v > 0 else 0.0

        alpha_u = compute_alpha(p_ij_u, G.degree(u))
        alpha_v = compute_alpha(p_ij_v, G.degree(v))

        if max(alpha_u, alpha_v) >= alpha:
            edges_to_remove.append((u, v))

    backbone.remove_edges_from(edges_to_remove)
    return backbone


def estimate_surround(pos, tree, offset=1 / 3 * np.pi, padding=100, countries=[]):
    """
    Fixed: Added 'tree' as an argument.
    """
    p = np.asarray(list(pos.values()))
    avg = p.mean(0)
    pos_centered = {k: np.array(v) - avg for k, v in pos.items()}

    quad_theta = np.linspace(0, 2 * np.pi, 4, 0)
    quads = np.stack([np.cos(quad_theta), np.sin(quad_theta)], axis=1)

    quad_members = {}
    for country in countries:
        if country not in tree:
            continue
        neighbor = next(tree.neighbors(country))
        target = np.array(pos_centered[neighbor])
        dists = np.linalg.norm(quads - target, axis=1)
        q = np.argmin(dists)
        quad_members.setdefault(q, []).append(country)
        tree.add_node(country)

    existing_positions = np.array(list(pos_centered.values()))
    max_distance = np.linalg.norm(existing_positions, axis=1).max()
    radius = max_distance + padding

    for q, members in quad_members.items():
        anchor_theta = quad_theta[q]

        neighbor_angles = []
        for m in members:
            neighbor = next(tree.neighbors(m))
            target_vec = pos_centered[neighbor]
            rel_vec = target_vec / np.linalg.norm(target_vec)
            angle = np.arctan2(rel_vec[1], rel_vec[0]) - anchor_theta
            neighbor_angles.append(angle)

        sorted_members = [m for _, m in sorted(zip(neighbor_angles, members))]

        angles = (
            np.linspace(-(np.pi / 2 - offset), np.pi / 2 - offset, len(sorted_members))
            + anchor_theta
        )
        semicircle_positions = radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

        for m, p_val in zip(sorted_members, semicircle_positions):
            pos[m] = p_val + avg  # Add avg back to restore original coordinates

    return pos


def nx_layout(graph, layout):
    data = [[node] + list(layout[node]) for node in graph.nodes]
    nodes = pd.DataFrame(data, columns=["id", "x", "y"]).set_index("id")
    edges = pd.DataFrame(list(graph.edges), columns=["source", "target"])
    return nodes, edges


def smart_wrap(text, width=10, max_lines=3):
    import textwrap

    return textwrap.fill(str(text), width=width, max_lines=max_lines, placeholder="...")
