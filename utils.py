from imojify import imojify
import countryflag, requests, pycountry, numpy as np
import ultraplot as plt
from PIL import Image
from io import BytesIO
from pathlib import Path
import pandas as pd, numpy as np


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


# def get_rca(df, total_counts) -> pd.DataFrame:
#     rca = {}
#     z = np.sum(df.sum())  # total votes
#     for idx, index in enumerate(df.index):
#         stripped = index.split("_rel")[0]
#         jdx = np.where(df.index == stripped)[0][0]
#         zi = (total_counts.iloc[jdx] / z).sum()
#         t = df.iloc[idx] / zi
#         stripped += "_rca"
#         rca[stripped] = t
#     return pd.DataFrame(rca)


# def get_fractional_interest(row):
#     z = float(row.sum())
#     d = {}
#     for col in row.index:
#         new_col = col + "_rel"
#         if z == 0:
#             row[col] = 0
#         else:
#             row[col] = row[col] / z
#     return row


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
