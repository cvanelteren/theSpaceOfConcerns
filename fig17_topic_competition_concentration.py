# %%
from matplotlib.pyplot import rc

from utils import get_rca, load_data

fp = "./antarctic-database-go/data/processed/document-summary.parquet"
df, submitted, countries, topics = load_data(fp)

# Convert topics to sorted list to ensure consistent ordering
# topics = sorted(list(topics))
# countries = sorted(list(countries))

# %%
import pandas as pd

from utils import generate_interaction_matrix, get_rca, standardize_index_labels

rcas = []
for idx, dfi in submitted.groupby("year"):
    counts_df = generate_interaction_matrix(dfi, countries, topics)
    final_df = standardize_index_labels(counts_df)
    rca = get_rca(final_df)
    rca = rca.sort_index(axis=0)
    rca = rca.sort_index(axis=1)
    row = dict(year=idx, rca=rca)  # Store as numpy array
    rcas.append(row)

rcas_df = pd.DataFrame(rcas)
topics = sorted(list(topics))
# %% Topic competition over time
"""
Question: Are topics more competitive or more concentrated as a function of time?

The relative competitive advantage (RCA) indicates whether a country is active in a topic. It can therefore be used to see how many countries are active in each topic over time, and thus how competitive topics are.
"""


def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


import numpy as np
import ultraplot as uplt

years = np.unique(rcas_df.year)

s = np.stack(rcas_df.rca)  # shape: (years, topics, countries)
print(f"RCA stack shape: {s.shape}")
print(f"Expected shape: ({len(years)}, {len(topics)}, {len(countries)})")
tmp = s.copy()
tmp[tmp >= 1] = 1  # Changed to >= to match RCA threshold
d = tmp.sum(-1) / len(countries)  # Sum over countries → (years, topics)
d = pd.DataFrame(d, index=years, columns=topics)  # Use topics directly (already a list)
print(f"Competition dataframe shape: {d.shape}")

# Compute Gini across countries for each topic at each year
# Result: (topics, years) matrix of Gini values
ginis_matrix = []
for topic_idx in range(s.shape[1]):  # Iterate over topics
    topic_ginis = []
    for year_idx in range(s.shape[0]):  # Iterate over years
        # Get RCA values for this topic across all countries for this year
        rca_values = s[year_idx, topic_idx, :]  # shape: (countries,)
        rca_nonzero = rca_values[rca_values > 0]  # Exclude zeros

        if len(rca_nonzero) > 1:
            topic_ginis.append(gini(rca_nonzero))
        else:
            topic_ginis.append(np.nan)
    ginis_matrix.append(topic_ginis)

ginis_matrix = np.array(ginis_matrix)  # shape: (topics, years)
ginis_df = pd.DataFrame(
    ginis_matrix, index=topics, columns=years
)  # Use topics directly

# For plotting: mean Gini across all topics per year
ginis = ginis_df.mean(axis=0).values  # Average across topics

# %% Hierarchical clustering to sort topics by temporal similarity
from scipy.cluster.hierarchy import dendrogram, linkage

# Cluster topics based on their temporal competition profiles
topic_profiles = d.T.values  # (topics, years)
linkage_matrix = linkage(topic_profiles, method="ward")

# Get the order from the dendrogram
dendro = dendrogram(linkage_matrix, no_plot=True)
topic_order = dendro["leaves"]

# Reorder dataframes
d_sorted = d.iloc[:, topic_order]
topics_sorted = d_sorted.columns
ginis_df_sorted = ginis_df.iloc[topic_order, :]
# %%
fig, ax = uplt.subplots()
ax.pcolormesh(
    d_sorted.T,
    colorbar="right",
    vmin=0,
    vmax=0.5,
    colorbar_kw=dict(label="General interest"),
)
ax.format(yticklabelsize=4.5)
tax = ax.panel("t")
tax.plot(years, ginis, label="Mean Gini")
tax.set_ylim(0, 1)
tax.set_title("Mean Gini (across topics)")
fig.savefig("./figures/fig17_topic_competition_concentration.pdf")
fig.savefig("./figures/fig17_topic_competition_concentration.png", transparent=True)
# %%
"""
We measure competition (general interest) through the fraction of countries with
RCA ≥ 1 for each topic. The heatmap reveals three key patterns:

1. **Temporal expansion**: Post-1990, significantly more topics attract broad
   participation, particularly operational topics (bottom cluster) which show
   40-50% of countries actively engaged by 2020.

2. **Persistent stratification**: Despite increasing participation, the mean Gini
   coefficient remains stable around 0.25-0.30 across six decades. This indicates
   that while MORE countries enter topics (breadth ↑), the INEQUALITY in their
   RCA values remains constant. New entrants join without disrupting the established
   hierarchy—a pattern of proportional expansion rather than democratization.

3. **Topic-specific dynamics**:
   - Tourism (1990s spike then decline) shows transient concerns
   - Operational/governance topics (bottom) show sustained, growing engagement
   - Historic/niche topics (top) remain specialist-dominated

The stable Gini amid expanding participation suggests the ATS accommodates new
members while preserving structural hierarchies—constrained diversification rather than convergence toward equality.
"""

# %%


# Optional debug plots (kept disabled in normal script execution).
if False:  # pragma: no cover
    fig, ax = uplt.subplots()
    ax.plot(sorted(d.iloc[:, 0].values))
    print(gini(d.iloc[:, 0].values))

    print(s.shape)
    fig, ax = uplt.subplots()
    ax.imshow(s.sum(1).T, colorbar="r")
    ax.set_xticks(np.arange(len(years)), labels=years, rotation=90)
    fig.show()
    print(submitted["submitted by"].unique())
