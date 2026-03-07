# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# Topics
# ----------------------------------------------------

topics = [
    "Area Protection and Management Plans General",
    "Biological Prospecting",
    "CEP Strategy Discussions",
    "Climate Change",
    "Comprehensive Environmental Evaluations",
    "Cooperation with Other Organisations",
    "Drilling",
    "Educational issues",
    "Emergency report and contingency planning",
    "Environmental Domains Analysis",
    "Environmental Impact Assessment EIA Other EIA Matters",
    "Environmental Monitoring and Reporting",
    "Environmental Protection General",
    "Exchange of Information",
    "Fauna and Flora General",
    "Historic Sites and Monuments",
    "Human Footprint and wilderness values",
    "Inspections",
    "Institutional and legal matters",
    "International Polar Year",
    "Liability",
    "Management Plans",
    "Marine Acoustics",
    "Marine Protected Areas",
    "Marine living resources",
    "Mineral resources",
    "Multiyear strategic workplan",
    "Nonnative Species and Quarantine",
    "Opening statements",
    "Operation of the Antarctic Treaty system General",
    "Operation of the Antarctic Treaty system Reports",
    "Operation of the Antarctic Treaty system The Secretariat",
    "Operation of the CEP",
    "Operational issues",
    "Prevention of marine pollution",
    "Repair and remediation of environmental damage",
    "Safety and Operations in Antarctica",
    "Science issues",
    "Search and Rescue",
    "Site Guidelines for Visitors",
    "Specially Protected Species",
    "State of the Antarctic Environment Report SAER",
    "Sub glacial Lakes",
    "Tourism and NG Activities",
    "Waste management and disposal",
]

# ----------------------------------------------------
# Compute embeddings
# ----------------------------------------------------

model = SentenceTransformer("all-mpnet-base-v2")
emb = model.encode(topics)

# ----------------------------------------------------
# Compute similarity
# ----------------------------------------------------

sim = cosine_similarity(emb)

# ----------------------------------------------------
# Hierarchical clustering
# ----------------------------------------------------

dist = pdist(emb, metric="cosine")
Z = linkage(dist, method="ward")

# ----------------------------------------------------
# Clustered heatmap with dendrogram
# ----------------------------------------------------

sns.clustermap(
    sim,
    row_linkage=Z,
    col_linkage=Z,
    cmap="viridis",
    xticklabels=topics,
    yticklabels=topics,
    figsize=(13, 13),
    dendrogram_ratio=(0.15, 0.15),
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
)

plt.title("Semantic clustering of Antarctic Treaty topics", pad=80)
plt.show()
