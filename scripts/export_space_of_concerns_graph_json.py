#!/usr/bin/env python3
"""
Export the ATS space-of-concerns graph to JSON for the D3 force-directed demo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import compute_product_space, get_rca, load_data

DATA_PATHS = [
    Path("antarctic-database-go/data/processed/document-summary.parquet"),
    Path("antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"),
    Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
    Path("document-summary.csv"),
]

OUT_PATH = Path("d3_space_of_concerns/data/space_of_concerns_graph.json")

THEME_COLORS = {
    "Environmental Protection": "#2E7D32",
    "Marine & Wildlife": "#0277BD",
    "Operations & Safety": "#F57C00",
    "Governance & Legal": "#6A1B9A",
    "Science & Research": "#C62828",
    "Tourism & Human Activity": "#D84315",
    "Infrastructure & Planning": "#5D4037",
    "Resource Extraction": "#00838F",
}

TOPIC_TO_THEME = {
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


def _normalize_topic(value: str) -> str:
    return " ".join(str(value).replace("_", " ").strip().split()).lower()


THEME_LOOKUP = {_normalize_topic(k): v for k, v in TOPIC_TO_THEME.items()}


def load_data_with_fallback():
    last_error = None
    for path in DATA_PATHS:
        if not path.exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Failed to load {path}: {exc}")
    raise FileNotFoundError("No usable data file found in DATA_PATHS.") from last_error


def build_graph():
    counts_df, _, _, _ = load_data_with_fallback()
    rca = get_rca(counts_df)
    phi = compute_product_space(rca)
    g = nx.from_pandas_adjacency(phi)
    mst = nx.maximum_spanning_tree(g)

    weights = np.array([d.get("weight", 1.0) for _, _, d in g.edges(data=True)])
    cutoff = np.percentile(weights, 95)

    merged = mst.copy()
    for u, v, d in g.edges(data=True):
        if d["weight"] >= cutoff:
            merged.add_edge(u, v, **d)
    return merged, mst, cutoff


def main() -> None:
    merged, mst, cutoff = build_graph()
    mst_edges = {tuple(sorted((u, v))) for u, v in mst.edges()}

    node_payload = []
    for node in sorted(merged.nodes()):
        theme = THEME_LOOKUP.get(_normalize_topic(node), "Governance & Legal")
        weighted_degree = float(
            sum(d.get("weight", 1.0) for _, _, d in merged.edges(node, data=True))
        )
        node_payload.append(
            {
                "id": node,
                "theme": theme,
                "color": THEME_COLORS.get(theme, THEME_COLORS["Governance & Legal"]),
                "weighted_degree": round(weighted_degree, 6),
                "degree": int(merged.degree(node)),
            }
        )

    edge_payload = []
    for u, v, d in sorted(merged.edges(data=True), key=lambda x: (x[0], x[1])):
        pair = tuple(sorted((u, v)))
        edge_payload.append(
            {
                "source": u,
                "target": v,
                "weight": round(float(d.get("weight", 1.0)), 6),
                "kind": "mst" if pair in mst_edges else "strong",
            }
        )

    payload = {
        "meta": {
            "n_nodes": merged.number_of_nodes(),
            "n_edges": merged.number_of_edges(),
            "strong_edge_percentile": 95,
            "strong_edge_cutoff": round(float(cutoff), 6),
        },
        "theme_colors": THEME_COLORS,
        "nodes": node_payload,
        "links": edge_payload,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {OUT_PATH}")
    print(payload["meta"])


if __name__ == "__main__":
    main()
