#!/usr/bin/env python3
"""Build an interactive browser linking formal outcomes to the concern space.

The browser keeps three concepts separate:

1. meeting activity: papers assigned to a concern at the focal ATCM or within a
   user-selected number of preceding ATCMs;
2. direct documentary evidence: concerns assigned to papers that the verified
   meeting record links to the selected outcome; and
3. inherited documentary evidence: concerns assigned to papers linked to
   earlier formal instruments cited by the selected outcome, recursively.

The fixed 45-node map uses all submitted papers in the archive. Formal outcomes
are not nodes in that map. Recursive overlays describe documented ancestry, not
causal influence or legal inheritance.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import normalize_topic_key  # noqa: E402
from scripts.primary_concern_sensitivity import corpus_objects, variants  # noqa: E402
from utils import (  # noqa: E402
    _is_excluded_topic_label,
    _normalize_topic_label,
    _split_multi_value,
    extract_unique_topics,
    standardize_index_labels,
)

LINEAGE_PATH = Path("/home/casper/ats_lineage/decision_map.json")
DIRECT_LINEAGE_PATH = Path("/home/casper/ats_lineage/direct_lineage.json")
OUT_DIR = ROOT / "output/outcome_concern_browser"
OUT_HTML = OUT_DIR / "index.html"
OUT_JSON = OUT_DIR / "browser_data.json"
OUT_EDGE_AUDIT = OUT_DIR / "paper_edge_audit.csv"
OUT_OUTCOME_AUDIT = OUT_DIR / "outcome_evidence_summary.csv"

TYPE_RE = re.compile(r"/(wp|ip|bp|sp|inf)/", re.I)
YEAR_RE = re.compile(r"\((\d{4})\)$")

THEME_BY_REGION = {
    1: ("Procedural and scientific core", "#23638f"),
    2: ("Environmental management", "#16877c"),
    3: ("Access, tourism, and safety", "#c15b24"),
    4: ("Resources and planning", "#597a3d"),
    5: ("Cooperation and reporting", "#ae8426"),
    6: ("Environmental domains and reporting", "#b83265"),
    7: ("Acoustics and sub-glacial lakes", "#70588c"),
}

OPERATIVE_RELATIONS = {
    "amends",
    "supersedes",
    "designates_under",
    "pursuant_to",
}

EVIDENCE_RANK = {
    "rejected": 0,
    "candidate": 1,
    "contextual": 2,
    "corroborated": 3,
    "confirmed": 4,
}


def canonical_topic(raw: str) -> str:
    """Apply the same topic spelling corrections as the main concern map."""
    normalized = _normalize_topic_label(raw)
    frame = standardize_index_labels(pd.DataFrame(index=[normalized]))
    return str(frame.index[0])


def paper_node_id(row: pd.Series) -> str | None:
    """Translate a paper record into the lineage graph's canonical identifier."""
    match = TYPE_RE.search(str(row.get("paper url", "")))
    if not match or pd.isna(row.get("meeting number")) or pd.isna(row.get("paper number")):
        return None
    paper_type = match.group(1).upper().replace("INF", "IP")
    return f"ATCM{int(row['meeting number'])}:{paper_type} {int(row['paper number'])}"


def topic_metadata(topics: list[str]) -> tuple[dict[str, str], dict[str, int]]:
    order = pd.read_csv(ROOT / "output/fig45_portfolio_space_ridgelines_topic_order.csv")
    display: dict[str, str] = {}
    region: dict[str, int] = {}
    for _, row in order.iterrows():
        key = normalize_topic_key(row["topic"])
        display[key] = str(row["topic"]).replace("_", " ")
        region[key] = int(row["region_id"])
    return (
        {topic: display.get(normalize_topic_key(topic), topic) for topic in topics},
        {topic: region.get(normalize_topic_key(topic), 1) for topic in topics},
    )


def build_map(submitted: pd.DataFrame) -> tuple[list[dict], list[dict], list[str]]:
    topics = sorted(
        standardize_index_labels(
            pd.DataFrame(index=sorted(extract_unique_topics(submitted)))
        ).index.unique()
    )
    topics = [t for t in topics if not _is_excluded_topic_label(t)]
    counts, phi_values, _, _, _ = corpus_objects(submitted, topics)
    topics = list(counts.index)
    phi = pd.DataFrame(phi_values, index=topics, columns=topics).reindex(
        index=topics, columns=topics, fill_value=0.0
    )

    graph = nx.from_pandas_adjacency(phi)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    positive = [float(d["weight"]) for _, _, d in graph.edges(data=True) if d["weight"] > 0]
    cutoff = float(np.percentile(positive, 92))
    mst = nx.maximum_spanning_tree(graph, weight="weight")
    layout_graph = mst.copy()
    for source, target, attrs in graph.edges(data=True):
        if float(attrs["weight"]) >= cutoff:
            layout_graph.add_edge(source, target, **attrs)
    pos = nx.spring_layout(layout_graph, weight="weight", seed=17, k=0.52, iterations=500)
    xs = np.array([pos[t][0] for t in topics])
    ys = np.array([pos[t][1] for t in topics])
    xspan = max(float(np.ptp(xs)), 1e-9)
    yspan = max(float(np.ptp(ys)), 1e-9)
    display, regions = topic_metadata(topics)

    mst_pairs = {tuple(sorted((a, b))) for a, b in mst.edges()}
    nodes = []
    for topic in topics:
        rid = regions[topic]
        label, color = THEME_BY_REGION[rid]
        nodes.append(
            {
                "id": topic,
                "label": display[topic],
                "region": rid,
                "region_label": label,
                "color": color,
                "x": round(0.04 + 0.92 * (float(pos[topic][0]) - float(xs.min())) / xspan, 5),
                "y": round(0.05 + 0.90 * (float(pos[topic][1]) - float(ys.min())) / yspan, 5),
            }
        )

    edges = []
    for source, target, attrs in graph.edges(data=True):
        weight = float(attrs.get("weight", 0.0))
        if weight <= 0:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": round(weight, 6),
                "mst": tuple(sorted((source, target))) in mst_pairs,
            }
        )
    edges.sort(key=lambda edge: edge["weight"], reverse=True)
    return nodes, edges, topics


def explode_papers(
    submitted: pd.DataFrame, topics: list[str]
) -> tuple[dict[str, dict], dict[str, dict[str, float]], dict[str, int]]:
    topic_set = set(topics)
    paper_topics: dict[str, set[str]] = defaultdict(set)
    paper_rows: dict[str, dict] = {}
    meeting_activity: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    unmatched_categories: defaultdict[str, int] = defaultdict(int)

    for _, row in submitted.iterrows():
        raw_topics = _split_multi_value(row.get("category"), delimiter="\t")
        cats = []
        for raw in raw_topics:
            topic = canonical_topic(raw)
            if topic in topic_set and topic not in cats:
                cats.append(topic)
            elif topic not in topic_set:
                unmatched_categories[topic] += 1
        if not cats:
            continue
        meeting = int(row["meeting number"])
        weight = 1.0 / len(cats)
        for topic in cats:
            meeting_activity[meeting][topic] += weight

        lineage_id = paper_node_id(row)
        if lineage_id:
            paper_topics[lineage_id].update(cats)
            paper_rows.setdefault(
                lineage_id,
                {
                    "id": lineage_id,
                    "meeting": meeting,
                    "title": str(row.get("paper name") or "").strip(),
                    "url": str(row.get("paper url") or "").strip(),
                },
            )

    papers = {}
    for paper_id, record in paper_rows.items():
        record["topics"] = sorted(paper_topics[paper_id])
        papers[paper_id] = record
    compact_activity = {
        str(meeting): {topic: round(value, 4) for topic, value in values.items()}
        for meeting, values in sorted(meeting_activity.items())
    }
    return papers, compact_activity, dict(unmatched_categories)


def display_outcome_id(node: dict) -> str:
    node_id = str(node["id"])
    year = int(node["year"])
    if YEAR_RE.search(node_id):
        return node_id
    return f"{node_id} ({year})"


def classify_paper_edge(edge: dict, outcome: dict) -> tuple[str, str]:
    """Classify relationship evidence without confusing endpoint and edge validity."""
    channel = edge.get("channel", "")
    if channel == "official_paragraph":
        return "confirmed", "official outcome paragraph identifies the paper relationship"

    if channel == "direct_citation":
        match = re.match(
            r"^(Measure|Decision|Resolution)\s+(\d+)\s+\((\d{4})\)",
            str(outcome["id"]),
        )
        evidence = str(edge.get("evidence") or "")
        if match:
            kind, number, focal_year = match.group(1), match.group(2), int(match.group(3))
            explicit_years = [
                int(year)
                for year in re.findall(
                    rf"\b{re.escape(kind)}\s+{number}\s*\((\d{{4}})\)",
                    evidence,
                    flags=re.I,
                )
            ]
            if explicit_years and focal_year not in explicit_years:
                years = ", ".join(map(str, sorted(set(explicit_years))))
                return "rejected", f"evidence names {kind} {number} from {years}, not {focal_year}"
            if focal_year in explicit_years:
                return "confirmed", "evidence names the year-qualified focal instrument"
        if edge.get("relation") == "direct_comention" or edge.get("confidence") == "low":
            return "candidate", "same-passage co-mention without a specific relationship"
        return "contextual", "same-passage reference lacks a year-qualified focal identity"

    if channel == "anchor_proximity":
        return "corroborated", "report anchor and nearby paper reference"
    if channel == "early_title":
        return "contextual", "historical paper and outcome titles align"
    if channel == "wp_title_window":
        score = float(edge.get("jaccard") or 0.0)
        lag = int(edge.get("lag_meetings") or 0)
        if score >= 0.5 and lag == 0:
            return "corroborated", f"strong same-meeting title match (Jaccard {score:.2f})"
        if score >= 0.5:
            return "contextual", f"strong title match after {lag} meetings (Jaccard {score:.2f})"
        return "candidate", f"weak title match (Jaccard {score:.2f})"
    return "candidate", f"unclassified lineage channel: {channel or 'unknown'}"


def build_lineage_payload(
    lineage: dict, paper_records: dict[str, dict]
) -> tuple[list[dict], dict[str, dict], dict[str, dict], list[dict]]:
    nodes = {node["id"]: node for node in lineage["nodes"]}
    outcome_nodes = {
        node_id: node
        for node_id, node in nodes.items()
        if node.get("kind") == "outcome"
        and node.get("year") is not None
        and not node.get("placeholder")
    }
    direct_evidence: dict[tuple[str, str, str], str] = {}
    if DIRECT_LINEAGE_PATH.exists():
        direct = json.loads(DIRECT_LINEAGE_PATH.read_text(encoding="utf-8"))
        outcomes_by_meeting_label = {
            (float(node["meeting"]), re.sub(r"\s+\(\d{4}\)$", "", node_id)): node_id
            for node_id, node in outcome_nodes.items()
        }
        for atcm_id, record in direct.items():
            meeting_match = re.search(r"ATCM(\d+)", atcm_id, re.I)
            if not meeting_match:
                continue
            meeting = float(meeting_match.group(1))
            for source_edge in record.get("edges", []):
                paper_match = re.match(r"([A-Za-z]+)\s*0*(\d+)", source_edge.get("paper", ""))
                if not paper_match:
                    continue
                source = f"ATCM{int(meeting)}:{paper_match.group(1).upper()} {int(paper_match.group(2))}"
                destination = outcomes_by_meeting_label.get((meeting, source_edge.get("output", "")))
                if destination:
                    direct_evidence[(source, destination, str(source_edge.get("para") or ""))] = str(
                        source_edge.get("evidence") or ""
                    )

    direct_papers: defaultdict[str, list[dict]] = defaultdict(list)
    incoming_outcomes: defaultdict[str, list[dict]] = defaultdict(list)
    relation_counts: defaultdict[str, int] = defaultdict(int)
    edge_audit: list[dict] = []

    for edge in lineage["edges"]:
        src = edge["src"]
        dst = edge["dst"]
        if dst not in outcome_nodes:
            continue
        if src in paper_records:
            working_edge = dict(edge)
            if working_edge.get("channel") == "direct_citation":
                full_evidence = direct_evidence.get(
                    (src, dst, str(working_edge.get("para") or ""))
                )
                if full_evidence:
                    working_edge["evidence"] = full_evidence
            level, reason = classify_paper_edge(working_edge, outcome_nodes[dst])
            paper_edge = {
                "paper": src,
                "relation": edge.get("relation", "documented_link"),
                "channel": edge.get("channel", ""),
                "source_tier": edge.get("tier", ""),
                "confidence": edge.get("confidence", ""),
                "evidence_level": level,
                "evidence_rank": EVIDENCE_RANK[level],
                "evidence_reason": reason,
                "jaccard": edge.get("jaccard"),
                "lag_meetings": edge.get("lag_meetings"),
                "evidence": str(working_edge.get("evidence") or "")[:420],
            }
            direct_papers[dst].append(paper_edge)
            edge_audit.append(
                {
                    "paper": src,
                    "outcome": dst,
                    "channel": paper_edge["channel"],
                    "source_tier": paper_edge["source_tier"],
                    "relation": paper_edge["relation"],
                    "evidence_level": level,
                    "evidence_rank": EVIDENCE_RANK[level],
                    "reason": reason,
                    "jaccard": paper_edge["jaccard"],
                    "lag_meetings": paper_edge["lag_meetings"],
                }
            )
            relation_counts[edge.get("relation", "unknown")] += 1
        elif src in outcome_nodes and edge.get("tier") in {"verified", "supported"}:
            incoming_outcomes[dst].append(
                {
                    "outcome": src,
                    "relation": edge.get("relation", "cites"),
                    "channel": edge.get("channel", ""),
                    "source_tier": edge.get("tier", ""),
                }
            )

    outcomes = []
    outcome_lookup = {}
    for node_id, node in outcome_nodes.items():
        record = {
            "id": node_id,
            "display_id": display_outcome_id(node),
            "meeting": float(node["meeting"]),
            "year": int(node["year"]),
            "type": node.get("outcome_type", "Outcome"),
            "title": str(node.get("title") or "Untitled instrument"),
            "url": str(node.get("official_detail_url") or ""),
            "direct_papers": sorted(direct_papers.get(node_id, []), key=lambda x: x["paper"]),
            "incoming_outcomes": sorted(
                incoming_outcomes.get(node_id, []), key=lambda x: x["outcome"]
            ),
        }
        outcomes.append(record)
        outcome_lookup[node_id] = record
    outcomes.sort(key=lambda x: (x["year"], x["meeting"], x["type"], x["display_id"]), reverse=True)
    return outcomes, outcome_lookup, dict(relation_counts), edge_audit


def validate_payload(payload: dict) -> dict:
    topics = {node["id"] for node in payload["nodes"]}
    outcomes = payload["outcomes"]
    outcome_ids = {outcome["id"] for outcome in outcomes}
    outcome_lookup = {outcome["id"]: outcome for outcome in outcomes}
    papers = payload["papers"]

    assert len(topics) == 45, f"Expected 45 concerns, found {len(topics)}"
    assert len(outcomes) == len(outcome_ids), "Outcome identifiers are not unique"
    assert all(YEAR_RE.search(outcome["display_id"]) for outcome in outcomes)
    assert all(set(paper["topics"]).issubset(topics) for paper in papers.values())

    forward_or_same = []
    formal_graph = nx.DiGraph()
    for outcome in outcomes:
        for edge in outcome["incoming_outcomes"]:
            source = outcome_lookup[edge["outcome"]]
            formal_graph.add_edge(source["id"], outcome["id"])
            if source["meeting"] > outcome["meeting"]:
                forward_or_same.append((source["id"], outcome["id"]))
        for edge in outcome["direct_papers"]:
            assert papers[edge["paper"]]["meeting"] <= outcome["meeting"], (
                f"Paper {edge['paper']} occurs after outcome {outcome['id']}"
            )
    assert not forward_or_same, f"Found forward-time ancestry edges: {forward_or_same[:3]}"
    ancestry_is_acyclic = nx.is_directed_acyclic_graph(formal_graph)

    direct_linked = {
        edge["paper"] for outcome in outcomes for edge in outcome["direct_papers"]
    }
    return {
        "nodes": len(topics),
        "positive_edges": len(payload["edges"]),
        "outcomes": len(outcomes),
        "linked_papers_with_categories": len(papers),
        "direct_linked_papers_with_categories": len(direct_linked),
        "outcomes_with_direct_paper_evidence": sum(bool(x["direct_papers"]) for x in outcomes),
        "outcomes_with_corroborated_or_confirmed_paper_routes": sum(
            any(edge["evidence_rank"] >= EVIDENCE_RANK["corroborated"] for edge in x["direct_papers"])
            for x in outcomes
        ),
        "outcomes_with_formal_ancestors": sum(bool(x["incoming_outcomes"]) for x in outcomes),
        "formal_ancestry_edges": formal_graph.number_of_edges(),
        "formal_ancestry_is_acyclic": ancestry_is_acyclic,
        "meetings_with_activity": len(payload["meeting_activity"]),
    }


def html_document(data: dict) -> str:
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Outcome pathways through the concern space</title>
<style>
:root{--ink:#17212b;--muted:#68737d;--paper:#f5f1e8;--panel:#fffdf8;--rule:#d4cec0;--activity:#178f83;--direct:#d84b2f;--inherited:#2f69a1;--quiet:#d8d5cc;--edge:#80909b;--focus:#191919}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Gill Sans",sans-serif}main{max-width:1500px;margin:0 auto;padding:20px}.mast{display:grid;grid-template-columns:minmax(300px,1fr) auto;gap:18px;align-items:end;border-bottom:1px solid var(--rule);padding-bottom:14px}.eyebrow{margin:0 0 3px;text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:var(--muted)}h1{margin:0;font-family:Georgia,serif;font-size:clamp(25px,3vw,42px);font-weight:400}.sub{margin:8px 0 0;max-width:860px;color:var(--muted);line-height:1.45}.nav{display:flex;gap:8px}.nav button,.controls select,.controls input[type=search]{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--rule);padding:8px 10px}.nav button{cursor:pointer}.controls{display:grid;grid-template-columns:minmax(230px,2fr) repeat(4,minmax(140px,1fr));gap:12px;margin:15px 0}.control{display:flex;flex-direction:column;gap:5px}.control label{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}.control input[type=range]{width:100%;accent-color:var(--activity)}.work{display:grid;grid-template-columns:minmax(0,1.9fr) minmax(320px,.85fr);gap:18px}.map-wrap{position:relative;min-height:690px;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}svg{display:block;width:100%;height:690px}.edge{stroke:var(--edge);stroke-opacity:.16;stroke-linecap:round}.edge.mst{stroke-opacity:.30}.node .base{stroke:var(--panel);stroke-width:1.4}.node .activity{fill:none;stroke:var(--activity);stroke-opacity:.72}.node .evidence{fill:none}.node.direct .evidence{stroke:var(--direct);stroke-width:4}.node.inherited .evidence{stroke:var(--inherited);stroke-width:3;stroke-dasharray:5 3}.node.direct.inherited .evidence{stroke:var(--direct);stroke-dasharray:none}.node text{font-family:"Avenir Next","Gill Sans",sans-serif;font-size:11px;fill:var(--ink);paint-order:stroke;stroke:var(--paper);stroke-width:3px;stroke-linejoin:round;pointer-events:none}.node:not(.salient) text{display:none}.node{cursor:pointer}.legend{position:absolute;left:12px;bottom:12px;background:rgba(255,253,248,.94);border:1px solid var(--rule);padding:9px 11px;font-size:12px;line-height:1.7}.key{display:inline-block;width:18px;height:10px;margin-right:6px;vertical-align:middle}.key.activity{border-top:3px solid var(--activity)}.key.direct{border-top:4px solid var(--direct)}.key.inherited{border-top:3px dashed var(--inherited)}.detail{border-top:4px solid var(--focus);padding-top:13px}.detail h2{font-family:Georgia,serif;font-size:25px;font-weight:400;line-height:1.15;margin:0 0 6px}.idline{font-size:13px;font-weight:600;letter-spacing:.04em}.meta{color:var(--muted);margin:5px 0 14px}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule);margin:12px 0}.metric{background:var(--panel);padding:10px}.metric b{display:block;font-size:22px;font-weight:500}.metric span{font-size:11px;color:var(--muted)}h3{font-size:12px;text-transform:uppercase;letter-spacing:.09em;margin:18px 0 8px}.evidence-list{margin:0;padding:0;list-style:none;max-height:330px;overflow:auto;border-top:1px solid var(--rule)}.evidence-list li{padding:9px 0;border-bottom:1px solid var(--rule);font-size:13px;line-height:1.35}.evidence-list .route{display:block;color:var(--muted);font-size:11px;margin-top:3px}.node-detail{min-height:56px;border-left:3px solid var(--activity);padding:7px 10px;background:var(--panel);font-size:13px;line-height:1.4}.note{font-size:12px;line-height:1.45;color:var(--muted);margin-top:16px}.tag{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:.06em;margin-right:6px;color:var(--muted)}a{color:inherit;text-decoration-thickness:1px}.empty{color:var(--muted);font-style:italic}@media(max-width:980px){.controls{grid-template-columns:repeat(2,minmax(0,1fr))}.work{grid-template-columns:1fr}.map-wrap,svg{min-height:580px;height:580px}.mast{grid-template-columns:1fr}.detail{margin-top:4px}}@media(max-width:560px){main{padding:12px}.controls{grid-template-columns:1fr}.map-wrap,svg{min-height:480px;height:480px}.summary{grid-template-columns:1fr}.legend{position:static;border-width:1px 0 0;margin-top:-1px}.node text{font-size:10px}}
</style>
</head>
<body>
<main>
  <header class="mast">
    <div><p class="eyebrow">Antarctic Treaty System</p><h1>Outcome pathways through the concern space</h1><p class="sub">Select a formal instrument to compare documentary concern evidence with attention elevated above earlier meetings. Windows count ATCM meetings, not calendar years.</p></div>
    <div class="nav"><button id="newer" type="button">Newer</button><button id="older" type="button">Older</button></div>
  </header>
  <section class="controls" aria-label="Browser controls">
    <div class="control"><label for="search">Find outcome</label><input id="search" type="search" placeholder="Title, type, number, or year"></div>
    <div class="control"><label for="outcome">Outcome, recent first</label><select id="outcome"></select></div>
    <div class="control"><label for="slack">Attention window: focal + <span id="slack-value">2</span> prior meetings</label><input id="slack" type="range" min="0" max="10" value="2"></div>
    <div class="control"><label for="depth">Ancestry depth: <span id="depth-value">2</span></label><input id="depth" type="range" min="0" max="6" value="2"></div>
    <div class="control"><label for="edge-cut">Map edges: top <span id="edge-value">8</span>%</label><input id="edge-cut" type="range" min="1" max="100" value="8"></div>
    <div class="control"><label for="paper-scope">Paper evidence shown</label><select id="paper-scope"><option value="3">Confirmed and corroborated</option><option value="2">Add contextual routes</option><option value="1">Add weak candidates</option></select></div>
    <div class="control"><label for="scope">Formal links followed</label><select id="scope"><option value="all">Supported formal citations</option><option value="operative">Operative links only</option></select></div>
  </section>
  <section class="work">
    <div class="map-wrap"><svg id="map" role="img" aria-labelledby="map-title map-desc"><title id="map-title">Space of concerns with relative attention and outcome evidence</title><desc id="map-desc">Forty-five concerns connected by full-record co-specialization probability.</desc></svg><div class="legend"><div><span class="key activity"></span>Attention above earlier-meeting baseline</div><div><span class="key direct"></span>Focal outcome paper route</div><div><span class="key inherited"></span>Inherited through formal ancestry</div></div></div>
    <aside class="detail"><div class="idline" id="outcome-id"></div><h2 id="outcome-title"></h2><div class="meta" id="outcome-meta"></div><div class="summary"><div class="metric"><b id="active-count">0</b><span>elevated concerns</span></div><div class="metric"><b id="direct-count">0</b><span>focal-route concerns</span></div><div class="metric"><b id="inherited-count">0</b><span>inherited concerns</span></div></div><h3>Selected concern</h3><div class="node-detail" id="node-detail">Select a node to inspect its attention and evidence.</div><h3>Documentary routes</h3><ul class="evidence-list" id="evidence-list"></ul><p class="note">The map is built from the full WP/IP archive. Attention lift compares the selected meeting window with all earlier meetings using a small smoothing prior. Formal instruments are not map nodes. Paper routes retain their lineage channel and relationship-evidence grade.</p></aside>
  </section>
</main>
<script>
const DATA=__DATA__;
const byId=new Map(DATA.outcomes.map(d=>[d.id,d]));
const paperById=DATA.papers;
const nodeById=new Map(DATA.nodes.map(d=>[d.id,d]));
const operative=new Set(DATA.meta.operative_relations);
const els=Object.fromEntries(['search','outcome','slack','depth','edge-cut','paper-scope','scope','newer','older','map','outcome-id','outcome-title','outcome-meta','active-count','direct-count','inherited-count','node-detail','evidence-list','slack-value','depth-value','edge-value'].map(id=>[id,document.getElementById(id)]));
let filtered=[...DATA.outcomes],selectedNode=null;

function optionLabel(o){return `${o.display_id} · ${o.title}`}
function rebuildOptions(keep){const q=els.search.value.trim().toLowerCase();filtered=DATA.outcomes.filter(o=>!q||`${o.display_id} ${o.type} ${o.title}`.toLowerCase().includes(q));els.outcome.innerHTML='';for(const o of filtered){const opt=document.createElement('option');opt.value=o.id;opt.textContent=optionLabel(o);els.outcome.appendChild(opt)}if(keep&&filtered.some(o=>o.id===keep))els.outcome.value=keep;else if(filtered.length)els.outcome.value=filtered[0].id;render()}
function ancestors(focal,maxDepth,scope){const found=new Map(),queue=new Array(...focal.incoming_outcomes.map(e=>({id:e.outcome,depth:1,route:[{from:e.outcome,to:focal.id,relation:e.relation}]})));while(queue.length){const item=queue.shift();if(item.depth>maxDepth)continue;if(scope==='operative'&&!item.route.every(e=>operative.has(e.relation)))continue;const old=found.get(item.id);if(old&&old.depth<=item.depth)continue;found.set(item.id,item);const src=byId.get(item.id);if(!src)continue;for(const edge of src.incoming_outcomes)queue.push({id:edge.outcome,depth:item.depth+1,route:[{from:edge.outcome,to:item.id,relation:edge.relation},...item.route]})}return found}
function evidence(focal){const direct=new Map(),inherited=new Map(),routes=[],minimum=+els['paper-scope'].value;for(const link of focal.direct_papers){if(link.evidence_rank<minimum)continue;const p=paperById[link.paper];if(!p)continue;for(const topic of p.topics)direct.set(topic,(direct.get(topic)||[]).concat([{paper:p,link,depth:0,via:focal.display_id}]));routes.push({kind:'direct',paper:p,link,depth:0,via:focal})}const anc=ancestors(focal,+els.depth.value,els.scope.value);for(const [id,path] of anc){const out=byId.get(id);for(const link of out.direct_papers){if(link.evidence_rank<minimum)continue;const p=paperById[link.paper];if(!p)continue;for(const topic of p.topics)if(!direct.has(topic))inherited.set(topic,(inherited.get(topic)||[]).concat([{paper:p,link,depth:path.depth,via:out.display_id,path:path.route}]));routes.push({kind:'inherited',paper:p,link,depth:path.depth,via:out,path:path.route})}}return{direct,inherited,routes,anc,rejected:focal.direct_papers.filter(link=>link.evidence_rank===0)}}
function activity(meeting,slack){const focal=Math.floor(meeting),start=Math.max(1,focal-slack),observed=new Map(),prior=new Map();let observedTotal=0,priorTotal=0;for(let m=start;m<=focal;m++){for(const [topic,value] of Object.entries(DATA.meeting_activity[String(m)]||{})){observed.set(topic,(observed.get(topic)||0)+value);observedTotal+=value}}for(let m=1;m<start;m++){for(const [topic,value] of Object.entries(DATA.meeting_activity[String(m)]||{})){prior.set(topic,(prior.get(topic)||0)+value);priorTotal+=value}}const result=new Map(),alpha=.5,denom=priorTotal+alpha*DATA.nodes.length;for(const node of DATA.nodes){const value=observed.get(node.id)||0,share=(prior.get(node.id)||0)+alpha,expected=observedTotal*share/Math.max(denom,1),lift=(value+.25)/(expected+.25);result.set(node.id,{observed:value,expected,lift})}return result}
function quantile(values,p){if(!values.length)return 0;const a=[...values].sort((a,b)=>a-b),i=Math.min(a.length-1,Math.max(0,Math.floor(p*(a.length-1))));return a[i]}
function createSvg(tag,attrs={}){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);return el}
function drawMap(active,ev){const svg=els.map,box=svg.getBoundingClientRect(),w=Math.max(320,box.width),h=Math.max(420,box.height);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.querySelectorAll('g').forEach(g=>g.remove());const edgeLayer=createSvg('g'),nodeLayer=createSvg('g');svg.append(edgeLayer,nodeLayer);const vals=DATA.edges.map(e=>e.weight),show=+els['edge-cut'].value/100,cut=quantile(vals,1-show);for(const e of DATA.edges){if(!e.mst&&e.weight<cut)continue;const a=nodeById.get(e.source),b=nodeById.get(e.target),line=createSvg('line',{x1:a.x*w,y1:a.y*h,x2:b.x*w,y2:b.y*h,'stroke-width':e.mst?1.2:Math.max(.5,3*e.weight),class:`edge${e.mst?' mst':''}`});edgeLayer.appendChild(line)}const maxLift=Math.max(1,...[...active.values()].map(a=>Math.max(0,Math.log2(a.lift))));for(const n of DATA.nodes){const attention=active.get(n.id),intensity=Math.max(0,Math.log2(attention.lift)),isDirect=ev.direct.has(n.id),isInherited=ev.inherited.has(n.id),salient=isDirect||isInherited;const g=createSvg('g',{class:`node${isDirect?' direct':''}${isInherited?' inherited':''}${salient?' salient':''}`,transform:`translate(${n.x*w},${n.y*h})`});const title=createSvg('title');title.textContent=`${n.label}: ${attention.observed.toFixed(2)} papers, ${attention.lift.toFixed(2)}x earlier-meeting expectation`;g.appendChild(title);const radius=6+4*Math.sqrt(intensity/maxLift);if(attention.lift>1)g.appendChild(createSvg('circle',{r:radius+5,class:'activity','stroke-width':1.3+2.2*Math.sqrt(intensity/maxLift)}));g.appendChild(createSvg('circle',{r:radius,class:'base',fill:n.color}));if(isDirect||isInherited)g.appendChild(createSvg('circle',{r:radius+2.5,class:'evidence'}));const text=createSvg('text',{x:radius+5,y:4});text.textContent=n.label;g.appendChild(text);g.addEventListener('click',()=>{selectedNode=n.id;showNode(n.id,active,ev)});nodeLayer.appendChild(g)}}
function showNode(topic,active,ev){const n=nodeById.get(topic),a=active.get(topic),d=ev.direct.get(topic)||[],i=ev.inherited.get(topic)||[];els['node-detail'].innerHTML=`<strong>${n.label}</strong><br>${a.observed.toFixed(2)} fractional papers; ${a.lift.toFixed(2)}× the earlier-meeting expectation. ${d.length} focal and ${i.length} inherited paper routes.`}
function showRoutes(ev){const rows=[...ev.routes].sort((a,b)=>b.link.evidence_rank-a.link.evidence_rank||a.depth-b.depth||a.paper.id.localeCompare(b.paper.id));els['evidence-list'].innerHTML='';if(!rows.length){els['evidence-list'].innerHTML='<li class="empty">No paper route meets the selected evidence threshold.</li>';return}for(const row of rows.slice(0,100)){const li=document.createElement('li'),kind=row.kind==='direct'?'Focal':'Inherited',topics=row.paper.topics.map(t=>nodeById.get(t)?.label||t).join('; '),score=row.link.jaccard!=null?` · Jaccard ${Number(row.link.jaccard).toFixed(2)}`:'',route=row.kind==='direct'?`${row.link.evidence_level} · ${row.link.channel}${score}`:`${row.link.evidence_level} · depth ${row.depth} via ${row.via.display_id}`;li.innerHTML=`<span class="tag">${kind}</span><strong>${row.paper.id}</strong> ${row.paper.title||''}<span class="route">${topics} · ${route}<br>${row.link.evidence_reason}</span>`;li.addEventListener('click',()=>{if(row.paper.topics[0]){selectedNode=row.paper.topics[0];const focal=byId.get(els.outcome.value);showNode(selectedNode,activity(focal.meeting,+els.slack.value),ev)}});els['evidence-list'].appendChild(li)}if(rows.length>100){const li=document.createElement('li');li.className='empty';li.textContent=`${rows.length-100} additional routes omitted from this list.`;els['evidence-list'].appendChild(li)}}
function render(){if(!filtered.length){els['outcome-title'].textContent='No matching outcomes';return}const focal=byId.get(els.outcome.value)||filtered[0];els.outcome.value=focal.id;const active=activity(focal.meeting,+els.slack.value),ev=evidence(focal),warning=ev.rejected.length?` · ${ev.rejected.length} year-mismatched direct reference${ev.rejected.length===1?'':'s'} excluded`:'';els['outcome-id'].textContent=focal.display_id;els['outcome-title'].textContent=focal.title;els['outcome-meta'].innerHTML=`${focal.type} · ATCM ${Number.isInteger(focal.meeting)?focal.meeting:focal.meeting.toFixed(1)}${focal.url?` · <a href="${focal.url}" target="_blank" rel="noreferrer">official record</a>`:''}${warning}`;els['active-count'].textContent=[...active.values()].filter(v=>v.lift>1).length;els['direct-count'].textContent=ev.direct.size;els['inherited-count'].textContent=ev.inherited.size;els['slack-value'].textContent=els.slack.value;els['depth-value'].textContent=els.depth.value;els['edge-value'].textContent=els['edge-cut'].value;drawMap(active,ev);showRoutes(ev);if(selectedNode&&nodeById.has(selectedNode))showNode(selectedNode,active,ev);else els['node-detail'].textContent='Select a node to inspect its attention and evidence.'}
function step(delta){const i=filtered.findIndex(o=>o.id===els.outcome.value),j=Math.max(0,Math.min(filtered.length-1,i+delta));if(filtered[j]){els.outcome.value=filtered[j].id;selectedNode=null;render()}}
els.search.addEventListener('input',()=>rebuildOptions(els.outcome.value));els.outcome.addEventListener('change',()=>{selectedNode=null;render()});for(const id of ['slack','depth','edge-cut','paper-scope','scope'])els[id].addEventListener('input',render);els.newer.addEventListener('click',()=>step(-1));els.older.addEventListener('click',()=>step(1));window.addEventListener('resize',render);rebuildOptions();
</script>
</body>
</html>'''
    return template.replace("__DATA__", encoded)


def main() -> None:
    if not LINEAGE_PATH.exists():
        raise FileNotFoundError(f"Missing lineage graph: {LINEAGE_PATH}")
    submitted = variants()["fractional_multilabel"]
    nodes, edges, topics = build_map(submitted)
    all_papers, meeting_activity, unmatched_categories = explode_papers(submitted, topics)
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    outcomes, _, relation_counts, edge_audit = build_lineage_payload(lineage, all_papers)
    linked_paper_ids = {
        edge["paper"] for outcome in outcomes for edge in outcome["direct_papers"]
    }
    papers = {paper_id: all_papers[paper_id] for paper_id in sorted(linked_paper_ids)}
    payload = {
        "meta": {
            "source": str(LINEAGE_PATH),
            "scope": "complete lineage graph with relationship-evidence reclassification",
            "activity_unit": "ATCM meeting",
            "operative_relations": sorted(OPERATIVE_RELATIONS),
            "paper_relation_counts": relation_counts,
            "unmatched_category_rows": unmatched_categories,
            "archive_papers_with_categories": len(all_papers),
        },
        "nodes": nodes,
        "edges": edges,
        "meeting_activity": meeting_activity,
        "papers": papers,
        "outcomes": outcomes,
    }
    payload["meta"]["validation"] = validate_payload(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_HTML.write_text(html_document(payload), encoding="utf-8")
    pd.DataFrame(edge_audit).to_csv(OUT_EDGE_AUDIT, index=False)
    outcome_audit = []
    for outcome in outcomes:
        counts = defaultdict(int)
        for edge in outcome["direct_papers"]:
            counts[edge["evidence_level"]] += 1
        outcome_audit.append(
            {
                "outcome": outcome["display_id"],
                "outcome_type": outcome["type"],
                "meeting": outcome["meeting"],
                "year": outcome["year"],
                "title": outcome["title"],
                "confirmed_edges": counts["confirmed"],
                "corroborated_edges": counts["corroborated"],
                "contextual_edges": counts["contextual"],
                "candidate_edges": counts["candidate"],
                "rejected_edges": counts["rejected"],
                "strong_route": int(counts["confirmed"] + counts["corroborated"] > 0),
                "formal_ancestor_edges": len(outcome["incoming_outcomes"]),
            }
        )
    pd.DataFrame(outcome_audit).to_csv(OUT_OUTCOME_AUDIT, index=False)
    print(json.dumps(payload["meta"]["validation"], indent=2))
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_EDGE_AUDIT}")
    print(f"Wrote {OUT_OUTCOME_AUDIT}")


if __name__ == "__main__":
    main()
