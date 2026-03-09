#!/usr/bin/env python3
"""Export a static D3 payload for the ATS space of concerns with actor RPA>1 placements."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from shutil import copy2
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import get_rca, load_data

APP_ROOT = ROOT / "d3_space_actor_interests"
GRAPH_PATH = ROOT / "d3_space_of_concerns/data/space_of_concerns_graph.json"
LAYOUT_PATH = ROOT / "d3_space_of_concerns/data/space_of_concerns_layout_saved.json"
OUT_PATH = APP_ROOT / "data/space_actor_interests.json"
FLAG_DIR = ROOT / "assets/flags"
APP_FLAG_DIR = APP_ROOT / "assets/flags"
CONTOUR_SRC = ROOT / "d3_space_of_concerns/assets/antarctica_contour.png"
CONTOUR_DST = APP_ROOT / "assets/antarctica_contour.png"

DATA_PATHS = [
    ROOT / "antarctic-database-go/data/processed/document-summary.parquet",
    ROOT / "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet",
    ROOT / "Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv",
    ROOT / "document-summary.csv",
]

EXPORT_RPA_FLOOR = 0.0
DEFAULT_RPA_THRESHOLD = 1.0


def _normalize_topic(value: str) -> str:
    return " ".join(str(value).replace("_", " ").strip().split()).lower()


def _load_data_with_fallback():
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


def _load_graph_payload() -> tuple[list[dict], list[dict], dict[str, str]]:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    layout_by_id = {rec["id"]: rec for rec in layout.get("nodes", [])}

    nodes = []
    for rec in graph["nodes"]:
        pos = layout_by_id.get(rec["id"], {})
        nodes.append(
            {
                **rec,
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
            }
        )
    return nodes, graph["links"], graph["theme_colors"]


def _actor_kind_and_icon(name: str) -> tuple[str, str | None]:
    mapped = name
    low = name.lower()
    if "korea (rok)" in low:
        mapped = "Korea"
    elif "korea (dprk)" in low:
        mapped = "Korea, Democratic People's Republic of"

    flag = FLAG_DIR / f"{mapped}_flag.png"
    if flag.exists():
        APP_FLAG_DIR.mkdir(parents=True, exist_ok=True)
        copy2(flag, APP_FLAG_DIR / flag.name)
        return "state", f"./assets/flags/{flag.name}"

    logo = FLAG_DIR / f"{name.lower()}_logo.png"
    if logo.exists():
        APP_FLAG_DIR.mkdir(parents=True, exist_ok=True)
        copy2(logo, APP_FLAG_DIR / logo.name)
        return "organization", f"./assets/flags/{logo.name}"

    return "other", None


def build_payload() -> dict:
    counts_df, submitted_df, actors, _ = _load_data_with_fallback()
    nodes, links, theme_colors = _load_graph_payload()
    node_by_id = {rec["id"]: rec for rec in nodes}
    canonical_topic = {_normalize_topic(rec["id"]): rec["id"] for rec in nodes}

    rpa = get_rca(counts_df)
    actor_type_map = {}
    if "submitted by" in submitted_df.columns and "party_type" in submitted_df.columns:
        actor_type_map = (
            submitted_df[["submitted by", "party_type"]]
            .dropna()
            .assign(**{"submitted by": lambda df: df["submitted by"].astype(str).str.strip()})
            .groupby("submitted by")["party_type"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .to_dict()
        )

    topic_supports: dict[str, list[dict]] = defaultdict(list)
    actor_rows = []
    actor_count = 0
    placement_count = 0

    for actor in sorted(rpa.columns):
        series = rpa[actor]
        active = series[series > EXPORT_RPA_FLOOR].sort_values(ascending=False)
        placements = []
        for raw_topic, value in active.items():
            topic_id = canonical_topic.get(_normalize_topic(raw_topic))
            if topic_id is None or topic_id not in node_by_id:
                continue
            node = node_by_id[topic_id]
            placement = {
                "topic": topic_id,
                "rpa": round(float(value), 4),
                "x": node["x"],
                "y": node["y"],
                "theme": node["theme"],
                "color": node["color"],
            }
            placements.append(placement)
            if float(value) > DEFAULT_RPA_THRESHOLD:
                topic_supports[topic_id].append(
                    {"actor": actor, "rpa": round(float(value), 4)}
                )

        if not placements:
            continue

        actor_count += 1
        placement_count += len(placements)
        kind, icon_path = _actor_kind_and_icon(actor)

        weights = pd.Series([rec["rpa"] for rec in placements], dtype=float)
        xs = pd.Series([rec["x"] for rec in placements], dtype=float)
        ys = pd.Series([rec["y"] for rec in placements], dtype=float)
        centroid_x = float((weights * xs).sum() / weights.sum())
        centroid_y = float((weights * ys).sum() / weights.sum())

        theme_totals: dict[str, float] = defaultdict(float)
        for rec in placements:
            theme_totals[rec["theme"]] += float(rec["rpa"])
        dominant_theme = max(theme_totals.items(), key=lambda kv: kv[1])[0]

        actor_rows.append(
            {
                "id": actor,
                "kind": kind,
                "source_type": actor_type_map.get(actor),
                "icon_path": icon_path,
                "support_size": len(placements),
                "mean_rpa": round(float(weights.mean()), 4),
                "max_rpa": round(float(weights.max()), 4),
                "dominant_theme": dominant_theme,
                "centroid": {"x": round(centroid_x, 3), "y": round(centroid_y, 3)},
                "topics": placements,
            }
        )

    for node in nodes:
        supports = topic_supports.get(node["id"], [])
        values = [float(rec["rpa"]) for rec in supports]
        node["support_count"] = len(supports)
        node["mean_support_rpa"] = round(sum(values) / len(values), 4) if values else 0.0
        node["top_actors"] = sorted(supports, key=lambda rec: rec["rpa"], reverse=True)[:8]

    actor_rows.sort(key=lambda rec: (-rec["support_size"], rec["id"]))

    return {
        "meta": {
            "n_topics": len(nodes),
            "n_links": len(links),
            "n_actors": actor_count,
            "n_placements": placement_count,
            "export_rpa_floor": EXPORT_RPA_FLOOR,
            "default_rpa_threshold": DEFAULT_RPA_THRESHOLD,
        },
        "theme_colors": theme_colors,
        "nodes": nodes,
        "links": links,
        "actors": actor_rows,
        "actor_kinds": sorted({rec["kind"] for rec in actor_rows}),
    }


def main() -> None:
    payload = build_payload()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTOUR_DST.parent.mkdir(parents=True, exist_ok=True)
    if CONTOUR_SRC.exists():
        copy2(CONTOUR_SRC, CONTOUR_DST)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(payload["meta"])


if __name__ == "__main__":
    main()
