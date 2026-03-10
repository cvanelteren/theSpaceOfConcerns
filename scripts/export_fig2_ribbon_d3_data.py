#!/usr/bin/env python3
"""Export Figure 2 ribbon-plot data for the interactive D3 app."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figS03_specialized_support_ribbons import (
    DATA_PATHS,
    DEFAULT_THEME,
    EDGE_WIDTH_METRIC,
    OTHER_LABEL,
    PERIOD_YEARS,
    RCA_THRESHOLD,
    START_YEAR,
    THEME_COLORS,
    TRANSITION_SOURCE,
    TOP_N_COUNTRIES,
    TOP_N_TOPICS,
    TOPIC_TO_THEME,
)
from utils import (
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

OUT_JSON = Path("d3_ribbon_plot/data/fig2_ribbon_data.json")


def _normalize_topic(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def _load_data_with_fallback() -> tuple[pd.DataFrame, pd.DataFrame, set[str], set[str]]:
    last_error = None
    for path in DATA_PATHS:
        if not Path(path).exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Failed to load {path}: {exc}")
    raise FileNotFoundError("No usable data file found in DATA_PATHS.") from last_error


def _build_periods(
    year_min: int, year_max: int, step: int
) -> list[tuple[int, int, str]]:
    start = year_min if START_YEAR is None else max(START_YEAR, year_min)
    out = []
    y = start
    while y <= year_max:
        end = min(y + step - 1, year_max)
        out.append((y, end, f"{y}-{end}"))
        y = end + 1
    return out


def _sanitize_years(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[col])


def _topic_theme(topic: str, lookup: dict[str, str]) -> str:
    return lookup.get(_normalize_topic(topic), DEFAULT_THEME)


# %%
counts_df, submitted_df, countries, topics = _load_data_with_fallback()
year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
if year_col not in submitted_df.columns:
    raise KeyError("No meeting year or year column found in source data.")

submitted_df = _sanitize_years(submitted_df, year_col)
year_min = int(submitted_df[year_col].min())
year_max = int(submitted_df[year_col].max())

periods = _build_periods(year_min, year_max, PERIOD_YEARS)
period_labels = [label for _, _, label in periods]
period_index = {label: i for i, label in enumerate(period_labels)}


# %%
# 1) Per-period topic assignments for each country.
rows = []
for start, end, label in periods:
    period_df = submitted_df[
        (submitted_df[year_col] >= start) & (submitted_df[year_col] <= end)
    ]
    if period_df.empty:
        continue
    interaction = generate_interaction_matrix(period_df, countries, topics)
    interaction = standardize_index_labels(interaction)
    if interaction.index.has_duplicates:
        interaction = interaction.groupby(level=0).sum()
    rca = get_rca(interaction)
    for country in rca.columns:
        series = rca[country]
        if series.sum() <= 0:
            continue
        if TRANSITION_SOURCE == "top_rca":
            top_topic = series.idxmax()
            top_rca = float(series.loc[top_topic])
            if top_rca < RCA_THRESHOLD:
                continue
            rows.append(
                {
                    "period": label,
                    "country": country,
                    "topic": str(top_topic),
                    "rca": top_rca,
                }
            )
        else:
            active = series[series >= RCA_THRESHOLD]
            if active.empty:
                continue
            for topic, value in active.items():
                rows.append(
                    {
                        "period": label,
                        "country": country,
                        "topic": str(topic),
                        "rca": float(value),
                    }
                )

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No RCA-qualified entries found for ribbon export.")

if TOP_N_COUNTRIES is not None:
    keep = df["country"].value_counts().head(TOP_N_COUNTRIES).index.tolist()
    df = df[df["country"].isin(keep)]

topic_counts = df["topic"].value_counts()
top_topics = topic_counts.head(TOP_N_TOPICS).index.tolist()
df["topic_group"] = df["topic"].where(df["topic"].isin(top_topics), OTHER_LABEL)

periods_present = [p for p in period_labels if p in set(df["period"])]
period_position = {p: i for i, p in enumerate(periods_present)}


# %%
# 2) Stable topic ordering (group by theme, then active span).
theme_lookup = {_normalize_topic(k): v for k, v in TOPIC_TO_THEME.items()}
topics_order = top_topics + ([OTHER_LABEL] if OTHER_LABEL in df["topic_group"].values else [])

period_counts = (
    df.groupby(["period", "topic_group"]).size().rename("count").reset_index()
)

period_span = {}
for topic in topics_order:
    pset = period_counts.loc[period_counts["topic_group"] == topic, "period"].unique()
    if pset.size == 0:
        period_span[topic] = 0
        continue
    idxs = [period_position[p] for p in pset]
    period_span[topic] = max(idxs) - min(idxs)

theme_groups = defaultdict(list)
for topic in topics_order:
    theme_groups[_topic_theme(topic, theme_lookup)].append(topic)

theme_groups_sorted = []
topics_sorted = []
for theme in sorted(theme_groups.keys()):
    group = sorted(theme_groups[theme], key=lambda t: period_span.get(t, 0), reverse=True)
    theme_groups_sorted.append((theme, group))
    topics_sorted.extend(group)
topics_order = topics_sorted

topic_color_map = {
    t: THEME_COLORS.get(_topic_theme(t, theme_lookup), THEME_COLORS[DEFAULT_THEME])
    for t in topics_order
}


# %%
# 3) Country paths and transitions between consecutive periods.
country_paths = {}
transitions = defaultdict(lambda: {"support": 0.0, "countries": set()})

if TRANSITION_SOURCE == "top_rca":
    for country, group in df.groupby("country"):
        group = group.copy()
        group["period_order"] = group["period"].map(period_index)
        group = group.sort_values("period_order")
        path = []
        for _, row in group.iterrows():
            path.append(
                {
                    "period": row["period"],
                    "topic": row["topic_group"],
                    "rca": float(row["rca"]),
                    "period_order": int(row["period_order"]),
                }
            )
        country_paths[country] = path

        for i in range(len(group) - 1):
            cur = group.iloc[i]
            nxt = group.iloc[i + 1]
            if int(nxt["period_order"]) != int(cur["period_order"]) + 1:
                continue
            key = (
                str(cur["period"]),
                str(cur["topic_group"]),
                str(nxt["period"]),
                str(nxt["topic_group"]),
            )
            transitions[key]["support"] += 1.0
            transitions[key]["countries"].add(country)
else:
    per_country_period = (
        df.groupby(["country", "period"])["topic_group"]
        .apply(lambda s: sorted(set(s)))
        .reset_index()
    )

    for country, group in df.groupby("country"):
        group = group.copy()
        group["period_order"] = group["period"].map(period_index)
        group = group.sort_values(["period_order", "topic_group"])
        path = []
        for _, row in group.iterrows():
            path.append(
                {
                    "period": row["period"],
                    "topic": row["topic_group"],
                    "rca": float(row["rca"]),
                    "period_order": int(row["period_order"]),
                }
            )
        country_paths[country] = path

    for country, group in per_country_period.groupby("country"):
        period_topics = {
            str(row["period"]): list(row["topic_group"]) for _, row in group.iterrows()
        }
        for i in range(len(periods_present) - 1):
            p0 = periods_present[i]
            p1 = periods_present[i + 1]
            topics0 = period_topics.get(p0, [])
            topics1 = period_topics.get(p1, [])
            if not topics0 or not topics1:
                continue
            w = 1.0 / (len(topics0) * len(topics1))
            for t0 in topics0:
                for t1 in topics1:
                    key = (str(p0), str(t0), str(p1), str(t1))
                    transitions[key]["support"] += w
                    transitions[key]["countries"].add(country)


# %%
# 4) Node records with start/end flags per topic.
nodes = []
topic_active_periods = {}
for topic in topics_order:
    pset = sorted(
        period_counts.loc[period_counts["topic_group"] == topic, "period"].unique().tolist(),
        key=lambda p: period_position.get(p, 10_000),
    )
    topic_active_periods[topic] = pset

count_lookup = {
    (str(r["period"]), str(r["topic_group"])): int(r["count"])
    for _, r in period_counts.iterrows()
}
node_country_lookup = (
    df.groupby(["period", "topic_group"])["country"]
    .apply(lambda s: sorted(set(map(str, s))))
    .to_dict()
)

for period in periods_present:
    for topic in topics_order:
        count = count_lookup.get((period, topic), 0)
        if count <= 0:
            continue
        pset = topic_active_periods.get(topic, [])
        first = pset[0] if pset else None
        last = pset[-1] if pset else None
        node_id = f"{period}::{topic}"
        theme = _topic_theme(topic, theme_lookup)
        nodes.append(
            {
                "id": node_id,
                "period": period,
                "topic": topic,
                "theme": theme,
                "color": topic_color_map.get(topic, "#888888"),
                "count": count,
                "is_start": period == first,
                "is_end": period == last,
                "period_index": int(period_position[period]),
                "topic_index": int(topics_order.index(topic)),
                "countries": node_country_lookup.get((str(period), str(topic)), []),
            }
        )


# %%
# 5) Link records.
links = []
for (p0, t0, p1, t1), payload in transitions.items():
    if p0 not in period_position or p1 not in period_position:
        continue
    if period_position[p1] != period_position[p0] + 1:
        continue
    actor_count = int(len(payload["countries"]))
    support_value = float(payload["support"])
    if EDGE_WIDTH_METRIC == "actor_count":
        value = float(actor_count)
    elif EDGE_WIDTH_METRIC == "weighted_support":
        value = support_value
    else:
        raise ValueError(f"Unsupported EDGE_WIDTH_METRIC: {EDGE_WIDTH_METRIC}")
    links.append(
        {
            "id": f"{p0}::{t0}-->{p1}::{t1}",
            "source": f"{p0}::{t0}",
            "target": f"{p1}::{t1}",
            "period0": p0,
            "period1": p1,
            "source_topic": t0,
            "target_topic": t1,
            "source_theme": _topic_theme(t0, theme_lookup),
            "target_theme": _topic_theme(t1, theme_lookup),
            "value": value,
            "support_value": support_value,
            "countries": sorted(payload["countries"]),
            "actor_count": actor_count,
        }
    )

links = sorted(
    links,
    key=lambda d: (
        period_position[d["period0"]],
        topics_order.index(d["source_topic"]),
        topics_order.index(d["target_topic"]),
    ),
)


# %%
# 6) Theme composition per period (bottom panel).
df_theme = df.copy()
df_theme["theme"] = df_theme["topic_group"].map(lambda t: _topic_theme(t, theme_lookup))
theme_order = [t for t in THEME_COLORS.keys() if t in set(df_theme["theme"])]
theme_counts = (
    df_theme.groupby(["period", "theme"]).size().unstack(fill_value=0).reindex(
        index=periods_present, columns=theme_order, fill_value=0
    )
)
theme_count_records = []
for period in periods_present:
    rec = {"period": period}
    for theme in theme_order:
        rec[theme] = int(theme_counts.loc[period, theme])
    theme_count_records.append(rec)


# %%
# 7) Export.
out = {
    "meta": {
        "description": "Interactive Figure 2 ribbon data export",
        "period_years": int(PERIOD_YEARS),
        "rca_threshold": float(RCA_THRESHOLD),
        "transition_source": str(TRANSITION_SOURCE),
        "edge_width_metric": str(EDGE_WIDTH_METRIC),
        "n_rows": int(len(df)),
        "n_topics": int(len(topics_order)),
        "n_countries": int(df["country"].nunique()),
    },
    "periods": periods_present,
    "topics_order": topics_order,
    "themes_order": theme_order,
    "theme_colors": THEME_COLORS,
    "nodes": nodes,
    "links": links,
    "country_paths": country_paths,
    "theme_counts": theme_count_records,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Saved {OUT_JSON}")
print(f"Nodes: {len(nodes)} | Links: {len(links)} | Countries: {len(country_paths)}")
