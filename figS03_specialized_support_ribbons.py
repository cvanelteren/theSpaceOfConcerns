"""Supplementary Figure S03. Shows the full RPA>1 ribbon view across adjacent periods. Complements the filtered main-text ribbon by exposing the broader pattern of specialized continuity and shift."""

from __future__ import annotations

# Topic-aligned ribbon plot with fixed y positions across periods.
#
# By default, transitions are built from all topics with RCA >= threshold per
# country-period. Set TRANSITION_SOURCE="top_rca" to recover the previous
# behavior. Ribbon width is driven by EDGE_WIDTH_METRIC. Running this script
# writes two outputs:
# 1) main-text dominant-flow view (`fig2_space_of_concerns_ribbon.*`)
# 2) full unfiltered appendix view (`fig2_space_of_concerns_ribbon_full.*`)

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib import patches as mpatches
from matplotlib import path as mpath
from scipy.optimize import linprog

from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

# -----------------------------
# Configuration
# -----------------------------
DATA_PATHS = [
    Path("antarctic-database-go/data/processed/document-summary.parquet"),
    Path("antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"),
    Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
    Path("document-summary.csv"),
]

START_YEAR = None  # None = use dataset min
PERIOD_YEARS = 10
RCA_THRESHOLD = 1.0
TOP_N_TOPICS = 1000
TOP_N_COUNTRIES = None  # None = use all
OTHER_LABEL = "Other topics"
DEFAULT_THEME = "Governance & Legal"
EMD_TOP_N = 100
TRANSITION_SOURCE = "all_rca"  # "all_rca" | "top_rca"
EDGE_WIDTH_METRIC = "weighted_support"  # "actor_count" | "weighted_support"
MAIN_MIN_ACTOR_COUNT = 2
MAIN_MAX_NONSELF_FLOWS_PER_PAIR = 35
MAIN_MAX_SELF_FLOWS_PER_PAIR = 20

FIG_WIDTH = "26cm"
FIG_HEIGHT = "16cm"

X_MARGIN = 0.1
Y_MARGIN = 0.08
ROW_HEIGHT_RATIO = 2.5
FLOW_CURVATURE = 0.5
FLOW_ALPHA = 0.6

NODE_WIDTH = 0.02
NODE_SIZE_MODE = "count_scaled"  # "count_scaled" | "fixed"
NODE_FIXED_HEIGHT = 0.02  # used when NODE_SIZE_MODE == "fixed" (axes units)
LABEL_OFFSET = 0.02
LABEL_SIZE = 8
LABEL_BOX = True
THEME_LABEL_OFFSET = 0.01
THEME_LABEL_SIZE = 9
THEME_LABEL_SCALE = 0.25
THEME_LABEL_JITTER = 0.5
THEME_LABEL_HEADER_OFFSET = 0.9
BAND_COLORS = ("white", "#f7f7f7")
GRID_COLOR = "k"  # "#d0d0d0"
GRID_ALPHA = 0.6
GRID_DASH = (0, (3, 3))
THEME_LABEL_X_OFFSETS = {
    "Environmental Protection": -0.02,
    "Governance & Legal": -0.04,
    "Infrastructure & Planning": -0.06,
    "Marine & Wildlife": -0.02,
    "Operations & Safety": -0.04,
    "Resource Extraction": -0.06,
    "Science & Research": -0.02,
    "Tourism & Human Activity": -0.04,
}

OUT_PNG = Path("output/fig07_specialized_support_ribbon.png")
OUT_PDF = Path("output/fig07_specialized_support_ribbon.pdf")
OUT_FULL_PNG = Path("figures/figS03_specialized_support_ribbons.png")
OUT_FULL_PDF = Path("figures/figS03_specialized_support_ribbons.pdf")

# Define thematic color mapping for nodes
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


def load_data_with_fallback() -> tuple[pd.DataFrame, pd.DataFrame, set[str], set[str]]:
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


def smartwrap(text, width):
    """Wrap text for better display in Sankey labels"""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def build_periods(
    year_min: int, year_max: int, step: int
) -> list[tuple[int, int, str]]:
    start = year_min if START_YEAR is None else max(START_YEAR, year_min)
    periods = []
    year = start
    while year <= year_max:
        end = min(year + step - 1, year_max)
        periods.append((year, end, f"{year}-{end}"))
        year = end + 1
    return periods


def sanitize_years(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    return df.dropna(subset=[year_col])


def _emd_distance(p: np.ndarray, q: np.ndarray, dist: np.ndarray) -> float:
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p = p / p.sum()
    q = q / q.sum()
    n = len(p)
    c = dist.flatten()
    A_eq = []
    b_eq = []
    # Row constraints
    for i in range(n):
        row = np.zeros(n * n)
        row[i * n : (i + 1) * n] = 1
        A_eq.append(row)
        b_eq.append(p[i])
    # Column constraints
    for j in range(n):
        col = np.zeros(n * n)
        col[j::n] = 1
        A_eq.append(col)
        b_eq.append(q[j])
    bounds = [(0, None)] * (n * n)
    res = linprog(
        c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs"
    )
    if not res.success:
        return np.nan
    return float(res.fun)


def _normalize_topic(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def filter_main_flows_for_pair(
    flows: list[tuple[str, str, float, float, float]],
) -> list[tuple[str, str, float, float, float]]:
    self_flows = [
        f for f in flows if f[0] == f[1] and float(f[3]) >= float(MAIN_MIN_ACTOR_COUNT)
    ]
    nonself_flows = [
        f for f in flows if f[0] != f[1] and float(f[3]) >= float(MAIN_MIN_ACTOR_COUNT)
    ]

    self_top = sorted(self_flows, key=lambda f: (f[2], f[3]), reverse=True)[
        : int(MAIN_MAX_SELF_FLOWS_PER_PAIR)
    ]
    nonself_top = sorted(nonself_flows, key=lambda f: (f[2], f[3]), reverse=True)[
        : int(MAIN_MAX_NONSELF_FLOWS_PER_PAIR)
    ]
    return self_top + nonself_top


def compute_transition_mechanism_counts(
    per_country_period: pd.DataFrame,
    periods_present: list[str],
) -> pd.DataFrame:
    country_topics = {}
    for country, group in per_country_period.groupby("country"):
        country_topics[str(country)] = {
            str(row["period"]): set(row["topic_group"]) for _, row in group.iterrows()
        }

    rows = []
    for idx in range(len(periods_present) - 1):
        p0 = periods_present[idx]
        p1 = periods_present[idx + 1]
        persistent = 0
        entries = 0
        exits = 0
        for period_map in country_topics.values():
            s0 = period_map.get(p0, set())
            s1 = period_map.get(p1, set())
            if not s0 and not s1:
                continue
            persistent += len(s0 & s1)
            entries += len(s1 - s0)
            exits += len(s0 - s1)
        rows.append(
            {
                "pair_idx": idx,
                "pair_label": f"{p0}->{p1}",
                "period0": p0,
                "period1": p1,
                "persistent": int(persistent),
                "entries": int(entries),
                "exits": int(exits),
            }
        )

    return pd.DataFrame(rows)


def ribbon_path(
    x0: float, y0: float, x1: float, y1: float, thickness: float, curvature: float
) -> mpath.Path:
    dx = max(x1 - x0, 1e-6)
    cx0 = x0 + dx * curvature
    cx1 = x1 - dx * curvature
    top0 = y0 + thickness / 2
    bot0 = y0 - thickness / 2
    top1 = y1 + thickness / 2
    bot1 = y1 - thickness / 2
    verts = [
        (x0, top0),
        (cx0, top0),
        (cx1, top1),
        (x1, top1),
        (x1, bot1),
        (cx1, bot1),
        (cx0, bot0),
        (x0, bot0),
        (x0, top0),
    ]
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CLOSEPOLY,
    ]
    return mpath.Path(verts, codes)


def main(view_mode: str = "main") -> None:
    if view_mode not in {"main", "full"}:
        raise ValueError(f"Unsupported view_mode '{view_mode}'. Use 'main' or 'full'.")
    out_png = OUT_PNG if view_mode == "main" else OUT_FULL_PNG
    out_pdf = OUT_PDF if view_mode == "main" else OUT_FULL_PDF

    counts_df, submitted_df, countries, topics = load_data_with_fallback()

    year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns:
        raise KeyError("No meeting year or year column found in data.")

    submitted_df = sanitize_years(submitted_df, year_col)
    year_min = int(submitted_df[year_col].min())
    year_max = int(submitted_df[year_col].max())

    periods = build_periods(year_min, year_max, PERIOD_YEARS)
    period_labels = [label for _, _, label in periods]
    period_index = {label: idx for idx, label in enumerate(period_labels)}

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
                top_rca_value = float(series.loc[top_topic])
                if top_rca_value < RCA_THRESHOLD:
                    continue
                rows.append(
                    {
                        "period": label,
                        "country": country,
                        "topic": top_topic,
                        "rca": top_rca_value,
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
                            "topic": topic,
                            "rca": float(value),
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No RCA-qualified entries found for ribbon plot.")

    if TOP_N_COUNTRIES is not None:
        top_countries = (
            df["country"].value_counts().head(TOP_N_COUNTRIES).index.tolist()
        )
        df = df[df["country"].isin(top_countries)]

    topic_counts = df["topic"].value_counts()
    top_topics = topic_counts.head(TOP_N_TOPICS).index.tolist()
    df["topic_group"] = df["topic"].where(df["topic"].isin(top_topics), OTHER_LABEL)

    periods_present = [
        label for label in period_labels if label in df["period"].unique()
    ]
    period_position = {label: idx for idx, label in enumerate(periods_present)}

    topics_order = top_topics + (
        [OTHER_LABEL] if OTHER_LABEL in df["topic_group"].values else []
    )
    topic_theme_lookup = {
        _normalize_topic(topic): theme for topic, theme in TOPIC_TO_THEME.items()
    }

    def _topic_theme(topic: str) -> str:
        return topic_theme_lookup.get(_normalize_topic(topic), DEFAULT_THEME)

    topic_color_map = {}
    for topic in topics_order:
        theme = _topic_theme(topic)
        topic_color_map[topic] = THEME_COLORS.get(theme, THEME_COLORS[DEFAULT_THEME])

    # Counts per period/topic for node sizing.
    period_counts = (
        df.groupby(["period", "topic_group"]).size().rename("count").reset_index()
    )
    max_count = period_counts["count"].max() if not period_counts.empty else 0
    per_country_period = (
        df.groupby(["country", "period"])["topic_group"]
        .apply(lambda s: sorted(set(s)))
        .reset_index()
    )

    # Reorder topics: group by theme, then by longest active span (desc).
    period_span = {}
    for topic in topics_order:
        topic_periods = period_counts.loc[
            period_counts["topic_group"] == topic, "period"
        ].unique()
        if topic_periods.size == 0:
            period_span[topic] = 0
            continue
        indices = [period_position[p] for p in topic_periods]
        period_span[topic] = max(indices) - min(indices)

    theme_groups = defaultdict(list)
    for topic in topics_order:
        theme = topic_theme_lookup.get(_normalize_topic(topic), DEFAULT_THEME)
        theme_groups[theme].append(topic)

    sorted_topics = []
    theme_order = sorted(theme_groups.keys())
    theme_groups_sorted = []
    for theme in theme_order:
        group = theme_groups[theme]
        group = sorted(group, key=lambda t: period_span.get(t, 0), reverse=True)
        theme_groups_sorted.append((theme, group))
        sorted_topics.extend(group)
    topics_order = sorted_topics

    # Global topic space for EMD (use top EMD_TOP_N topics by overall count).
    emd_topics = topic_counts.head(EMD_TOP_N).index.tolist()
    overall_interaction = generate_interaction_matrix(submitted_df, countries, topics)
    overall_interaction = standardize_index_labels(overall_interaction)
    if overall_interaction.index.has_duplicates:
        overall_interaction = overall_interaction.groupby(level=0).sum()
    overall_rca = get_rca(overall_interaction)
    phi = compute_product_space(overall_rca)
    phi = phi.reindex(index=emd_topics, columns=emd_topics, fill_value=0.0)
    dist = 1 - phi.to_numpy()

    # Transitions between consecutive periods.
    # In all_rca mode, each country contributes:
    # - actor support: membership in each crossed topic pair
    # - weighted support: total mass 1 per adjacent period pair distributed over pairs
    transitions = defaultdict(lambda: {"support": 0.0, "countries": set()})
    if TRANSITION_SOURCE == "top_rca":
        for country, group in df.groupby("country"):
            group = group.copy()
            group["period_order"] = group["period"].map(period_index)
            group = group.sort_values("period_order")
            for i in range(len(group) - 1):
                curr = group.iloc[i]
                nxt = group.iloc[i + 1]
                if nxt["period_order"] != curr["period_order"] + 1:
                    continue
                key = (
                    curr["period"],
                    curr["topic_group"],
                    nxt["period"],
                    nxt["topic_group"],
                )
                transitions[key]["support"] += 1.0
                transitions[key]["countries"].add(country)
    else:
        for country, group in per_country_period.groupby("country"):
            period_topics = {
                row["period"]: row["topic_group"] for _, row in group.iterrows()
            }
            for i in range(len(periods_present) - 1):
                p0 = periods_present[i]
                p1 = periods_present[i + 1]
                topics0 = period_topics.get(p0, [])
                topics1 = period_topics.get(p1, [])
                if not topics0 or not topics1:
                    continue
                weight = 1.0 / (len(topics0) * len(topics1))
                for t0 in topics0:
                    for t1 in topics1:
                        transitions[(p0, t0, p1, t1)]["support"] += weight
                        transitions[(p0, t0, p1, t1)]["countries"].add(country)

    fig, axs = uplt.subplots(
        nrows=2,
        refwidth=FIG_WIDTH,
        refheight=FIG_HEIGHT,
        share=0,
        hratios=[3, 0.5],
    )
    ax = axs[0]

    n_topics = max(len(topics_order), 1)
    row_gap = (1.0 - 2 * Y_MARGIN) / n_topics
    row_height = row_gap * ROW_HEIGHT_RATIO
    topic_y = {
        topic: 1.0 - Y_MARGIN - (idx + 0.5) * row_gap
        for idx, topic in enumerate(topics_order)
    }

    x_positions = np.linspace(X_MARGIN, 1.0 - X_MARGIN, len(periods_present))
    period_x = {label: x_positions[idx] for idx, label in enumerate(periods_present)}

    # Alternating background bands per period column.
    if len(periods_present) > 1:
        dx = x_positions[1] - x_positions[0]
    else:
        dx = 1.0 - 2 * X_MARGIN
    for idx, label in enumerate(periods_present):
        x0 = max(0.0, period_x[label] - dx / 2)
        x1 = min(1.0, period_x[label] + dx / 2)
        # ax.add_patch(
        #     mpatches.Rectangle(
        #         (x0, Y_MARGIN / 2),
        #         x1 - x0,
        #         1.0 - Y_MARGIN,
        #         facecolor=BAND_COLORS[idx % 2],
        #         edgecolor="none",
        #         zorder=0,
        #     )
        # )

    # Draw nodes (fixed y positions).
    if NODE_SIZE_MODE == "count_scaled":
        if max_count > 0:
            node_scale = row_height * 2 / max_count
        else:
            node_scale = 0.0
    elif NODE_SIZE_MODE == "fixed":
        node_scale = None
    else:
        raise ValueError(f"Unsupported NODE_SIZE_MODE: {NODE_SIZE_MODE}")

    for _, row in period_counts.iterrows():
        period = row["period"]
        topic = row["topic_group"]
        if period not in period_x or topic not in topic_y:
            continue
        if NODE_SIZE_MODE == "fixed":
            height = min(float(NODE_FIXED_HEIGHT), row_height * 1.95)
        else:
            height = row["count"] * float(node_scale)
        if height <= 0:
            continue
        y_center = topic_y[topic]
        x_center = period_x[period]
        topic_periods = (
            period_counts.loc[period_counts["topic_group"] == topic, "period"]
            .unique()
            .tolist()
        )
        first_active = topic_periods[0] if topic_periods else None
        last_active = topic_periods[-1] if topic_periods else None
        is_start = period == first_active
        is_end = period == last_active
        edgecolor = "black" if (is_start or is_end) else "none"
        linewidth = 1.0 if edgecolor == "black" else 0.0
        hatch = "////" if is_start else ("\\\\\\\\" if is_end else None)
        patch = mpatches.FancyBboxPatch(
            (x_center - NODE_WIDTH / 2, y_center - height / 2),
            NODE_WIDTH,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.008",
            facecolor=topic_color_map.get(topic, "0.7"),
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
        )
        ax.add_patch(patch)

    # Soft dashed horizontal guides through topic centers.
    for topic in topics_order:
        y = topic_y.get(topic, None)
        if y is None:
            continue
        topic_periods = period_counts.loc[
            period_counts["topic_group"] == topic, "period"
        ].unique()
        if topic_periods.size == 0:
            continue
        x_min = min(period_x[p] for p in topic_periods) - NODE_WIDTH / 2
        x_max = max(period_x[p] for p in topic_periods) + NODE_WIDTH / 2
        ax.hlines(
            y,
            x_min,
            x_max,
            colors=GRID_COLOR,
            alpha=GRID_ALPHA,
            linestyles=GRID_DASH,
            linewidth=0.7,
            zorder=0,
        )

    # Draw ribbons between consecutive periods.
    transitions_by_pair = defaultdict(list)
    for (p0, t0, p1, t1), payload in transitions.items():
        if p0 not in period_position or p1 not in period_position:
            continue
        if period_position[p1] != period_position[p0] + 1:
            continue
        actor_count = float(len(payload["countries"]))
        support = float(payload["support"])
        if EDGE_WIDTH_METRIC == "actor_count":
            value = actor_count
        elif EDGE_WIDTH_METRIC == "weighted_support":
            value = support
        else:
            raise ValueError(f"Unsupported EDGE_WIDTH_METRIC: {EDGE_WIDTH_METRIC}")
        transitions_by_pair[(p0, p1)].append((t0, t1, value, actor_count, support))

    n_flows_full = int(sum(len(v) for v in transitions_by_pair.values()))
    if view_mode == "main":
        filtered_by_pair = defaultdict(list)
        for pair, flows in transitions_by_pair.items():
            kept = filter_main_flows_for_pair(flows)
            if kept:
                filtered_by_pair[pair] = kept
        transitions_by_pair = filtered_by_pair
    n_flows_kept = int(sum(len(v) for v in transitions_by_pair.values()))

    for (p0, p1), flows in transitions_by_pair.items():
        x0 = period_x[p0]
        x1 = period_x[p1]
        # Scale thickness within each topic row.
        totals_by_src = defaultdict(float)
        totals_by_tgt = defaultdict(float)
        for t0, t1, value, _, _ in flows:
            totals_by_src[t0] += value
            totals_by_tgt[t1] += value
        max_total = max(totals_by_src.values()) if totals_by_src else 0.0
        scale = row_height * 0.9 / max_total if max_total > 0 else 0.0

        # Source offsets.
        src_offsets = {}
        for src, total in totals_by_src.items():
            y_center = topic_y.get(src, 0.5)
            src_offsets[src] = y_center - (total * scale) / 2
        # Target offsets.
        tgt_offsets = {}
        for tgt, total in totals_by_tgt.items():
            y_center = topic_y.get(tgt, 0.5)
            tgt_offsets[tgt] = y_center - (total * scale) / 2

        # Order for stability.
        flows_sorted = sorted(
            flows, key=lambda f: (topics_order.index(f[0]), topics_order.index(f[1]))
        )
        src_pos = {}
        tgt_pos = {}
        for src, tgt, value, _, _ in flows_sorted:
            thickness = value * scale
            sy = src_offsets[src] + thickness / 2
            src_offsets[src] += thickness
            src_pos[(src, tgt)] = (sy, thickness)
        for tgt, src, value, _, _ in sorted(
            [(f[1], f[0], f[2], f[3], f[4]) for f in flows_sorted],
            key=lambda f: (topics_order.index(f[0]), topics_order.index(f[1])),
        ):
            thickness = value * scale
            ty = tgt_offsets[tgt] + thickness / 2
            tgt_offsets[tgt] += thickness
            tgt_pos[(src, tgt)] = (ty, thickness)

        for src, tgt, value, _, _ in flows_sorted:
            if src not in topic_y or tgt not in topic_y:
                continue
            sy, thickness = src_pos[(src, tgt)]
            ty, _ = tgt_pos[(src, tgt)]
            if thickness <= 0:
                continue
            path = ribbon_path(x0, sy, x1, ty, thickness, FLOW_CURVATURE)
            patch = mpatches.PathPatch(
                path,
                facecolor=topic_color_map.get(src, "0.7"),
                edgecolor="none",
                alpha=FLOW_ALPHA,
                zorder=0,
            )
            ax.add_patch(patch)

    # Theme labels on the left at the start of each block.
    left_label_x = period_x[periods_present[0]] - LABEL_OFFSET
    max_group_size = max((len(t) for _, t in theme_groups_sorted), default=1)

    def _label_height_units(text, fontsize):
        # Rough estimate of vertical span for rotated text (length-dominated).
        return row_gap * (0.6 + 0.035 * max(1, len(str(text))))

    theme_label_items = []
    for theme, topics in theme_groups_sorted:
        if not topics:
            continue
        y_positions = [topic_y[t] for t in topics if t in topic_y]
        if not y_positions:
            continue
        y_center = (max(y_positions) + min(y_positions)) * 0.5
        size_scale = 1 + THEME_LABEL_SCALE * (len(topics) / max_group_size)
        est_height = _label_height_units(theme, THEME_LABEL_SIZE * size_scale)
        theme_label_items.append((theme, y_center, size_scale, est_height))

    theme_label_items.sort(key=lambda x: x[1], reverse=True)
    occupied = []
    theme_texts = []
    for idx, (theme, y_center, size_scale, est_height) in enumerate(theme_label_items):
        step = max(row_gap * 0.5, est_height * 0.6) * THEME_LABEL_JITTER
        y_label = y_center
        for attempt in range(12):
            if attempt == 0:
                y_label = y_center
            else:
                sign = 1 if attempt % 2 else -1
                mag = (attempt + 1) // 2
                y_label = y_center + sign * mag * step
            y_label = min(1.0 - Y_MARGIN, max(Y_MARGIN, y_label))
            y_min = y_label - est_height / 2
            y_max = y_label + est_height / 2
            if not any(not (y_max < a or y_min > b) for (a, b) in occupied):
                break
        x_label = (
            left_label_x - THEME_LABEL_OFFSET + THEME_LABEL_X_OFFSETS.get(theme, 0.0)
        )
        theme_color = THEME_COLORS.get(theme, "#333333")
        ax.text(
            x_label,
            y_label,
            theme,
            ha="right",
            va="center",
            fontsize=THEME_LABEL_SIZE * size_scale * 0.85,
            color=theme_color,
            rotation=90,
        )
        occupied.append((y_min, y_max))
        theme_texts.append((theme, theme_color))

    if theme_texts:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [
            t for t in ax.texts if t.get_text() in {label for label, _ in theme_texts}
        ]
        if texts:
            inv = ax.transData.inverted()
            dx_pixels = 30
            dx_data = abs(inv.transform((dx_pixels, 0))[0] - inv.transform((0, 0))[0])
            base_x = left_label_x - THEME_LABEL_OFFSET
            max_shift = 16 * dx_data
            for pass_idx in range(3):
                dx_step = dx_data * (1 + pass_idx)
                for _ in range(15):
                    moved = False
                    bboxes = [
                        t.get_window_extent(renderer).expanded(1.02, 1.05)
                        for t in texts
                    ]
                    for i in range(len(texts)):
                        for j in range(i + 1, len(texts)):
                            if bboxes[i].overlaps(bboxes[j]):
                                xi = texts[i].get_position()[0]
                                xj = texts[j].get_position()[0]
                                if xj <= xi:
                                    new_x = max(base_x - max_shift, xj - dx_step)
                                    texts[j].set_x(new_x)
                                else:
                                    new_x = min(base_x + max_shift, xi + dx_step)
                                    texts[i].set_x(new_x)
                                moved = True
                    if not moved:
                        break

    # Topic labels on the rightmost column only.
    for topic in topics_order:
        y = topic_y.get(topic, None)
        if y is None or not str(topic).strip():
            continue
        color = topic_color_map.get(topic, "black")
        topic_periods = period_counts.loc[
            period_counts["topic_group"] == topic, "period"
        ].unique()
        if topic_periods.size == 0:
            continue
        right_period = periods_present[-1]
        ax.text(
            period_x[right_period] + LABEL_OFFSET,
            y,
            topic,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE,
            color=color,
            bbox=(
                dict(
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.6,
                    pad=0.4,
                    alpha=0.85,
                )
                if LABEL_BOX
                else None
            ),
        )

    # Period labels.
    for label in periods_present:
        ax.text(
            period_x[label],
            1.0 - Y_MARGIN / 2,
            label,
            ha="center",
            va="bottom",
            fontsize=LABEL_SIZE + 1,
        )

    if view_mode == "main":
        title = "Space of Concerns: Dominant Topic Transitions by Period"
    else:
        title = "Space of Concerns: Full Topic Transition Map by Period"
    ax.format(title=title, grid=False)
    ax.format(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    # Start/end hatch legend.
    start_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="////", label="Topic start"
    )
    end_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="\\\\\\\\", label="Topic end"
    )
    ax.legend(
        handles=[start_patch, end_patch],
        loc="ll",
        framealpha=0.0,
        fontsize=8,
        bbox_to_anchor=(0.25, 0.02),
    )

    ax2 = axs[1]
    if view_mode == "main":
        mechanism_df = compute_transition_mechanism_counts(
            per_country_period=per_country_period,
            periods_present=periods_present,
        )
        x = np.arange(len(mechanism_df))
        persistent = mechanism_df["persistent"].to_numpy(dtype=float)
        entries = mechanism_df["entries"].to_numpy(dtype=float)
        exits = mechanism_df["exits"].to_numpy(dtype=float)

        ax2.bar(
            x,
            persistent,
            color="#4CAF50",
            alpha=0.9,
            label="Persistent (A->A)",
        )
        ax2.bar(
            x,
            entries,
            bottom=persistent,
            color="#1E88E5",
            alpha=0.9,
            label="Entries",
        )
        ax2.bar(
            x,
            exits,
            bottom=persistent + entries,
            color="#EF6C00",
            alpha=0.9,
            label="Exits",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            mechanism_df["pair_label"].tolist(),
            rotation=20,
            ha="right",
            fontsize=8,
        )
        ax2.set_ylabel("Topic-set changes", fontsize=9)
        ax2.legend(
            ncols=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            framealpha=0.0,
            fontsize=7,
        )
        ax2.grid(axis="y", alpha=0.3)
    else:
        # Theme composition panel (RCA>1 counts per theme, period blocks).
        df["theme"] = df["topic_group"].map(_topic_theme)
        theme_counts = df.groupby(["period", "theme"]).size().unstack(fill_value=0)
        theme_order = [t for t in THEME_COLORS.keys() if t in theme_counts.columns]
        theme_counts = theme_counts.reindex(index=periods_present, columns=theme_order)
        x = np.arange(len(periods_present))
        stacks = [theme_counts[col].to_numpy() for col in theme_order]
        colors = [THEME_COLORS[theme] for theme in theme_order]
        ax2.stackplot(x, stacks, colors=colors, alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods_present, rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("RCA>1 topics", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(
        f"Saved ribbon plot to {out_png} and {out_pdf} "
        f"(view mode: {view_mode}, transition mode: {TRANSITION_SOURCE}, "
        f"width metric: {EDGE_WIDTH_METRIC}, node size mode: {NODE_SIZE_MODE}, "
        f"flows kept: {n_flows_kept}/{n_flows_full})"
    )


if __name__ == "__main__":
    main(view_mode="main")
    main(view_mode="full")
