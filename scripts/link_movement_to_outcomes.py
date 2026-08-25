#!/usr/bin/env python3
"""Connect stable concern-space position, actor movement, and formal outcomes.

The analysis has three deliberately separate pieces:

* actor-bootstrap the concern matrix and retain only repeatedly strong edges;
* place conservatively documented formal outcomes on that stable scaffold;
* ask whether actors whose papers feed an outcome were already positioned near
  that outcome's concerns, and show those events on example trajectories.

No paper count is treated as a proposal denominator. An "outcome contribution"
here means that a final report explicitly connects a submitted paper to adoption
or approval, or that an official paragraph documents the paper's contribution.
"""

from __future__ import annotations

import collections
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import PercentFormatter
from statsmodels.discrete.conditional_models import ConditionalLogit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle
from analysis.data_loading import load_submitted_with_fallback
from fig01_space_of_concerns_topology import (
    build_graphs,
    draw_silhouette,
    fitted_layout,
    load_topic_meta,
)
from scripts import explore_lineage_space as lineage
from utils import (
    compute_product_space,
    compute_rolling_rca,
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
    standardize_index_labels,
)


OUTDIR = ROOT / "output" / "movement_outcomes"
FIGDIR = ROOT / "figures"

N_BOOTSTRAP = 1000
BOOTSTRAP_TOP_K = 5
STABILITY_THRESHOLD = 0.60
RPA_THRESHOLD = 1.0
PRIOR_WINDOW = 5
RANDOM_SEED = 20260812
EXAMPLE_ACTORS = ["Australia", "Netherlands", "Ukraine"]
TOPIC_POSITION_CSV = ROOT / "output" / "fig45_portfolio_space_ridgelines_topic_order.csv"

INSTRUMENTS = lineage.INSTRUMENTS
INSTRUMENT_COLORS = lineage.INSTRUMENT_COLORS


def _canonicalize_topic_matrix(
    matrix: pd.DataFrame,
    topics: list[str],
    topic_lookup: dict[str, str],
    actors: list[str],
) -> pd.DataFrame:
    matrix = standardize_index_labels(matrix.copy())
    mapped = []
    for raw in matrix.index:
        mapped.append(topic_lookup.get(lineage._normalize(raw)))
    matrix["__topic"] = mapped
    matrix = matrix.dropna(subset=["__topic"]).groupby("__topic").sum(numeric_only=True)
    return matrix.reindex(index=topics, columns=actors, fill_value=0.0)


def actor_bootstrap_edges(
    counts: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    top_k: int = BOOTSTRAP_TOP_K,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Resample actors and record edge weight and endpoint-top-k selection."""
    topics = list(counts.index)
    values = counts.to_numpy(dtype=float)
    n_topics, n_actors = values.shape
    first, second = np.triu_indices(n_topics, 1)
    edge_lookup = {
        (int(i), int(j)): index
        for index, (i, j) in enumerate(zip(first, second))
    }
    weights = np.empty((n_bootstrap, len(first)), dtype=np.float32)
    selected = np.zeros(len(first), dtype=np.int32)
    rng = np.random.default_rng(seed)

    for draw in range(n_bootstrap):
        sample = rng.integers(0, n_actors, size=n_actors)
        sampled = pd.DataFrame(
            values[:, sample], index=topics,
            columns=[f"bootstrap_actor_{i}" for i in range(n_actors)],
        )
        phi = compute_product_space(get_rca(sampled)).to_numpy(dtype=float)
        weights[draw] = phi[first, second]
        chosen: set[tuple[int, int]] = set()
        for i in range(n_topics):
            order = np.argsort(phi[i])[::-1]
            positive = [int(j) for j in order if j != i and phi[i, j] > 0]
            for j in positive[:top_k]:
                chosen.add(tuple(sorted((i, j))))
        for edge in chosen:
            selected[edge_lookup[edge]] += 1

    result = pd.DataFrame(
        {
            "concern_a": [topics[i] for i in first],
            "concern_b": [topics[j] for j in second],
            "phi_median": np.median(weights, axis=0),
            "phi_low": np.quantile(weights, 0.025, axis=0),
            "phi_high": np.quantile(weights, 0.975, axis=0),
            "topk_frequency": selected / n_bootstrap,
        }
    )
    result["stable"] = (
        (result["topk_frequency"] >= STABILITY_THRESHOLD)
        & (result["phi_low"] > 0)
    )
    return result.sort_values(
        ["stable", "topk_frequency", "phi_median"], ascending=False
    )


def build_stable_graph(topics: list[str], edge_table: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(topics)
    for row in edge_table[edge_table["stable"]].itertuples(index=False):
        graph.add_edge(
            row.concern_a, row.concern_b,
            weight=float(row.phi_median),
            stability=float(row.topk_frequency),
            low=float(row.phi_low), high=float(row.phi_high),
        )
    return graph


def paper_sponsors(submitted: pd.DataFrame) -> dict[str, set[str]]:
    """Recover sponsors using meeting, paper type, and paper number."""
    data = submitted.copy()
    data["__meeting"] = pd.to_numeric(data["meeting number"], errors="coerce")
    data["__number"] = pd.to_numeric(data["paper number"], errors="coerce")
    data["__type"] = data["paper url"].astype(str).str.extract(
        r"/([^/]+)/ATCM\d+_", flags=re.IGNORECASE
    )[0].str.upper()
    data = data.dropna(subset=["__meeting", "__number", "__type"])
    sponsors: dict[str, set[str]] = {}
    for (meeting, paper_type, number), group in data.groupby(
        ["__meeting", "__type", "__number"]
    ):
        actors: set[str] = set()
        for value in group["submitted by"].dropna():
            actors.update(
                actor.strip() for actor in str(value).split(",") if actor.strip()
            )
        sponsors[f"ATCM{int(meeting)}:{paper_type} {int(number)}"] = actors
    return sponsors


def documented_outcome_events(
    nodes: dict[str, dict],
    links: dict[tuple[str, str], dict],
    paper_categories: dict[str, set[str]],
    sponsors: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    """Return actor events, outcome concerns, and outcome contributor sets."""
    outcome_topics: dict[str, set[str]] = collections.defaultdict(set)
    contributors: dict[str, set[str]] = collections.defaultdict(set)
    actor_outcome_topics: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    source_papers: dict[tuple[str, str], set[str]] = collections.defaultdict(set)

    for (paper, outcome), edge in links.items():
        if edge["level"] < 3:
            continue
        categories = paper_categories[paper]
        outcome_topics[outcome].update(categories)
        for actor in sponsors.get(paper, set()):
            contributors[outcome].add(actor)
            actor_outcome_topics[(actor, outcome)].update(categories)
            source_papers[(actor, outcome)].add(paper)

    rows = []
    for (actor, outcome), topics in actor_outcome_topics.items():
        node = nodes[outcome]
        rows.append(
            {
                "actor": actor,
                "outcome_id": outcome,
                "year": int(node["year"]),
                "instrument": node.get("outcome_type"),
                "title": node.get("title"),
                "concerns": " | ".join(sorted(topics)),
                "n_concerns": len(topics),
                "source_papers": " | ".join(sorted(source_papers[(actor, outcome)])),
            }
        )
    events = pd.DataFrame(rows).sort_values(
        ["actor", "year", "outcome_id"]
    ).reset_index(drop=True)
    return events, dict(outcome_topics), dict(contributors)


def rolling_actor_centroids(
    submitted: pd.DataFrame,
    positions: dict[str, tuple[float, float]],
    topic_lookup: dict[str, str],
) -> pd.DataFrame:
    rolling = compute_rolling_rca(submitted, window_years=5)
    rolling["canonical_topic"] = rolling["topic"].map(
        lambda raw: topic_lookup.get(lineage._normalize(raw))
    )
    rolling = rolling.dropna(subset=["canonical_topic"]).copy()
    rolling["rca"] = pd.to_numeric(rolling["rca"], errors="coerce").fillna(0.0)
    rolling = rolling[rolling["rca"] >= RPA_THRESHOLD].copy()
    rolling["x"] = rolling["canonical_topic"].map(lambda topic: positions[topic][0])
    rolling["y"] = rolling["canonical_topic"].map(lambda topic: positions[topic][1])

    rows = []
    for (actor, year), group in rolling.groupby(["country", "year"], sort=True):
        weights = group["rca"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            continue
        rows.append(
            {
                "actor": actor,
                "year": int(year),
                "x": float(np.average(group["x"], weights=weights)),
                "y": float(np.average(group["y"], weights=weights)),
                "breadth": int(group["canonical_topic"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["actor", "year"])


def rolling_actor_positions_1d(
    submitted: pd.DataFrame,
    topic_positions: pd.DataFrame,
    topic_lookup: dict[str, str],
) -> pd.DataFrame:
    """Five-year RPA-weighted actor positions on the continuous concern axis."""
    x_of = topic_positions.set_index("topic")["x_plot"].astype(float).to_dict()
    rolling = compute_rolling_rca(submitted, window_years=5)
    rolling["canonical_topic"] = rolling["topic"].map(
        lambda raw: topic_lookup.get(lineage._normalize(raw))
    )
    rolling = rolling.dropna(subset=["canonical_topic"]).copy()
    rolling["rca"] = pd.to_numeric(rolling["rca"], errors="coerce").fillna(0.0)
    rolling = rolling[rolling["rca"] >= RPA_THRESHOLD].copy()
    rolling["position"] = rolling["canonical_topic"].map(x_of)
    rolling = rolling.dropna(subset=["position"])

    rows = []
    for (actor, year), group in rolling.groupby(["country", "year"], sort=True):
        weights = group["rca"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            continue
        rows.append(
            {
                "actor": actor,
                "year": int(year),
                "position": float(np.average(group["position"], weights=weights)),
                "breadth": int(group["canonical_topic"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["actor", "year"])


def _prior_state(
    submitted: pd.DataFrame,
    year: int,
    topics: list[str],
    actors: list[str],
    topic_lookup: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    all_actors_raw = extract_unique_countries(submitted)
    all_topics_raw = extract_unique_topics(submitted)
    year_values = pd.to_numeric(submitted["year"], errors="coerce")
    recent = submitted[(year_values > year - PRIOR_WINDOW - 1) & (year_values <= year - 1)]
    history = submitted[year_values <= year - 1]

    recent_inter = generate_interaction_matrix(recent, all_actors_raw, all_topics_raw)
    recent_inter = _canonicalize_topic_matrix(
        recent_inter, topics, topic_lookup, actors
    )
    recent_rca = get_rca(recent_inter).reindex(index=topics, columns=actors, fill_value=0.0)

    history_inter = generate_interaction_matrix(history, all_actors_raw, all_topics_raw)
    history_inter = _canonicalize_topic_matrix(
        history_inter, topics, topic_lookup, actors
    )
    history_phi = compute_product_space(get_rca(history_inter)).to_numpy(dtype=float)
    np.fill_diagonal(history_phi, 1.0)
    volume = recent_inter.sum(axis=0).reindex(actors, fill_value=0.0)
    return recent_rca, history_phi, volume


def outcome_candidate_panel(
    submitted: pd.DataFrame,
    topics: list[str],
    actors: list[str],
    topic_lookup: dict[str, str],
    outcome_topics: dict[str, set[str]],
    contributors: dict[str, set[str]],
    nodes: dict[str, dict],
) -> tuple[pd.DataFrame, dict]:
    topic_index = {topic: i for i, topic in enumerate(topics)}
    rows = []
    state_cache: dict[int, tuple[pd.DataFrame, np.ndarray, pd.Series]] = {}
    source_actor_total = 0
    source_actor_in_universe = 0
    source_actor_with_prior_state = 0

    for outcome, target_topics in outcome_topics.items():
        year = int(nodes[outcome]["year"])
        if year not in state_cache:
            state_cache[year] = _prior_state(
                submitted, year, topics, actors, topic_lookup
            )
        recent_rca, history_phi, volume = state_cache[year]
        target_indices = [topic_index[t] for t in target_topics if t in topic_index]
        if not target_indices:
            continue
        source_set = contributors.get(outcome, set())
        source_actor_total += len(source_set)
        source_actor_in_universe += len(source_set.intersection(actors))

        for actor in actors:
            held = np.flatnonzero(
                recent_rca[actor].to_numpy(dtype=float) >= RPA_THRESHOLD
            )
            if held.size == 0 or float(volume[actor]) <= 0:
                continue
            if actor in source_set:
                source_actor_with_prior_state += 1
            proximity = float(history_phi[np.ix_(target_indices, held)].max())
            rows.append(
                {
                    "outcome_id": outcome,
                    "year": year,
                    "actor": actor,
                    "contributor": int(actor in source_set),
                    "distance": 1.0 - proximity,
                    "proximity": proximity,
                    "breadth": int(held.size),
                    "prior_papers": float(volume[actor]),
                    "n_outcome_concerns": len(target_indices),
                }
            )

    panel = pd.DataFrame(rows)
    # Conditional comparisons require both contributors and noncontributors.
    usable = panel.groupby("outcome_id")["contributor"].agg(["sum", "count"])
    usable_ids = usable[(usable["sum"] > 0) & (usable["sum"] < usable["count"])].index
    panel = panel[panel["outcome_id"].isin(usable_ids)].copy()
    diagnostics = {
        "source_actor_outcome_incidences": source_actor_total,
        "source_actor_outcome_incidences_in_actor_universe": source_actor_in_universe,
        "source_actor_outcome_incidences_with_prior_specialization": source_actor_with_prior_state,
        "usable_outcomes": int(panel["outcome_id"].nunique()),
        "candidate_rows": int(len(panel)),
        "contributor_rows": int(panel["contributor"].sum()),
    }
    return panel, diagnostics


def within_outcome_rank_test(
    panel: pd.DataFrame,
    value: str,
    n_permutations: int = 9999,
    seed: int = RANDOM_SEED,
) -> dict:
    data = panel.copy()
    # Higher percentile always means "more" of the named variable. For
    # proximity this means nearer; for breadth it means broader.
    data["percentile"] = data.groupby("outcome_id")[value].rank(
        method="average", pct=True
    )
    groups = []
    observed_by_group = []
    for _, group in data.groupby("outcome_id", sort=False):
        values = group["percentile"].to_numpy(dtype=float)
        labels = group["contributor"].to_numpy(dtype=bool)
        k = int(labels.sum())
        if k == 0 or k == len(group):
            continue
        groups.append((values, k))
        observed_by_group.append(float(values[labels].mean()))
    observed = float(np.mean(observed_by_group))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for draw in range(n_permutations):
        group_means = [
            float(values[rng.choice(len(values), size=k, replace=False)].mean())
            for values, k in groups
        ]
        null[draw] = float(np.mean(group_means))
    return {
        "value": value,
        "observed_mean_percentile": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "upper_tail_p": float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1)),
        "n_outcomes": len(groups),
        "null": null,
    }


def fit_outcome_position_model(panel: pd.DataFrame) -> pd.DataFrame:
    """Outcome-fixed-effects conditional logit with standardized predictors."""
    data = panel.copy()
    data["log_breadth"] = np.log1p(data["breadth"])
    data["log_prior_papers"] = np.log1p(data["prior_papers"])
    columns = ["proximity", "log_breadth", "log_prior_papers"]
    design = data[columns].copy()
    design = (design - design.mean()) / design.std(ddof=0)
    result = ConditionalLogit(
        data["contributor"], design, groups=data["outcome_id"]
    ).fit(method="bfgs", disp=False, maxiter=300)
    labels = {
        "proximity": "Proximity to outcome concern",
        "log_breadth": "Prior portfolio breadth",
        "log_prior_papers": "Prior paper volume",
    }
    rows = []
    for term in columns:
        coefficient = float(result.params[term])
        se = float(result.bse[term])
        rows.append(
            {
                "term": term,
                "label": labels[term],
                "coefficient": coefficient,
                "se": se,
                "odds_ratio": math.exp(coefficient),
                "ci_low": math.exp(coefficient - 1.96 * se),
                "ci_high": math.exp(coefficient + 1.96 * se),
                "p_value": float(result.pvalues[term]),
                "scale": "one sample standard deviation",
            }
        )
    return pd.DataFrame(rows)


def draw_stable_edges(ax, graph: nx.Graph, positions, alpha_scale: float = 1.0) -> None:
    for source, target, attrs in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        stability = float(attrs["stability"])
        weight = float(attrs["weight"])
        ax.plot(
            [x0, x1], [y0, y1],
            color="#667085", lw=0.35 + 2.3 * weight,
            alpha=alpha_scale * (0.15 + 0.78 * stability),
            zorder=1, solid_capstyle="round",
        )


def _node_radii(series: pd.Series, minimum=0.11, maximum=0.62) -> pd.Series:
    maximum_value = float(series.max())
    if maximum_value <= 0:
        return pd.Series(minimum, index=series.index)
    return minimum + (maximum - minimum) * np.sqrt(series / maximum_value)


def draw_attention_outcome_glyphs(
    ax,
    positions,
    paper_counts: pd.Series,
    direct_table: pd.DataFrame,
) -> None:
    """Paper-volume cores with instrument-segmented formal-outcome rings."""
    paper_radius = _node_radii(paper_counts, 0.10, 0.31)
    outcome_total = direct_table.sum(axis=1)
    max_outcomes = max(float(outcome_total.max()), 1.0)
    for topic in paper_counts.index:
        x, y = positions[topic]
        core_radius = float(paper_radius[topic])
        ax.add_patch(
            Circle(
                (x, y), core_radius,
                facecolor="#40566B", edgecolor="white",
                linewidth=0.65, zorder=3,
            )
        )
        total = int(outcome_total[topic])
        if total == 0:
            continue
        ring_radius = max(
            core_radius + 0.10,
            0.26 + 0.24 * math.sqrt(total / max_outcomes),
        )
        ring_width = min(0.105, max(0.065, ring_radius - core_radius - 0.025))
        angle = 90.0
        for instrument in INSTRUMENTS:
            value = int(direct_table.loc[topic, instrument])
            if value <= 0:
                continue
            extent = 360.0 * value / total
            ax.add_patch(
                Wedge(
                    (x, y), ring_radius, angle, angle + extent,
                    width=ring_width,
                    facecolor=INSTRUMENT_COLORS[instrument],
                    edgecolor="white", linewidth=0.45, zorder=4,
                )
            )
            angle += extent


def draw_selected_network_labels(ax, positions, topics: list[str]) -> None:
    """Place selected labels around the map as in Figure 1."""
    center = np.mean(np.asarray(list(positions.values()), dtype=float), axis=0)
    sides: dict[str, list[str]] = {"top": [], "right": [], "bottom": [], "left": []}
    for topic in topics:
        x, y = positions[topic]
        angle = float(np.degrees(np.arctan2(y - center[1], x - center[0]))) % 360
        if 45 <= angle < 135:
            sides["top"].append(topic)
        elif 135 <= angle < 225:
            sides["left"].append(topic)
        elif 225 <= angle < 315:
            sides["bottom"].append(topic)
        else:
            sides["right"].append(topic)
    for side, members in sides.items():
        members.sort(key=lambda topic: positions[topic][0 if side in ("top", "bottom") else 1])
        anchors = np.linspace(0.09, 0.91, len(members)) if members else []
        for topic, anchor in zip(members, anchors):
            x, y = positions[topic]
            if side == "top":
                target, ha, va, rotation = (float(anchor), 1.035), "center", "bottom", 90
            elif side == "bottom":
                target, ha, va, rotation = (float(anchor), -0.035), "center", "top", 90
            elif side == "left":
                target, ha, va, rotation = (-0.025, float(anchor)), "right", "center", 0
            else:
                target, ha, va, rotation = (1.025, float(anchor)), "left", "center", 0
            label = "\n".join(textwrap.wrap(lineage._short_label(topic), 20))
            annotation = ax.annotate(
                label, xy=(x, y), xytext=target,
                xycoords="data", textcoords="axes fraction",
                ha=ha, va=va, rotation=rotation,
                fontsize=7.1, color="#263442", annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": "#98A2B3", "lw": 0.42},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.2},
                zorder=8,
            )
            annotation.set_clip_on(False)


def make_stable_outcome_figure(
    stable_graph: nx.Graph,
    layout_backbone: nx.Graph,
    positions,
    land,
    projection_extent,
    paper_counts: pd.Series,
    direct_table: pd.DataFrame,
) -> None:
    fig, axes = uplt.subplots(
        [[0, 1, 1, 1, 1, 0]], figwidth=12.5, refaspect=1.45,
        share=False,
    )
    ax_map = axes[0]
    draw_silhouette(ax_map, land, projection_extent)

    # Keep Figure 1's exact coordinates. Backbone edges that fail the actor
    # bootstrap are retained only as a dotted layout scaffold, never as data.
    stable_pairs = {frozenset((u, v)) for u, v in stable_graph.edges()}
    for source, target in layout_backbone.edges():
        if frozenset((source, target)) in stable_pairs:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax_map.plot(
            [x0, x1], [y0, y1], color="#C8CED6", lw=0.62,
            ls=(0, (2.0, 2.7)), alpha=0.72, zorder=0,
        )
    draw_stable_edges(ax_map, stable_graph, positions)
    draw_attention_outcome_glyphs(ax_map, positions, paper_counts, direct_table)
    ax_map.format(
        aspect="equal", grid=False, xlocator=[], ylocator=[],
        xspineloc="neither", yspineloc="neither",
    )

    direct_total = direct_table.sum(axis=1)
    isolates = [topic for topic, degree in stable_graph.degree() if degree == 0]
    label_topics = direct_total.sort_values(ascending=False).head(12).index.tolist()
    label_topics += [topic for topic in isolates if topic not in label_topics]
    draw_selected_network_labels(ax_map, positions, label_topics)

    handles = [
        Line2D([0], [0], color="#667085", lw=2.0,
               label="bootstrap-supported tie"),
        Line2D([0], [0], color="#C8CED6", lw=0.9, ls=(0, (2.0, 2.7)),
               label="Figure 1 scaffold"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor="#40566B", markeredgecolor="white",
               label="paper volume"),
    ] + [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=INSTRUMENT_COLORS[name], markeredgecolor="white",
               label=name)
        for name in INSTRUMENTS
    ]
    fig.legend(handles=handles, loc="b", ncols=7, frame=False, fontsize=7.0)
    fig.format(
        suptitle="Formal action occupies the same institutional concern space",
        suptitlesize=14,
    )
    figstyle.apply_typography([ax_map])
    fig.save(FIGDIR / "exploratory_stable_space_outcomes.pdf")
    fig.save(FIGDIR / "exploratory_stable_space_outcomes.png", dpi=260)
    uplt.close(fig)


def make_outcome_position_figure(
    proximity_test: dict,
    model_table: pd.DataFrame,
) -> None:
    ordered = model_table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(ordered))
    odds = ordered["odds_ratio"].to_numpy(dtype=float)
    low = ordered["ci_low"].to_numpy(dtype=float)
    high = ordered["ci_high"].to_numpy(dtype=float)
    colors = ["#6A3D9A" if term == "proximity" else "#667085" for term in ordered["term"]]

    fig, ax = uplt.subplots(ncols=1, refwidth=4.6, refaspect=1.25)
    for yi, point, lower, upper, color in zip(y, odds, low, high, colors):
        ax.errorbar(
            [point], [yi], xerr=np.array([[point - lower], [upper - point]]),
            fmt="none", ecolor=color, elinewidth=2.0, capsize=3.0, zorder=2,
        )
    ax.scatter(
        odds, y, s=60, color=colors, edgecolor="white", linewidth=0.65, zorder=3,
    )
    ax.axvline(1.0, color="#98A2B3", lw=0.9, ls="--", zorder=1)
    ax.format(
        xlabel="Odds ratio per 1 SD increase", ylabel="",
        yticks=y, yticklabels=["Paper volume", "Portfolio breadth", "Outcome proximity"],
        xscale="log", xlim=(0.85, max(high) * 1.16),
        xlocator=[1.0, 1.5, 2.0, 3.0, 4.0], xformatter="{x:g}", grid=False,
    )
    ax.text(
        0.98, 0.78,
        f"Contributors: {proximity_test['observed_mean_percentile']:.0%} proximity percentile\n"
        f"Random assignment within outcome: {proximity_test['null_mean']:.0%}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=figstyle.FS_ANNOT, color="#40566B",
    )
    fig.format(
        suptitle="Documented contributors were already positioned near the outcome concern",
        suptitlesize=13,
    )
    figstyle.apply_typography([ax])
    fig.save(FIGDIR / "exploratory_outcome_position_model.pdf")
    fig.save(FIGDIR / "exploratory_outcome_position_model.png", dpi=260)
    uplt.close(fig)


def _nearest_centroid(trajectory: pd.DataFrame, year: int) -> pd.Series | None:
    if trajectory.empty:
        return None
    exact = trajectory[trajectory["year"] == year]
    if not exact.empty:
        return exact.iloc[0]
    distance = (trajectory["year"] - year).abs()
    if distance.min() <= 2:
        return trajectory.loc[distance.idxmin()]
    return None


def make_trajectory_figure(
    actor_positions: pd.DataFrame,
    events: pd.DataFrame,
    topic_positions: pd.DataFrame,
) -> None:
    x_of = topic_positions.set_index("topic")["x_plot"].astype(float).to_dict()
    fig, axes = uplt.subplots(
        ncols=3, refwidth=2.85, refaspect=1.0, share=False, wspace=3.4,
    )
    for ax, actor in zip(axes, EXAMPLE_ACTORS):
        actor_color = figstyle.ACTOR_EXAMPLE_COLORS[actor]
        trajectory = actor_positions[
            (actor_positions["actor"] == actor)
            & (actor_positions["year"] >= 1991)
            & (actor_positions["year"] <= 2025)
        ].copy()
        blocks = trajectory["year"].diff().fillna(1).gt(1).cumsum()
        for _, block in trajectory.groupby(blocks):
            ax.plot(
                block["year"], block["position"], color=actor_color,
                lw=2.15, alpha=0.95, zorder=4,
            )
        ax.scatter(
            trajectory["year"], trajectory["position"], s=13,
            color=actor_color, edgecolor="white", linewidth=0.35, zorder=5,
        )

        actor_events = events[events["actor"] == actor].copy()
        year_counts = actor_events.groupby("year").size().to_dict()
        year_seen: collections.Counter = collections.Counter()
        for event in actor_events.itertuples(index=False):
            year = int(event.year)
            instrument = str(event.instrument)
            point = _nearest_centroid(trajectory, year)
            if point is None:
                continue
            target_topics = [
                topic for topic in str(event.concerns).split(" | ") if topic in x_of
            ]
            if not target_topics:
                continue
            target_position = float(np.mean([x_of[topic] for topic in target_topics]))
            count = int(year_counts[year])
            rank = int(year_seen[year])
            year_seen[year] += 1
            event_year = year + (rank - (count - 1) / 2) * 0.32
            ax.plot(
                [event_year, event_year], [point["position"], target_position],
                color=INSTRUMENT_COLORS[instrument], lw=0.85,
                alpha=0.48, zorder=2,
            )
            ax.scatter(
                [event_year], [target_position], marker="D", s=31,
                color=INSTRUMENT_COLORS[instrument], edgecolor="white",
                linewidth=0.5, zorder=7,
            )
            ax.scatter(
                [event_year], [point["position"]], marker="o", s=18,
                color=INSTRUMENT_COLORS[instrument], edgecolor="white",
                linewidth=0.45, zorder=7,
            )

        outcome_count = int(actor_events["outcome_id"].nunique())
        noun = "link" if outcome_count == 1 else "links"
        title = f"{actor}: {outcome_count} documented outcome {noun}"
        ax.format(
            title=title, grid=False, xlabel="Year",
            xlim=(1990, 2026), xlocator=[1991, 2000, 2010, 2020, 2025],
            ylim=(-0.03, 1.03), ylocator=[0.0, 0.25, 0.5, 0.75, 1.0],
            ylabel="Position on the continuous\nconcern axis",
            titlesize=figstyle.FS_TITLE,
        )
    for ax in axes[1:]:
        ax.format(ylabel="", yticklabels=[])

    handles = [
        Line2D([0], [0], marker="D", linestyle="none", markersize=6,
               markerfacecolor=INSTRUMENT_COLORS[name], markeredgecolor="white",
               label=name)
        for name in INSTRUMENTS
    ]
    handles += [
        Line2D([0], [0], color="#98A2B3", lw=0.9, marker="D", markersize=5,
               markerfacecolor="#667085", markeredgecolor="white",
               label="portfolio-to-outcome distance"),
    ]
    axes[1].legend(handles=handles, loc="b", ncols=5, frame=False, fontsize=7.2)
    fig.format(
        abc="A", abcloc="ul",
        suptitle="Formal-outcome links connect actor movement to specific parts of the concern space",
        abcsize=figstyle.FS_PANEL, suptitlesize=14,
    )
    figstyle.apply_typography(axes)
    fig.save(FIGDIR / "exploratory_movement_to_outcomes.pdf")
    fig.save(FIGDIR / "exploratory_movement_to_outcomes.png", dpi=260)
    uplt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    backbone, mst, pooled_graph, counts, _ = build_graphs()
    topics = list(counts.index)
    actors = list(counts.columns)
    _, mode_of, _ = load_topic_meta()
    positions, land, projection_extent = fitted_layout(mst, pooled_graph, mode_of)
    topic_lookup = lineage._canonical_topic_lookup(topics)

    edge_table = actor_bootstrap_edges(counts)
    edge_table.to_csv(OUTDIR / "bootstrap_edge_stability.csv", index=False)
    stable_graph = build_stable_graph(topics, edge_table)

    submitted = load_submitted_with_fallback()
    sponsors = paper_sponsors(submitted)
    decision_map = lineage._load_json(lineage.LINEAGE_ROOT / "decision_map.json")
    inventory = lineage._load_json(lineage.LINEAGE_ROOT / "atcm_inventory.json")
    paper_categories = lineage._paper_categories(inventory, topic_lookup)
    nodes, links = lineage._paper_outcome_links(decision_map, paper_categories)
    events, outcome_topics, contributors = documented_outcome_events(
        nodes, links, paper_categories, sponsors
    )
    events.to_csv(OUTDIR / "actor_documented_outcome_events.csv", index=False)

    direct_topics = lineage._outcome_topic_sets(links, paper_categories, minimum_level=3)
    direct_topics = lineage._filter_outcome_topics_by_year(direct_topics, nodes, 1991, 2025)
    direct_table = lineage._topic_instrument_table(direct_topics, nodes, topics)
    paper_counts = lineage._period_paper_counts(topics, topic_lookup, 1991, 2025)

    candidate_panel, panel_diagnostics = outcome_candidate_panel(
        submitted, topics, actors, topic_lookup,
        outcome_topics, contributors, nodes,
    )
    candidate_panel.to_csv(OUTDIR / "outcome_actor_candidate_panel.csv", index=False)
    proximity_test = within_outcome_rank_test(candidate_panel, "proximity")
    breadth_test = within_outcome_rank_test(candidate_panel, "breadth")
    model_table = fit_outcome_position_model(candidate_panel)
    model_table.to_csv(OUTDIR / "outcome_position_conditional_logit.csv", index=False)

    centroids = rolling_actor_centroids(submitted, positions, topic_lookup)
    centroids.to_csv(OUTDIR / "actor_space_centroids_5y.csv", index=False)
    topic_positions = pd.read_csv(TOPIC_POSITION_CSV)
    actor_positions = rolling_actor_positions_1d(
        submitted, topic_positions, topic_lookup,
    )
    actor_positions.to_csv(OUTDIR / "actor_concern_axis_positions_5y.csv", index=False)

    make_stable_outcome_figure(
        stable_graph, backbone, positions, land, projection_extent,
        paper_counts, direct_table,
    )
    make_outcome_position_figure(proximity_test, model_table)
    make_trajectory_figure(actor_positions, events, topic_positions)

    degrees = dict(stable_graph.degree())
    components = sorted(
        nx.connected_components(stable_graph), key=lambda members: (-len(members), sorted(members))
    )
    pd.DataFrame(
        [
            {"component": component_id, "component_size": len(members), "concern": concern}
            for component_id, members in enumerate(components, start=1)
            for concern in sorted(members)
        ]
    ).to_csv(OUTDIR / "bootstrap_stable_components.csv", index=False)
    threshold_rows = []
    for threshold in (0.50, 0.60, 0.70):
        keep = (edge_table["topk_frequency"] >= threshold) & (edge_table["phi_low"] > 0)
        graph = nx.Graph()
        graph.add_nodes_from(topics)
        graph.add_edges_from(edge_table.loc[keep, ["concern_a", "concern_b"]].itertuples(index=False, name=None))
        threshold_rows.append(
            {
                "threshold": threshold,
                "edges": int(graph.number_of_edges()),
                "isolated_concerns": int(sum(degree == 0 for _, degree in graph.degree())),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(
        OUTDIR / "bootstrap_threshold_sensitivity.csv", index=False
    )

    summary = {
        "stable_space": {
            "actor_bootstrap_samples": N_BOOTSTRAP,
            "endpoint_top_k": BOOTSTRAP_TOP_K,
            "selection_frequency_threshold": STABILITY_THRESHOLD,
            "requires_positive_95pct_lower_bound": True,
            "stable_edges": int(stable_graph.number_of_edges()),
            "isolated_concerns": int(sum(value == 0 for value in degrees.values())),
            "connected_components": len(components),
            "component_sizes": [len(members) for members in components],
            "figure_1_scaffold_edges": int(backbone.number_of_edges()),
            "layout_coordinates_match_figure_1": True,
        },
        "documented_outcomes": {
            "outcomes": int(events["outcome_id"].nunique()),
            "actor_outcome_incidences": int(len(events)),
            "actors": int(events["actor"].nunique()),
            "example_actor_outcomes": {
                actor: int(events.loc[events["actor"] == actor, "outcome_id"].nunique())
                for actor in EXAMPLE_ACTORS
            },
        },
        "outcome_position_comparison": {
            **panel_diagnostics,
            "proximity": {k: v for k, v in proximity_test.items() if k != "null"},
            "breadth": {k: v for k, v in breadth_test.items() if k != "null"},
            "conditional_logit": model_table.to_dict(orient="records"),
        },
    }
    (OUTDIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote figures/exploratory_stable_space_outcomes.[pdf|png]")
    print("wrote figures/exploratory_outcome_position_model.[pdf|png]")
    print("wrote figures/exploratory_movement_to_outcomes.[pdf|png]")


if __name__ == "__main__":
    main()
