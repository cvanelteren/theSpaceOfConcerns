#!/usr/bin/env python3
"""Feasibility analysis linking the concern space to formal ATCM outcomes.

This script deliberately separates three objects that are easy to conflate:

1. documentary attention: all papers classified under a Secretariat concern;
2. documented formal action: outcomes tied to papers by explicit final-report
   adoption/approval language or an officially documented contribution;
3. supported association: the broader paper--outcome graph after excluding bare
   candidate title-window matches.

The script does not treat the absence of a formal outcome as policy failure.
Different Secretariat concerns and legal instruments perform different roles.
Instead, it asks where formal action is located in the existing concern space
and whether concerns feeding the same outcome are unusually close in that
space, relative to a degree-preserving bipartite null.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import (
    build_graphs,
    fitted_layout,
    load_topic_meta,
    normalize_topic_key,
)
from analysis.data_loading import load_submitted_with_fallback
from utils import compute_product_space


LINEAGE_ROOT = ROOT.parent / "ats_lineage"
OUTDIR = ROOT / "output" / "lineage_space"
FIGDIR = ROOT / "figures"

INSTRUMENTS = ["Recommendation", "Measure", "Decision", "Resolution"]
INSTRUMENT_COLORS = {
    "Recommendation": "#0072B2",
    "Measure": "#D55E00",
    "Decision": "#CC79A7",
    "Resolution": "#009E73",
}

# The strongest category is based on language in a final report explicitly
# connecting a paper to adoption/approval, or on an official paragraph that
# documents the paper's contribution. Direct discussion is informative but is
# kept in the broader supported layer.
DIRECT_ACTION_RELATIONS = {
    "direct_adoption_or_approval",
    "documented_contribution",
}
DIRECT_DISCUSSION_RELATIONS = {"direct_proposal_or_discussion"}


def _normalize(name: object) -> str:
    text = normalize_topic_key(name)
    return " ".join(text.replace("_", " ").split())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _canonical_topic_lookup(topics: list[str]) -> dict[str, str]:
    lookup = {_normalize(topic): topic for topic in topics}
    # Known harmless typography variants in the two exports.
    aliases = {
        "tourism and ng activities": "Tourism and NG_Activities",
        "fauna and flora general": "Fauna and Flora_General",
    }
    for alias, canonical in aliases.items():
        if canonical in topics:
            lookup[alias] = canonical
    return lookup


def _paper_categories(inventory: dict, topic_lookup: dict[str, str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for record in inventory.values():
        meeting = record.get("meeting_number")
        if meeting is None:
            continue
        for paper in record.get("submitted_inputs", []):
            paper_label = paper.get("paper")
            if not paper_label:
                continue
            cats = {
                topic_lookup[key]
                for raw in (paper.get("categories") or [])
                if (key := _normalize(raw)) in topic_lookup
            }
            if cats:
                result[f"ATCM{meeting}:{paper_label}"] = cats
    return result


def _edge_level(edge: dict) -> int:
    """Return 3=direct action, 2=supported, 1=candidate, 0=exclude."""
    relation = edge.get("relation")
    if relation in DIRECT_ACTION_RELATIONS:
        return 3
    if relation in DIRECT_DISCUSSION_RELATIONS:
        return 2
    if edge.get("tier") in {"verified", "supported"}:
        return 2
    if edge.get("tier") == "candidate":
        return 1
    return 0


def _paper_outcome_links(
    decision_map: dict,
    paper_categories: dict[str, set[str]],
) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    nodes = {node["id"]: node for node in decision_map["nodes"]}
    best: dict[tuple[str, str], dict] = {}
    for edge in decision_map["edges"]:
        source = nodes.get(edge.get("src"), {})
        target = nodes.get(edge.get("dst"), {})
        if source.get("kind") != "paper" or target.get("kind") != "outcome":
            continue
        if edge["src"] not in paper_categories or target.get("placeholder"):
            continue
        level = _edge_level(edge)
        if level == 0:
            continue
        key = (edge["src"], edge["dst"])
        record = dict(edge)
        record["level"] = level
        previous = best.get(key)
        if previous is None or level > previous["level"]:
            best[key] = record
    return nodes, best


def _outcome_topic_sets(
    links: dict[tuple[str, str], dict],
    paper_categories: dict[str, set[str]],
    minimum_level: int,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = collections.defaultdict(set)
    for (paper, outcome), edge in links.items():
        if edge["level"] >= minimum_level:
            result[outcome].update(paper_categories[paper])
    return dict(result)


def _topic_instrument_table(
    outcome_topics: dict[str, set[str]],
    nodes: dict[str, dict],
    topics: list[str],
) -> pd.DataFrame:
    table = pd.DataFrame(0, index=topics, columns=INSTRUMENTS, dtype=int)
    for outcome, assigned_topics in outcome_topics.items():
        instrument = nodes.get(outcome, {}).get("outcome_type")
        if instrument not in table.columns:
            continue
        for topic in assigned_topics:
            table.loc[topic, instrument] += 1
    return table


def _filter_outcome_topics_by_year(
    outcome_topics: dict[str, set[str]],
    nodes: dict[str, dict],
    start: int,
    end: int,
) -> dict[str, set[str]]:
    return {
        outcome: topics
        for outcome, topics in outcome_topics.items()
        if start <= int(nodes.get(outcome, {}).get("year") or -1) <= end
    }


def _period_paper_counts(
    topics: list[str],
    topic_lookup: dict[str, str],
    start: int,
    end: int,
) -> pd.Series:
    submitted = load_submitted_with_fallback().copy()
    year_column = "year" if "year" in submitted.columns else "meeting year"
    submitted[year_column] = pd.to_numeric(submitted[year_column], errors="coerce")
    submitted = submitted[
        submitted[year_column].between(start, end, inclusive="both")
    ].copy()
    submitted["canonical_concern"] = submitted["category"].map(
        lambda raw: topic_lookup.get(_normalize(raw))
    )
    submitted = submitted.dropna(subset=["canonical_concern"])
    if "paper id" in submitted.columns:
        counts = submitted.groupby("canonical_concern")["paper id"].nunique()
    else:
        counts = submitted.groupby("canonical_concern").size()
    return counts.reindex(topics, fill_value=0).astype(int)


def _action_pair_counts(outcome_topics: dict[str, set[str]]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for assigned_topics in outcome_topics.values():
        ordered = sorted(assigned_topics)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                counts[(first, second)] += 1
    return counts


def _mean_within_outcome_phi(
    incidence: set[tuple[int, int]],
    n_outcomes: int,
    phi: np.ndarray,
) -> float:
    rows: list[list[int]] = [[] for _ in range(n_outcomes)]
    for outcome, topic in incidence:
        rows[outcome].append(topic)
    outcome_means = []
    for row in rows:
        if len(row) < 2:
            continue
        values = [phi[a, b] for i, a in enumerate(row) for b in row[i + 1 :]]
        if values:
            outcome_means.append(float(np.mean(values)))
    return float(np.mean(outcome_means)) if outcome_means else float("nan")


def _swap_bipartite(
    incidence: set[tuple[int, int]],
    rng: np.random.Generator,
    accepted_target: int,
) -> int:
    """Degree-preserving double-edge swaps, mutating ``incidence`` in place."""
    accepted = 0
    attempts = 0
    max_attempts = max(accepted_target * 40, 1000)
    while accepted < accepted_target and attempts < max_attempts:
        attempts += 1
        edges = tuple(incidence)
        first, second = rng.choice(len(edges), size=2, replace=False)
        o1, t1 = edges[int(first)]
        o2, t2 = edges[int(second)]
        if o1 == o2 or t1 == t2:
            continue
        cross1, cross2 = (o1, t2), (o2, t1)
        if cross1 in incidence or cross2 in incidence:
            continue
        incidence.remove((o1, t1))
        incidence.remove((o2, t2))
        incidence.add(cross1)
        incidence.add(cross2)
        accepted += 1
    return accepted


def _degree_preserving_null(
    outcome_topics: dict[str, set[str]],
    topics: list[str],
    phi: np.ndarray,
    n_samples: int = 999,
    seed: int = 20260812,
) -> tuple[float, np.ndarray, float, dict]:
    topic_index = {topic: i for i, topic in enumerate(topics)}
    eligible = sorted(o for o, cats in outcome_topics.items() if len(cats) >= 2)
    outcome_index = {outcome: i for i, outcome in enumerate(eligible)}
    observed_edges = {
        (outcome_index[outcome], topic_index[topic])
        for outcome in eligible
        for topic in outcome_topics[outcome]
    }
    observed = _mean_within_outcome_phi(observed_edges, len(eligible), phi)

    rng = np.random.default_rng(seed)
    randomized = set(observed_edges)
    edge_count = len(randomized)
    burn_target = max(20 * edge_count, 200)
    burn_accepted = _swap_bipartite(randomized, rng, burn_target)
    between_target = max(3 * edge_count, 50)
    null = np.empty(n_samples, dtype=float)
    accepted = []
    for sample in range(n_samples):
        accepted.append(_swap_bipartite(randomized, rng, between_target))
        null[sample] = _mean_within_outcome_phi(randomized, len(eligible), phi)
    p_upper = float((1 + np.count_nonzero(null >= observed)) / (n_samples + 1))
    diagnostics = {
        "n_multi_topic_outcomes": len(eligible),
        "n_incidence_edges": edge_count,
        "burn_swaps_accepted": burn_accepted,
        "mean_between_sample_swaps_accepted": float(np.mean(accepted)),
    }
    return observed, null, p_upper, diagnostics


def _centrality_footprint_test(
    phi: np.ndarray,
    paper_counts: pd.Series,
    action_counts: pd.Series,
    n_samples: int = 9999,
    seed: int = 20260812,
) -> dict:
    """Compare outcome-weighted topology strength with a paper-volume baseline.

    This is an exploratory reference distribution, not an adoption model. It
    asks where the observed formal-action footprint lies if its incidences were
    redistributed in proportion to the documentary attention already present.
    """
    matrix = np.array(phi, dtype=float, copy=True)
    np.fill_diagonal(matrix, 0.0)
    strength = matrix.sum(axis=1)
    paper_weights = paper_counts.to_numpy(dtype=float)
    action_weights = action_counts.to_numpy(dtype=float)
    observed = float(np.average(strength, weights=action_weights))
    attention_baseline = float(np.average(strength, weights=paper_weights))
    probabilities = paper_weights / paper_weights.sum()
    total = int(round(action_weights.sum()))
    rng = np.random.default_rng(seed)
    null = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        draw = rng.multinomial(total, probabilities)
        null[i] = np.average(strength, weights=draw)
    return {
        "outcome_weighted_phi_strength": observed,
        "attention_weighted_phi_strength": attention_baseline,
        "relative_difference": observed / attention_baseline - 1.0,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "upper_tail_p": float((1 + np.count_nonzero(null >= observed)) / (n_samples + 1)),
        "reference": "multinomial redistribution of action incidences proportional to submitted-paper volume",
    }


def _node_sizes(values: pd.Series, minimum: float = 22, maximum: float = 430) -> np.ndarray:
    values = values.astype(float).to_numpy()
    vmax = float(np.max(values)) if len(values) else 0.0
    if vmax <= 0:
        return np.full_like(values, minimum, dtype=float)
    return minimum + (maximum - minimum) * values / vmax


def _effective_number(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    if array.sum() <= 0:
        return float("nan")
    shares = array[array > 0] / array.sum()
    return float(np.exp(-np.sum(shares * np.log(shares))))


def _top_share(values: pd.Series, n: int = 5) -> float:
    array = values.to_numpy(dtype=float)
    if array.sum() <= 0:
        return float("nan")
    return float(np.sort(array)[-n:].sum() / array.sum())


def _draw_backbone(ax, backbone, positions, alpha: float = 0.22) -> None:
    for source, target, attrs in backbone.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        weight = float(attrs.get("weight", 0.0))
        ax.plot(
            [x0, x1], [y0, y1],
            color="#8A93A0", lw=0.35 + 1.25 * weight,
            alpha=alpha, zorder=0, solid_capstyle="round",
        )


def _selected_labels(
    paper_counts: pd.Series,
    action_counts: pd.Series,
    maximum: int = 8,
) -> list[str]:
    score = (
        paper_counts.rank(pct=True)
        + 2.2 * action_counts.rank(pct=True)
        + 0.4 * (action_counts > 0).astype(float)
    )
    return score.sort_values(ascending=False).head(maximum).index.tolist()


def _short_label(topic: str) -> str:
    replacements = {
        "Area Protection and Management Plans General": "Area protection",
        "Operation of the Antarctic Treaty system General": "ATS operation",
        "Operation of the Antarctic Treaty system Reports": "ATS reports",
        "Operation of the Antarctic Treaty system The Secretariat": "Secretariat",
        "Environmental Impact Assessment EIA Other EIA Matters": "EIA",
        "Tourism and NG_Activities": "Tourism",
        "Historic Sites and Monuments": "Historic sites",
        "Fauna and Flora_General": "Fauna and flora",
        "Environmental Protection General": "Environmental protection",
        "Safety and Operations in Antarctica": "Safety and operations",
        "Exchange of Information": "Information exchange",
        "Human Footprint and wilderness values": "Human footprint",
    }
    return replacements.get(topic, topic)


def _draw_labels(ax, positions, labels: list[str]) -> None:
    offsets = {
        "Area Protection and Management Plans General": (10, -13),
        "Tourism and NG_Activities": (7, -11),
        "Historic Sites and Monuments": (32, 34),
        "Operation of the Antarctic Treaty system Reports": (6, 12),
        "Operation of the Antarctic Treaty system General": (-5, -12),
        "Management Plans": (-28, -16),
        "Environmental Impact Assessment EIA Other EIA Matters": (-31, 22),
        "Liability": (-22, 2),
        "Climate Change": (7, 11),
        "Environmental Protection General": (-8, -12),
    }
    for topic in labels:
        x, y = positions[topic]
        dx, dy = offsets.get(topic, (5, 7))
        ax.annotate(
            _short_label(topic),
            xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=7.0, color="#263238",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 0.35},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
            zorder=8,
        )


def _dominant_instrument_colors(table: pd.DataFrame) -> list[str]:
    colors = []
    for _, row in table.iterrows():
        if int(row.sum()) == 0:
            colors.append("#D1D5DB")
        else:
            colors.append(INSTRUMENT_COLORS[str(row.idxmax())])
    return colors


def _make_space_figure(
    topics: list[str],
    positions: dict[str, tuple[float, float]],
    backbone,
    paper_counts: pd.Series,
    direct_table: pd.DataFrame,
    supported_table: pd.DataFrame,
    action_pairs: collections.Counter,
    n_multi_outcomes: int,
    comparison_start: int,
    comparison_end: int,
) -> None:
    direct_counts = direct_table.sum(axis=1)
    supported_counts = supported_table.sum(axis=1)
    labels = [
        "Operation of the Antarctic Treaty system Reports",
        "Historic Sites and Monuments",
        "Liability",
        "Environmental Impact Assessment EIA Other EIA Matters",
        "Management Plans",
        "Area Protection and Management Plans General",
        "Tourism and NG_Activities",
        "Operation of the Antarctic Treaty system General",
    ]
    coords = np.array([positions[t] for t in topics])

    fig, axes = uplt.subplots(ncols=3, refwidth=2.65, aspect=1, share=False)
    titles = [
        f"Documentary attention, {comparison_start}–{comparison_end}",
        f"Documented formal-action footprint, {comparison_start}–{comparison_end}",
        f"Only {n_multi_outcomes} documented outcomes span concerns",
    ]
    for ax, title in zip(axes, titles):
        _draw_backbone(ax, backbone, positions)
        ax.format(
            title=title, aspect="equal", grid=False,
            xlocator=[], ylocator=[],
            xspineloc="neither", yspineloc="neither",
        )

    axes[0].scatter(
        coords[:, 0], coords[:, 1],
        s=_node_sizes(paper_counts, 24, 360),
        c="#3F566B", edgecolor="white", linewidth=0.75, zorder=3,
    )

    axes[1].scatter(
        coords[:, 0], coords[:, 1],
        s=_node_sizes(direct_counts, 18, 500),
        c=_dominant_instrument_colors(direct_table),
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    # A thin outer ring shows how far the footprint expands when all supported
    # (but not bare candidate) links are admitted.
    axes[1].scatter(
        coords[:, 0], coords[:, 1],
        s=_node_sizes(supported_counts, 28, 610),
        facecolors="none", edgecolors="#4B5563", linewidths=0.65,
        alpha=0.65, zorder=2,
    )

    if action_pairs:
        maximum = max(action_pairs.values())
        for (source, target), count in action_pairs.most_common():
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            axes[2].plot(
                [x0, x1], [y0, y1],
                color="#6A3D9A", lw=0.55 + 2.7 * math.sqrt(count / maximum),
                alpha=0.38 + 0.48 * math.sqrt(count / maximum),
                zorder=2, solid_capstyle="round",
            )
    axes[2].scatter(
        coords[:, 0], coords[:, 1],
        s=_node_sizes(direct_counts, 18, 420),
        c=_dominant_instrument_colors(direct_table),
        edgecolor="white", linewidth=0.75, zorder=4,
    )

    # One shared set of labels is enough because all panels use identical
    # coordinates. Repeating labels in each square obscures the outcome layer.
    _draw_labels(axes[0], positions, labels)

    # Compact semantic legends: one for instrument hue, one for evidence ring.
    handles = [
        axes[1].scatter([], [], s=55, color=INSTRUMENT_COLORS[name],
                        edgecolor="white", label=name)
        for name in INSTRUMENTS
    ]
    handles.append(
        axes[1].scatter([], [], s=70, facecolors="none", edgecolors="#4B5563",
                        linewidths=0.8, label="supported-link extent")
    )
    axes[1].legend(handles=handles, loc="b", ncols=3, frame=False, fontsize=7.5)
    axes[2].plot([], [], color="#6A3D9A", lw=2.0,
                 label="shared documented outcome")
    axes[2].legend(loc="b", frame=False, fontsize=7.5)

    fig.format(
        abc="A", abcloc="ul",
        suptitle="From documentary attention to formal action in the same concern space",
    )
    fig.save(FIGDIR / "exploratory_lineage_space_map.pdf")
    fig.save(FIGDIR / "exploratory_lineage_space_map.png", dpi=260)
    uplt.close(fig)


def _make_diagnostic_figure(
    paper_counts: pd.Series,
    direct_table: pd.DataFrame,
    supported_table: pd.DataFrame,
    observed_phi: float,
    null_phi: np.ndarray,
    p_upper: float,
) -> None:
    direct_counts = direct_table.sum(axis=1)
    supported_counts = supported_table.sum(axis=1)
    top = direct_counts.sort_values(ascending=False).head(13).index[::-1]

    fig, axes = uplt.subplots(ncols=3, refwidth=2.65, aspect=1, share=False)

    # A. Concern-specific instrument mix among directly documented actions.
    left = np.zeros(len(top), dtype=float)
    y = np.arange(len(top))
    for instrument in INSTRUMENTS:
        values = direct_table.loc[top, instrument].to_numpy(dtype=float)
        axes[0].barh(
            y, values, left=left,
            color=INSTRUMENT_COLORS[instrument],
            edgecolor="white", linewidth=0.35, label=instrument,
        )
        left += values
    axes[0].format(
        title="Formal instruments divide the work",
        xlabel="Documented formal outcomes", ylabel="",
        yticks=y, yticklabels=[_short_label(t) for t in top],
        xlim=(0, max(left) * 1.05 if len(left) else 1),
        grid=False,
    )
    axes[0].legend(loc="b", ncols=2, frame=False, fontsize=7.5)

    # B. Compare normalized concentration across the same 45 categories. This
    # describes the breadth of each layer without treating papers as a proposal
    # denominator or low-output concerns as failures.
    concentration_series = {
        "documentary attention": (paper_counts, "#3F566B"),
        "documented action": (direct_counts, "#6A3D9A"),
        "supported association": (supported_counts, "#C4699E"),
    }
    ranks = np.arange(1, len(paper_counts) + 1)
    for label, (series, color) in concentration_series.items():
        ordered = np.sort(series.to_numpy(dtype=float))[::-1]
        cumulative = np.cumsum(ordered) / max(float(ordered.sum()), 1.0)
        axes[1].plot(ranks, cumulative, color=color, lw=2.0, label=label)
    axes[1].axhline(0.5, color="#9CA3AF", lw=0.8, ls="--")
    axes[1].format(
        title="Formal action occupies a narrower footprint",
        xlabel="Concerns, ranked within each layer", ylabel="Cumulative share",
        xlim=(1, len(paper_counts)), ylim=(0, 1.02),
        yticks=np.arange(0, 1.01, 0.2), yformatter=PercentFormatter(1.0), grid=False,
    )
    axes[1].legend(loc="b", ncols=1, frame=False, fontsize=7.5)

    # C. Degree-preserving null for the concern combinations feeding outcomes.
    axes[2].hist(
        null_phi, bins=26, density=True,
        color="#CBD5E1", edgecolor="white", linewidth=0.45,
    )
    axes[2].axvline(observed_phi, color="#6A3D9A", lw=2.1,
                    label=f"observed = {observed_phi:.3f}")
    axes[2].axvline(float(np.mean(null_phi)), color="#475569", lw=1.2, ls="--",
                    label=f"null mean = {np.mean(null_phi):.3f}")
    axes[2].format(
        title="Cross-concern action is not clearly local",
        xlabel="Mean concern proximity within an outcome", ylabel="Null density",
        grid=False,
    )
    axes[2].legend(loc="b", frame=False, fontsize=7.5,
                   title=f"degree-preserving null, p={p_upper:.3g}")

    fig.format(
        abc="A", abcloc="ul",
        suptitle="What the outcome layer adds to the concern space",
    )
    fig.save(FIGDIR / "exploratory_lineage_space_diagnostics.pdf")
    fig.save(FIGDIR / "exploratory_lineage_space_diagnostics.png", dpi=260)
    uplt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    backbone, mst, graph, counts_df, rca = build_graphs()
    topics = list(counts_df.index)
    _, mode_of, _ = load_topic_meta()
    positions, _, _ = fitted_layout(mst, graph, mode_of)
    phi_df = compute_product_space(rca).reindex(index=topics, columns=topics).fillna(0.0)
    phi = phi_df.to_numpy(dtype=float)

    decision_map = _load_json(LINEAGE_ROOT / "decision_map.json")
    inventory = _load_json(LINEAGE_ROOT / "atcm_inventory.json")
    topic_lookup = _canonical_topic_lookup(topics)
    paper_categories = _paper_categories(inventory, topic_lookup)
    nodes, links = _paper_outcome_links(decision_map, paper_categories)

    direct_topics_raw = _outcome_topic_sets(links, paper_categories, minimum_level=3)
    supported_topics_raw = _outcome_topic_sets(links, paper_categories, minimum_level=2)
    candidate_topics_raw = _outcome_topic_sets(links, paper_categories, minimum_level=1)
    comparison_start = min(int(nodes[o]["year"]) for o in direct_topics_raw)
    comparison_end = max(int(nodes[o]["year"]) for o in direct_topics_raw)
    direct_topics = _filter_outcome_topics_by_year(
        direct_topics_raw, nodes, comparison_start, comparison_end
    )
    supported_topics = _filter_outcome_topics_by_year(
        supported_topics_raw, nodes, comparison_start, comparison_end
    )
    candidate_topics = _filter_outcome_topics_by_year(
        candidate_topics_raw, nodes, comparison_start, comparison_end
    )

    direct_table = _topic_instrument_table(direct_topics, nodes, topics)
    supported_table = _topic_instrument_table(supported_topics, nodes, topics)
    candidate_table = _topic_instrument_table(candidate_topics, nodes, topics)
    paper_counts = _period_paper_counts(
        topics, topic_lookup, comparison_start, comparison_end
    )
    action_pairs = _action_pair_counts(direct_topics)

    direct_multi = sum(len(cats) >= 2 for cats in direct_topics.values())
    supported_multi = sum(len(cats) >= 2 for cats in supported_topics.values())
    candidate_multi = sum(len(cats) >= 2 for cats in candidate_topics.values())

    observed_phi, null_phi, p_upper, null_diag = _degree_preserving_null(
        direct_topics, topics, phi,
    )
    centrality_test = _centrality_footprint_test(
        phi, paper_counts, direct_table.sum(axis=1)
    )

    summary = pd.DataFrame(index=topics)
    summary.index.name = "concern"
    summary["submitted_papers"] = paper_counts
    summary["documented_action_outcomes"] = direct_table.sum(axis=1)
    summary["supported_association_outcomes"] = supported_table.sum(axis=1)
    summary["all_including_candidate_outcomes"] = candidate_table.sum(axis=1)
    summary["dominant_documented_instrument"] = [
        row.idxmax() if int(row.sum()) else "None" for _, row in direct_table.iterrows()
    ]
    summary = summary.join(
        direct_table.add_prefix("documented_")
    ).join(
        supported_table.add_prefix("supported_")
    )
    summary.to_csv(OUTDIR / "concern_outcome_footprint.csv")

    direct_sources: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for (paper, outcome), edge in links.items():
        if edge["level"] >= 3:
            direct_sources[outcome].append((paper, edge))
    outcome_rows = []
    for outcome in sorted(direct_topics, key=lambda oid: (nodes[oid].get("year") or 0, oid)):
        sources = direct_sources[outcome]
        outcome_rows.append(
            {
                "outcome_id": outcome,
                "year": nodes[outcome].get("year"),
                "instrument": nodes[outcome].get("outcome_type"),
                "title": nodes[outcome].get("title"),
                "concerns": " | ".join(sorted(direct_topics[outcome])),
                "n_concerns": len(direct_topics[outcome]),
                "source_papers": " | ".join(sorted({paper for paper, _ in sources})),
                "relations": " | ".join(sorted({edge.get("relation", "") for _, edge in sources})),
                "reports": " | ".join(sorted({edge.get("report", "") for _, edge in sources if edge.get("report")})),
            }
        )
    pd.DataFrame(outcome_rows).to_csv(
        OUTDIR / "documented_action_outcomes_audit.csv", index=False
    )

    pair_rows = [
        {"concern_a": pair[0], "concern_b": pair[1], "n_documented_outcomes": count,
         "phi": float(phi_df.loc[pair[0], pair[1]])}
        for pair, count in action_pairs.most_common()
    ]
    pd.DataFrame(pair_rows).to_csv(OUTDIR / "documented_action_pairs.csv", index=False)
    pd.DataFrame({"null_mean_phi": null_phi}).to_csv(
        OUTDIR / "degree_preserving_null.csv", index=False
    )

    analysis = {
        "definitions": {
            "documented_action": sorted(DIRECT_ACTION_RELATIONS),
            "supported_association": "paper--outcome edges at verified or supported tier, including direct discussion; excludes bare candidate title-window matches",
            "candidate": "includes candidate title-window and co-mention links",
            "comparison_window": f"{comparison_start}-{comparison_end}; the pooled 1961-2025 concern topology is retained only as a fixed coordinate system",
        },
        "counts": {
            "concerns": len(topics),
            "papers_with_mapped_categories": len(paper_categories),
            "collapsed_paper_outcome_links": len(links),
            "supported_association_outcomes_all_years": len(supported_topics_raw),
            "all_including_candidate_outcomes_all_years": len(candidate_topics_raw),
            "documented_action_outcomes": len(direct_topics),
            "supported_association_outcomes": len(supported_topics),
            "all_including_candidate_outcomes": len(candidate_topics),
            "documented_multi_concern_outcomes": null_diag["n_multi_topic_outcomes"],
            "supported_multi_concern_outcomes": supported_multi,
            "all_including_candidate_multi_concern_outcomes": candidate_multi,
            "documented_action_by_instrument": dict(collections.Counter(
                nodes[outcome].get("outcome_type") for outcome in direct_topics
            )),
        },
        "footprint_concentration": {
            "documentary_attention_effective_concerns": _effective_number(paper_counts),
            "documented_action_effective_concerns": _effective_number(direct_table.sum(axis=1)),
            "supported_association_effective_concerns": _effective_number(supported_table.sum(axis=1)),
            "documentary_attention_top5_share": _top_share(paper_counts),
            "documented_action_top5_share": _top_share(direct_table.sum(axis=1)),
            "supported_association_top5_share": _top_share(supported_table.sum(axis=1)),
        },
        "cross_concern_share": {
            "documented_action": direct_multi / max(len(direct_topics), 1),
            "supported_association": supported_multi / max(len(supported_topics), 1),
            "all_including_candidate": candidate_multi / max(len(candidate_topics), 1),
        },
        "locality_test": {
            "observed_mean_within_outcome_phi": observed_phi,
            "null_mean": float(np.mean(null_phi)),
            "null_sd": float(np.std(null_phi, ddof=1)),
            "upper_tail_p": p_upper,
            **null_diag,
        },
        "exploratory_topological_concentration": centrality_test,
    }
    (OUTDIR / "analysis_summary.json").write_text(json.dumps(analysis, indent=2) + "\n")

    _make_space_figure(
        topics, positions, backbone, paper_counts,
        direct_table, supported_table, action_pairs, direct_multi,
        comparison_start, comparison_end,
    )
    _make_diagnostic_figure(
        paper_counts, direct_table, supported_table,
        observed_phi, null_phi, p_upper,
    )

    print(json.dumps(analysis, indent=2))
    print(f"wrote {OUTDIR.relative_to(ROOT)}")
    print("wrote figures/exploratory_lineage_space_map.[pdf|png]")
    print("wrote figures/exploratory_lineage_space_diagnostics.[pdf|png]")


if __name__ == "__main__":
    main()
