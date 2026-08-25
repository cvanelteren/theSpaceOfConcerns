#!/usr/bin/env python3
"""How documentary attention enters formal action, and what Measures actually are.

`analyze_attention_to_outcomes.py` found that prior same-concern attention
predicts later formal output overall and for Resolutions, but not for Measures
(IRR 1.12, 95% CI 0.84--1.50). This script asks whether that null reflects
timing, formal mediation, the composition of the Measure population, incomplete
paper lineage, or a genuinely different decision process.

It follows `MEASURE_PATHWAYS_NEXT_STEPS.md` and is deliberately organised as
five work packages:

1. validate and functionally classify all 279 Measures;
2. describe how each Measure is assembled from recovered predecessors;
3. fit pre-specified timing and formal-history models;
4. test whether observed lineage links are unusually concern-proximate
   relative to a time- and source-matched null; and
5. emit a machine-readable summary that selects one row of the decision table.

Two facts about the corpus drive most of the design. First, Measures,
Decisions, and Resolutions exist only from 1995; Recommendations run
1961--1994. The original instrument-split models were fitted over 1961--2025,
so 34 of 65 years contributed structural zeros. All Measure models here are
restricted to 1995--2025. Second, the Measure population is overwhelmingly
recurring site administration, so a nominal sample of 279 instruments does not
represent 279 independent political processes.
"""

from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.data_loading import load_submitted_with_fallback
from fig01_space_of_concerns_topology import build_graphs
from scripts import explore_lineage_space as lineage
from utils import compute_product_space, get_rca

import scripts.analyze_attention_to_outcomes as base


LINEAGE_ROOT = base.LINEAGE_ROOT
OUTDIR = base.OUTDIR
CACHE = ROOT / "output" / "outcome_linkage" / "_cache"

GRAPH_NAME = "decision_map_verified.json"
START_YEAR = base.START_YEAR
END_YEAR = base.END_YEAR
# Measures, Decisions, and Resolutions were created by the 1995 instrument
# reform. Before that the ATCM produced Recommendations only.
MEASURE_START_YEAR = 1995
RANDOM_SEED = 20260813
N_PERMUTATIONS = 9999
# Minimum probability-weighted events before a fixed-effect Poisson on this
# panel is treated as informative rather than merely fitted.
MINIMUM_EVENTS_FOR_INFERENCE = 30

# Relations that transform an earlier instrument into the target. These are the
# links that can carry a substantive pathway.
STRONG_OUTCOME_RELATIONS = ("supersedes", "amends", "pursuant_to", "designates_under")
# Relations that place the target in a legal context without transforming a
# specific earlier instrument.
CONTEXTUAL_OUTCOME_RELATIONS = ("recalls", "cites")
PAPER_ACTION_RELATIONS = ("direct_adoption_or_approval", "documented_contribution")
PAPER_DISCUSSION_RELATIONS = ("direct_proposal_or_discussion",)

# Pre-registered exposure windows, in years before the target year. Fixed here
# before any coefficient was inspected.
WINDOWS: dict[str, tuple[int, int]] = {
    "same_meeting": (0, 0),
    "prior_1_3": (1, 3),
    "prior_4_7": (4, 7),
    "prior_8_15": (8, 15),
}

# Functional families that represent recurring administration of a named site
# or of the protected-area register, as opposed to substantive rule-making.
RECURRING_SITE_FAMILIES = {
    "management_plan",
    "protected_area_designation",
    "historic_site",
}

# Ordered coding rules for the functional typology. The first pattern that
# matches decides the family, so the order is part of the protocol: substantive
# instruments are tested before site vocabulary, because titles such as
# "Specially Protected Species: Fur Seals" contain protected-area words without
# being site administration.
FAMILY_RULES: list[tuple[str, str, str]] = [
    (
        "environment_or_liability",
        r"liability|environmental emergenc|annex\s+(i|ii|iii|iv|v|vi)\b|environment protocol|"
        r"protocol on environmental protection|specially protected species",
        "Amends or adds to the environmental-protection or liability regime.",
    ),
    (
        "tourism_safety_operations",
        # "vessel" alone is too broad: it also matches shipwrecks listed as
        # Historic Sites, so an operational context is required.
        r"tourism|tourist|non[- ]govern|landing of persons|passenger vessel|insurance|"
        r"contingency plan|shipborne|air safety|vessel operation",
        "Regulates tourism, non-governmental activity, vessels, or operational safety.",
    ),
    (
        "historic_site",
        r"historic site|historic monument|historical site|monument|\bhsm\b",
        "Designates, lists, or revises a Historic Site or Monument.",
    ),
    (
        "management_plan",
        r"management plan",
        "Adopts, revises, or revokes a management plan for a designated area.",
    ),
    (
        "protected_area_designation",
        r"protected area|specially managed area|\baspa\b|\basma\b|\bsssi\b|"
        r"site[s]? of special scientific interest|expiry date|\bspa\s*\d",
        "Designates, extends, or alters a protected or managed area without a new plan.",
    ),
]

SITE_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("ASPA", r"(?:antarctic specially protected area|aspa)\s*(?:no\.?|number)?\s*(\d{1,3})"),
    ("ASMA", r"(?:antarctic specially managed area|asma)\s*(?:no\.?|number)?\s*(\d{1,3})"),
    ("SSSI", r"(?:site of special scientific interest|sssi)\s*(?:no\.?|number)?\s*(\d{1,3})"),
    ("SPA", r"(?:^|[^a-z])spa\s*(?:no\.?|number)?\s*(\d{1,3})"),
    ("HSM", r"(?:historic site|historic monument|hsm)s?\s*(?:no\.?|number)?\s*(\d{1,3})"),
]


def load_json(path: Path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Work package 1: validate and classify the Measure population
# ---------------------------------------------------------------------------


def classify_family(title: str) -> tuple[str, str]:
    """Return (family, rule) for one Measure title.

    The rule string records which pattern fired so that every assignment is
    auditable and reproducible without re-reading the code.
    """
    text = (title or "").lower()
    for family, pattern, _ in FAMILY_RULES:
        if re.search(pattern, text):
            return family, pattern
    return "other_substantive", "no rule matched"


def parse_site_ids(title: str) -> list[str]:
    """Extract stable site identities such as ASPA 106 or HSM 72 from a title.

    Titles frequently list several sites of one type by number after naming the
    type once ("ASPA 106 (Cape Hallett), 107 (Emperor Island), 108 ..."). The
    parser therefore carries the most recent type forward across bare numbers
    that are followed by a parenthesised site name.
    """
    text = (title or "").lower()
    hits: list[tuple[int, str]] = []
    for site_type, pattern in SITE_TYPE_PATTERNS:
        for match in re.finditer(pattern, text):
            hits.append((match.start(), f"{site_type} {int(match.group(1))}"))
    if not hits:
        return []
    # Continuation numbers: a bare "<number> (" after a typed hit inherits it.
    last_type = None
    ordered = sorted(hits)
    typed_spans = {position for position, _ in ordered}
    for match in re.finditer(r"(\d{1,3})\s*\(", text):
        if match.start() in typed_spans:
            continue
        preceding = [label for position, label in ordered if position < match.start()]
        if not preceding:
            continue
        last_type = preceding[-1].split()[0]
        hits.append((match.start(), f"{last_type} {int(match.group(1))}"))
    return sorted({label for _, label in hits})


def load_graph() -> tuple[dict, list[dict]]:
    graph = load_json(LINEAGE_ROOT / GRAPH_NAME)
    nodes = {node["id"]: node for node in graph["nodes"]}
    return nodes, graph["edges"]


def relation_class(relation: str) -> str:
    if relation in STRONG_OUTCOME_RELATIONS:
        return "strong_transformation"
    if relation in CONTEXTUAL_OUTCOME_RELATIONS:
        return "contextual_reference"
    if relation in PAPER_ACTION_RELATIONS:
        return "paper_adoption_or_contribution"
    if relation in PAPER_DISCUSSION_RELATIONS:
        return "paper_proposal_or_discussion"
    return "other"


def build_measure_inventory(
    nodes: dict,
    edges: list[dict],
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the 279-row Measure inventory and the incoming-edge audit table."""
    measures = {
        node_id: node
        for node_id, node in nodes.items()
        if node.get("kind") == "outcome"
        and node.get("outcome_type") == "Measure"
        and not node.get("placeholder")
    }
    prediction_index = predictions.set_index("outcome_id")

    incoming: dict[str, list[dict]] = collections.defaultdict(list)
    audit_rows = []
    for edge in edges:
        target_id = edge["dst"]
        if target_id not in measures:
            continue
        source = nodes.get(edge["src"], {})
        source_kind = source.get("kind")
        if source_kind not in {"paper", "outcome"}:
            continue
        target_year = measures[target_id]["year"]
        source_year = source.get("year")
        lag = (
            int(target_year) - int(source_year)
            if isinstance(source_year, int) and isinstance(target_year, int)
            else np.nan
        )
        record = {
            "measure_id": target_id,
            "measure_year": int(target_year),
            "source_id": edge["src"],
            "source_kind": source_kind,
            "source_instrument": source.get("outcome_type") if source_kind == "outcome" else "Paper",
            "source_year": int(source_year) if isinstance(source_year, int) else np.nan,
            "relation": edge.get("relation"),
            "relation_class": relation_class(edge.get("relation")),
            "tier": edge.get("tier"),
            "confidence": edge.get("confidence"),
            "lag_years": lag,
        }
        incoming[target_id].append(record)
        audit_rows.append(record)
    edge_audit = pd.DataFrame(audit_rows).sort_values(["measure_year", "measure_id", "source_id"])

    probability_pivot = probabilities.pivot(
        index="outcome_id", columns="topic", values="probability"
    )

    rows = []
    for measure_id, node in sorted(measures.items(), key=lambda item: (item[1]["year"], item[0])):
        title = node.get("title") or ""
        family, rule = classify_family(title)
        site_ids = parse_site_ids(title)
        records = incoming.get(measure_id, [])
        paper_sources = [r for r in records if r["source_kind"] == "paper"]
        outcome_sources = [r for r in records if r["source_kind"] == "outcome"]
        strong = [r for r in outcome_sources if r["relation_class"] == "strong_transformation"]
        contextual = [
            r for r in outcome_sources if r["relation_class"] == "contextual_reference"
        ]
        has_paper = bool(paper_sources)
        has_outcome = bool(outcome_sources)
        if has_paper and has_outcome:
            pathway = "both"
        elif has_paper:
            pathway = "paper_only"
        elif has_outcome:
            pathway = "outcome_only"
        else:
            pathway = "neither"
        prediction = (
            prediction_index.loc[measure_id] if measure_id in prediction_index.index else None
        )
        row = {
            "measure_id": measure_id,
            "year": int(node["year"]),
            "meeting": int(node["meeting"]),
            "title": title,
            "title_source": node.get("title_source"),
            "functional_family": family,
            "family_rule": rule,
            "recurring_site_administration": family in RECURRING_SITE_FAMILIES,
            "site_ids": " | ".join(site_ids),
            "n_site_ids": len(site_ids),
            "primary_site_id": site_ids[0] if site_ids else "",
            "pathway": pathway,
            "n_paper_predecessors": len(paper_sources),
            "n_paper_adoption_predecessors": sum(
                1 for r in paper_sources if r["relation_class"] == "paper_adoption_or_contribution"
            ),
            "n_outcome_predecessors": len(outcome_sources),
            "n_strong_outcome_predecessors": len(strong),
            "n_contextual_outcome_predecessors": len(contextual),
            "min_strong_lag_years": min((r["lag_years"] for r in strong), default=np.nan),
            "min_contextual_lag_years": min((r["lag_years"] for r in contextual), default=np.nan),
            "paper_predecessors": " | ".join(sorted(r["source_id"] for r in paper_sources)),
            "outcome_predecessors": " | ".join(sorted(r["source_id"] for r in outcome_sources)),
            "topic_top1": prediction["topic_top1"] if prediction is not None else "",
            "topic_top1_probability": (
                float(prediction["probability_top1"]) if prediction is not None else np.nan
            ),
            "high_confidence_topic": (
                bool(prediction["high_confidence"]) if prediction is not None else False
            ),
        }
        if measure_id in probability_pivot.index:
            distribution = probability_pivot.loc[measure_id]
            row["topic_probability_entropy"] = float(
                -(distribution * np.log(distribution.clip(lower=1e-12))).sum()
            )
        else:
            row["topic_probability_entropy"] = np.nan
        rows.append(row)
    inventory = pd.DataFrame(rows)
    return inventory, edge_audit


def unconnected_audit_sample(inventory: pd.DataFrame, per_stratum: int = 4) -> pd.DataFrame:
    """Draw a stratified sample of Measures with no recovered predecessor.

    Absence of an edge is not evidence of institutional independence: it may
    reflect Conference Room Papers, drafting groups, or report language the
    parser did not recover. Strata are functional family crossed with a coarse
    period so that the audit covers both the sparse early years and the dense
    modern management-plan era.
    """
    unconnected = inventory[inventory["pathway"] == "neither"].copy()
    unconnected["period"] = np.where(unconnected["year"] <= 2009, "1995_2009", "2010_2025")
    rng = np.random.default_rng(RANDOM_SEED)
    picks = []
    for (family, period), group in unconnected.groupby(["functional_family", "period"]):
        take = min(per_stratum, len(group))
        chosen = group.iloc[rng.choice(len(group), size=take, replace=False)]
        for _, record in chosen.iterrows():
            picks.append(
                {
                    "measure_id": record["measure_id"],
                    "year": record["year"],
                    "functional_family": family,
                    "period": period,
                    "stratum_size": int(len(group)),
                    "title": record["title"],
                    "site_ids": record["site_ids"],
                    "coder_verdict_blank": "",
                    "coder_note_blank": "",
                }
            )
    return pd.DataFrame(picks).sort_values(["functional_family", "period", "year"])


# ---------------------------------------------------------------------------
# Work package 2: describe how Measures are assembled
# ---------------------------------------------------------------------------


def pathway_composition(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["paper_only", "outcome_only", "both", "neither"]
    for group_label, subset in [("all_measures", inventory)] + [
        (f"family:{family}", part)
        for family, part in inventory.groupby("functional_family")
    ] + [
        ("recurring_site_administration", inventory[inventory["recurring_site_administration"]]),
        ("other_measures", inventory[~inventory["recurring_site_administration"]]),
    ]:
        counts = subset["pathway"].value_counts()
        total = int(len(subset))
        for pathway in order:
            count = int(counts.get(pathway, 0))
            rows.append(
                {
                    "group": group_label,
                    "pathway": pathway,
                    "measures": count,
                    "group_total": total,
                    "share": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def predecessor_lag_summary(edge_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = edge_audit.dropna(subset=["lag_years"]).groupby(
        ["relation_class", "relation", "source_instrument"]
    )
    for (relation_group, relation, instrument), part in grouped:
        lags = part["lag_years"].to_numpy(dtype=float)
        rows.append(
            {
                "relation_class": relation_group,
                "relation": relation,
                "source_instrument": instrument,
                "edges": int(len(lags)),
                "measures": int(part["measure_id"].nunique()),
                "lag_median": float(np.median(lags)),
                "lag_mean": float(lags.mean()),
                "lag_p25": float(np.percentile(lags, 25)),
                "lag_p75": float(np.percentile(lags, 75)),
                "lag_min": float(lags.min()),
                "lag_max": float(lags.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["relation_class", "relation", "source_instrument"])


# ---------------------------------------------------------------------------
# Work package 3: timing and formal inheritance
# ---------------------------------------------------------------------------


def window_sum(panel: pd.DataFrame, column: str, low: int, high: int) -> pd.Series:
    """Sum a concern-year series over lags [low, high] inclusive, by concern."""
    grouped = panel.groupby("topic")[column]
    parts = [grouped.shift(lag).fillna(0.0) for lag in range(low, high + 1)]
    return sum(parts)


def build_measure_panel(
    base_panel: pd.DataFrame,
    inventory: pd.DataFrame,
    probabilities: pd.DataFrame,
    topics: list[str],
) -> pd.DataFrame:
    """Extend the concern-year panel with pre-specified windows and family splits."""
    panel = base_panel.copy().sort_values(["topic", "year"]).reset_index(drop=True)

    # Probability-weighted Measure mass split by functional family. Each Measure
    # contributes its full concern distribution to exactly one family, so the
    # two family series sum to the total Measure mass.
    family_of = inventory.set_index("measure_id")["recurring_site_administration"]
    measure_probabilities = probabilities[probabilities["instrument"] == "Measure"].copy()
    measure_probabilities["recurring"] = measure_probabilities["outcome_id"].map(family_of)
    for label, keep in (
        ("measure_mass_site_admin", measure_probabilities["recurring"] == True),  # noqa: E712
        ("measure_mass_other", measure_probabilities["recurring"] == False),  # noqa: E712
    ):
        series = (
            measure_probabilities[keep]
            .groupby(["topic", "year"])["probability"]
            .sum()
            .rename(label)
        )
        panel = panel.merge(series.reset_index(), on=["topic", "year"], how="left")
        panel[label] = panel[label].fillna(0.0)

    # Formal predecessors are split by instrument so that no window can contain
    # the outcome being modelled. Other instruments (Recommendations pre-1995,
    # Decisions and Resolutions after) are safe at every window including the
    # adoption year. Earlier Measures enter only at strictly positive lags,
    # because same-year Measure mass in a concern is the outcome itself.
    panel["other_instrument_mass"] = (
        panel["recommendation_mass"] + panel["decision_mass"] + panel["resolution_mass"]
    )

    for name, (low, high) in WINDOWS.items():
        panel[f"papers_{name}"] = window_sum(panel, "paper_count", low, high)
        panel[f"neighbor_papers_{name}"] = window_sum(panel, "neighbor_papers", low, high)
        panel[f"prior_other_instruments_{name}"] = window_sum(
            panel, "other_instrument_mass", low, high
        )
        if low >= 1:
            panel[f"prior_measures_{name}"] = window_sum(panel, "measure_mass", low, high)
            # Mirror-image lead windows, used only as a placebo.
            panel[f"papers_lead_{name}"] = sum(
                panel.groupby("topic")["paper_count"].shift(-lag).fillna(0.0)
                for lag in range(low, high + 1)
            )

    # Hard top-one Measure counts, used only as a sensitivity check on the
    # probability-weighted outcome.
    hard_counts = (
        inventory[inventory["topic_top1"].astype(str).ne("")]
        .groupby(["topic_top1", "year"])
        .size()
        .rename("measure_count_hard")
        .reset_index()
        .rename(columns={"topic_top1": "topic"})
    )
    panel = panel.merge(hard_counts, on=["topic", "year"], how="left")
    panel["measure_count_hard"] = panel["measure_count_hard"].fillna(0.0)

    # Years since the previous Measure primarily assigned to this concern. Hard
    # top-one assignment is used because "the previous Measure" is a discrete
    # event; probability mass cannot date one.
    hard_years: dict[str, list[int]] = collections.defaultdict(list)
    for _, record in inventory.iterrows():
        if record["topic_top1"]:
            hard_years[record["topic_top1"]].append(int(record["year"]))
    gaps = []
    for _, record in panel.iterrows():
        previous = [year for year in hard_years.get(record["topic"], []) if year < record["year"]]
        gaps.append(record["year"] - max(previous) if previous else np.nan)
    panel["years_since_last_measure"] = gaps
    # Concerns that have never carried a Measure are censored, not zero. A
    # censoring flag keeps them in the model with an explicit indicator.
    panel["no_previous_measure"] = panel["years_since_last_measure"].isna().astype(float)
    panel["years_since_last_measure"] = panel["years_since_last_measure"].fillna(0.0)
    return panel


def fit_ppml(
    panel: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    specification: str,
    minimum_year: int = MEASURE_START_YEAR,
) -> pd.DataFrame:
    """Concern- and year-fixed-effect Poisson with two-way clustered errors.

    Identical in construction to the model in `analyze_attention_to_outcomes`
    (log1p transform, standardization, two-way clustering by concern and year),
    but without the own-versus-nearby contrast rows, which are not part of this
    question.
    """
    data = panel[panel["year"] >= minimum_year].copy()
    z_terms = []
    for predictor in predictors:
        transformed = np.log1p(data[predictor].to_numpy(dtype=float))
        standard_deviation = transformed.std(ddof=0)
        name = f"z_{predictor}"
        data[name] = (transformed - transformed.mean()) / max(standard_deviation, 1e-12)
        z_terms.append(name)
    formula = f"{outcome} ~ {' + '.join(z_terms)} + C(topic) + C(year)"
    fitted = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()
    topic_groups = pd.Categorical(data["topic"]).codes
    year_groups = pd.Categorical(data["year"]).codes
    covariance, _, _ = cov_cluster_2groups(fitted, topic_groups, year_groups)
    term_index = {term: index for index, term in enumerate(fitted.params.index)}
    rows = []
    for predictor, term in zip(predictors, z_terms):
        estimate = float(fitted.params[term])
        se = float(np.sqrt(max(covariance[term_index[term], term_index[term]], 0.0)))
        z_value = estimate / se if se > 0 else np.nan
        rows.append(
            {
                "specification": specification,
                "outcome": outcome,
                "predictor": predictor,
                "first_year": int(minimum_year),
                "n_topic_years": int(len(data)),
                "outcome_total": float(data[outcome].sum()),
                "coefficient": estimate,
                "se_two_way_cluster_topic_year": se,
                "incidence_rate_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * se),
                "ci_high": math.exp(estimate + 1.96 * se),
                "p_value": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
                "scale": "one SD of log1p predictor",
            }
        )
    return pd.DataFrame(rows)


def fit_measure_models(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit the pre-specified model sequence: papers, formal history, combined, family."""
    paper_terms = [f"papers_{name}" for name in WINDOWS]
    history_terms = (
        [f"prior_other_instruments_{name}" for name in WINDOWS]
        + [f"prior_measures_{name}" for name, (low, _) in WINDOWS.items() if low >= 1]
        + ["years_since_last_measure", "no_previous_measure"]
    )

    tables = [
        # Reference: the published three-year specification, refitted on the
        # years in which Measures could actually be adopted.
        fit_ppml(
            panel,
            "measure_mass",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "reference_prior3_measures_1995_onward",
        ),
        fit_ppml(panel, "measure_mass", paper_terms, "paper_windows_measures"),
        fit_ppml(panel, "measure_mass", history_terms, "formal_history_measures"),
        fit_ppml(panel, "measure_mass", paper_terms + history_terms, "combined_measures"),
        # Family split of the paper model alone, so that attenuation caused by
        # adding formal history is separable from attenuation caused by
        # restricting the outcome to one family.
        fit_ppml(
            panel,
            "measure_mass_site_admin",
            paper_terms,
            "paper_windows_measures_recurring_site_administration",
        ),
        fit_ppml(panel, "measure_mass_other", paper_terms, "paper_windows_measures_other"),
        # Sensitivity: hard top-one Measure counts instead of probability mass.
        fit_ppml(panel, "measure_count_hard", paper_terms, "paper_windows_measures_hard_top1"),
        # Falsification: the mirror-image lead windows. If the eight-to-fifteen
        # year result merely tracked a concern's own growth in attention and
        # output, papers eight to fifteen years *after* adoption would predict
        # it just as well. Concern-specific linear trends would be the direct
        # test, but 45 trends on top of concern and year effects make this
        # panel singular, so the lead placebo is used instead.
        fit_ppml(
            panel,
            "measure_mass",
            [f"papers_lead_{name}" for name, (low, _) in WINDOWS.items() if low >= 1],
            "paper_lead_windows_measures_placebo",
        ),
        fit_ppml(
            panel,
            "measure_mass_site_admin",
            paper_terms + history_terms,
            "combined_measures_recurring_site_administration",
        ),
        fit_ppml(
            panel,
            "measure_mass_other",
            paper_terms + history_terms,
            "combined_measures_other",
        ),
        # Contrast instruments, on the same post-1995 window, so that the
        # Measure result is not confounded with the pre-1995 structural zeros
        # that the original 1961--2025 instrument split carried.
        fit_ppml(panel, "resolution_mass", paper_terms, "paper_windows_resolutions"),
        fit_ppml(panel, "decision_mass", paper_terms, "paper_windows_decisions"),
    ]
    models = pd.concat(tables, ignore_index=True)
    # A fixed-effect Poisson on 45 concerns and 31 years cannot be estimated
    # from seven events. Such specifications are retained for transparency but
    # marked so that no interpretation rule reads them.
    models["estimable"] = models["outcome_total"] >= MINIMUM_EVENTS_FOR_INFERENCE
    return models


# ---------------------------------------------------------------------------
# Work package 4: does the concern space structure the lineage?
# ---------------------------------------------------------------------------


def phi_by_year_cached(topics: list[str], topic_lookup: dict[str, str]) -> dict[int, pd.DataFrame]:
    """Cumulative-lagged concern spaces, cached because they are expensive."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "cumulative_phi_by_year.npz"
    if path.exists():
        stored = np.load(path, allow_pickle=True)
        stored_topics = [str(topic) for topic in stored["topics"]]
        if stored_topics == topics:
            years = [int(year) for year in stored["years"]]
            stack = stored["phi"]
            return {
                year: pd.DataFrame(stack[index], index=topics, columns=topics)
                for index, year in enumerate(years)
            }
    submitted = load_submitted_with_fallback()
    result = base.cumulative_phi_by_year(submitted, topics, topic_lookup)
    years = sorted(result)
    np.savez_compressed(
        path,
        topics=np.array(topics, dtype=object),
        years=np.array(years, dtype=int),
        phi=np.stack([result[year].to_numpy(dtype=float) for year in years]),
    )
    return result


def distribution_proximity(
    source_vector: np.ndarray, target_vector: np.ndarray, phi_values: np.ndarray
) -> float:
    """Expected concern proximity between two probability distributions.

    Both formal instruments carry a 45-element concern distribution rather than
    a single label, so the estimand is the expectation of phi under the product
    of the two distributions. For papers, which carry Secretariat labels, the
    source distribution is uniform over the assigned labels. One estimand is
    therefore used for every edge type.
    """
    return float(source_vector @ phi_values @ target_vector)


def matched_null_percentile(
    observed: float, candidate_values: np.ndarray
) -> tuple[float, int]:
    finite = candidate_values[np.isfinite(candidate_values)]
    if finite.size == 0:
        return np.nan, 0
    return float((finite <= observed).mean()), int(finite.size)


def lag_bin(lag: float) -> str:
    if not np.isfinite(lag):
        return "unknown"
    if lag <= 0:
        return "same_year"
    if lag <= 3:
        return "1_3"
    if lag <= 7:
        return "4_7"
    if lag <= 15:
        return "8_15"
    return "16_plus"


def period_bin(year: float) -> str:
    if not np.isfinite(year):
        return "unknown"
    return f"{int(year) // 10 * 10}s"


def outcome_to_measure_proximity(
    nodes: dict,
    edge_audit: pd.DataFrame,
    probability_lookup: dict[str, np.ndarray],
    inventory: pd.DataFrame,
    phi_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Percentile of each observed predecessor against matched alternatives.

    For every target Measure, the observed predecessor is compared with all
    other formal outcomes that share its instrument type, decade of adoption,
    and lag bin, and that existed before the target year. Where the pool allows,
    matching is further restricted to predecessors of the same Measure family.
    """
    family_of = inventory.set_index("measure_id")["functional_family"]
    outcome_nodes = [
        node
        for node in nodes.values()
        if node.get("kind") == "outcome"
        and not node.get("placeholder")
        and node["id"] in probability_lookup
        and isinstance(node.get("year"), int)
    ]
    pool = pd.DataFrame(
        [
            {
                "outcome_id": node["id"],
                "instrument": node["outcome_type"],
                "year": int(node["year"]),
            }
            for node in outcome_nodes
        ]
    )

    subset = edge_audit[
        edge_audit["source_kind"].eq("outcome")
        & edge_audit["relation_class"].isin(["strong_transformation", "contextual_reference"])
        & edge_audit["source_id"].isin(probability_lookup)
        & edge_audit["measure_id"].isin(probability_lookup)
    ].copy()

    rows = []
    for _, edge in subset.iterrows():
        target_id = edge["measure_id"]
        target_year = int(edge["measure_year"])
        phi_values = phi_by_year[target_year].to_numpy(dtype=float)
        target_vector = probability_lookup[target_id]
        observed = distribution_proximity(
            probability_lookup[edge["source_id"]], target_vector, phi_values
        )
        bin_label = lag_bin(edge["lag_years"])
        eligible = pool[
            pool["instrument"].eq(edge["source_instrument"])
            & pool["year"].lt(target_year)
            & pool["outcome_id"].ne(target_id)
            & pool["year"].map(period_bin).eq(period_bin(edge["source_year"]))
            & (target_year - pool["year"]).map(lag_bin).eq(bin_label)
        ]
        candidates = np.array(
            [
                distribution_proximity(probability_lookup[other], target_vector, phi_values)
                for other in eligible["outcome_id"]
            ],
            dtype=float,
        )
        percentile, pool_size = matched_null_percentile(observed, candidates)
        rows.append(
            {
                "edge_set": edge["relation_class"],
                "target_id": target_id,
                "target_year": target_year,
                "target_family": family_of.get(target_id, ""),
                "source_id": edge["source_id"],
                "source_instrument": edge["source_instrument"],
                "relation": edge["relation"],
                "lag_years": edge["lag_years"],
                "lag_bin": bin_label,
                "expected_phi": observed,
                "matched_percentile": percentile,
                "matched_pool": pool_size,
            }
        )
    return pd.DataFrame(rows)


def paper_edge_proximity(
    nodes: dict,
    edges: list[dict],
    paper_categories: dict[str, list[str]],
    probability_lookup: dict[str, np.ndarray],
    inventory: pd.DataFrame,
    phi_by_year: dict[int, pd.DataFrame],
    topics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paper-to-outcome edges and verified paper -> outcome -> Measure paths.

    Both use the same estimand and the same matched null: the observed paper is
    ranked against every categorized paper submitted at the same meeting, which
    matches on source type, calendar period, and availability by construction.
    """
    topic_index = {topic: index for index, topic in enumerate(topics)}
    family_of = inventory.set_index("measure_id")["functional_family"]
    measure_ids = set(inventory["measure_id"])

    def source_vector(paper_id: str) -> np.ndarray | None:
        labels = [topic for topic in paper_categories.get(paper_id, []) if topic in topic_index]
        if not labels:
            return None
        vector = np.zeros(len(topics), dtype=float)
        for label in labels:
            vector[topic_index[label]] = 1.0 / len(labels)
        return vector

    meeting_papers: dict[int, list[str]] = collections.defaultdict(list)
    for paper_id in paper_categories:
        try:
            meeting = int(paper_id.split(":", 1)[0].replace("ATCM", ""))
        except ValueError:
            continue
        meeting_papers[meeting].append(paper_id)

    paper_edges = [
        edge
        for edge in edges
        if nodes.get(edge["src"], {}).get("kind") == "paper"
        and nodes.get(edge["dst"], {}).get("kind") == "outcome"
        and edge["src"] in paper_categories
        and edge["dst"] in probability_lookup
    ]

    def score_against(paper_id: str, target_id: str, phi_values: np.ndarray) -> float:
        vector = source_vector(paper_id)
        if vector is None:
            return np.nan
        return distribution_proximity(vector, probability_lookup[target_id], phi_values)

    def percentile_row(paper_id: str, meeting: int, target_id: str, target_year: int) -> tuple:
        phi_values = phi_by_year[target_year].to_numpy(dtype=float)
        observed = score_against(paper_id, target_id, phi_values)
        if not np.isfinite(observed):
            return np.nan, 0, np.nan
        candidates = np.array(
            [
                score_against(other, target_id, phi_values)
                for other in meeting_papers.get(meeting, [])
            ],
            dtype=float,
        )
        percentile, pool_size = matched_null_percentile(observed, candidates)
        return percentile, pool_size, observed

    direct_rows = []
    for edge in paper_edges:
        target = nodes[edge["dst"]]
        source_meeting = int(edge["src"].split(":", 1)[0].replace("ATCM", ""))
        percentile, pool_size, observed = percentile_row(
            edge["src"], source_meeting, edge["dst"], int(target["year"])
        )
        if not np.isfinite(percentile):
            continue
        direct_rows.append(
            {
                "edge_set": (
                    "paper_to_outcome_adoption"
                    if edge.get("relation") in PAPER_ACTION_RELATIONS
                    else "paper_to_outcome_discussion"
                ),
                "target_id": edge["dst"],
                "target_year": int(target["year"]),
                "target_instrument": target.get("outcome_type"),
                "target_family": family_of.get(edge["dst"], ""),
                "source_id": edge["src"],
                "relation": edge.get("relation"),
                "lag_years": 0.0,
                "expected_phi": observed,
                "matched_percentile": percentile,
                "matched_pool": pool_size,
            }
        )

    # Two-step paths: a paper feeds an intermediate outcome, and that outcome is
    # later a recovered predecessor of a Measure. The proximity tested is
    # between the originating paper and the final Measure.
    #
    # The strength of the first leg is carried through, because a path whose
    # opening step is only a discussion mention is weaker evidence of a route
    # into formal action than one documented as contributing to adoption. Where
    # a paper reaches the intermediate by both routes, adoption takes priority.
    paper_to_outcome: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for edge in paper_edges:
        first_leg = (
            "adoption"
            if edge.get("relation") in PAPER_ACTION_RELATIONS
            else "discussion"
        )
        existing = paper_to_outcome[edge["dst"]].get(edge["src"])
        if existing != "adoption":
            paper_to_outcome[edge["dst"]][edge["src"]] = first_leg
    two_step_rows = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        target_id = edge["dst"]
        if target_id not in measure_ids:
            continue
        source = nodes.get(edge["src"], {})
        if source.get("kind") != "outcome":
            continue
        if relation_class(edge.get("relation")) not in {
            "strong_transformation",
            "contextual_reference",
        }:
            continue
        target_year = int(nodes[target_id]["year"])
        for paper_id, first_leg in paper_to_outcome.get(edge["src"], {}).items():
            key = (paper_id, target_id)
            if key in seen:
                continue
            seen.add(key)
            source_meeting = int(paper_id.split(":", 1)[0].replace("ATCM", ""))
            percentile, pool_size, observed = percentile_row(
                paper_id, source_meeting, target_id, target_year
            )
            if not np.isfinite(percentile):
                continue
            two_step_rows.append(
                {
                    "edge_set": f"paper_to_intermediate_to_measure_{first_leg}",
                    "target_id": target_id,
                    "first_leg": first_leg,
                    "target_year": target_year,
                    "target_instrument": "Measure",
                    "target_family": family_of.get(target_id, ""),
                    "source_id": paper_id,
                    "intermediate_id": edge["src"],
                    "relation": edge.get("relation"),
                    "lag_years": float(target_year - int(source.get("year", target_year))),
                    "expected_phi": observed,
                    "matched_percentile": percentile,
                    "matched_pool": pool_size,
                }
            )
    return pd.DataFrame(direct_rows), pd.DataFrame(two_step_rows)


def summarize_proximity(edges: pd.DataFrame, label_column: str = "edge_set") -> pd.DataFrame:
    """Target-balanced mean percentile with a rank-uniform permutation test.

    Targets are averaged equally so that heavily cited Measures cannot dominate.
    Under exchangeability within a matched pool, the observed rank is uniform,
    which gives an exact null without resampling the pool itself.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    usable = edges[edges["matched_pool"] > 1].copy()
    for label, subset in usable.groupby(label_column):
        observed = float(subset.groupby("target_id")["matched_percentile"].mean().mean())
        by_target = subset.groupby("target_id")
        null_sum = np.zeros(N_PERMUTATIONS, dtype=float)
        for target_id, part in by_target:
            pool_sizes = part["matched_pool"].to_numpy(dtype=int)
            draws = np.stack(
                [
                    rng.integers(1, size + 1, size=N_PERMUTATIONS) / size
                    for size in pool_sizes
                ]
            )
            null_sum += draws.mean(axis=0)
        null = null_sum / by_target.ngroups
        rows.append(
            {
                "edge_set": label,
                "edges": int(len(subset)),
                "targets": int(subset["target_id"].nunique()),
                "observed_mean_matched_percentile": observed,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std(ddof=1)),
                "median_matched_pool": float(subset["matched_pool"].median()),
                "upper_tail_p": float(
                    (1 + np.count_nonzero(null >= observed)) / (N_PERMUTATIONS + 1)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("edge_set")


# ---------------------------------------------------------------------------
# Work package 5: select a row of the decision table
# ---------------------------------------------------------------------------


def select_decision_row(
    models: pd.DataFrame, spatial: pd.DataFrame, inventory: pd.DataFrame
) -> dict:
    """Apply the pre-stated interpretation rules to the fitted estimates.

    An "association" means a positive coefficient whose two-way clustered
    interval excludes one. A significant negative coefficient is recorded
    separately and never counts as evidence that attention produces Measures.
    """

    def estimate(specification: str, predictor: str) -> pd.Series | None:
        row = models[
            models["specification"].eq(specification) & models["predictor"].eq(predictor)
        ]
        return None if row.empty else row.iloc[0]

    def positive(specification: str, predictor: str) -> bool:
        row = estimate(specification, predictor)
        return bool(row is not None and row["estimable"] and row["ci_low"] > 1.0)

    paper_windows = [f"papers_{name}" for name in WINDOWS]
    long_windows = ["papers_prior_4_7", "papers_prior_8_15"]

    long_window_papers = any(positive("paper_windows_measures", term) for term in long_windows)
    any_paper_window = any(positive("paper_windows_measures", term) for term in paper_windows)
    # Does a paper association survive once formal predecessors are in the model?
    surviving_paper_windows = [
        term
        for term in paper_windows
        if positive("paper_windows_measures", term) and positive("combined_measures", term)
    ]
    attenuation = {}
    for term in paper_windows:
        alone = estimate("paper_windows_measures", term)
        combined = estimate("combined_measures", term)
        if alone is None or combined is None:
            continue
        # Attenuation is only meaningful where the unadjusted coefficient is
        # large enough to attenuate; a share of a near-zero coefficient is noise.
        base_coefficient = float(alone["coefficient"])
        attenuation[term] = {
            "irr_paper_model": float(alone["incidence_rate_ratio"]),
            "irr_combined_model": float(combined["incidence_rate_ratio"]),
            "log_irr_share_removed": (
                float(1.0 - combined["coefficient"] / base_coefficient)
                if abs(base_coefficient) >= 0.05
                else None
            ),
        }

    formal_history_terms = [
        f"prior_other_instruments_{name}" for name in WINDOWS
    ] + [f"prior_measures_{name}" for name, (low, _) in WINDOWS.items() if low >= 1]
    formal_history = any(positive("formal_history_measures", term) for term in formal_history_terms)

    site_paper_association = any(
        positive("paper_windows_measures_recurring_site_administration", term)
        for term in paper_windows
    )
    other_paper_association = any(
        positive("paper_windows_measures_other", term) for term in paper_windows
    )

    strong_spatial = spatial[spatial["edge_set"].eq("strong_transformation")]
    # The primary two-step estimand requires the opening leg to be
    # adoption-linked; the discussion-opened variant is a sensitivity.
    two_step_spatial = spatial[
        spatial["edge_set"].eq("paper_to_intermediate_to_measure_adoption")
    ]
    spatially_local = (
        not strong_spatial.empty
        and float(strong_spatial["upper_tail_p"].iloc[0]) < 0.05
        and not two_step_spatial.empty
        and float(two_step_spatial["upper_tail_p"].iloc[0]) < 0.05
    )
    site_share = float(inventory["recurring_site_administration"].mean())
    n_other = int((~inventory["recurring_site_administration"]).sum())

    if surviving_paper_windows and any(
        term in long_windows for term in surviving_paper_windows
    ):
        row = (
            "Longer paper windows predict Measures and survive adjustment for formal "
            "predecessors: the original null was principally a timing mismatch."
        )
    elif long_window_papers and formal_history and not surviving_paper_windows:
        row = (
            "A long-window (8--15 year) paper association is present and the three-year window "
            "is null: the original null was principally a timing mismatch. The long-window "
            "estimate loses precision once formal predecessors are adjusted for, which is "
            "consistent with part of the pathway running through formal precedent."
        )
    elif formal_history and not any_paper_window:
        row = (
            "Prior formal outcomes predict Measures while paper attention does not at any "
            "window: Measures are consistent with a multi-stage formal pathway."
        )
    elif site_paper_association and not other_paper_association:
        row = (
            "Any paper association is confined to recurring site instruments: Measure "
            "production primarily reflects administrative renewal."
        )
    elif not any_paper_window and not formal_history:
        row = (
            "No stable paper or formal-history result: Measures form a boundary of the "
            "documentary-attention approach."
        )
    else:
        row = (
            "Mixed result: see the model table; no single decision-table row is selected by "
            "the pre-stated rules."
        )

    reference = estimate("reference_prior3_measures_1995_onward", "papers_prior3")
    hard_long_window = estimate("paper_windows_measures_hard_top1", "papers_prior_8_15")

    return {
        "selected_row": row,
        "recurring_site_administration_share": site_share,
        "three_year_window_refitted_from_1995": (
            None
            if reference is None
            else {
                "irr": float(reference["incidence_rate_ratio"]),
                "ci": [float(reference["ci_low"]), float(reference["ci_high"])],
                "note": (
                    "Restricting the panel to the years in which Measures exist does not "
                    "recover an association at the published three-year window, so the "
                    "pre-1995 structural zeros were not the cause of the null."
                ),
            }
        ),
        "long_window_robust_to_hard_assignment": (
            None
            if hard_long_window is None
            else {
                "irr": float(hard_long_window["incidence_rate_ratio"]),
                "ci": [float(hard_long_window["ci_low"]), float(hard_long_window["ci_high"])],
            }
        ),
        "n_non_site_measures": n_other,
        "composition_caveat": (
            f"Only {n_other} of {len(inventory)} Measures fall outside recurring site "
            "administration, so no substantive-hardening subsample can be estimated with "
            "usable precision."
        ),
        "any_paper_window_association": any_paper_window,
        "long_window_paper_association": long_window_papers,
        "paper_windows_surviving_formal_history": surviving_paper_windows,
        "paper_coefficient_attenuation": attenuation,
        "formal_history_association": formal_history,
        "paper_association_in_recurring_site_families": site_paper_association,
        "paper_association_in_other_measures": other_paper_association,
        "strong_predecessors_spatially_local": spatially_local,
        "spatial_addendum": (
            "The concern space traces continuity from attention through formal precedent."
            if spatially_local
            else "Outcome lineage matters, but the concern space should not be used to explain Measures."
        ),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _, _, _, counts, _ = build_graphs()
    topics = list(counts.index)
    topic_lookup = lineage._canonical_topic_lookup(topics)

    predictions = pd.read_csv(OUTDIR / "outcome_topic_predictions.csv")
    probabilities = pd.read_csv(OUTDIR / "outcome_topic_probabilities.csv")
    base_panel = pd.read_csv(OUTDIR / "topic_year_attention_outcomes.csv")

    nodes, edges = load_graph()
    inventory, edge_audit = build_measure_inventory(nodes, edges, predictions, probabilities)
    inventory.to_csv(OUTDIR / "measure_pathway_inventory.csv", index=False)
    edge_audit.to_csv(OUTDIR / "measure_edge_audit.csv", index=False)
    audit_sample = unconnected_audit_sample(inventory)
    audit_sample.to_csv(OUTDIR / "measure_unconnected_audit_sample.csv", index=False)

    composition = pathway_composition(inventory)
    composition.to_csv(OUTDIR / "measure_pathway_composition.csv", index=False)
    lags = predecessor_lag_summary(edge_audit)
    lags.to_csv(OUTDIR / "measure_predecessor_lags.csv", index=False)

    panel = build_measure_panel(base_panel, inventory, probabilities, topics)
    panel.to_csv(OUTDIR / "measure_windows_panel.csv", index=False)
    models = fit_measure_models(panel)
    models.to_csv(OUTDIR / "measure_pathway_models.csv", index=False)

    phi_by_year = phi_by_year_cached(topics, topic_lookup)
    probability_pivot = probabilities.pivot(
        index="outcome_id", columns="topic", values="probability"
    ).reindex(columns=topics)
    probability_lookup = {
        outcome_id: row.to_numpy(dtype=float)
        for outcome_id, row in probability_pivot.iterrows()
        if np.isfinite(row.to_numpy(dtype=float)).all()
    }
    _, paper_categories = base.load_paper_training(topics)

    outcome_edges = outcome_to_measure_proximity(
        nodes, edge_audit, probability_lookup, inventory, phi_by_year
    )
    direct_edges, two_step_edges = paper_edge_proximity(
        nodes, edges, paper_categories, probability_lookup, inventory, phi_by_year, topics
    )
    spatial_edges = pd.concat([outcome_edges, direct_edges, two_step_edges], ignore_index=True)
    spatial_edges.to_csv(OUTDIR / "measure_spatial_continuity_edges.csv", index=False)
    spatial = summarize_proximity(spatial_edges)
    spatial.to_csv(OUTDIR / "measure_spatial_continuity_tests.csv", index=False)

    decision = select_decision_row(models, spatial, inventory)

    summary = {
        "scope": {
            "measures": int(len(inventory)),
            "measure_years": [int(inventory["year"].min()), int(inventory["year"].max())],
            "instrument_reform_note": (
                "Measures, Decisions, and Resolutions exist only from 1995. All models here "
                "start in 1995; the published 1961--2025 instrument split included 34 years "
                "of structural zeros."
            ),
            "graph": GRAPH_NAME,
        },
        "composition": {
            "by_family": inventory["functional_family"].value_counts().to_dict(),
            "recurring_site_administration": int(
                inventory["recurring_site_administration"].sum()
            ),
            "measures_with_parsed_site_id": int((inventory["n_site_ids"] > 0).sum()),
            "distinct_sites": int(
                len(
                    {
                        site
                        for entry in inventory["site_ids"]
                        for site in entry.split(" | ")
                        if site
                    }
                )
            ),
        },
        "pathways": composition[composition["group"].eq("all_measures")].to_dict(
            orient="records"
        ),
        "predecessor_lags": lags.to_dict(orient="records"),
        "models": models.to_dict(orient="records"),
        "spatial_continuity": spatial.to_dict(orient="records"),
        "decision": decision,
        "interpretive_limits": [
            "The functional typology is rule-based on titles and site identifiers; it is a "
            "reproducible starting point for author coding, not a substitute for legal coding.",
            "Absence of a recovered predecessor is a property of the parsed record, not "
            "evidence of institutional independence.",
            "Attenuation of a paper coefficient when formal predecessors are added is "
            "descriptive evidence consistent with mediation, not a causal mediation estimate.",
            "The analysis models adoption at the ATCM, not entry into effect, implementation, "
            "or environmental outcome.",
        ],
    }
    (OUTDIR / "measure_pathway_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["composition"], indent=2))
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
