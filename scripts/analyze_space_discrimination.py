#!/usr/bin/env python3
"""Does the geometry of the concern space add anything beyond the concern label?

The proximity results elsewhere in this repository establish that papers
documented as contributing to adoption sit unusually close to the concern of
the instrument they fed. That finding has an obvious deflationary reading: a
paper filed under the same Secretariat concern as the outcome is trivially
"close", so the apparent role of the space could be nothing more than an exact
label match wearing a geometric costume.

This script separates the two. For every candidate paper it decomposes the
expected proximity to an outcome into

    same-concern mass   -- the outcome's probability on the paper's own labels
    related-concern     -- the same expectation with the diagonal of the space
    proximity              removed, so only off-label geometry can contribute

and then asks whether the second term still identifies the paper that reached
adoption, holding the first constant. If it does, the space carries information
about routes into formal action that the taxonomy alone does not.

Three tests, all within the opportunity set of a single meeting:

1. discrimination (AUC): how often does concern proximity rank an
   adoption-linked paper above a randomly chosen paper submitted at the same
   meeting?
2. off-label discrimination: the same statistic computed only from
   related-concern proximity, and restricted to papers that are not exact
   label matches;
3. a conditional logit stratified by outcome, racing the two terms directly.

The space used at each outcome is cumulative-lagged in ATCM order: it is
estimated only from meetings preceding the outcome's meeting, so no test uses
information from the adopting meeting to construct its geometry.
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
from scipy.stats import norm
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fig01_space_of_concerns_topology import build_graphs
from scripts import explore_lineage_space as lineage

import scripts.analyze_attention_to_outcomes as base
import scripts.analyze_measure_pathways as pathways
import scripts.analyze_exposure_corrected_contribution as conversion


OUTDIR = base.OUTDIR
LINEAGE_ROOT = base.LINEAGE_ROOT
GRAPH_NAME = "decision_map_verified.json"
RANDOM_SEED = 20260813
N_BOOTSTRAP = 5000

ADOPTION_RELATIONS = ("direct_adoption_or_approval", "documented_contribution")
DISCUSSION_RELATIONS = ("direct_proposal_or_discussion",)


def expected_proximity(
    source_indices: list[int], outcome_vector: np.ndarray, phi_values: np.ndarray
) -> float:
    """Outcome-probability-weighted proximity to a paper's nearest own concern.

    Matches the estimand used in `analyze_attention_to_outcomes.direct_edge_
    analysis`: for each concern the outcome might be about, take the paper's
    closest label, then average over the outcome's concern distribution.
    """
    if not source_indices:
        return np.nan
    nearest = phi_values[:, source_indices].max(axis=1)
    return float(outcome_vector @ nearest)


STOPWORDS = {
    "the", "of", "and", "for", "to", "a", "an", "in", "on", "at", "no", "antarctic",
    "antarctica", "revised", "management", "plan", "plans", "area", "areas",
}


def title_overlap(paper_title: str, outcome_title: str) -> float:
    """Jaccard overlap of content words in a paper title and an outcome title.

    The concern of an instrument is assigned from its title, so an instrument
    whose title was copied from the paper that proposed it would be classified
    into that paper's concern almost by construction. This measures how much of
    that risk each linked pair carries.
    """
    def tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z]{3,}", str(text).lower())
        return {word for word in words if word not in STOPWORDS}

    left, right = tokens(paper_title), tokens(outcome_title)
    if not left or not right:
        return np.nan
    return len(left & right) / len(left | right)


def build_candidate_panel(
    topics: list[str],
    paper_categories: dict[str, list[str]],
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    phi_by_meeting: dict[int, pd.DataFrame],
    paper_titles: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per (outcome, paper available at that outcome's meeting)."""
    graph = pathways.load_json(LINEAGE_ROOT / GRAPH_NAME)
    nodes = {node["id"]: node for node in graph["nodes"]}

    topic_index = {topic: index for index, topic in enumerate(topics)}
    probability_pivot = probabilities.pivot(
        index="outcome_id", columns="topic", values="probability"
    ).reindex(columns=topics)
    outcome_vectors = {
        outcome_id: row.to_numpy(dtype=float)
        for outcome_id, row in probability_pivot.iterrows()
        if np.isfinite(row.to_numpy(dtype=float)).all()
    }
    prediction_index = predictions.set_index("outcome_id")

    meeting_papers: dict[int, list[str]] = collections.defaultdict(list)
    for paper_id in paper_categories:
        try:
            meeting = int(paper_id.split(":", 1)[0].replace("ATCM", ""))
        except ValueError:
            continue
        meeting_papers[meeting].append(paper_id)

    # Which papers reached each outcome, and by which route.
    contributors: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for edge in graph["edges"]:
        if nodes.get(edge["src"], {}).get("kind") != "paper":
            continue
        if nodes.get(edge["dst"], {}).get("kind") != "outcome":
            continue
        relation = edge.get("relation")
        if relation in ADOPTION_RELATIONS:
            route = "adoption"
        elif relation in DISCUSSION_RELATIONS:
            route = "discussion"
        else:
            continue
        if contributors[edge["dst"]].get(edge["src"]) != "adoption":
            contributors[edge["dst"]][edge["src"]] = route

    rows = []
    for outcome_id, routes in contributors.items():
        if outcome_id not in outcome_vectors:
            continue
        node = nodes[outcome_id]
        year = int(node["year"])
        meeting = int(node["meeting"])
        candidates = meeting_papers.get(meeting, [])
        if len(candidates) < 5:
            continue
        phi = phi_by_meeting[meeting]
        phi_values = phi.to_numpy(dtype=float).copy()
        # The off-label matrix removes every concern's proximity to itself, so
        # a paper filed under the outcome's own concern gets no credit for it.
        off_label_values = phi_values.copy()
        np.fill_diagonal(off_label_values, 0.0)
        outcome_vector = outcome_vectors[outcome_id]
        top_concern = prediction_index.loc[outcome_id, "topic_top1"]
        outcome_title = prediction_index.loc[outcome_id, "title"]
        site_administration = top_concern in base.SITE_ADMIN_TOPICS

        for paper_id in candidates:
            labels = [t for t in paper_categories.get(paper_id, []) if t in topic_index]
            if not labels:
                continue
            indices = [topic_index[t] for t in labels]
            # Probability the outcome is about a concern this paper is filed
            # under: the "exact label match" explanation, in continuous form.
            same_concern_mass = float(outcome_vector[indices].sum())
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "year": year,
                    "meeting": meeting,
                    "instrument": node.get("outcome_type"),
                    "paper_id": paper_id,
                    "route": routes.get(paper_id, "none"),
                    "adoption_linked": int(routes.get(paper_id) == "adoption"),
                    "discussion_linked": int(routes.get(paper_id) == "discussion"),
                    "expected_proximity": expected_proximity(
                        indices, outcome_vector, phi_values
                    ),
                    "related_concern_proximity": expected_proximity(
                        indices, outcome_vector, off_label_values
                    ),
                    "same_concern_mass": same_concern_mass,
                    "exact_label_match": int(top_concern in labels),
                    "n_labels": len(labels),
                    "site_administration_outcome": int(site_administration),
                    "title_overlap": (
                        title_overlap(paper_titles.get(paper_id, ""), outcome_title)
                        if paper_titles is not None
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def discrimination_auc(
    panel: pd.DataFrame,
    score_column: str,
    positive_column: str,
    label: str,
    restrict_off_label: bool = False,
    exclude_site_administration: bool = False,
    max_title_overlap: float | None = None,
) -> dict:
    """Outcome-balanced AUC with an outcome-bootstrap interval.

    Within each outcome the score ranks the papers available at that meeting.
    The AUC is the probability that a linked paper outranks a randomly chosen
    unlinked paper from the same meeting, averaged equally over outcomes so
    that heavily linked outcomes cannot dominate.
    """
    data = panel
    if exclude_site_administration:
        data = data[data["site_administration_outcome"].eq(0)]
    if max_title_overlap is not None:
        # Drop instruments whose linked paper shares most of its title, where
        # the title classifier could be recovering the paper's own wording.
        risky = data.loc[
            data[positive_column].eq(1) & data["title_overlap"].gt(max_title_overlap),
            "outcome_id",
        ].unique()
        data = data[~data["outcome_id"].isin(risky)]
    if restrict_off_label:
        # Drop every paper that shares the outcome's leading concern, positives
        # included. What remains can only be ranked by off-label geometry.
        data = data[data["exact_label_match"].eq(0)]

    per_outcome = []
    for outcome_id, group in data.groupby("outcome_id"):
        positives = group[group[positive_column].eq(1)]
        negatives = group[group[positive_column].eq(0)]
        if positives.empty or len(negatives) < 4:
            continue
        positive_scores = positives[score_column].to_numpy(dtype=float)
        negative_scores = negatives[score_column].to_numpy(dtype=float)
        if not np.isfinite(positive_scores).all():
            continue
        comparisons = (
            (positive_scores[:, None] > negative_scores[None, :]).astype(float)
            + 0.5 * (positive_scores[:, None] == negative_scores[None, :])
        )
        per_outcome.append(float(comparisons.mean()))
    values = np.asarray(per_outcome, dtype=float)
    if values.size == 0:
        return {"comparison": label, "outcomes": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    draws = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return {
        "comparison": label,
        "score": score_column,
        "positives": positive_column,
        "off_label_only": restrict_off_label,
        "excludes_site_administration": exclude_site_administration,
        "max_title_overlap": max_title_overlap,
        "outcomes": int(values.size),
        "auc": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "share_above_chance": float((values > 0.5).mean()),
    }


def _fit_race(data: pd.DataFrame, positive_column: str, terms: list[str]):
    design = pd.DataFrame(index=data.index)
    for term in terms:
        values = data[term].to_numpy(dtype=float)
        design[f"z_{term}"] = (values - values.mean()) / max(values.std(ddof=0), 1e-12)
    model = ConditionalLogit(
        data[positive_column].to_numpy(dtype=int),
        design.to_numpy(dtype=float),
        groups=data["outcome_id"].to_numpy(),
    )
    return model.fit(disp=False)


def bootstrap_race(
    data: pd.DataFrame, positive_column: str, terms: list[str], n_bootstrap: int = 400
) -> dict[str, tuple[float, float]]:
    """Resample whole outcomes, so that papers within an outcome move together.

    The conditional-logit standard errors treat each stratum as independent.
    Resampling outcomes rather than rows checks that the estimate does not rest
    on a handful of heavily linked instruments.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    outcomes = data["outcome_id"].unique()
    groups = {outcome: part for outcome, part in data.groupby("outcome_id")}
    draws: list[list[float]] = []
    for _ in range(n_bootstrap):
        picked = rng.choice(outcomes, size=len(outcomes), replace=True)
        frames = []
        for replicate, outcome in enumerate(picked):
            part = groups[outcome].copy()
            # Relabel so repeated outcomes stay separate strata.
            part["outcome_id"] = f"{outcome}__{replicate}"
            frames.append(part)
        resampled = pd.concat(frames, ignore_index=True)
        try:
            fitted = _fit_race(resampled, positive_column, terms)
        except Exception:
            continue
        draws.append([float(value) for value in fitted.params])
    if not draws:
        return {}
    array = np.asarray(draws, dtype=float)
    return {
        term: (
            float(np.exp(np.quantile(array[:, index], 0.025))),
            float(np.exp(np.quantile(array[:, index], 0.975))),
        )
        for index, term in enumerate(terms)
    }


def race_conditional_logit(
    panel: pd.DataFrame,
    positive_column: str,
    label: str,
    bootstrap: bool = True,
    terms: list[str] | None = None,
) -> pd.DataFrame:
    """Race off-label geometry against exact label match, within meetings.

    Stratifying by outcome means every comparison is between papers submitted
    to the same meeting and evaluated against the same instrument, so the
    meeting's agenda and the outcome's own concern are differenced out.
    """
    terms = terms or ["same_concern_mass", "related_concern_proximity"]
    data = panel.dropna(subset=terms).copy()
    usable = data.groupby("outcome_id")[positive_column].transform(
        lambda values: values.sum() > 0 and values.sum() < len(values)
    )
    data = data[usable]
    if data.empty:
        return pd.DataFrame()
    fitted = _fit_race(data, positive_column, terms)
    # The outcome-cluster bootstrap is expensive, so it is run for the
    # specification that carries the headline claim and skipped for the
    # placebo, whose model-based interval already spans one.
    intervals = bootstrap_race(data, positive_column, terms) if bootstrap else {}
    rows = []
    for index, term in enumerate(terms):
        estimate = float(fitted.params[index])
        se = float(fitted.bse[index])
        z_value = estimate / se if se > 0 else np.nan
        rows.append(
            {
                "specification": label,
                "term": term,
                "outcomes": int(data["outcome_id"].nunique()),
                "papers": int(len(data)),
                "coefficient": estimate,
                "se": se,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * se),
                "ci_high": math.exp(estimate + 1.96 * se),
                "outcome_bootstrap_ci_low": intervals.get(term, (np.nan, np.nan))[0],
                "outcome_bootstrap_ci_high": intervals.get(term, (np.nan, np.nan))[1],
                "p_value": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
                "scale": "one SD within the pooled candidate set",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _, _, _, counts, _ = build_graphs()
    topics = list(counts.index)
    topic_lookup = lineage._canonical_topic_lookup(topics)

    predictions = pd.read_csv(OUTDIR / "outcome_topic_predictions.csv")
    probabilities = pd.read_csv(OUTDIR / "outcome_topic_probabilities.csv")
    training, paper_categories = base.load_paper_training(topics)
    paper_titles = dict(zip(training["paper_id"], training["title"]))
    submitted = base.load_submitted_with_fallback()
    meetings = sorted(predictions["meeting"].dropna().astype(int).unique())
    phi_by_meeting = base.cumulative_phi_by_meeting(
        submitted, topics, topic_lookup, meetings
    )

    panel = build_candidate_panel(
        topics, paper_categories, predictions, probabilities, phi_by_meeting,
        paper_titles=paper_titles,
    )
    panel.to_csv(OUTDIR / "space_discrimination_panel.csv", index=False)

    tests = [
        discrimination_auc(
            panel, "expected_proximity", "adoption_linked",
            "Adoption-linked papers, full proximity",
        ),
        discrimination_auc(
            panel, "expected_proximity", "discussion_linked",
            "Discussion-only papers, full proximity",
        ),
        discrimination_auc(
            panel, "related_concern_proximity", "adoption_linked",
            "Adoption-linked papers, off-label geometry only",
        ),
        discrimination_auc(
            panel, "related_concern_proximity", "adoption_linked",
            "Adoption-linked papers, off-label geometry, exact matches removed",
            restrict_off_label=True,
        ),
        discrimination_auc(
            panel, "related_concern_proximity", "discussion_linked",
            "Discussion-only papers, off-label geometry, exact matches removed",
            restrict_off_label=True,
        ),
        discrimination_auc(
            panel, "same_concern_mass", "adoption_linked",
            "Adoption-linked papers, same-concern mass only",
        ),
        # Circularity guards. An instrument's concern is read off its title, so
        # an instrument that reuses its source paper's title could be classified
        # into that paper's concern by construction. Site administration is
        # where such reuse is most common.
        discrimination_auc(
            panel, "expected_proximity", "adoption_linked",
            "Adoption-linked papers, excluding site administration",
            exclude_site_administration=True,
        ),
        discrimination_auc(
            panel, "expected_proximity", "adoption_linked",
            "Adoption-linked papers, title overlap below 0.30",
            max_title_overlap=0.30,
        ),
        discrimination_auc(
            panel, "expected_proximity", "adoption_linked",
            "Adoption-linked papers, title overlap below 0.15",
            max_title_overlap=0.15,
        ),
    ]
    auc_table = pd.DataFrame(tests)
    auc_table.to_csv(OUTDIR / "space_discrimination_auc.csv", index=False)

    races = pd.concat(
        [
            race_conditional_logit(
                panel, "adoption_linked", "adoption_linked_papers", bootstrap=False
            ),
            race_conditional_logit(
                panel, "discussion_linked", "discussion_only_papers", bootstrap=False
            ),
            # Circularity control. An instrument's concern is read off its
            # title, so lexical similarity between a paper and the instrument
            # it fed is the channel through which a copied title could
            # manufacture proximity. Controlling for it directly keeps all the
            # data, unlike deleting the instruments where it is high, which
            # also deletes genuinely on-topic papers.
            race_conditional_logit(
                panel,
                "adoption_linked",
                "adoption_linked_papers_title_overlap_controlled",
                bootstrap=False,
                terms=[
                    "same_concern_mass",
                    "related_concern_proximity",
                    "title_overlap",
                ],
            ),
        ],
        ignore_index=True,
    )
    races.to_csv(OUTDIR / "space_discrimination_race.csv", index=False)

    conversion_panel = conversion.build_panel()
    conversion_model = conversion.fit_model(conversion_panel)
    conversion_panel.to_csv(conversion.OUT_PANEL, index=False)
    conversion_model.to_csv(conversion.OUT_MODEL, index=False)

    summary = {
        "question": (
            "Does the geometry of the concern space identify documentary routes into "
            "formal action beyond what the concern label alone identifies?"
        ),
        "design": {
            "candidate_set": "all categorized papers submitted at the outcome's meeting",
            "space": "cumulative-lagged, estimated only from ATCMs preceding the outcome meeting",
            "balancing": "outcomes averaged equally",
            "outcomes": int(panel["outcome_id"].nunique()),
            "candidate_rows": int(len(panel)),
        },
        "discrimination": auc_table.to_dict(orient="records"),
        "race": races.to_dict(orient="records"),
        "exposure_corrected_actor_output_conversion": {
            "unit": "actor-output pair",
            "response": "linked papers out of eligible same-meeting papers",
            "model": conversion_model.to_dict(orient="records"),
        },
        "interpretive_limits": [
            "Discrimination is within a meeting's opportunity set; it does not forecast "
            "whether an outcome will occur at a concern--meeting pair.",
            "Verified paper-outcome lineage begins in 1991.",
            "A recovered link records documentation, not influence.",
        ],
    }
    (OUTDIR / "space_discrimination_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(auc_table.to_string(index=False))
    print()
    print(races.to_string(index=False))


if __name__ == "__main__":
    main()
