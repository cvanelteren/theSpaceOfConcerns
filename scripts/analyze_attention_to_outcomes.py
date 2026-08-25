#!/usr/bin/env python3
"""Relate documentary attention to regular-ATCM outputs.

The analysis keeps three evidentiary layers separate.

1. Outcome titles are projected onto the inferred primary concerns by a text
   classifier trained on paper titles.
2. The full topic--meeting panel provides indirect evidence. Fixed-effect PPML
   models ask whether own-concern and local-neighbour attention precede formal
   output mass in ATCM order, using a concern space built from earlier meetings.

This estimates documentary association and predictive locality, not intent,
causal political influence, implementation, or legal effectiveness.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parents[1]
LINEAGE_ROOT = ROOT.parent / "ats_lineage"
OUTDIR = ROOT / "output" / "outcome_linkage"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.data_loading import load_submitted_with_fallback
from fig01_space_of_concerns_topology import build_graphs, load_topic_meta
from scripts import explore_lineage_space as lineage
from scripts.official_regular_atcm_outputs import load_official_regular_outputs
from utils import (
    compute_product_space,
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
    standardize_index_labels,
)


START_YEAR = 1995
END_YEAR = 2025
START_MEETING = 19
END_MEETING = 47
RANDOM_SEED = 20260812
N_CV_SPLITS = 5
LOCAL_K = 5
HIGH_CONFIDENCE_MARGIN = 0.60
SITE_ADMIN_TOPICS = {
    "Management Plans",
    "Area Protection and Management Plans General",
    "Historic Sites and Monuments",
    "Site Guidelines for Visitors",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def load_paper_training(topics: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    submitted = load_submitted_with_fallback()
    topic_lookup = lineage._canonical_topic_lookup(topics)
    rows: list[dict] = []
    paper_categories: dict[str, list[str]] = {}
    submitted = submitted.drop_duplicates("paper id")
    for _, record in submitted.iterrows():
        raw_title = record.get("paper name")
        title = "" if pd.isna(raw_title) else str(raw_title).strip()
        meeting = pd.to_numeric(record.get("meeting number"), errors="coerce")
        year = pd.to_numeric(record.get("meeting year"), errors="coerce")
        categories = []
        raw_categories = record.get("category")
        category_text = "" if pd.isna(raw_categories) else str(raw_categories)
        for raw_category in category_text.split("\t"):
            category = topic_lookup.get(lineage._normalize(raw_category))
            if category and category not in categories:
                categories.append(category)
        if not title or pd.isna(meeting) or not categories:
            continue
        paper_id = f"paper:{int(record['paper id'])}"
        paper_categories[paper_id] = categories
        rows.append(
            {
                "paper_id": paper_id,
                "meeting": int(meeting),
                "year": int(year) if not pd.isna(year) else np.nan,
                "title": title,
                "topics": categories,
            }
        )
    return pd.DataFrame(rows), paper_categories


def load_outcomes() -> pd.DataFrame:
    graph = load_json(LINEAGE_ROOT / "decision_map.json")
    rows = []
    for node in graph["nodes"]:
        if node.get("kind") != "outcome" or node.get("placeholder"):
            continue
        year = node.get("year")
        meeting = node.get("meeting")
        if not isinstance(year, int) or not START_YEAR <= year <= END_YEAR:
            continue
        if not isinstance(meeting, (int, float)) or not float(meeting).is_integer():
            continue
        meeting = int(meeting)
        if not START_MEETING <= meeting <= END_MEETING:
            continue
        if node.get("id") == "Resolution 6 (1996)":
            continue
        rows.append(
            {
                "outcome_id": node["id"],
                "meeting": meeting,
                "year": year,
                "instrument": node.get("outcome_type"),
                "title": node.get("title"),
                "title_source": node.get("title_source"),
            }
        )
    outcomes = pd.DataFrame(rows).sort_values(["year", "instrument", "outcome_id"])
    official_titles = load_official_regular_outputs().set_index("output_id")
    if set(outcomes["outcome_id"]) != set(official_titles.index):
        raise ValueError("Graph output IDs do not match the authoritative ATS export")
    outcomes["title"] = outcomes["outcome_id"].map(official_titles["title"])
    outcomes["official_categories"] = outcomes["outcome_id"].map(
        official_titles["official_categories"]
    )
    outcomes["official_concerns"] = outcomes["outcome_id"].map(
        official_titles["official_concerns"]
    )
    outcomes["title_source"] = "official_ats_inventory_subject"
    official = load_json(LINEAGE_ROOT / "official_outcome_counts.json")
    expected = official["sources"]["atcm_19_47"]
    actual_by_meeting = outcomes.groupby("meeting").size().to_dict()
    expected_by_meeting = {
        int(meeting): int(count)
        for meeting, count in official["counts_by_atcm"].items()
        if START_MEETING <= int(meeting) <= END_MEETING
    }
    actual_by_type = outcomes["instrument"].value_counts().to_dict()
    if len(outcomes) != int(expected["total"]):
        raise ValueError(f"Output count mismatch: {len(outcomes)} != {expected['total']}")
    if actual_by_meeting != expected_by_meeting:
        raise ValueError("Per-meeting output counts do not match the official audit")
    if actual_by_type != expected["type_counts"]:
        raise ValueError("Output-type counts do not match the official audit")
    if outcomes["title"].isna().any() or outcomes["title"].astype(str).str.strip().eq("").any():
        raise ValueError("Every regular-ATCM output must have a usable title")
    return outcomes


def official_output_allocations(
    outcomes: pd.DataFrame,
    topics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Allocate outputs from their official ATS instrument categories.

    The instrument register uses 15 broad categories that map directly onto 15
    members of the 45-concern paper vocabulary.  Sixty outputs have more than
    one official category; as with multi-category papers, each receives weight
    ``1/k`` across its returned categories.
    """
    topic_set = set(topics)
    prediction_rows: list[dict] = []
    probability_rows: list[dict] = []
    for record in outcomes.to_dict(orient="records"):
        concerns = list(dict.fromkeys(record["official_concerns"]))
        if not concerns or not set(concerns).issubset(topic_set):
            raise AssertionError(
                f"Official concerns are missing from the paper vocabulary: {concerns}"
            )
        weight = 1.0 / len(concerns)
        primary = dict(record)
        primary.update(
            {
                "topic_top1": concerns[0],
                "topic_top2": concerns[1] if len(concerns) > 1 else "",
                "topic_top3": concerns[2] if len(concerns) > 2 else "",
                "probability_top1": weight,
                "margin_top1_top2": np.nan,
                "high_confidence": True,
                "crossfit_fold": np.nan,
                "allocation_source": "official_ats_instrument_category",
            }
        )
        prediction_rows.append(primary)
        for topic in topics:
            probability_rows.append(
                {
                    "outcome_id": record["outcome_id"],
                    "year": int(record["year"]),
                    "meeting": int(record["meeting"]),
                    "instrument": record["instrument"],
                    "topic": topic,
                    "probability": weight if topic in concerns else 0.0,
                    "allocation_source": "official_ats_instrument_category",
                }
            )
    probabilities = pd.DataFrame(probability_rows)
    sums = probabilities.groupby("outcome_id")["probability"].sum()
    if not np.allclose(sums.to_numpy(), 1.0):
        raise AssertionError("Official output weights must sum to one per output")
    metrics = {
        "n_outputs": int(len(outcomes)),
        "n_official_instrument_categories": int(
            len({value for values in outcomes["official_categories"] for value in values})
        ),
        "n_multi_category_outputs": int(outcomes["official_concerns"].map(len).gt(1).sum()),
        "allocation": "equal fractional weight across official ATS instrument categories",
    }
    return pd.DataFrame(prediction_rows), probabilities, metrics


def title_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    sublinear_tf=True,
                    max_features=70_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=70_000,
                ),
            ),
        ]
    )


def calibrate_temperature(scores: np.ndarray, targets: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        probabilities = softmax(scores / temperature, axis=1)
        target_mass = (probabilities * targets).sum(axis=1) / targets.sum(axis=1)
        return float(-np.log(np.clip(target_mass, 1e-12, 1)).mean())

    fitted = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(fitted.x))


def nested_training_temperature(
    titles: pd.Series,
    targets: np.ndarray,
    groups: pd.Series,
) -> float:
    """Fit temperature from predictions cross-fitted inside an outer training fold."""
    n_splits = min(4, int(pd.Series(groups).nunique()))
    if n_splits < 2:
        raise ValueError("Temperature calibration requires at least two meetings")
    splitter = GroupKFold(n_splits=n_splits)
    nested_scores = np.full(targets.shape, np.nan, dtype=float)
    for inner_train, inner_test in splitter.split(titles, groups=groups):
        features = title_features()
        x_train = features.fit_transform(titles.iloc[inner_train])
        x_test = features.transform(titles.iloc[inner_test])
        for class_index in range(targets.shape[1]):
            target = targets[inner_train, class_index]
            if np.unique(target).size < 2:
                nested_scores[inner_test, class_index] = (
                    -20.0 if target[0] == 0 else 20.0
                )
                continue
            classifier = LinearSVC(
                C=1.3, class_weight="balanced", max_iter=10_000, tol=1e-3
            )
            classifier.fit(x_train, target)
            nested_scores[inner_test, class_index] = classifier.decision_function(
                x_test
            )
    if np.isnan(nested_scores).any():
        raise AssertionError("Nested calibration left unscored paper titles")
    return calibrate_temperature(nested_scores, targets)


def classify_outcomes(
    training: pd.DataFrame,
    outcomes: pd.DataFrame,
    phi: pd.DataFrame,
    x_position: dict[str, float],
    region_of: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    usable_outcomes = outcomes[outcomes["title"].notna()].copy()
    classes = np.asarray(phi.index.tolist())
    encoder = MultiLabelBinarizer(classes=classes)
    targets = encoder.fit_transform(training["topics"])
    splitter = GroupKFold(n_splits=N_CV_SPLITS)
    oof_scores = np.full((len(training), len(classes)), np.nan, dtype=float)
    oof_probabilities = np.full((len(training), len(classes)), np.nan, dtype=float)
    oof_folds = np.zeros(len(training), dtype=int)
    # Cross-fit formal outputs by the same held-out meeting groups used for
    # paper validation.  An output is therefore never labelled by a model that
    # learned from paper titles at its own meeting.
    outcome_scores = np.full(
        (len(usable_outcomes), len(classes)), np.nan, dtype=float
    )
    outcome_probabilities = np.full(
        (len(usable_outcomes), len(classes)), np.nan, dtype=float
    )
    outcome_folds = np.zeros(len(usable_outcomes), dtype=int)
    fold_temperatures: dict[int, float] = {}
    for fold, (train_index, test_index) in enumerate(
        splitter.split(training["title"], groups=training["meeting"]),
        start=1,
    ):
        features = title_features()
        x_train = features.fit_transform(training.iloc[train_index]["title"])
        x_test = features.transform(training.iloc[test_index]["title"])
        held_meetings = set(training.iloc[test_index]["meeting"].astype(int))
        outcome_index = np.flatnonzero(
            usable_outcomes["meeting"].astype(int).isin(held_meetings).to_numpy()
        )
        x_outcome = (
            features.transform(usable_outcomes.iloc[outcome_index]["title"])
            if len(outcome_index)
            else None
        )
        fold_scores = np.empty((len(test_index), len(classes)), dtype=float)
        for class_index in range(len(classes)):
            target = targets[train_index, class_index]
            if np.unique(target).size < 2:
                fold_scores[:, class_index] = -20.0 if target[0] == 0 else 20.0
                if len(outcome_index):
                    outcome_scores[outcome_index, class_index] = (
                        -20.0 if target[0] == 0 else 20.0
                    )
                continue
            classifier = LinearSVC(
                C=1.3, class_weight="balanced", max_iter=10_000, tol=1e-3
            )
            classifier.fit(x_train, target)
            fold_scores[:, class_index] = classifier.decision_function(x_test)
            if len(outcome_index):
                outcome_scores[outcome_index, class_index] = (
                    classifier.decision_function(x_outcome)
                )
        oof_scores[test_index] = fold_scores
        fold_temperature = nested_training_temperature(
            training.iloc[train_index]["title"].reset_index(drop=True),
            targets[train_index],
            training.iloc[train_index]["meeting"].reset_index(drop=True),
        )
        fold_temperatures[fold] = fold_temperature
        oof_probabilities[test_index] = softmax(
            fold_scores / fold_temperature, axis=1
        )
        if len(outcome_index):
            outcome_probabilities[outcome_index] = softmax(
                outcome_scores[outcome_index] / fold_temperature, axis=1
            )
        oof_folds[test_index] = fold
        outcome_folds[outcome_index] = fold

    if (
        np.isnan(outcome_scores).any()
        or np.isnan(outcome_probabilities).any()
        or np.isnan(oof_probabilities).any()
        or (outcome_folds == 0).any()
    ):
        raise ValueError("Every output must receive a held-out-meeting prediction")

    oof_prediction_index = oof_scores.argmax(axis=1)
    oof_predictions = classes[oof_prediction_index]
    oof_order = np.argsort(oof_scores, axis=1)[:, ::-1]
    ordered_scores = np.sort(oof_scores, axis=1)
    margins = ordered_scores[:, -1] - ordered_scores[:, -2]

    phi_values = phi.to_numpy(dtype=float)
    predicted_phi = np.asarray(
        [phi_values[np.flatnonzero(targets[i]), oof_prediction_index[i]].max() for i in range(len(training))]
    )
    x_error = np.asarray(
        [min(abs(x_position[topic] - x_position[oof_predictions[i]]) for topic in training.iloc[i]["topics"]) for i in range(len(training))]
    )
    region_match = np.asarray(
        [any(region_of[topic] == region_of[oof_predictions[i]] for topic in training.iloc[i]["topics"]) for i in range(len(training))]
    )
    top1_hit = targets[np.arange(len(training)), oof_prediction_index].astype(bool)
    top3 = oof_order[:, :3]
    top3_hit = np.asarray([targets[i, top3[i]].any() for i in range(len(training))])
    threshold_predictions = (oof_scores > 0).astype(int)
    cv_metrics = {
        "n_papers": int(len(training)),
        "n_concerns": int(len(classes)),
        "grouped_folds": N_CV_SPLITS,
        "top1_matches_any_official_category": float(top1_hit.mean()),
        "top3_contains_any_official_category": float(top3_hit.mean()),
        "macro_f1_at_zero_threshold": float(f1_score(targets, threshold_predictions, average="macro", zero_division=0)),
        "same_descriptive_region": float(region_match.mean()),
        "mean_true_predicted_phi": float(predicted_phi.mean()),
        "median_concern_axis_absolute_error": float(np.median(x_error)),
        "calibration": "temperature nested-cross-fitted inside each outer training fold",
        "fold_temperatures": fold_temperatures,
        "high_confidence_margin": HIGH_CONFIDENCE_MARGIN,
        "high_confidence_coverage": float((margins >= HIGH_CONFIDENCE_MARGIN).mean()),
        "high_confidence_top1_hit": float(top1_hit[margins >= HIGH_CONFIDENCE_MARGIN].mean()),
    }

    scores = outcome_scores
    probabilities = outcome_probabilities
    order = np.argsort(scores, axis=1)[:, ::-1]
    sorted_scores = np.sort(scores, axis=1)
    outcome_margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    prediction_rows = []
    probability_rows = []
    for index, outcome in usable_outcomes.reset_index(drop=True).iterrows():
        ranked = order[index]
        top = classes[ranked[:3]]
        record = outcome.to_dict()
        record.update(
            {
                "topic_top1": top[0],
                "topic_top2": top[1],
                "topic_top3": top[2],
                "probability_top1": float(probabilities[index, ranked[0]]),
                "margin_top1_top2": float(outcome_margin[index]),
                "high_confidence": bool(outcome_margin[index] >= HIGH_CONFIDENCE_MARGIN),
                "crossfit_fold": int(outcome_folds[index]),
                "expected_x": float(
                    sum(probabilities[index, j] * x_position[topic] for j, topic in enumerate(classes))
                ),
            }
        )
        prediction_rows.append(record)
        for class_index, topic in enumerate(classes):
            probability_rows.append(
                {
                    "outcome_id": outcome["outcome_id"],
                    "year": int(outcome["year"]),
                    "meeting": int(outcome["meeting"]),
                    "instrument": outcome["instrument"],
                    "topic": topic,
                    "probability": float(probabilities[index, class_index]),
                }
            )

    oof_export = training[["paper_id", "meeting"]].copy()
    oof_export["fold"] = oof_folds
    oof_export["true_topics"] = training["topics"].map(" | ".join)
    oof_export["predicted_topic"] = oof_predictions
    oof_export["predicted_topic_top1"] = classes[oof_order[:, 0]]
    oof_export["predicted_topic_top2"] = classes[oof_order[:, 1]]
    oof_export["predicted_topic_top3"] = classes[oof_order[:, 2]]
    oof_export["allocation_weight_top1"] = oof_probabilities[
        np.arange(len(training)), oof_order[:, 0]
    ]
    oof_export["true_category_probability_mass"] = np.asarray(
        [
            oof_probabilities[index, targets[index].astype(bool)].sum()
            for index in range(len(training))
        ]
    )
    oof_export["margin_top1_top2"] = margins
    oof_export["true_predicted_phi"] = predicted_phi
    oof_export["concern_axis_absolute_error"] = x_error
    oof_export["same_descriptive_region"] = region_match
    oof_export.to_csv(OUTDIR / "paper_title_classifier_oof.csv", index=False)
    return pd.DataFrame(prediction_rows), pd.DataFrame(probability_rows), cv_metrics


def canonicalize_topic_matrix(
    matrix: pd.DataFrame,
    topics: list[str],
    topic_lookup: dict[str, str],
    actors: list[str],
) -> pd.DataFrame:
    matrix = standardize_index_labels(matrix.copy())
    matrix["__topic"] = [topic_lookup.get(lineage._normalize(raw)) for raw in matrix.index]
    matrix = matrix.dropna(subset=["__topic"]).groupby("__topic").sum(numeric_only=True)
    return matrix.reindex(index=topics, columns=actors, fill_value=0.0)


def cumulative_phi_by_meeting(
    submitted: pd.DataFrame,
    topics: list[str],
    topic_lookup: dict[str, str],
    meetings: list[int],
) -> dict[int, pd.DataFrame]:
    """Build the concern space from meetings preceding each focal ATCM."""
    actors_raw = extract_unique_countries(submitted)
    topics_raw = extract_unique_topics(submitted)
    actors = sorted({str(actor) for actor in actors_raw})
    meeting_values = pd.to_numeric(submitted["meeting number"], errors="coerce")
    result = {}
    for meeting in meetings:
        history = submitted[meeting_values < meeting]
        interaction = generate_interaction_matrix(history, actors_raw, topics_raw)
        counts = canonicalize_topic_matrix(interaction, topics, topic_lookup, actors)
        phi = (
            compute_product_space(get_rca(counts))
            .reindex(index=topics, columns=topics, fill_value=0.0)
            .fillna(0.0)
        )
        np.fill_diagonal(phi.values, 1.0)
        result[meeting] = phi
    return result


def local_weight_matrix(phi: pd.DataFrame, k: int = LOCAL_K) -> np.ndarray:
    values = phi.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, 0.0)
    weights = np.zeros_like(values)
    for row in range(values.shape[0]):
        order = np.argsort(values[row])[::-1]
        keep = [index for index in order if values[row, index] > 0][:k]
        if keep:
            weights[row, keep] = values[row, keep]
            weights[row] /= weights[row].sum()
    return weights


def paper_attention_meeting_panel(
    training: pd.DataFrame,
    topics: list[str],
    meetings: list[int],
) -> pd.DataFrame:
    data = training[training["meeting"].isin(meetings)].copy()
    data["paper_weight"] = data["topics"].map(lambda values: 1.0 / len(values))
    data = data.explode("topics").rename(columns={"topics": "topic"})
    counts = data.groupby(["topic", "meeting"])["paper_weight"].sum().rename("paper_count")
    index = pd.MultiIndex.from_product(
        [topics, meetings], names=["topic", "meeting"]
    )
    return counts.reindex(index, fill_value=0).reset_index()


def outcome_mass_meeting_panel(
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    topics: list[str],
    meetings: list[int],
) -> pd.DataFrame:
    """Aggregate classified output mass by ATCM number rather than year."""
    index = pd.MultiIndex.from_product(
        [topics, meetings], names=["topic", "meeting"]
    )
    soft = (
        probabilities.groupby(["topic", "meeting"])["probability"]
        .sum()
        .rename("outcome_mass")
    )
    hard = (
        predictions.groupby(["topic_top1", "meeting"])
        .size()
        .rename("outcome_count_hard")
    )
    hard.index.names = ["topic", "meeting"]
    high = (
        predictions[predictions["high_confidence"]]
        .groupby(["topic_top1", "meeting"])
        .size()
        .rename("outcome_count_high_confidence")
    )
    high.index.names = ["topic", "meeting"]
    by_instrument = []
    for instrument in ("Recommendation", "Measure", "Decision", "Resolution"):
        by_instrument.append(
            probabilities[probabilities["instrument"].eq(instrument)]
            .groupby(["topic", "meeting"])["probability"]
            .sum()
            .rename(f"{instrument.lower()}_mass")
        )
    return (
        pd.concat([soft, hard, high, *by_instrument], axis=1)
        .reindex(index, fill_value=0)
        .reset_index()
    )


def write_outcome_coverage(
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    topics: list[str],
) -> pd.DataFrame:
    """Export the exact node encodings used by the annual-output map.

    ``primary_outcomes`` is a hard, mutually exclusive title-classifier count.
    ``expected_outcome_mass`` retains classification uncertainty and is used in
    the regression analysis, not to decide whether a map node is hollow.
    """
    hard = predictions["topic_top1"].value_counts()
    high = predictions.loc[predictions["high_confidence"], "topic_top1"].value_counts()
    mass = probabilities.groupby("topic")["probability"].sum()
    coverage = pd.DataFrame(
        {
            "topic": topics,
            "primary_outcomes": [int(hard.get(topic, 0)) for topic in topics],
            "high_confidence_primary_outcomes": [int(high.get(topic, 0)) for topic in topics],
            "expected_outcome_mass": [float(mass.get(topic, 0.0)) for topic in topics],
        }
    )
    coverage["has_primary_annual_output"] = coverage["primary_outcomes"] > 0
    coverage.to_csv(OUTDIR / "annual_output_topic_coverage.csv", index=False)
    return coverage


def write_scope_anchors() -> pd.DataFrame:
    """Record external adoption events used only to explain corpus boundaries.

    These rows are not appended to annual ATCM output counts or regressions.
    They mirror the scope-aware audits in ``../ats_lineage/case_studies.json``.
    """
    cases = {
        case["case_id"]: case
        for case in load_json(LINEAGE_ROOT / "case_studies.json")["cases"]
    }
    rows = [
        {
            "event_id": "madrid_protocol",
            "label": "Madrid Protocol",
            "year": 1991,
            "adopting_forum": "SATCM XI-4",
            "scope_status": cases["madrid_protocol"]["scope_status"],
            "in_annual_atcm_outcome_corpus": False,
            "topic": "Mineral resources",
            "relation": "Article 7 mineral-resource prohibition",
            "mapping_role": "primary provision-level context",
        },
        {
            "event_id": "madrid_protocol",
            "label": "Madrid Protocol",
            "year": 1991,
            "adopting_forum": "SATCM XI-4",
            "scope_status": cases["madrid_protocol"]["scope_status"],
            "in_annual_atcm_outcome_corpus": False,
            "topic": "Drilling",
            "relation": "Article 7 scientific-research exception",
            "mapping_role": "secondary legal context; not an annual-output assignment",
        },
        {
            "event_id": "ross_sea_mpa",
            "label": "Ross Sea region MPA",
            "year": 2016,
            "adopting_forum": "CCAMLR",
            "scope_status": cases["ross_sea_mpa"]["scope_status"],
            "in_annual_atcm_outcome_corpus": False,
            "topic": "Marine Protected Areas",
            "relation": "CCAMLR Conservation Measure 91-05 adoption",
            "mapping_role": "primary external-adoption context",
        },
    ]
    anchors = pd.DataFrame(rows)
    anchors.to_csv(OUTDIR / "external_outcome_scope_anchors.csv", index=False)
    return anchors


def write_outcome_validation_sample(
    predictions: pd.DataFrame,
    sample_size: int = 120,
) -> None:
    """Write an outcome-blind, instrument/confidence-stratified audit packet."""
    data = predictions.copy()
    data["confidence_band"] = np.where(data["high_confidence"], "high", "lower")
    pieces = []
    per_stratum = max(1, sample_size // max(data.groupby(["instrument", "confidence_band"]).ngroups, 1))
    for _, group in data.groupby(["instrument", "confidence_band"], sort=True):
        pieces.append(
            group.sample(n=min(per_stratum, len(group)), random_state=RANDOM_SEED)
        )
    sample = pd.concat(pieces).drop_duplicates("outcome_id")
    if len(sample) < sample_size:
        remainder = data[~data["outcome_id"].isin(sample["outcome_id"])]
        sample = pd.concat(
            [
                sample,
                remainder.sample(
                    n=min(sample_size - len(sample), len(remainder)),
                    random_state=RANDOM_SEED + 1,
                ),
            ]
        )
    sample = sample.sort_values(["instrument", "year", "outcome_id"]).head(sample_size)
    blind = sample[["outcome_id", "year", "instrument", "title"]].copy()
    blind["coder_primary_concern"] = ""
    blind["coder_secondary_concern"] = ""
    blind["coder_confidence"] = ""
    blind["coder_notes"] = ""
    blind.to_csv(OUTDIR / "outcome_topic_validation_blind.csv", index=False)
    sample[[
        "outcome_id", "topic_top1", "topic_top2", "topic_top3",
        "probability_top1", "margin_top1_top2", "high_confidence",
    ]].to_csv(OUTDIR / "outcome_topic_validation_model_key.csv", index=False)


def build_topic_meeting_panel(
    attention: pd.DataFrame,
    outcome_mass: pd.DataFrame,
    phi_by_meeting: dict[int, pd.DataFrame],
    topics: list[str],
) -> pd.DataFrame:
    """Construct a complete concern-by-meeting panel in ATCM order."""
    panel = attention.merge(outcome_mass, on=["topic", "meeting"], how="left").fillna(0)
    topic_index = {topic: index for index, topic in enumerate(topics)}
    panel["neighbor_papers"] = 0.0
    for meeting, group_index in panel.groupby("meeting").groups.items():
        indices = list(group_index)
        ordered = panel.loc[indices].sort_values("topic")
        paper_vector = (
            ordered.set_index("topic")["paper_count"]
            .reindex(topics)
            .to_numpy(dtype=float)
        )
        neighbor_vector = local_weight_matrix(phi_by_meeting[int(meeting)]) @ paper_vector
        panel.loc[indices, "neighbor_papers"] = [
            neighbor_vector[topic_index[topic]] for topic in panel.loc[indices, "topic"]
        ]

    panel = panel.sort_values(["topic", "meeting"]).reset_index(drop=True)
    for source, target in (
        ("paper_count", "papers_prior3"),
        ("neighbor_papers", "neighbor_papers_prior3"),
        ("outcome_mass", "outcomes_prior3"),
    ):
        panel[target] = (
            panel.groupby("topic")[source]
            .transform(lambda values: values.shift(1).rolling(3, min_periods=1).sum())
            .fillna(0.0)
        )
    for lag in range(1, 6):
        panel[f"papers_lag{lag}"] = (
            panel.groupby("topic")["paper_count"].shift(lag).fillna(0.0)
        )
        panel[f"neighbor_papers_lag{lag}"] = (
            panel.groupby("topic")["neighbor_papers"].shift(lag).fillna(0.0)
        )
    return panel


def fit_ppml(
    panel: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    specification: str,
    minimum_year: int = START_YEAR,
    period_column: str = "year",
) -> pd.DataFrame:
    data = panel[panel[period_column] >= minimum_year].copy()
    z_terms = []
    for predictor in predictors:
        transformed = np.log1p(data[predictor].to_numpy(dtype=float))
        standard_deviation = transformed.std(ddof=0)
        name = f"z_{predictor}"
        data[name] = (transformed - transformed.mean()) / max(standard_deviation, 1e-12)
        z_terms.append(name)
    formula = f"{outcome} ~ {' + '.join(z_terms)} + C(topic) + C({period_column})"
    fitted = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()
    topic_groups = pd.Categorical(data["topic"]).codes
    period_groups = pd.Categorical(data[period_column]).codes
    covariance, _, _ = cov_cluster_2groups(fitted, topic_groups, period_groups)
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
                f"n_topic_{period_column}s": int(len(data)),
                "coefficient": estimate,
                f"se_two_way_cluster_topic_{period_column}": se,
                "incidence_rate_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * se),
                "ci_high": math.exp(estimate + 1.96 * se),
                "p_value": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
                "scale": "one SD of log1p predictor",
            }
        )
    contrast_pairs = [
        ("paper_count", "neighbor_papers"),
        ("papers_prior3", "neighbor_papers_prior3"),
    ]
    for own_predictor, neighbor_predictor in contrast_pairs:
        if own_predictor not in predictors or neighbor_predictor not in predictors:
            continue
        own_term = f"z_{own_predictor}"
        neighbor_term = f"z_{neighbor_predictor}"
        own_index = term_index[own_term]
        neighbor_index = term_index[neighbor_term]
        estimate = float(fitted.params[own_term] - fitted.params[neighbor_term])
        variance = float(
            covariance[own_index, own_index]
            + covariance[neighbor_index, neighbor_index]
            - 2 * covariance[own_index, neighbor_index]
        )
        se = math.sqrt(max(variance, 0.0))
        z_value = estimate / se if se > 0 else np.nan
        rows.append(
            {
                "specification": specification,
                "outcome": outcome,
                "predictor": f"{own_predictor}_minus_{neighbor_predictor}",
                f"n_topic_{period_column}s": int(len(data)),
                "coefficient": estimate,
                f"se_two_way_cluster_topic_{period_column}": se,
                "incidence_rate_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * se),
                "ci_high": math.exp(estimate + 1.96 * se),
                "p_value": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
                "scale": "ratio of own-attention IRR to nearby-attention IRR, each per one SD",
            }
        )
    return pd.DataFrame(rows)


def fit_meeting_lag_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate attention--output associations over ordered ATCM meetings.

    A lag of one denotes the immediately preceding meeting, even when the
    elapsed calendar time is longer than one year. Each lag is fitted in a
    separate model with concern and meeting fixed effects.
    """
    tables = []
    same_meeting = fit_ppml(
        panel,
        "outcome_mass",
        ["paper_count", "neighbor_papers"],
        "meeting_lag_profile_0",
        minimum_year=1,
        period_column="meeting",
    )
    same_meeting["lag_meetings"] = 0
    tables.append(same_meeting)
    for lag in range(1, 6):
        fitted = fit_ppml(
            panel,
            "outcome_mass",
            [f"papers_lag{lag}", f"neighbor_papers_lag{lag}", "outcomes_prior3"],
            f"meeting_lag_profile_{lag}",
            minimum_year=1 + lag,
            period_column="meeting",
        )
        fitted["lag_meetings"] = lag
        tables.append(fitted)
    profile = pd.concat(tables, ignore_index=True)
    profile.to_csv(OUTDIR / "attention_outcome_meeting_lag_profile.csv", index=False)
    return profile


def expected_topic_proximity(
    source_topics: list[str],
    outcome_id: str,
    probability_lookup: dict[str, np.ndarray],
    phi: pd.DataFrame,
) -> float:
    if not source_topics or outcome_id not in probability_lookup:
        return np.nan
    source_indices = [phi.index.get_loc(topic) for topic in source_topics if topic in phi.index]
    if not source_indices:
        return np.nan
    proximity_to_sources = phi.to_numpy(dtype=float)[:, source_indices].max(axis=1)
    return float(probability_lookup[outcome_id] @ proximity_to_sources)


def direct_edge_analysis(
    paper_categories: dict[str, list[str]],
    outcomes: pd.DataFrame,
    probabilities: pd.DataFrame,
    phi: pd.DataFrame,
    graph_name: str,
    phi_by_meeting: dict[int, pd.DataFrame] | None = None,
    geometry: str = "pooled_figure1_space",
    n_permutations: int = 10_000,
) -> tuple[pd.DataFrame, list[dict]]:
    graph = load_json(LINEAGE_ROOT / graph_name)
    nodes = {node["id"]: node for node in graph["nodes"]}
    topics = list(phi.index)
    probability_pivot = probabilities.pivot(index="outcome_id", columns="topic", values="probability").reindex(columns=topics)
    probability_lookup = {outcome: row.to_numpy(dtype=float) for outcome, row in probability_pivot.iterrows()}
    outcome_set = set(outcomes["outcome_id"])
    outcome_meta = outcomes.set_index("outcome_id")
    meeting_papers: dict[int, list[str]] = collections.defaultdict(list)
    for paper_id in paper_categories:
        try:
            meeting = int(paper_id.split(":", 1)[0].replace("ATCM", ""))
        except ValueError:
            continue
        meeting_papers[meeting].append(paper_id)

    rows = []
    for edge in graph["edges"]:
        source = nodes.get(edge["src"], {})
        target = nodes.get(edge["dst"], {})
        if source.get("kind") != "paper" or target.get("kind") != "outcome":
            continue
        if edge["dst"] not in outcome_set or edge["src"] not in paper_categories:
            continue
        edge_phi = (
            phi_by_meeting[int(target["meeting"])]
            if phi_by_meeting is not None
            else phi
        )
        proximity = expected_topic_proximity(
            paper_categories[edge["src"]], edge["dst"], probability_lookup, edge_phi
        )
        if np.isnan(proximity):
            continue
        candidates = meeting_papers[int(target["meeting"])]
        candidate_proximities = np.asarray(
            [
                expected_topic_proximity(paper_categories[paper], edge["dst"], probability_lookup, edge_phi)
                for paper in candidates
            ],
            dtype=float,
        )
        rows.append(
            {
                "graph": graph_name,
                "geometry": geometry,
                "paper_id": edge["src"],
                "outcome_id": edge["dst"],
                "meeting": int(target["meeting"]),
                "year": int(target["year"]),
                "tier": edge.get("tier"),
                "channel": edge.get("channel"),
                "relation": edge.get("relation"),
                "source_topics": " | ".join(paper_categories[edge["src"]]),
                "expected_phi": proximity,
                "within_meeting_percentile": float((candidate_proximities <= proximity).mean()),
                "candidate_papers": int(len(candidate_proximities)),
                "outcome_topic_top1": outcome_meta.loc[edge["dst"], "topic_top1"],
                "outcome_instrument": outcome_meta.loc[edge["dst"], "instrument"],
                "outcome_high_confidence": bool(outcome_meta.loc[edge["dst"], "high_confidence"]),
            }
        )
    edges = pd.DataFrame(rows)

    tests = []
    subsets = {
        "all_documented_edges": np.ones(len(edges), dtype=bool),
        "adoption_or_contribution": edges["relation"].isin(
            ["direct_adoption_or_approval", "documented_contribution"]
        ).to_numpy(),
        "adoption_or_contribution_excluding_site_administration": (
            edges["relation"].isin(["direct_adoption_or_approval", "documented_contribution"])
            & ~edges["outcome_topic_top1"].isin(SITE_ADMIN_TOPICS)
        ).to_numpy(),
        "adoption_or_contribution_high_confidence_outcomes": (
            edges["relation"].isin(["direct_adoption_or_approval", "documented_contribution"])
            & edges["outcome_high_confidence"]
        ).to_numpy(),
        "proposal_or_discussion": edges["relation"].eq("direct_proposal_or_discussion").to_numpy(),
    }
    rng = np.random.default_rng(RANDOM_SEED)
    for label, keep in subsets.items():
        subset = edges.loc[keep].copy()
        if subset.empty:
            continue
        observed = float(subset.groupby("outcome_id")["within_meeting_percentile"].mean().mean())
        null_sum = np.zeros(n_permutations, dtype=float)
        candidate_sizes = subset.groupby("outcome_id")["candidate_papers"].first()
        edge_counts = subset.groupby("outcome_id").size()
        for outcome_id, n_edges in edge_counts.items():
            n_candidates = int(candidate_sizes[outcome_id])
            # Under exchangeability, ranks among the meeting's candidate
            # papers are discrete uniforms. Random keys generate an exact
            # without-replacement sample for every permutation at once.
            n_sample = min(int(n_edges), n_candidates)
            keys = rng.random((n_permutations, n_candidates), dtype=np.float32)
            ranks = np.argpartition(keys, n_sample - 1, axis=1)[:, :n_sample] + 1
            null_sum += ranks.mean(axis=1) / n_candidates
        null = null_sum / len(edge_counts)
        tests.append(
            {
                "graph": graph_name,
                "geometry": geometry,
                "subset": label,
                "edges": int(len(subset)),
                "outcomes": int(subset["outcome_id"].nunique()),
                "observed_mean_within_meeting_percentile": observed,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std(ddof=1)),
                "upper_tail_p": float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1)),
            }
        )
    return edges, tests


def prior_actor_state(
    submitted: pd.DataFrame,
    meeting: int,
    topics: list[str],
    actors: list[str],
    topic_lookup: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_actors = extract_unique_countries(submitted)
    raw_topics = extract_unique_topics(submitted)
    meeting_values = pd.to_numeric(submitted["meeting number"], errors="coerce")
    ordered_meetings = sorted(
        int(value) for value in meeting_values.dropna().unique() if int(value) < meeting
    )
    # "Prior portfolio" means the immediately preceding ATCM, not a calendar
    # year or a fixed number of elapsed years.
    recent_meetings = ordered_meetings[-1:]
    recent = submitted[meeting_values.isin(recent_meetings)]
    history = submitted[meeting_values < meeting]
    recent_counts = canonicalize_topic_matrix(
        generate_interaction_matrix(recent, raw_actors, raw_topics), topics, topic_lookup, actors
    )
    recent_rca = get_rca(recent_counts).reindex(index=topics, columns=actors, fill_value=0.0)
    history_counts = canonicalize_topic_matrix(
        generate_interaction_matrix(history, raw_actors, raw_topics), topics, topic_lookup, actors
    )
    history_phi = (
        compute_product_space(get_rca(history_counts))
        .reindex(index=topics, columns=topics, fill_value=0.0)
        .fillna(0.0)
    )
    np.fill_diagonal(history_phi.values, 1.0)
    volume = recent_counts.sum(axis=0).reindex(actors, fill_value=0.0)
    return recent_rca, history_phi, volume


def actor_outcome_panel(
    submitted: pd.DataFrame,
    outcomes: pd.DataFrame,
    probabilities: pd.DataFrame,
    paper_categories: dict[str, list[str]],
    topics: list[str],
    topic_lookup: dict[str, str],
) -> pd.DataFrame:
    graph = load_json(LINEAGE_ROOT / "decision_map_verified.json")
    nodes = {node["id"]: node for node in graph["nodes"]}
    sponsors = paper_sponsors(submitted)
    target_outcomes = set(outcomes["outcome_id"])
    contributors: dict[str, set[str]] = collections.defaultdict(set)
    for edge in graph["edges"]:
        if edge.get("relation") not in ("direct_adoption_or_approval", "documented_contribution"):
            continue
        if edge["dst"] not in target_outcomes or edge["src"] not in paper_categories:
            continue
        contributors[edge["dst"]].update(sponsors.get(edge["src"], set()))

    probability_pivot = probabilities.pivot(index="outcome_id", columns="topic", values="probability").reindex(columns=topics)
    actors = sorted(
        {
            actor.strip()
            for value in submitted["submitted by"].dropna()
            for actor in str(value).split(",")
            if actor.strip()
        }
    )
    state_cache = {}
    rows = []
    outcome_meta = outcomes.set_index("outcome_id")
    for outcome_id, source_actors in contributors.items():
        if not source_actors or outcome_id not in probability_pivot.index:
            continue
        year = int(outcome_meta.loc[outcome_id, "year"])
        meeting = int(outcome_meta.loc[outcome_id, "meeting"])
        if meeting not in state_cache:
            state_cache[meeting] = prior_actor_state(
                submitted, meeting, topics, actors, topic_lookup
            )
        recent_rca, history_phi, volume = state_cache[meeting]
        target_probability = probability_pivot.loc[outcome_id].to_numpy(dtype=float)
        for actor in actors:
            held = np.flatnonzero(recent_rca[actor].to_numpy(dtype=float) >= 1.0)
            if held.size == 0 or float(volume[actor]) <= 0:
                continue
            proximity_by_target = history_phi.to_numpy(dtype=float)[:, held].max(axis=1)
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "year": year,
                    "meeting": meeting,
                    "actor": actor,
                    "contributor": int(actor in source_actors),
                    "expected_proximity": float(target_probability @ proximity_by_target),
                    "breadth": int(held.size),
                    "prior_papers": float(volume[actor]),
                }
            )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    usable = panel.groupby("outcome_id")["contributor"].agg(["sum", "count"])
    usable = usable[(usable["sum"] > 0) & (usable["sum"] < usable["count"])].index
    return panel[panel["outcome_id"].isin(usable)].copy()


def fit_actor_outcome_model(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["log_breadth"] = np.log1p(data["breadth"])
    data["log_prior_papers"] = np.log1p(data["prior_papers"])
    terms = ["expected_proximity", "log_breadth", "log_prior_papers"]
    design = (data[terms] - data[terms].mean()) / data[terms].std(ddof=0)
    fitted = ConditionalLogit(
        data["contributor"], design, groups=data["outcome_id"]
    ).fit(method="bfgs", disp=False, maxiter=400)
    rows = []
    for term in terms:
        estimate = float(fitted.params[term])
        se = float(fitted.bse[term])
        rows.append(
            {
                "term": term,
                "coefficient": estimate,
                "se_model_based": se,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * se),
                "ci_high": math.exp(estimate + 1.96 * se),
                "p_value": float(fitted.pvalues[term]),
                "n_outcomes": int(data["outcome_id"].nunique()),
                "n_actor_outcome_rows": int(len(data)),
                "n_contributor_rows": int(data["contributor"].sum()),
                "scale": "one sample SD",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _, _, _, counts, _ = build_graphs()
    topics = list(counts.index)
    pooled_phi = compute_product_space(get_rca(counts)).reindex(index=topics, columns=topics)
    np.fill_diagonal(pooled_phi.values, 1.0)
    _, region_raw, x_raw = load_topic_meta()
    region_of = {topic: int(region_raw[lineage.normalize_topic_key(topic)]) for topic in topics}
    x_position = {topic: float(x_raw[lineage.normalize_topic_key(topic)]) for topic in topics}

    training, paper_categories = load_paper_training(topics)
    outcomes = load_outcomes()
    predictions, probabilities, classifier_metrics = classify_outcomes(
        training, outcomes, pooled_phi, x_position, region_of
    )
    predictions.to_csv(OUTDIR / "outcome_topic_predictions.csv", index=False)
    probabilities.to_csv(OUTDIR / "outcome_topic_probabilities.csv", index=False)
    coverage = write_outcome_coverage(predictions, probabilities, topics)
    scope_anchors = write_scope_anchors()
    write_outcome_validation_sample(predictions)
    (OUTDIR / "title_classifier_metrics.json").write_text(
        json.dumps(classifier_metrics, indent=2) + "\n"
    )

    submitted = load_submitted_with_fallback()
    topic_lookup = lineage._canonical_topic_lookup(topics)
    meetings = list(range(START_MEETING, END_MEETING + 1))
    phi_by_meeting = cumulative_phi_by_meeting(
        submitted, topics, topic_lookup, meetings
    )

    meeting_attention = paper_attention_meeting_panel(training, topics, meetings)
    meeting_outcomes = outcome_mass_meeting_panel(
        predictions, probabilities, topics, meetings
    )
    meeting_panel = build_topic_meeting_panel(
        meeting_attention, meeting_outcomes, phi_by_meeting, topics
    )
    meeting_panel.to_csv(OUTDIR / "topic_meeting_attention_outcomes.csv", index=False)
    meeting_lag_profile = fit_meeting_lag_profile(meeting_panel)

    model_tables = [
        fit_ppml(
            meeting_panel, "outcome_mass", ["paper_count", "neighbor_papers"],
            "same_meeting_soft_assignment", minimum_year=1,
            period_column="meeting",
        ),
        fit_ppml(
            meeting_panel, "outcome_mass",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "prospective_prior3_meetings_soft_assignment",
            minimum_year=4, period_column="meeting",
        ),
        fit_ppml(
            meeting_panel, "outcome_count_hard",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "prospective_prior3_meetings_hard_top1",
            minimum_year=4, period_column="meeting",
        ),
        fit_ppml(
            meeting_panel, "outcome_count_high_confidence",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "prospective_prior3_meetings_high_confidence",
            minimum_year=4, period_column="meeting",
        ),
        fit_ppml(
            meeting_panel, "measure_mass",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "prospective_prior3_meetings_measures",
            minimum_year=4, period_column="meeting",
        ),
        fit_ppml(
            meeting_panel, "decision_mass",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "prospective_prior3_meetings_decisions",
            minimum_year=4, period_column="meeting",
        ),
        fit_ppml(
            meeting_panel, "resolution_mass",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "prospective_prior3_meetings_resolutions",
            minimum_year=4, period_column="meeting",
        ),
        fit_ppml(
            meeting_panel[~meeting_panel["topic"].isin(SITE_ADMIN_TOPICS)],
            "outcome_mass",
            ["papers_prior3", "neighbor_papers_prior3", "outcomes_prior3"],
            "prospective_prior3_meetings_excluding_site_administration",
            minimum_year=4, period_column="meeting",
        ),
    ]
    ppml = pd.concat(model_tables, ignore_index=True)
    ppml.to_csv(OUTDIR / "attention_outcome_ppml.csv", index=False)

    summary = {
        "scope": {
            "years": [START_YEAR, END_YEAR],
            "regular_atcm_outputs": int(len(outcomes)),
            "outcomes_with_titles": int(len(predictions)),
            "constitutional_events_outside_atcm_outcome_graph": ["Madrid Protocol adoption at SATCM XI-4"],
            "topics_with_primary_annual_output": int(coverage["has_primary_annual_output"].sum()),
            "topics_without_primary_annual_output": int((~coverage["has_primary_annual_output"]).sum()),
            "external_scope_anchor_rows": int(len(scope_anchors)),
        },
        "independent_outcome_coding": classifier_metrics,
        "indirect_topic_meeting_models": ppml.to_dict(orient="records"),
        "attention_outcome_meeting_lag_profile": meeting_lag_profile.to_dict(orient="records"),
        "interpretive_limits": [
            "Output concern probabilities come from titles, not full legal-text coding.",
            "Paper labels are inferred primary concerns derived from archive-category bundles.",
            "PPML coefficients are within-topic associations with topic and meeting fixed effects.",
            "The analysis does not identify intent, bargaining power, implementation, or legal effectiveness.",
        ],
    }
    (OUTDIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
