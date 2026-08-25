#!/usr/bin/env python3
"""Prospective test of whether concern-space attention forecasts Resolutions.

Papers are observable before an ATCM adopts its formal outputs. For each rolling-origin
meeting, this analysis predicts how each output type is distributed across the
15 broad ATS categories. It compares the category's earlier output history,
direct paper attention, and paper attention in nearby categories of a concern
space constructed only from earlier meetings.

The output-history and direct-attention settings are selected from pooled output
forecasts for ATCMs 25--28, before the ATCM 29--47 test period.  The network
extension adds current attention from every non-focal category, weighted by the
pre-meeting concern map, without another fitted neighbourhood-size parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.analyze_output_category_families import (
    family_phi_by_meeting,
    local_weight_matrix,
    paper_family_relations,
)
from scripts.forecast_output_allocation import (
    BOOTSTRAP_DRAWS,
    SEED,
    TEST_END,
    TEST_START,
    TRAIN_START,
    prepare_panel,
)
from scripts.official_regular_atcm_outputs import (
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
    load_official_regular_outputs,
)
from scripts.primary_concern_sensitivity import variants
from utils import _split_multi_value, compute_product_space, get_rca


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "attention_output_signal"
TYPE_SCORES_PATH = OUTDIR / "type_meeting_scores.csv"
TYPE_SUMMARY_PATH = OUTDIR / "type_summary.csv"
TYPE_COMPOSITION_PATH = OUTDIR / "type_category_composition.csv"
DOMINANT_EXCLUSION_PATH = OUTDIR / "type_dominant_category_exclusion.csv"
TYPE_CONTRAST_PATH = OUTDIR / "type_contrast.csv"
PRETEST_SELECTION_PATH = OUTDIR / "pretest_selected_specifications.csv"
TEMPORAL_PATH = OUTDIR / "resolution_temporal_robustness.csv"
OTHER_ACTIVITY_PATH = OUTDIR / "resolution_other_activity_control.csv"
MEASURE_RECURRENCE_PATH = OUTDIR / "measure_recurrence_ablation.csv"
MEASURE_CONTENT_PATH = OUTDIR / "measure_content_summary.json"
HORIZON_PATH = OUTDIR / "resolution_horizon_robustness.csv"
ALPHA_PATH = OUTDIR / "resolution_regularization_robustness.csv"
NETWORK_K_PATH = OUTDIR / "resolution_network_k_robustness.csv"
LEAVE_ONE_OUT_PATH = OUTDIR / "resolution_leave_one_category_out.csv"
GEOMETRY_NULL_PATH = OUTDIR / "resolution_geometry_null.csv"
FINE_PROJECTION_PATH = OUTDIR / "resolution_fine_space_projection.csv"
SUMMARY_PATH = OUTDIR / "summary.json"

PRIMARY_HISTORY = 5
PRIMARY_ATTENTION = 1
PRIMARY_ALPHA = 0.1
PRIMARY_NETWORK_K = 14
PRETEST_START = 25
PRETEST_END = 28
HISTORY_WINDOWS = (3, 5, 8, 10)
ATTENTION_WINDOWS = (1, 2, 3, 5)
ALPHAS = (0.1, 0.2, 0.5, 1.0)
NETWORK_K = (3, 5, 7, 10, 14)
OUTPUT_COLUMNS = {
    "Measure": "measure_mass",
    "Decision": "decision_mass",
    "Resolution": "resolution_mass",
}
AREA_CATEGORY = "Area protection and management"
PERMUTATIONS = 200


def add_features() -> pd.DataFrame:
    panel = prepare_panel()
    relations = paper_family_relations(variants()["fractional_multilabel"])
    current_reach = (
        relations.groupby(["family", "meeting"])["actor"]
        .nunique()
        .rename("current_actor_reach")
        .reset_index()
        .rename(columns={"family": "topic"})
    )
    panel = panel.merge(current_reach, on=["topic", "meeting"], how="left")
    panel["current_actor_reach"] = panel["current_actor_reach"].fillna(0.0)
    panel["meeting_papers"] = panel.groupby("meeting")["paper_count"].transform("sum")
    panel["other_papers"] = panel["meeting_papers"] - panel["paper_count"]

    for output in OUTPUT_COLUMNS.values():
        for horizon in HISTORY_WINDOWS:
            panel[f"{output}_history_{horizon}"] = (
                panel.groupby("topic")[output]
                .transform(
                    lambda values: values.shift(1).rolling(horizon, min_periods=1).sum()
                )
                .fillna(0.0)
            )

    for horizon in HISTORY_WINDOWS:
        panel[f"outcome_mass_history_{horizon}"] = (
            panel.groupby("topic")["outcome_mass"]
            .transform(
                lambda values: values.shift(1).rolling(horizon, min_periods=1).sum()
            )
            .fillna(0.0)
        )

    families = sorted(panel["topic"].unique())
    meetings = sorted(panel["meeting"].unique())
    phi = family_phi_by_meeting(relations, families, meetings)
    for k in NETWORK_K:
        panel[f"neighbor_papers_k{k}"] = 0.0
    for meeting, row_index in panel.groupby("meeting").groups.items():
        indices = list(row_index)
        current = (
            panel.loc[indices]
            .set_index("topic")["paper_count"]
            .reindex(families)
            .to_numpy(float)
        )
        topics = panel.loc[indices, "topic"].tolist()
        topic_index = {topic: index for index, topic in enumerate(families)}
        for k in NETWORK_K:
            nearby = local_weight_matrix(phi[int(meeting)], k=k) @ current
            panel.loc[indices, f"neighbor_papers_k{k}"] = [
                nearby[topic_index[topic]] for topic in topics
            ]
    if not np.allclose(panel["neighbor_papers_k5"], panel["neighbor_papers"]):
        raise AssertionError("Rebuilt five-neighbour attention differs from the panel")
    return panel


def add_fine_space_projection(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Project current attention through the original 45-concern geometry."""
    submitted = variants()["fractional_multilabel"]
    rows = []
    for record in submitted.drop_duplicates("paper id").to_dict(orient="records"):
        categories = _split_multi_value(record.get("category"), "\t")
        actors = _split_multi_value(record.get("submitted by"))
        meeting = pd.to_numeric(record.get("meeting number"), errors="coerce")
        if not categories or not actors or pd.isna(meeting):
            continue
        for category in categories:
            for actor in actors:
                rows.append(
                    {
                        "paper_id": int(record["paper id"]),
                        "meeting": int(meeting),
                        "concern": category,
                        "actor": actor,
                        "weight": 1.0 / len(categories),
                    }
                )
    relations = pd.DataFrame(rows)
    papers = relations.drop_duplicates(["paper_id", "concern"])
    concerns = sorted(relations["concern"].unique())
    actors = sorted(relations["actor"].unique())
    families = sorted(panel["topic"].unique())
    specifications = [("max", 5), ("max", 10), ("max", 40), ("mean", 5), ("mean", 10), ("mean", 40)]
    names = [f"fine_neighbor_{aggregation}_k{k}" for aggregation, k in specifications]
    result = panel.copy()
    for name in names:
        result[name] = 0.0

    for meeting in sorted(result["meeting"].unique()):
        history = (
            relations[relations["meeting"].lt(meeting)]
            .groupby(["concern", "actor"])["weight"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=concerns, columns=actors, fill_value=0.0)
        )
        phi = compute_product_space(get_rca(history)).reindex(
            index=concerns, columns=concerns, fill_value=0.0
        )
        current = (
            papers[papers["meeting"].eq(meeting)]
            .groupby("concern")["weight"]
            .sum()
            .reindex(concerns, fill_value=0.0)
            .to_numpy(float)
        )
        for family in families:
            own = [
                index
                for index, concern in enumerate(concerns)
                if PAPER_CONCERN_TO_INSTRUMENT_CATEGORY[concern] == family
            ]
            outside = [index for index in range(len(concerns)) if index not in own]
            for aggregation, k in specifications:
                proximity = (
                    phi.iloc[own].max(axis=0)
                    if aggregation == "max"
                    else phi.iloc[own].mean(axis=0)
                ).to_numpy(float)
                proximity[own] = 0.0
                keep = [
                    index
                    for index in sorted(outside, key=lambda item: proximity[item], reverse=True)
                    if proximity[index] > 0
                ][:k]
                weights = np.zeros(len(concerns))
                if keep:
                    weights[keep] = proximity[keep] / proximity[keep].sum()
                name = f"fine_neighbor_{aggregation}_k{k}"
                result.loc[
                    result["meeting"].eq(meeting) & result["topic"].eq(family), name
                ] = float(weights @ current)
    return result, names


def model_features(
    output: str,
    history: int = PRIMARY_HISTORY,
    attention: int = PRIMARY_ATTENTION,
    neighbor: str | None = None,
    include_focal: bool = True,
) -> list[str]:
    features = [f"{output}_history_{history}"]
    if include_focal:
        features.extend(
            [
                f"paper_history_{attention}",
                f"actor_reach_{attention}",
                "paper_count",
                "current_actor_reach",
            ]
        )
    if neighbor is not None:
        features.append(neighbor)
    return features


def forecast_meetings(
    panel: pd.DataFrame,
    output: str,
    features: list[str],
    alpha: float = PRIMARY_ALPHA,
    test_start: int = TEST_START,
    test_end: int = TEST_END,
) -> pd.DataFrame:
    rows = []
    for meeting in range(test_start, test_end + 1):
        train = panel[
            panel["meeting"].ge(TRAIN_START) & panel["meeting"].lt(meeting)
        ]
        test = panel[panel["meeting"].eq(meeting)]
        if float(test[output].sum()) <= 0:
            continue
        transform = ColumnTransformer(
            [
                ("topic", OneHotEncoder(handle_unknown="ignore"), ["topic"]),
                ("numeric", StandardScaler(), features),
            ]
        )
        model = make_pipeline(
            transform,
            PoissonRegressor(alpha=alpha, max_iter=2_000, tol=1e-9),
        )
        design = ["topic", *features]
        model.fit(train[design], train[output])
        predicted = np.maximum(model.predict(test[design]), 1e-12)
        observed_share = test[output].to_numpy(float) / float(test[output].sum())
        predicted_share = predicted / predicted.sum()
        rows.append(
            {
                "meeting": meeting,
                "allocation_log_score": float(
                    -np.sum(observed_share * np.log(predicted_share))
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_summary(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float]:
    joined = baseline.merge(candidate, on="meeting", suffixes=("_baseline", "_candidate"))
    difference = (
        joined["allocation_log_score_candidate"]
        - joined["allocation_log_score_baseline"]
    ).to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(
        difference, size=(BOOTSTRAP_DRAWS, len(difference)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_candidate_score": float(
            joined["allocation_log_score_candidate"].mean()
        ),
        "mean_baseline_score": float(
            joined["allocation_log_score_baseline"].mean()
        ),
        "mean_difference": float(difference.mean()),
        "bootstrap_low": float(low),
        "bootstrap_high": float(high),
        "meetings_improved": int((difference < 0).sum()),
        "meetings": int(len(difference)),
    }


def select_pretest_specifications(panel: pd.DataFrame) -> pd.DataFrame:
    """Choose forecast settings using pooled output before the test period."""

    rows = []
    for history in HISTORY_WINDOWS:
        for alpha in ALPHAS:
            score = forecast_meetings(
                panel,
                "outcome_mass",
                [f"outcome_mass_history_{history}"],
                alpha=alpha,
                test_start=PRETEST_START,
                test_end=PRETEST_END,
            )["allocation_log_score"].mean()
            rows.append(
                {
                    "model": "output history",
                    "history": history,
                    "attention": np.nan,
                    "alpha": alpha,
                    "mean_pretest_score": float(score),
                }
            )
    for history in HISTORY_WINDOWS:
        for attention in ATTENTION_WINDOWS:
            for alpha in ALPHAS:
                features = [
                    f"outcome_mass_history_{history}",
                    f"paper_history_{attention}",
                    f"actor_reach_{attention}",
                    "paper_count",
                    "current_actor_reach",
                ]
                score = forecast_meetings(
                    panel,
                    "outcome_mass",
                    features,
                    alpha=alpha,
                    test_start=PRETEST_START,
                    test_end=PRETEST_END,
                )["allocation_log_score"].mean()
                rows.append(
                    {
                        "model": "history + direct attention",
                        "history": history,
                        "attention": attention,
                        "alpha": alpha,
                        "mean_pretest_score": float(score),
                    }
                )
    grid = pd.DataFrame(rows)
    selected = (
        grid.sort_values(
            ["model", "mean_pretest_score", "history", "attention", "alpha"],
            na_position="first",
        )
        .groupby("model", as_index=False)
        .first()
    )
    expected = {
        "output history": (PRIMARY_HISTORY, np.nan, PRIMARY_ALPHA),
        "history + direct attention": (
            PRIMARY_HISTORY,
            PRIMARY_ATTENTION,
            PRIMARY_ALPHA,
        ),
    }
    for record in selected.to_dict(orient="records"):
        history, attention, alpha = expected[record["model"]]
        if int(record["history"]) != history or not np.isclose(record["alpha"], alpha):
            raise AssertionError(f"Unexpected pretest selection: {record}")
        if np.isnan(attention):
            if not pd.isna(record["attention"]):
                raise AssertionError(f"Unexpected pretest attention setting: {record}")
        elif int(record["attention"]) != int(attention):
            raise AssertionError(f"Unexpected pretest attention setting: {record}")
    return selected.assign(
        selection_meetings=f"ATCM {PRETEST_START}-{PRETEST_END}",
        network_extension=(
            "all non-focal categories weighted by pre-meeting proximity; no tuning"
        ),
    )


def type_comparison(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_tables = []
    summaries = []
    for instrument, output in OUTPUT_COLUMNS.items():
        specifications = {
            "output history": model_features(output, include_focal=False),
            "history + direct attention": model_features(output),
            "history + direct + network attention": model_features(
                output, neighbor=f"neighbor_papers_k{PRIMARY_NETWORK_K}"
            ),
        }
        scores = {
            name: forecast_meetings(panel, output, features)
            for name, features in specifications.items()
        }
        for name, table in scores.items():
            score_tables.append(table.assign(instrument=instrument, model=name))
        baseline = scores["output history"]
        for name in (
            "history + direct attention",
            "history + direct + network attention",
        ):
            summaries.append(
                {
                    "instrument": instrument,
                    "comparison": f"{name} vs output history",
                    **paired_summary(scores[name], baseline),
                }
            )
        summaries.append(
            {
                "instrument": instrument,
                "comparison": "network attention vs direct attention",
                **paired_summary(
                    scores["history + direct + network attention"],
                    scores["history + direct attention"],
                ),
            }
        )
    return pd.concat(score_tables, ignore_index=True), pd.DataFrame(summaries)


def type_category_composition(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instrument, output in OUTPUT_COLUMNS.items():
        mass = panel.groupby("topic")[output].sum().sort_values(ascending=False)
        shares = mass / mass.sum()
        positive = shares[shares > 0]
        rows.append(
            {
                "instrument": instrument,
                "total_output": float(mass.sum()),
                "dominant_category": str(mass.index[0]),
                "dominant_category_share": float(shares.iloc[0]),
                "effective_categories": float(
                    np.exp(-np.sum(positive * np.log(positive)))
                ),
            }
        )
    return pd.DataFrame(rows)


def dominant_category_exclusion(
    panel: pd.DataFrame, composition: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for record in composition.to_dict(orient="records"):
        instrument = record["instrument"]
        output = OUTPUT_COLUMNS[instrument]
        retained = panel[~panel["topic"].eq(record["dominant_category"])]
        history = forecast_meetings(
            retained, output, model_features(output, include_focal=False)
        )
        focal = forecast_meetings(retained, output, model_features(output))
        network = forecast_meetings(
            retained,
            output,
            model_features(
                output, neighbor=f"neighbor_papers_k{PRIMARY_NETWORK_K}"
            ),
        )
        for comparison, candidate, baseline in (
            ("direct vs history", focal, history),
            ("direct + network vs history", network, history),
            ("network vs direct", network, focal),
        ):
            rows.append(
                {
                    "instrument": instrument,
                    "excluded_category": record["dominant_category"],
                    "remaining_output_mass": float(retained[output].sum()),
                    "comparison": comparison,
                    **paired_summary(candidate, baseline),
                }
            )
    return pd.DataFrame(rows)


def instrument_contrasts(type_scores: pd.DataFrame) -> pd.DataFrame:
    wide = type_scores.pivot(
        index=["instrument", "meeting"], columns="model", values="allocation_log_score"
    )
    deltas = {
        "full attention vs history": (
            wide["history + direct + network attention"] - wide["output history"]
        ),
        "network vs direct": (
            wide["history + direct + network attention"]
            - wide["history + direct attention"]
        ),
    }
    rows = []
    for comparison, values in deltas.items():
        resolution = values.loc["Resolution"]
        for other in ("Measure", "Decision"):
            difference = (resolution - values.loc[other]).to_numpy(float)
            rng = np.random.default_rng(SEED)
            draws = rng.choice(
                difference, size=(BOOTSTRAP_DRAWS, len(difference)), replace=True
            ).mean(axis=1)
            rows.append(
                {
                    "comparison": comparison,
                    "contrast": f"Resolution minus {other}",
                    "mean_difference": float(difference.mean()),
                    "bootstrap_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_high": float(np.quantile(draws, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def temporal_robustness(type_scores: pd.DataFrame) -> pd.DataFrame:
    resolution = type_scores[type_scores["instrument"].eq("Resolution")].pivot(
        index="meeting", columns="model", values="allocation_log_score"
    )
    difference = (
        resolution["history + direct + network attention"]
        - resolution["history + direct attention"]
    )
    rows = []
    for label, meetings in (
        ("ATCM 29-38", range(29, 39)),
        ("ATCM 39-47", range(39, 48)),
    ):
        values = difference.loc[list(meetings)].to_numpy(float)
        rng = np.random.default_rng(SEED)
        draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                "check": "period",
                "setting": label,
                "mean_difference": float(values.mean()),
                "bootstrap_low": float(np.quantile(draws, 0.025)),
                "bootstrap_high": float(np.quantile(draws, 0.975)),
            }
        )
    values = difference.to_numpy(float)
    for block in (1, 2, 3, 4):
        rng = np.random.default_rng(SEED + block)
        starts = np.arange(len(values) - block + 1)
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled = []
            while len(sampled) < len(values):
                start = int(rng.choice(starts))
                sampled.extend(values[start : start + block])
            draws.append(float(np.mean(sampled[: len(values)])))
        rows.append(
            {
                "check": "moving-block bootstrap",
                "setting": f"block {block}",
                "mean_difference": float(values.mean()),
                "bootstrap_low": float(np.quantile(draws, 0.025)),
                "bootstrap_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def other_activity_control(panel: pd.DataFrame) -> pd.DataFrame:
    output = "resolution_mass"
    focal_features = model_features(output)
    specifications = {
        "direct attention": focal_features,
        "direct + all other papers": [*focal_features, "other_papers"],
        "direct + network attention": [
            *focal_features,
            f"neighbor_papers_k{PRIMARY_NETWORK_K}",
        ],
        "direct + network + all other papers": [
            *focal_features,
            f"neighbor_papers_k{PRIMARY_NETWORK_K}",
            "other_papers",
        ],
    }
    scores = {
        name: forecast_meetings(panel, output, features)
        for name, features in specifications.items()
    }
    baseline = scores["direct attention"]
    return pd.DataFrame(
        [
            {"model": name, **paired_summary(candidate, baseline)}
            for name, candidate in scores.items()
            if name != "direct attention"
        ]
    )


def measure_recurrence_ablation(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data[f"other_output_history_{PRIMARY_HISTORY}"] = (
        data[f"decision_mass_history_{PRIMARY_HISTORY}"]
        + data[f"resolution_mass_history_{PRIMARY_HISTORY}"]
    )
    data[f"all_output_history_{PRIMARY_HISTORY}"] = (
        data[f"measure_mass_history_{PRIMARY_HISTORY}"]
        + data[f"other_output_history_{PRIMARY_HISTORY}"]
    )
    specifications = {
        "prior Measures": [f"measure_mass_history_{PRIMARY_HISTORY}"],
        "prior Decisions + Resolutions": [
            f"other_output_history_{PRIMARY_HISTORY}"
        ],
        "all prior outputs": [f"all_output_history_{PRIMARY_HISTORY}"],
        "prior Measures + other outputs": [
            f"measure_mass_history_{PRIMARY_HISTORY}",
            f"other_output_history_{PRIMARY_HISTORY}",
        ],
        "direct attention only": [
            f"paper_history_{PRIMARY_ATTENTION}",
            f"actor_reach_{PRIMARY_ATTENTION}",
            "paper_count",
            "current_actor_reach",
        ],
        "direct + network attention only": [
            f"paper_history_{PRIMARY_ATTENTION}",
            f"actor_reach_{PRIMARY_ATTENTION}",
            "paper_count",
            "current_actor_reach",
            f"neighbor_papers_k{PRIMARY_NETWORK_K}",
        ],
        "prior Measures + direct + network attention": [
            f"measure_mass_history_{PRIMARY_HISTORY}",
            f"paper_history_{PRIMARY_ATTENTION}",
            f"actor_reach_{PRIMARY_ATTENTION}",
            "paper_count",
            "current_actor_reach",
            f"neighbor_papers_k{PRIMARY_NETWORK_K}",
        ],
    }
    scores = {
        name: forecast_meetings(data, "measure_mass", features)
        for name, features in specifications.items()
    }
    baseline = scores["prior Measures"]
    return pd.DataFrame(
        [
            {
                "model": name,
                "mean_score": float(table["allocation_log_score"].mean()),
                **(
                    {
                        "mean_difference_vs_prior_measures": 0.0,
                        "bootstrap_low": 0.0,
                        "bootstrap_high": 0.0,
                        "meetings_improved": 0,
                        "meetings": int(len(table)),
                    }
                    if name == "prior Measures"
                    else {
                        "mean_difference_vs_prior_measures": paired_summary(
                            table, baseline
                        )["mean_difference"],
                        "bootstrap_low": paired_summary(table, baseline)[
                            "bootstrap_low"
                        ],
                        "bootstrap_high": paired_summary(table, baseline)[
                            "bootstrap_high"
                        ],
                        "meetings_improved": paired_summary(table, baseline)[
                            "meetings_improved"
                        ],
                        "meetings": paired_summary(table, baseline)["meetings"],
                    }
                ),
            }
            for name, table in scores.items()
        ]
    )


def measure_content_summary() -> dict:
    measures = load_official_regular_outputs()
    measures = measures[measures["instrument"].eq("Measure")]
    area = measures["official_categories"].map(
        lambda values: "Area protection and management" in values
    )
    management = measures["title"].str.contains("management plan", case=False)
    revised = measures["title"].str.contains("revised management plan", case=False)
    return {
        "measures": int(len(measures)),
        "official_area_category_count": int(area.sum()),
        "official_area_category_share": float(area.mean()),
        "management_plan_count": int(management.sum()),
        "management_plan_share": float(management.mean()),
        "revised_management_plan_count": int(revised.sum()),
        "revised_management_plan_share": float(revised.mean()),
    }


def resolution_pair(
    panel: pd.DataFrame,
    history: int = PRIMARY_HISTORY,
    attention: int = PRIMARY_ATTENTION,
    alpha: float = PRIMARY_ALPHA,
    neighbor: str = f"neighbor_papers_k{PRIMARY_NETWORK_K}",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = "resolution_mass"
    focal = forecast_meetings(
        panel, output, model_features(output, history, attention), alpha
    )
    nearby = forecast_meetings(
        panel,
        output,
        model_features(output, history, attention, neighbor=neighbor),
        alpha,
    )
    return focal, nearby


def robustness_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon_rows = []
    for history in HISTORY_WINDOWS:
        for attention in ATTENTION_WINDOWS:
            focal, nearby = resolution_pair(panel, history=history, attention=attention)
            horizon_rows.append(
                {"history": history, "attention": attention, **paired_summary(nearby, focal)}
            )

    alpha_rows = []
    for alpha in ALPHAS:
        focal, nearby = resolution_pair(panel, alpha=alpha)
        alpha_rows.append({"alpha": alpha, **paired_summary(nearby, focal)})

    network_rows = []
    for k in NETWORK_K:
        focal, nearby = resolution_pair(panel, neighbor=f"neighbor_papers_k{k}")
        network_rows.append({"k": k, **paired_summary(nearby, focal)})
    return pd.DataFrame(horizon_rows), pd.DataFrame(alpha_rows), pd.DataFrame(network_rows)


def leave_one_category_out(panel: pd.DataFrame) -> pd.DataFrame:
    def one(topic: str) -> dict:
        retained = panel[~panel["topic"].eq(topic)].copy()
        focal, nearby = resolution_pair(retained)
        return {"excluded_category": topic, **paired_summary(nearby, focal)}

    rows = Parallel(n_jobs=8)(delayed(one)(topic) for topic in sorted(panel["topic"].unique()))
    return pd.DataFrame(rows)


def geometry_null(panel: pd.DataFrame, exclude_area: bool) -> pd.DataFrame:
    topics = sorted(panel["topic"].unique())
    meetings = sorted(panel["meeting"].unique())
    relations = paper_family_relations(variants()["fractional_multilabel"])
    phi_by_meeting = family_phi_by_meeting(relations, topics, meetings)
    weights_by_meeting = {
        int(meeting): local_weight_matrix(
            phi_by_meeting[int(meeting)], k=PRIMARY_NETWORK_K
        )
        for meeting in meetings
    }
    papers_by_meeting = {
        int(meeting): (
            group.set_index("topic")["paper_count"]
            .reindex(topics, fill_value=0.0)
            .to_numpy(float)
        )
        for meeting, group in panel.groupby("meeting")
    }
    topic_index = {topic: index for index, topic in enumerate(topics)}
    retained = panel if not exclude_area else panel[~panel["topic"].eq(AREA_CATEGORY)]
    focal, observed = resolution_pair(retained)
    observed_difference = paired_summary(observed, focal)["mean_difference"]

    def one(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(topics))
        permuted = panel.copy()
        permuted["random_neighbor"] = 0.0
        for meeting, row_index in permuted.groupby("meeting").groups.items():
            matrix = weights_by_meeting[int(meeting)]
            permuted_matrix = matrix[np.ix_(permutation, permutation)]
            nearby = permuted_matrix @ papers_by_meeting[int(meeting)]
            indices = list(row_index)
            permuted.loc[indices, "random_neighbor"] = [
                nearby[topic_index[topic]] for topic in permuted.loc[indices, "topic"]
            ]
        if exclude_area:
            permuted = permuted[~permuted["topic"].eq(AREA_CATEGORY)]
        _, random_nearby = resolution_pair(permuted, neighbor="random_neighbor")
        return {
            "permutation": seed,
            "exclude_area": exclude_area,
            "mean_difference_vs_direct": paired_summary(random_nearby, focal)[
                "mean_difference"
            ],
            "observed_difference_vs_direct": observed_difference,
        }

    return pd.DataFrame(
        Parallel(n_jobs=8)(delayed(one)(seed) for seed in range(PERMUTATIONS))
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = add_features()
    pretest_selection = select_pretest_specifications(panel)
    pretest_selection.to_csv(PRETEST_SELECTION_PATH, index=False)
    type_scores, type_summary = type_comparison(panel)
    type_scores.to_csv(TYPE_SCORES_PATH, index=False)
    type_summary.to_csv(TYPE_SUMMARY_PATH, index=False)
    instrument_contrasts(type_scores).to_csv(TYPE_CONTRAST_PATH, index=False)
    temporal_robustness(type_scores).to_csv(TEMPORAL_PATH, index=False)
    other_activity_control(panel).to_csv(OTHER_ACTIVITY_PATH, index=False)
    measure_recurrence_ablation(panel).to_csv(MEASURE_RECURRENCE_PATH, index=False)
    MEASURE_CONTENT_PATH.write_text(
        json.dumps(measure_content_summary(), indent=2) + "\n"
    )
    composition = type_category_composition(panel)
    composition.to_csv(TYPE_COMPOSITION_PATH, index=False)
    dominant_exclusion = dominant_category_exclusion(panel, composition)
    dominant_exclusion.to_csv(DOMINANT_EXCLUSION_PATH, index=False)

    horizons, alphas, network_k = robustness_tables(panel)
    horizons.to_csv(HORIZON_PATH, index=False)
    alphas.to_csv(ALPHA_PATH, index=False)
    network_k.to_csv(NETWORK_K_PATH, index=False)

    leave_out = leave_one_category_out(panel)
    leave_out.to_csv(LEAVE_ONE_OUT_PATH, index=False)

    fine_panel, fine_features = add_fine_space_projection(panel)
    fine_rows = []
    fine_focal, _ = resolution_pair(fine_panel)
    for feature in fine_features:
        _, fine_nearby = resolution_pair(fine_panel, neighbor=feature)
        fine_rows.append({"projection": feature, **paired_summary(fine_nearby, fine_focal)})
    fine_projection = pd.DataFrame(fine_rows)
    fine_projection.to_csv(FINE_PROJECTION_PATH, index=False)

    null = pd.concat(
        [geometry_null(panel, False), geometry_null(panel, True)], ignore_index=True
    )
    null.to_csv(GEOMETRY_NULL_PATH, index=False)

    primary = type_summary[
        type_summary["instrument"].eq("Resolution")
        & type_summary["comparison"].eq("network attention vs direct attention")
    ].iloc[0]
    no_area_panel = panel[~panel["topic"].eq(AREA_CATEGORY)]
    no_area_focal, no_area_nearby = resolution_pair(no_area_panel)
    no_area = paired_summary(no_area_nearby, no_area_focal)
    null_summary = {}
    for exclusion, group in null.groupby("exclude_area"):
        observed = float(group["observed_difference_vs_direct"].iloc[0])
        values = group["mean_difference_vs_direct"].to_numpy(float)
        null_summary[str(bool(exclusion)).lower()] = {
            "observed_difference": observed,
            "null_mean": float(values.mean()),
            "null_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
            "permutation_p_lower_tail": float(
                (1 + np.sum(values <= observed)) / (1 + len(values))
            ),
        }
    summary = {
        "design": "settings selected on pooled ATCM 25-28 forecasts, then frozen for rolling-origin ATCM 29-47 output-type forecasts",
        "primary_specification": {
            "output_history_meetings": PRIMARY_HISTORY,
            "prior_direct_attention_meetings": PRIMARY_ATTENTION,
            "current_meeting_direct_attention": True,
            "network_attention": "current papers in all 14 non-focal broad categories, weighted by a map built before the meeting",
            "alpha": PRIMARY_ALPHA,
            "selection_meetings": [PRETEST_START, PRETEST_END],
        },
        "resolution_network_vs_direct": primary.to_dict(),
        "resolution_network_vs_direct_without_area_category": no_area,
        "geometry_permutation_null": null_summary,
        "leave_one_category_out": {
            "minimum_difference": float(leave_out["mean_difference"].min()),
            "maximum_difference": float(leave_out["mean_difference"].max()),
            "exclusions_with_improvement": int((leave_out["mean_difference"] < 0).sum()),
            "total_exclusions": int(len(leave_out)),
        },
        "fine_space_projection": {
            "configurations_with_improvement": int(
                (fine_projection["mean_difference"] < 0).sum()
            ),
            "total_configurations": int(len(fine_projection)),
            "difference_range": [
                float(fine_projection["mean_difference"].min()),
                float(fine_projection["mean_difference"].max()),
            ],
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
