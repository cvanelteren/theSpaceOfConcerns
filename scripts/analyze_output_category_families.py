#!/usr/bin/env python3
"""Relate papers and outputs at the same 15-category resolution.

The paper archive uses 45 fine concerns, while the ATS instrument register uses
15 broad categories.  This analysis aggregates paper weights and the concern
space to those 15 families before comparing focal and neighbouring activity.
It avoids treating a direct subtopic, such as Management Plans, as a neighbour
of its own broad output category, Area protection and management.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.primary_concern_sensitivity import variants
from scripts.official_regular_atcm_outputs import (
    INSTRUMENT_EXPORT,
    INSTRUMENT_EXPORT_SHA256,
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
    load_official_regular_outputs,
)
from scripts import analyze_output_type_interactions as interaction
from utils import _split_multi_value, compute_product_space, get_rca


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "category_treatment_comparison"
PANEL_PATH = OUTDIR / "output_family_topic_meeting_panel.csv"
HORIZONS_PATH = OUTDIR / "output_family_pooled_horizons.csv"
COMMON_SUPPORT_PATH = OUTDIR / "output_family_common_support_horizons.csv"
SENSITIVITY_PATH = OUTDIR / "output_family_exclusion_sensitivity.csv"
CROSSWALK_PATH = OUTDIR / "output_family_crosswalk.csv"
CATEGORY_MASS_PATH = OUTDIR / "output_family_category_mass.csv"
SUMMARY_PATH = OUTDIR / "output_family_summary.json"
HORIZONS = (1, 2, 3, 5, 8, 10)
START_MEETING = 19
END_MEETING = 47
LOCAL_K = 5


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


def paper_family_relations(submitted: pd.DataFrame) -> pd.DataFrame:
    """Return one weighted row per paper, family, meeting, and submitting actor."""
    rows = []
    for record in submitted.drop_duplicates("paper id").to_dict(orient="records"):
        raw_categories = _split_multi_value(record.get("category"), "\t")
        if not raw_categories:
            continue
        family_weight: dict[str, float] = {}
        for category in raw_categories:
            family = PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.get(category)
            if family is None:
                raise KeyError(f"Unmapped paper concern: {category}")
            family_weight[family] = family_weight.get(family, 0.0) + 1.0 / len(
                raw_categories
            )
        actors = _split_multi_value(record.get("submitted by"))
        meeting = pd.to_numeric(record.get("meeting number"), errors="coerce")
        if pd.isna(meeting) or not actors:
            continue
        for family, weight in family_weight.items():
            for actor in actors:
                rows.append(
                    {
                        "paper_id": int(record["paper id"]),
                        "meeting": int(meeting),
                        "family": family,
                        "actor": actor,
                        "paper_weight": weight,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No paper-family relations were constructed")
    paper_sums = (
        result.drop_duplicates(["paper_id", "family"])
        .groupby("paper_id")["paper_weight"]
        .sum()
    )
    if not np.allclose(paper_sums.to_numpy(), 1.0):
        raise AssertionError("Fine-category weights must sum to one per paper")
    return result


def family_phi_by_meeting(
    relations: pd.DataFrame, families: list[str], meetings: list[int]
) -> dict[int, pd.DataFrame]:
    actors = sorted(relations["actor"].unique())
    result = {}
    for meeting in meetings:
        history = relations[relations["meeting"].lt(meeting)]
        counts = (
            history.groupby(["family", "actor"])["paper_weight"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=families, columns=actors, fill_value=0.0)
        )
        phi = compute_product_space(get_rca(counts)).reindex(
            index=families, columns=families, fill_value=0.0
        )
        np.fill_diagonal(phi.values, 1.0)
        result[meeting] = phi
    return result


def paper_attention_panel(
    relations: pd.DataFrame, families: list[str], meetings: list[int]
) -> pd.DataFrame:
    # Sponsor duplication should not multiply the total documentary weight of a
    # co-sponsored paper in the output association.
    papers = relations.drop_duplicates(["paper_id", "family"])
    counts = papers.groupby(["family", "meeting"])["paper_weight"].sum()
    index = pd.MultiIndex.from_product(
        [families, meetings], names=["topic", "meeting"]
    )
    counts.index.names = ["topic", "meeting"]
    return counts.rename("paper_count").reindex(index, fill_value=0.0).reset_index()


def output_family_allocations(
    families: list[str], meetings: list[int], require_complete: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = load_official_regular_outputs().rename(columns={"output_id": "outcome_id"})
    predictions = []
    weights = []
    for record in outcomes.to_dict(orient="records"):
        categories = list(dict.fromkeys(record["official_categories"]))
        weight = 1.0 / len(categories)
        predictions.append(
            {
                **record,
                "topic_top1": categories[0],
                "high_confidence": True,
            }
        )
        for family in families:
            weights.append(
                {
                    "outcome_id": record["outcome_id"],
                    "year": int(record["year"]),
                    "meeting": int(record["meeting"]),
                    "instrument": record["instrument"],
                    "topic": family,
                    "probability": weight if family in categories else 0.0,
                }
            )
    probabilities = pd.DataFrame(weights)
    totals = probabilities.groupby("outcome_id")["probability"].sum().to_numpy()
    if require_complete and not np.allclose(totals, 1.0):
        raise AssertionError("Output-family weights must sum to one")
    if (totals > 1 + 1e-12).any():
        raise AssertionError("Output-family weights cannot exceed one")
    index = pd.MultiIndex.from_product(
        [families, meetings], names=["topic", "meeting"]
    )
    mass = (
        probabilities.groupby(["topic", "meeting"])["probability"]
        .sum()
        .rename("outcome_mass")
        .reindex(index, fill_value=0.0)
        .reset_index()
    )
    for instrument in ("Measure", "Decision", "Resolution"):
        instrument_mass = (
            probabilities[probabilities["instrument"].eq(instrument)]
            .groupby(["topic", "meeting"])["probability"]
            .sum()
            .reindex(index, fill_value=0.0)
            .to_numpy()
        )
        mass[f"{instrument.lower()}_mass"] = instrument_mass
    return pd.DataFrame(predictions), mass


def build_family_panel(
    relations: pd.DataFrame, families: list[str], meetings: list[int]
) -> pd.DataFrame:
    retained = relations[relations["family"].isin(families)].copy()
    attention = paper_attention_panel(retained, families, meetings)
    _, output_mass = output_family_allocations(
        families, meetings, require_complete=len(families) == 15
    )
    phi_by_meeting = family_phi_by_meeting(retained, families, meetings)
    panel = attention.merge(output_mass, on=["topic", "meeting"], how="left").fillna(0)
    topic_index = {topic: index for index, topic in enumerate(families)}
    panel["neighbor_papers"] = 0.0
    for meeting, row_index in panel.groupby("meeting").groups.items():
        indices = list(row_index)
        paper_vector = (
            panel.loc[indices].set_index("topic")["paper_count"]
            .reindex(families)
            .to_numpy(dtype=float)
        )
        neighbor_vector = local_weight_matrix(phi_by_meeting[int(meeting)]) @ paper_vector
        panel.loc[indices, "neighbor_papers"] = [
            neighbor_vector[topic_index[topic]] for topic in panel.loc[indices, "topic"]
        ]
    return panel.sort_values(["topic", "meeting"]).reset_index(drop=True)


def fit_horizons(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    tables = []
    stocked = interaction.add_prior_stocks(panel)
    for horizon in HORIZONS:
        data = interaction.pooled_panel(stocked, horizon)
        fitted, covariance, design = interaction.fit_ppml(data, stacked=False)
        table = interaction.pooled_results(
            data, fitted, covariance, design, horizon, label
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def fit_common_support_horizons(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    """Fit every horizon on the meetings available to the longest window."""
    stocked = interaction.add_prior_stocks(panel)
    horizon_panels = {
        horizon: interaction.pooled_panel(stocked, horizon) for horizon in HORIZONS
    }
    common_meetings = set.intersection(
        *(set(data["meeting"].unique()) for data in horizon_panels.values())
    )
    if not common_meetings:
        raise ValueError("No meetings are shared by every output horizon")
    tables = []
    for horizon, data in horizon_panels.items():
        data = data[data["meeting"].isin(common_meetings)].copy()
        fitted, covariance, design = interaction.fit_ppml(data, stacked=False)
        table = interaction.pooled_results(
            data, fitted, covariance, design, horizon, label
        )
        table["support"] = "common_meetings"
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    submitted = variants()["fractional_multilabel"]
    relations = paper_family_relations(submitted)
    families = sorted(set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values()))
    if len(families) != 15 or set(relations["family"]) != set(families):
        raise AssertionError("The hierarchy must cover all 15 output categories")
    meetings = list(range(START_MEETING, END_MEETING + 1))
    panel = build_family_panel(relations, families, meetings)
    panel.to_csv(PANEL_PATH, index=False)
    (
        panel.groupby("topic", as_index=False)["outcome_mass"]
        .sum()
        .rename(columns={"outcome_mass": "category_weight"})
        .to_csv(CATEGORY_MASS_PATH, index=False)
    )

    horizons = fit_horizons(panel, "15 broad ATS categories")
    horizons.to_csv(HORIZONS_PATH, index=False)
    common_support = fit_common_support_horizons(
        panel, "15 broad ATS categories; common meeting support"
    )
    common_support.to_csv(COMMON_SUPPORT_PATH, index=False)

    sensitivity_tables = []
    exclusions = {
        "none": set(),
        "exclude_general_matters": {"General matters"},
        "exclude_area_protection": {"Area protection and management"},
        "exclude_both": {"General matters", "Area protection and management"},
    }
    for name, excluded in exclusions.items():
        sensitivity_families = [
            family for family in families if family not in excluded
        ]
        sensitivity_panel = build_family_panel(
            relations, sensitivity_families, meetings
        )
        table = fit_horizons(
            sensitivity_panel,
            f"15-category family model; {name}",
        )
        table.insert(0, "sensitivity", name)
        table.insert(1, "n_families", 15 - len(excluded))
        sensitivity_tables.append(table)
    sensitivity = pd.concat(sensitivity_tables, ignore_index=True)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)

    crosswalk = pd.DataFrame(
        sorted(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.items()),
        columns=["paper_concern", "official_output_category_family"],
    )
    crosswalk.to_csv(CROSSWALK_PATH, index=False)

    summary = {
        "paper_count": int(relations["paper_id"].nunique()),
        "n_paper_concerns": len(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY),
        "n_output_category_families": len(families),
        "meetings": [min(meetings), max(meetings)],
        "paper_weighting": "fine-category weights summed within broad family; one total unit per paper",
        "output_weighting": "equal fractional weight across official instrument categories",
        "official_instrument_export": str(INSTRUMENT_EXPORT.relative_to(ROOT)),
        "official_instrument_export_sha256": INSTRUMENT_EXPORT_SHA256,
        "primary_interpretation": "same broad category versus neighbouring broad categories",
        "outputs": {
            "panel": str(PANEL_PATH.relative_to(ROOT)),
            "horizons": str(HORIZONS_PATH.relative_to(ROOT)),
            "common_support_horizons": str(COMMON_SUPPORT_PATH.relative_to(ROOT)),
            "sensitivities": str(SENSITIVITY_PATH.relative_to(ROOT)),
            "crosswalk": str(CROSSWALK_PATH.relative_to(ROOT)),
            "category_mass": str(CATEGORY_MASS_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(horizons.to_string(index=False))
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
