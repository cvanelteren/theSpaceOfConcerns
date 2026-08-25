#!/usr/bin/env python3
"""Make official ATS instrument categories the primary output allocation.

The official instrument register supplies a Category field for every regular
ATCM Measure, Decision, and Resolution in the study period.  This script keeps
the held-meeting-out title allocation as a sensitivity, replaces the primary
output weights with equal fractional weights across official categories, and
rebuilds every existing concern-by-meeting formal-output panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import analyze_attention_accumulation as accumulation
from scripts import analyze_attention_to_outcomes as outcomes_analysis


ROOT = Path(__file__).resolve().parents[1]
OUTROOT = ROOT / "output" / "category_treatment_comparison"
TREATMENTS = ("inferred_primary", "fractional_multilabel")


def preserve_title_sensitivity(directory: Path) -> None:
    pairs = (
        ("outcome_topic_predictions.csv", "outcome_topic_predictions_title_model.csv"),
        ("outcome_topic_probabilities.csv", "outcome_topic_probabilities_title_model.csv"),
    )
    for primary_name, sensitivity_name in pairs:
        primary = directory / primary_name
        sensitivity = directory / sensitivity_name
        if not primary.exists():
            continue
        frame = pd.read_csv(primary)
        if "allocation_source" not in frame.columns or not frame[
            "allocation_source"
        ].eq("official_ats_instrument_category").all():
            frame.to_csv(sensitivity, index=False)


def main() -> None:
    outcomes = outcomes_analysis.load_outcomes()
    allocations: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for treatment in TREATMENTS:
        directory = OUTROOT / treatment
        preserve_title_sensitivity(directory)
        # Topics are read from the already completed concern--meeting panel so
        # that the allocation follows the exact vocabulary/order used by that
        # category treatment.
        panel_path = (
            OUTROOT
            / "formal_outputs"
            / f"attention_{treatment}__coding_{treatment}"
            / "topic_meeting_panel.csv"
        )
        topics = sorted(pd.read_csv(panel_path, usecols=["topic"])["topic"].unique())
        predictions, probabilities, metrics = (
            outcomes_analysis.official_output_allocations(outcomes, topics)
        )
        predictions.to_csv(directory / "outcome_topic_predictions.csv", index=False)
        probabilities.to_csv(directory / "outcome_topic_probabilities.csv", index=False)
        (directory / "official_output_allocation_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        allocations[treatment] = (predictions, probabilities)

    for attention_treatment in TREATMENTS:
        for coding_treatment in TREATMENTS:
            directory = (
                OUTROOT
                / "formal_outputs"
                / f"attention_{attention_treatment}__coding_{coding_treatment}"
            )
            panel_path = directory / "topic_meeting_panel.csv"
            panel = pd.read_csv(panel_path)
            topics = sorted(panel["topic"].unique())
            meetings = sorted(panel["meeting"].astype(int).unique())
            base_panel = panel.copy()
            predictions, probabilities = allocations[coding_treatment]
            output_mass = outcomes_analysis.outcome_mass_meeting_panel(
                predictions, probabilities, topics, meetings
            )
            replace = [
                column
                for column in output_mass.columns
                if column not in {"topic", "meeting"}
            ]
            panel = panel.drop(columns=[column for column in replace if column in panel])
            panel = panel.merge(output_mass, on=["topic", "meeting"], how="left")
            panel[replace] = panel[replace].fillna(0.0)
            panel.to_csv(panel_path, index=False)
            stocks = accumulation.add_attention_stocks(panel)
            models = accumulation.fit_all_models(stocks, {})
            models.to_csv(directory / "attention_accumulation_models.csv", index=False)

            title_predictions_path = (
                OUTROOT
                / coding_treatment
                / "outcome_topic_predictions_title_model.csv"
            )
            title_probabilities_path = (
                OUTROOT
                / coding_treatment
                / "outcome_topic_probabilities_title_model.csv"
            )
            if title_predictions_path.exists() and title_probabilities_path.exists():
                title_output_mass = outcomes_analysis.outcome_mass_meeting_panel(
                    pd.read_csv(title_predictions_path),
                    pd.read_csv(title_probabilities_path),
                    topics,
                    meetings,
                )
                title_panel = base_panel.drop(
                    columns=[column for column in replace if column in base_panel]
                ).merge(title_output_mass, on=["topic", "meeting"], how="left")
                title_panel[replace] = title_panel[replace].fillna(0.0)
                title_panel.to_csv(
                    directory / "topic_meeting_panel_title_model.csv", index=False
                )
                title_models = accumulation.fit_all_models(
                    accumulation.add_attention_stocks(title_panel), {}
                )
                title_models.to_csv(
                    directory / "attention_accumulation_models_title_model.csv",
                    index=False,
                )

    summary_path = OUTROOT / "major_results_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        formal = summary.setdefault("formal_outputs", {})
        previous = formal.pop("classifier_treatment_agreement", None)
        if previous is not None:
            formal["title_classifier_treatment_agreement"] = previous
        formal["primary_output_allocation"] = {
            "source": "official ATS instrument Category field",
            "n_outputs": int(len(outcomes)),
            "n_categories": int(
                len(
                    {
                        value
                        for values in outcomes["official_categories"]
                        for value in values
                    }
                )
            ),
            "n_multi_category_outputs": int(
                outcomes["official_categories"].map(len).gt(1).sum()
            ),
        }
        formal["official_output_allocation_treatment_agreement"] = {
            "n_outputs": int(len(outcomes)),
            "top1_agreement": 1.0,
            "reason": "official output allocation is independent of paper category treatment",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Applied official ATS instrument categories to all formal-output panels")


if __name__ == "__main__":
    main()
