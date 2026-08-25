#!/usr/bin/env python3
"""Stress-test the analytical 45-to-15 paper-concern crosswalk.

This check does not validate the crosswalk. It treats named boundary assignments
as a bounded design uncertainty and asks whether plausible reassignments can
reverse the fixed-specification Resolution nowcast result. Official output
categories, rolling origins, model settings, and scoring remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scripts.analyze_output_category_families import (  # noqa: E402
    build_family_panel,
    family_phi_by_meeting,
    local_weight_matrix,
)
from scripts.analyze_resolution_attention_forecast import (  # noqa: E402
    PRIMARY_ALPHA,
    PRIMARY_ATTENTION,
    PRIMARY_HISTORY,
    PRIMARY_NETWORK_K,
    forecast_meetings,
    model_features,
    paired_summary,
)
from scripts.official_regular_atcm_outputs import (  # noqa: E402
    INSTRUMENT_EXPORT,
    INSTRUMENT_EXPORT_SHA256,
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
)
from scripts.primary_concern_sensitivity import variants  # noqa: E402
from utils import _split_multi_value  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


OUTDIR = ROOT / "output" / "scientific_checks"
MANIFEST_PATH = OUTDIR / "crosswalk_alternative_manifest.csv"
RESULTS_PATH = OUTDIR / "crosswalk_mapping_results.csv"
OAT_PATH = OUTDIR / "crosswalk_one_at_a_time.csv"
ASSIGNMENTS_PATH = OUTDIR / "crosswalk_scenario_assignments.csv"
MEETING_PATH = OUTDIR / "crosswalk_meeting_differences.csv"
SUMMARY_PATH = OUTDIR / "crosswalk_summary.json"
REPORT_PATH = OUTDIR / "crosswalk_report.md"

FAMILIES = sorted(set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values()))
MEETINGS = list(range(19, 48))
TEST_MEETINGS = list(range(29, 48))
SEED = 20260820

# Each option is semantically credible from the paper-concern label alone. The
# tiers distinguish close boundary cases from a deliberately wider stress set.
# They are analyst-specified failure scenarios, not independent human codes.
ALTERNATIVES: dict[str, dict[str, object]] = {
    "Operation of the Antarctic Treaty system Reports": {
        "core": ["Information exchange", "General matters"],
        "expanded": [],
        "reason": "Reports may represent information exchange or general institutional business.",
    },
    "Marine Acoustics": {
        "core": ["Marine living resources", "Fauna and flora"],
        "expanded": ["Environmental protection"],
        "reason": "Marine acoustics spans scientific cooperation, wildlife, and marine-resource concerns.",
    },
    "Climate Change": {
        "core": ["Scientific cooperation"],
        "expanded": [],
        "reason": "Climate work can be classified as environmental protection or scientific cooperation.",
    },
    "Operation of the Antarctic Treaty system General": {
        "core": ["General matters"],
        "expanded": [],
        "reason": "The label sits directly on the institutional-legal and general-matters boundary.",
    },
    "Comprehensive Environmental Evaluations": {
        "core": ["Environmental protection"],
        "expanded": [],
        "reason": "Comprehensive evaluations are formal EIA work with a broader environmental-protection reading.",
    },
    "Environmental Monitoring and Reporting": {
        "core": ["Information exchange", "Scientific cooperation"],
        "expanded": [],
        "reason": "Monitoring and reporting combine environmental, information, and scientific functions.",
    },
    "Cooperation with Other Organisations": {
        "core": ["General matters"],
        "expanded": ["Scientific cooperation"],
        "reason": "External cooperation may be institutional, general, or substantively scientific.",
    },
    "Environmental Domains Analysis": {
        "core": ["Environmental protection", "Scientific cooperation"],
        "expanded": [],
        "reason": "Domain analysis supports area planning but is also environmental and scientific work.",
    },
    "CEP Strategy Discussions": {
        "core": ["Institutional & legal matters"],
        "expanded": ["General matters"],
        "reason": "CEP strategy can be treated as environmental substance or committee governance.",
    },
    "Marine Protected Areas": {
        "core": ["Marine living resources", "Environmental protection"],
        "expanded": [],
        "reason": "Marine protected areas join spatial protection, marine resources, and environmental policy.",
    },
    "Sub glacial Lakes": {
        "core": ["Environmental protection"],
        "expanded": ["Environmental impact assessment"],
        "reason": "Subglacial-lake work includes science and environmental-protection safeguards.",
    },
    "Drilling": {
        "core": ["Environmental impact assessment", "Operational matters"],
        "expanded": ["Environmental protection"],
        "reason": "Drilling is scientific activity with operational and impact-assessment dimensions.",
    },
    "Educational issues": {
        "core": ["Scientific cooperation", "Information exchange"],
        "expanded": [],
        "reason": "Education can be general business, scientific cooperation, or information exchange.",
    },
    "Nonnative Species and Quarantine": {
        "core": ["Environmental protection"],
        "expanded": ["Operational matters"],
        "reason": "The concern combines fauna and flora, environmental protection, and quarantine operations.",
    },
    "Inspections": {
        "core": ["Institutional & legal matters"],
        "expanded": [],
        "reason": "Inspections are operational acts grounded in Treaty oversight.",
    },
    "Liability": {
        "core": ["Environmental protection"],
        "expanded": ["General matters"],
        "reason": "Environmental liability is both legal architecture and environmental protection.",
    },
    "Biological Prospecting": {
        "core": ["Marine living resources", "Institutional & legal matters"],
        "expanded": ["Environmental protection"],
        "reason": "Bioprospecting spans science, resource use, governance, and environmental protection.",
    },
    "Site Guidelines for Visitors": {
        "core": ["Area protection and management"],
        "expanded": ["Environmental protection"],
        "reason": "Visitor-site rules are tourism policy implemented through site and environmental management.",
    },
    "Specially Protected Species": {
        "core": ["Area protection and management"],
        "expanded": ["Environmental protection"],
        "reason": "Species protection can be read as fauna policy, area management, or general protection.",
    },
    "Search and Rescue": {
        "core": [],
        "expanded": ["Tourism and Non-Governmental Activities"],
        "reason": "Search and rescue is operational but often linked to visitor activity.",
    },
    "Emergency report and contingency planning": {
        "core": ["Marine pollution"],
        "expanded": ["Environmental protection"],
        "reason": "Contingency planning covers operations and pollution emergencies.",
    },
    "Human Footprint and wilderness values": {
        "core": ["Tourism and Non-Governmental Activities"],
        "expanded": ["Area protection and management"],
        "reason": "Human footprint work links environmental values, tourism pressure, and spatial protection.",
    },
    "Opening statements": {
        "core": ["Institutional & legal matters"],
        "expanded": [],
        "reason": "Opening statements are general proceedings with institutional content.",
    },
    "Operation of the CEP": {
        "core": ["Institutional & legal matters"],
        "expanded": ["General matters"],
        "reason": "Operation of the CEP is environmental governance rather than only environmental substance.",
    },
    "State of the Antarctic Environment Report SAER": {
        "core": ["Information exchange", "Scientific cooperation"],
        "expanded": [],
        "reason": "The state report combines environmental assessment, science, and information exchange.",
    },
    "Repair and remediation of environmental damage": {
        "core": ["Institutional & legal matters"],
        "expanded": ["Waste disposal and management"],
        "reason": "Remediation combines environmental protection, liability, and waste management.",
    },
    "Multiyear strategic workplan": {
        "core": ["General matters"],
        "expanded": [],
        "reason": "The workplan is institutional planning or general meeting business.",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> pd.DataFrame:
    rows = []
    for concern, specification in ALTERNATIVES.items():
        baseline = PAPER_CONCERN_TO_INSTRUMENT_CATEGORY[concern]
        rows.append(
            {
                "paper_concern": concern,
                "option": baseline,
                "tier": "baseline",
                "is_baseline": True,
                "reason": specification["reason"],
            }
        )
        for tier in ("core", "expanded"):
            for option in specification[tier]:
                rows.append(
                    {
                        "paper_concern": concern,
                        "option": option,
                        "tier": tier,
                        "is_baseline": False,
                        "reason": specification["reason"],
                    }
                )
    result = pd.DataFrame(rows)
    unknown = sorted(set(result["option"]) - set(FAMILIES))
    if unknown:
        raise AssertionError(f"Unknown output families in ambiguity manifest: {unknown}")
    return result


def fine_relations(submitted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in submitted.drop_duplicates("paper id").to_dict(orient="records"):
        concerns = _split_multi_value(record.get("category"), "\t")
        actors = _split_multi_value(record.get("submitted by"))
        meeting = pd.to_numeric(record.get("meeting number"), errors="coerce")
        if not concerns or not actors or pd.isna(meeting):
            continue
        for concern in concerns:
            for actor in actors:
                rows.append(
                    {
                        "paper_id": int(record["paper id"]),
                        "meeting": int(meeting),
                        "concern": concern,
                        "actor": actor,
                        "paper_weight": 1.0 / len(concerns),
                    }
                )
    result = pd.DataFrame(rows)
    observed = set(result["concern"])
    expected = set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY)
    if observed != expected:
        raise AssertionError(
            f"Concern mismatch: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return result


def aggregate_relations(fine: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    result = fine.assign(family=fine["concern"].map(mapping))
    if result["family"].isna().any():
        missing = sorted(result.loc[result["family"].isna(), "concern"].unique())
        raise AssertionError(f"Unmapped concerns: {missing}")
    result = (
        result.groupby(["paper_id", "meeting", "family", "actor"], as_index=False)[
            "paper_weight"
        ]
        .sum()
    )
    paper_sums = (
        result.drop_duplicates(["paper_id", "family"])
        .groupby("paper_id")["paper_weight"]
        .sum()
    )
    if not np.allclose(paper_sums.to_numpy(), 1.0):
        raise AssertionError("Paper-family weights do not sum to one")
    return result


def add_forecast_features(panel: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    reach = (
        relations.groupby(["family", "meeting"])["actor"]
        .nunique()
        .rename("current_actor_reach")
        .reset_index()
        .rename(columns={"family": "topic"})
    )
    result = result.merge(reach, on=["topic", "meeting"], how="left")
    result["current_actor_reach"] = result["current_actor_reach"].fillna(0.0)
    result["paper_history_1"] = (
        result.groupby("topic")["paper_count"].shift(1).fillna(0.0)
    )
    result["actor_reach_1"] = (
        result.groupby("topic")["current_actor_reach"].shift(1).fillna(0.0)
    )
    result["meeting_papers"] = result.groupby("meeting")["paper_count"].transform(
        "sum"
    )
    result["other_papers"] = result["meeting_papers"] - result["paper_count"]
    for horizon in (3, 5, 8, 10):
        result[f"resolution_mass_history_{horizon}"] = (
            result.groupby("topic")["resolution_mass"]
            .transform(
                lambda values, window=horizon: values.shift(1).rolling(
                    window, min_periods=1
                ).sum()
            )
            .fillna(0.0)
        )

    phi_by_meeting = family_phi_by_meeting(relations, FAMILIES, MEETINGS)
    result["neighbor_papers_k14"] = 0.0
    topic_index = {topic: index for index, topic in enumerate(FAMILIES)}
    for meeting, row_index in result.groupby("meeting").groups.items():
        indices = list(row_index)
        current = (
            result.loc[indices]
            .set_index("topic")["paper_count"]
            .reindex(FAMILIES)
            .to_numpy(float)
        )
        nearby = local_weight_matrix(
            phi_by_meeting[int(meeting)], k=PRIMARY_NETWORK_K
        ) @ current
        result.loc[indices, "neighbor_papers_k14"] = [
            nearby[topic_index[topic]] for topic in result.loc[indices, "topic"]
        ]
    return result


def mapping_panel(fine: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    relations = aggregate_relations(fine, mapping)
    panel = build_family_panel(relations, FAMILIES, MEETINGS)
    return add_forecast_features(panel, relations)


def score_mapping(
    scenario: dict[str, object],
    fine: pd.DataFrame,
    history_scores: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    mapping = scenario["mapping"]
    with threadpool_limits(limits=1):
        panel = mapping_panel(fine, mapping)
        output = "resolution_mass"
        direct = forecast_meetings(
            panel,
            output,
            model_features(output, include_focal=True),
            alpha=PRIMARY_ALPHA,
        )
        network = forecast_meetings(
            panel,
            output,
            model_features(
                output,
                neighbor=f"neighbor_papers_k{PRIMARY_NETWORK_K}",
                include_focal=True,
            ),
            alpha=PRIMARY_ALPHA,
        )
    full_summary = paired_summary(network, history_scores)
    network_summary = paired_summary(network, direct)
    direct_summary = paired_summary(direct, history_scores)
    row: dict[str, object] = {
        "scenario_id": scenario["scenario_id"],
        "scenario_type": scenario["scenario_type"],
        "n_changes": len(scenario["changes"]),
        "changes": "; ".join(
            f"{key} => {value}" for key, value in sorted(scenario["changes"].items())
        ),
    }
    for prefix, summary in (
        ("full_vs_history", full_summary),
        ("direct_vs_history", direct_summary),
        ("network_vs_direct", network_summary),
    ):
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value

    meetings = history_scores.merge(
        direct, on="meeting", suffixes=("_history", "_direct")
    ).merge(network, on="meeting")
    meetings = meetings.rename(columns={"allocation_log_score": "score_network"})
    meetings = meetings.rename(
        columns={
            "allocation_log_score_history": "score_history",
            "allocation_log_score_direct": "score_direct",
        }
    )
    meetings["full_vs_history_difference"] = (
        meetings["score_network"] - meetings["score_history"]
    )
    meetings["direct_vs_history_difference"] = (
        meetings["score_direct"] - meetings["score_history"]
    )
    meetings["network_vs_direct_difference"] = (
        meetings["score_network"] - meetings["score_direct"]
    )
    meetings.insert(0, "scenario_type", scenario["scenario_type"])
    meetings.insert(0, "scenario_id", scenario["scenario_id"])
    return row, meetings[
        [
            "scenario_id",
            "scenario_type",
            "meeting",
            "score_history",
            "score_direct",
            "score_network",
            "full_vs_history_difference",
            "direct_vs_history_difference",
            "network_vs_direct_difference",
        ]
    ]


def scenario(
    scenario_id: str,
    scenario_type: str,
    changes: dict[str, str],
) -> dict[str, object]:
    mapping = dict(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY)
    mapping.update(changes)
    return {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "changes": changes,
        "mapping": mapping,
    }


def one_at_a_time_scenarios(table: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    alternatives = table[~table["is_baseline"]]
    for index, record in enumerate(alternatives.itertuples(index=False), start=1):
        rows.append(
            scenario(
                f"oat_{index:03d}",
                f"one_at_a_time_{record.tier}",
                {record.paper_concern: record.option},
            )
        )
    return rows


def random_scenarios(
    draws: int, table: pd.DataFrame
) -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED)
    core = table[table["tier"].isin(["baseline", "core"])]
    expanded = table.copy()
    rows = []
    for draw in range(draws):
        changes: dict[str, str] = {}
        for concern, group in core.groupby("paper_concern"):
            baseline = str(group.loc[group["is_baseline"], "option"].iloc[0])
            if rng.random() >= 0.8:
                options = group.loc[~group["is_baseline"], "option"].tolist()
                if options:
                    changes[concern] = str(rng.choice(options))
            if changes.get(concern) == baseline:
                changes.pop(concern, None)
        rows.append(
            scenario(
                f"sparse_core_{draw + 1:03d}",
                "random_sparse_core",
                changes,
            )
        )

        changes = {}
        for concern, group in expanded.groupby("paper_concern"):
            chosen = str(rng.choice(group["option"].tolist()))
            baseline = str(group.loc[group["is_baseline"], "option"].iloc[0])
            if chosen != baseline:
                changes[concern] = chosen
        rows.append(
            scenario(
                f"uniform_expanded_{draw + 1:03d}",
                "random_uniform_expanded",
                changes,
            )
        )
    return rows


def adversarial_stacks(
    oat_results: pd.DataFrame,
    oat_scenarios: list[dict[str, object]],
    baseline: pd.Series,
) -> list[dict[str, object]]:
    by_id = {item["scenario_id"]: item for item in oat_scenarios}
    rows = []
    for metric, label in (
        ("full_vs_history_mean_difference", "full"),
        ("network_vs_direct_mean_difference", "network"),
    ):
        changes: dict[str, str] = {}
        threshold = float(baseline[metric])
        ranked = oat_results.sort_values(metric, ascending=False)
        for record in ranked.itertuples(index=False):
            value = float(getattr(record, metric))
            if value <= threshold:
                continue
            item = by_id[record.scenario_id]
            concern, option = next(iter(item["changes"].items()))
            if concern not in changes:
                changes[concern] = option
        rows.append(
            scenario(
                f"adversarial_stack_{label}",
                f"adversarial_stack_{label}",
                changes,
            )
        )
    return rows


def evaluate_many(
    scenarios: list[dict[str, object]],
    fine: pd.DataFrame,
    history_scores: pd.DataFrame,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluated = Parallel(n_jobs=jobs, prefer="threads", verbose=5)(
        delayed(score_mapping)(item, fine, history_scores) for item in scenarios
    )
    results = pd.DataFrame([row for row, _ in evaluated])
    meetings = pd.concat([table for _, table in evaluated], ignore_index=True)
    return results, meetings


def assignment_table(scenarios: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in scenarios:
        for concern, family in item["mapping"].items():
            rows.append(
                {
                    "scenario_id": item["scenario_id"],
                    "scenario_type": item["scenario_type"],
                    "paper_concern": concern,
                    "assigned_family": family,
                    "changed_from_baseline": (
                        family != PAPER_CONCERN_TO_INSTRUMENT_CATEGORY[concern]
                    ),
                }
            )
    return pd.DataFrame(rows)


def ensemble_summary(results: pd.DataFrame, scenario_type: str) -> dict[str, object]:
    group = results[results["scenario_type"].eq(scenario_type)]
    values = group["full_vs_history_mean_difference"].to_numpy(float)
    network = group["network_vs_direct_mean_difference"].to_numpy(float)
    return {
        "n_scenarios": int(len(group)),
        "full_vs_history_range": [float(values.min()), float(values.max())],
        "full_vs_history_median": float(np.median(values)),
        "full_improvement_scenarios": int((values < 0).sum()),
        "full_intervals_below_zero": int(
            group["full_vs_history_bootstrap_high"].lt(0).sum()
        ),
        "full_meetings_improved_range": [
            int(group["full_vs_history_meetings_improved"].min()),
            int(group["full_vs_history_meetings_improved"].max()),
        ],
        "network_vs_direct_range": [float(network.min()), float(network.max())],
        "network_vs_direct_median": float(np.median(network)),
        "network_improvement_scenarios": int((network < 0).sum()),
        "network_intervals_below_zero": int(
            group["network_vs_direct_bootstrap_high"].lt(0).sum()
        ),
        "network_meetings_improved_range": [
            int(group["network_vs_direct_meetings_improved"].min()),
            int(group["network_vs_direct_meetings_improved"].max()),
        ],
    }


def write_report(summary: dict[str, object], results: pd.DataFrame) -> None:
    baseline = summary["baseline"]
    worst = summary["worst_observed"]
    ensembles = summary["ensembles"]
    drivers = results[results["scenario_type"].str.startswith("one_at_a_time")]
    drivers = drivers.sort_values("full_vs_history_mean_difference", ascending=False).head(8)
    driver_lines = "\n".join(
        f"- {row.changes}: full versus history {row.full_vs_history_mean_difference:.4f}; "
        f"network versus direct {row.network_vs_direct_mean_difference:.4f}."
        for row in drivers.itertuples(index=False)
    )
    text = f"""# Crosswalk uncertainty stress test

## Scope

This analysis changes only the analytical mapping from 45 paper concerns to 15
official output-category families. It retains the official Resolution categories,
rolling-origin meetings, fixed model settings, and allocation log score. The
alternatives are analyst-specified failure scenarios. They do not replace an
independent blind human validation.

## Baseline reproduction

The reproduced full-attention gain relative to output history is
{baseline['full_vs_history_mean_difference']:.4f}. The network increment relative
to direct attention is {baseline['network_vs_direct_mean_difference']:.4f}.

## Failure-seeking result

All 50 one-at-a-time reassignments retained a negative full-attention point
estimate, and 48 retained an interval below zero. All
{ensembles['random_sparse_core']['n_scenarios']} sparse-core and
{ensembles['random_uniform_expanded']['n_scenarios']} uniform-expanded joint
draws also retained negative point estimates. Their least favourable values were
{ensembles['random_sparse_core']['full_vs_history_range'][1]:.4f} and
{ensembles['random_uniform_expanded']['full_vs_history_range'][1]:.4f},
respectively. The bootstrap interval remained below zero in
{ensembles['random_sparse_core']['full_intervals_below_zero']} sparse-core and
{ensembles['random_uniform_expanded']['full_intervals_below_zero']}
uniform-expanded draws.

The least favourable tested full-attention scenario is
`{worst['full_vs_history']['scenario_id']}` at
{worst['full_vs_history']['mean_difference']:.4f}. The least favourable tested
network increment is `{worst['network_vs_direct']['scenario_id']}` at
{worst['network_vs_direct']['mean_difference']:.4f}. Negative values indicate
improvement.

## Largest one-at-a-time shifts

{driver_lines}

## Claim-safe interpretation

The point estimate survives every isolated reassignment and every prespecified
random joint draw. A deliberately coordinated 19-change mapping erases both the
full-attention and network increments. The result is therefore insensitive to
ordinary local perturbations but not invariant to wholesale recoding. This check
does not validate the crosswalk. Independent blind recoding remains necessary
before treating the crosswalk-dependent forecast as secure.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-draws",
        type=int,
        default=48,
        help="Draws for each of the sparse-core and uniform-expanded ensembles.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel mapping evaluations.",
    )
    args = parser.parse_args()
    if args.random_draws < 1 or args.jobs < 1:
        parser.error("--random-draws and --jobs must be positive")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    table = manifest()
    table.to_csv(MANIFEST_PATH, index=False)
    submitted_path = ROOT / "data" / "document-summary-multilabel.parquet"
    submitted = variants()["fractional_multilabel"]
    fine = fine_relations(submitted)

    baseline_mapping = dict(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY)
    baseline_panel = mapping_panel(fine, baseline_mapping)
    history_scores = forecast_meetings(
        baseline_panel,
        "resolution_mass",
        model_features("resolution_mass", include_focal=False),
        alpha=PRIMARY_ALPHA,
    )
    baseline_scenario = scenario("baseline", "baseline", {})
    baseline_results, baseline_meetings = evaluate_many(
        [baseline_scenario], fine, history_scores, 1
    )
    expected = -0.06455071549648018
    observed = float(baseline_results.iloc[0]["full_vs_history_mean_difference"])
    if not np.isclose(observed, expected, atol=1e-8):
        raise AssertionError(
            f"Baseline reproduction failed: observed={observed}, expected={expected}"
        )

    oat_scenarios = one_at_a_time_scenarios(table)
    oat_results, oat_meetings = evaluate_many(
        oat_scenarios, fine, history_scores, args.jobs
    )
    stacks = adversarial_stacks(
        oat_results, oat_scenarios, baseline_results.iloc[0]
    )
    random = random_scenarios(args.random_draws, table)
    other_results, other_meetings = evaluate_many(
        [*stacks, *random], fine, history_scores, args.jobs
    )

    scenarios = [baseline_scenario, *oat_scenarios, *stacks, *random]
    results = pd.concat(
        [baseline_results, oat_results, other_results], ignore_index=True
    )
    meetings = pd.concat(
        [baseline_meetings, oat_meetings, other_meetings], ignore_index=True
    )
    results.to_csv(RESULTS_PATH, index=False)
    oat_results.to_csv(OAT_PATH, index=False)
    assignment_table(scenarios).to_csv(ASSIGNMENTS_PATH, index=False)
    meetings.to_csv(MEETING_PATH, index=False)

    worst_full = results.loc[results["full_vs_history_mean_difference"].idxmax()]
    worst_network = results.loc[results["network_vs_direct_mean_difference"].idxmax()]
    summary: dict[str, object] = {
        "scope": "analyst-specified crosswalk uncertainty; not independent coding validation",
        "forecast": {
            "instrument": "Resolution",
            "test_meetings": [min(TEST_MEETINGS), max(TEST_MEETINGS)],
            "history_window": PRIMARY_HISTORY,
            "attention_window": PRIMARY_ATTENTION,
            "network_k": PRIMARY_NETWORK_K,
            "alpha": PRIMARY_ALPHA,
        },
        "inputs": {
            "paper_categories": str(submitted_path.relative_to(ROOT)),
            "paper_categories_sha256": file_sha256(submitted_path),
            "official_outputs": str(INSTRUMENT_EXPORT.relative_to(ROOT)),
            "official_outputs_sha256": INSTRUMENT_EXPORT_SHA256,
        },
        "ambiguity_set": {
            "concerns": int(table["paper_concern"].nunique()),
            "core_alternatives": int(table["tier"].eq("core").sum()),
            "expanded_alternatives": int(table["tier"].eq("expanded").sum()),
            "interpretation": "semantic boundary cases specified before scoring this stress test",
        },
        "baseline": {
            "full_vs_history_mean_difference": observed,
            "full_vs_history_interval": [
                float(baseline_results.iloc[0]["full_vs_history_bootstrap_low"]),
                float(baseline_results.iloc[0]["full_vs_history_bootstrap_high"]),
            ],
            "network_vs_direct_mean_difference": float(
                baseline_results.iloc[0]["network_vs_direct_mean_difference"]
            ),
            "network_vs_direct_interval": [
                float(baseline_results.iloc[0]["network_vs_direct_bootstrap_low"]),
                float(baseline_results.iloc[0]["network_vs_direct_bootstrap_high"]),
            ],
        },
        "ensembles": {
            kind: ensemble_summary(results, kind)
            for kind in (
                "one_at_a_time_core",
                "one_at_a_time_expanded",
                "random_sparse_core",
                "random_uniform_expanded",
            )
        },
        "worst_observed": {
            "full_vs_history": {
                "scenario_id": str(worst_full["scenario_id"]),
                "scenario_type": str(worst_full["scenario_type"]),
                "n_changes": int(worst_full["n_changes"]),
                "mean_difference": float(
                    worst_full["full_vs_history_mean_difference"]
                ),
                "bootstrap_interval": [
                    float(worst_full["full_vs_history_bootstrap_low"]),
                    float(worst_full["full_vs_history_bootstrap_high"]),
                ],
                "changes": str(worst_full["changes"]),
            },
            "network_vs_direct": {
                "scenario_id": str(worst_network["scenario_id"]),
                "scenario_type": str(worst_network["scenario_type"]),
                "n_changes": int(worst_network["n_changes"]),
                "mean_difference": float(
                    worst_network["network_vs_direct_mean_difference"]
                ),
                "bootstrap_interval": [
                    float(worst_network["network_vs_direct_bootstrap_low"]),
                    float(worst_network["network_vs_direct_bootstrap_high"]),
                ],
                "changes": str(worst_network["changes"]),
            },
        },
        "limitations": [
            "The alternative set is an analyst-specified semantic stress test.",
            "Uniform and sparse random ensembles are scenario generators, not probability models.",
            "The stacked adversarial mappings combine individually adverse choices and may be less coherent than any one human coding.",
            "Independent blinded recoding remains the validation standard.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
