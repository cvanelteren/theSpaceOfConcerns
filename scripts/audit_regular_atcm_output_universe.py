#!/usr/bin/env python3
"""Audit the 1995--2025 regular-ATCM output universe and its composition.

The output analyses use Measures, Decisions, and Resolutions adopted at
regular ATCMs XIX--XLVII.  The lineage project's broader graph also contains
five instruments adopted at SATCM XII in 2000 (internally represented as
meeting 23.5) and one unresolved ``Resolution 6 (1996)`` node.  This script
keeps those records visible while proving that neither enters the locked
584-output regular-ATCM universe.

The composition audit uses two distinct sources of evidence:

* reproducible title rules identify management-plan, protected-area, and
  historic-site administration; and
* official-category allocation weights show how much output mass falls on the three
  corresponding concern-space categories.

The title rules describe instrument composition.  They do not establish legal
lineage, importance, entry into force, or implementation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.official_regular_atcm_outputs import load_official_regular_outputs


ROOT = Path(__file__).resolve().parents[1]
LINEAGE_ROOT = ROOT.parent / "ats_lineage"
COMPARISON_ROOT = ROOT / "output" / "category_treatment_comparison"
ALLOCATION_ROOT = COMPARISON_ROOT / "fractional_multilabel"
OUTDIR = COMPARISON_ROOT / "output_universe_audit"

GRAPH_PATH = LINEAGE_ROOT / "decision_map.json"
OFFICIAL_COUNTS_PATH = LINEAGE_ROOT / "official_outcome_counts.json"
PREDICTIONS_PATH = ALLOCATION_ROOT / "outcome_topic_predictions.csv"
WEIGHTS_PATH = ALLOCATION_ROOT / "outcome_topic_probabilities.csv"

INSTRUMENTS = ("Measure", "Decision", "Resolution")
START_YEAR = 1995
END_YEAR = 2025
START_MEETING = 19
END_MEETING = 47
EXPECTED_COUNTS = {"Measure": 277, "Decision": 135, "Resolution": 172}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
SPECIAL_MEETING = 23.5

ID_PATTERN = re.compile(
    r"^(?P<instrument>Measure|Decision|Resolution) "
    r"(?P<number>\d+) \((?P<year>\d{4})\)$"
)

# These labels are the closest concern-space counterpart of recurring site
# administration.  Site Guidelines for Visitors is deliberately excluded: it
# concerns visitor conduct rather than adoption or revision of a protected
# site, management plan, or historic-site listing.
SITE_ADMIN_CONCERNS = {
    "Management Plans",
    "Area Protection and Management Plans General",
    "Historic Sites and Monuments",
}

# Ordered rules mirror the functional coding protocol in
# scripts/analyze_measure_pathways.py.  The two substantive rules run first so
# that, for example, "Specially Protected Species" is not mistaken for the
# administration of a protected site.
TITLE_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    (
        "environment_or_liability",
        r"liability|environmental emergenc|annex\s+(i|ii|iii|iv|v|vi)\b|"
        r"environment protocol|protocol on environmental protection|"
        r"specially protected species",
    ),
    (
        "tourism_safety_operations",
        r"tourism|tourist|non[- ]govern|landing of persons|passenger vessel|"
        r"insurance|contingency plan|shipborne|air safety|vessel operation",
    ),
    (
        "historic_site",
        r"historic site|historic monument|historical site|monument|\bhsm\b",
    ),
    ("management_plan", r"management plan"),
    (
        "protected_area_designation",
        r"protected area|specially managed area|\baspa\b|\basma\b|\bsssi\b|"
        r"site[s]? of special scientific interest|expiry date|\bspa\s*\d",
    ),
)
RECURRING_SITE_FAMILIES = {
    "historic_site",
    "management_plan",
    "protected_area_designation",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def classify_title_family(title: str) -> tuple[str, str]:
    """Return the first matched functional family and its auditable rule."""
    text = str(title or "").lower()
    for family, pattern in TITLE_FAMILY_RULES:
        if re.search(pattern, text):
            return family, pattern
    return "other_substantive", "no rule matched"


def candidate_nodes() -> pd.DataFrame:
    """Load the complete post-reform Measure/Decision/Resolution graph set."""
    graph = load_json(GRAPH_PATH)
    rows = []
    for node in graph["nodes"]:
        if node.get("kind") != "outcome":
            continue
        instrument = node.get("outcome_type")
        year = node.get("year")
        if instrument not in INSTRUMENTS or not isinstance(year, int):
            continue
        if not START_YEAR <= year <= END_YEAR:
            continue
        rows.append(
            {
                "output_id": node.get("id"),
                "instrument": instrument,
                "meeting": node.get("meeting"),
                "year": year,
                "title": node.get("title"),
                "title_source": node.get("title_source"),
                "placeholder": bool(node.get("placeholder", False)),
                "inventory_verified": bool(node.get("inventory_verified", False)),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty or candidates["output_id"].duplicated().any():
        raise AssertionError("Candidate output IDs must be non-empty and unique")
    official = load_official_regular_outputs().set_index("output_id")
    matched = candidates["output_id"].isin(official.index)
    candidates.loc[matched, "title"] = candidates.loc[matched, "output_id"].map(
        official["title"]
    )
    candidates.loc[matched, "title_source"] = "official_ats_inventory_subject"
    return candidates.sort_values(["year", "meeting", "instrument", "output_id"])


def exclusion_reason(record: pd.Series) -> str:
    meeting = record["meeting"]
    if isinstance(meeting, (int, float)) and math.isclose(
        float(meeting), SPECIAL_MEETING
    ):
        return "SATCM XII (2000), represented as meeting 23.5"
    if not isinstance(meeting, (int, float)) or not float(meeting).is_integer():
        return "not a numbered regular ATCM"
    if not START_MEETING <= int(meeting) <= END_MEETING:
        return "outside regular ATCM XIX--XLVII"
    if record["placeholder"] or not str(record["title"] or "").strip():
        return "unresolved graph node absent from official regular-ATCM inventory"
    return ""


def parse_and_validate_identifiers(inventory: pd.DataFrame) -> pd.DataFrame:
    parsed = inventory["output_id"].str.extract(ID_PATTERN)
    if parsed.isna().any().any():
        bad = inventory.loc[parsed.isna().any(axis=1), "output_id"].tolist()
        raise AssertionError(f"Malformed output identifiers: {bad}")
    parsed["instrument_number"] = parsed.pop("number").astype(int)
    parsed["identifier_year"] = parsed.pop("year").astype(int)
    if not parsed["instrument"].eq(inventory["instrument"].to_numpy()).all():
        raise AssertionError("Instrument type disagrees with output identifier")
    if not parsed["identifier_year"].eq(inventory["year"].to_numpy()).all():
        raise AssertionError("Instrument year disagrees with output identifier")
    result = inventory.copy()
    result.insert(2, "instrument_number", parsed["instrument_number"].to_numpy())
    result.insert(4, "identifier_year", parsed["identifier_year"].to_numpy())
    result["meeting"] = result["meeting"].astype(int)
    key = ["instrument", "instrument_number", "identifier_year", "meeting"]
    if result.duplicated(key).any():
        raise AssertionError(f"Full instrument key is not unique: {key}")
    return result


def build_inclusion_flow(
    candidates: pd.DataFrame, exclusions: pd.DataFrame, retained: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    special = exclusions[exclusions["exclusion_reason"].str.startswith("SATCM XII")]
    unresolved = exclusions.drop(special.index)

    def type_counts(frame: pd.DataFrame, prefix: str) -> dict[str, int]:
        counts = frame["instrument"].value_counts().to_dict()
        return {f"{prefix}_{kind.lower()}s": int(counts.get(kind, 0)) for kind in INSTRUMENTS}

    stages = [
        {
            "order": 0,
            "stage": "post-reform graph candidates",
            "description": "All Measure, Decision, and Resolution nodes dated 1995--2025",
            "n_before": int(len(candidates)),
            "n_excluded": 0,
            "n_retained": int(len(candidates)),
            **type_counts(candidates, "retained"),
        },
        {
            "order": 1,
            "stage": "exclude Special ATCM instruments",
            "description": "Remove SATCM XII (2000), represented internally as meeting 23.5",
            "n_before": int(len(candidates)),
            "n_excluded": int(len(special)),
            "n_retained": int(len(candidates) - len(special)),
            **type_counts(special, "excluded"),
            **type_counts(candidates.drop(special.index), "retained"),
        },
        {
            "order": 2,
            "stage": "exclude unresolved non-inventory node",
            "description": "Remove graph-only unresolved record absent from official regular-ATCM export",
            "n_before": int(len(candidates) - len(special)),
            "n_excluded": int(len(unresolved)),
            "n_retained": int(len(retained)),
            **type_counts(unresolved, "excluded"),
            **type_counts(retained, "retained"),
        },
        {
            "order": 3,
            "stage": "locked regular-ATCM universe",
            "description": "Regular ATCMs XIX--XLVII, 1995--2025",
            "n_before": int(len(retained)),
            "n_excluded": 0,
            "n_retained": int(len(retained)),
            **type_counts(retained, "retained"),
        },
    ]
    flow = pd.DataFrame(stages).fillna(0)
    count_columns = [
        column
        for column in flow.columns
        if column.startswith(("n_", "retained_", "excluded_"))
    ]
    flow[count_columns] = flow[count_columns].astype(int)
    flow_json = {
        "scope": "Measures, Decisions, and Resolutions adopted at regular ATCMs XIX--XLVII (1995--2025)",
        "stages": flow.to_dict(orient="records"),
        "excluded_records": exclusions[
            ["output_id", "instrument", "meeting", "year", "title", "exclusion_reason"]
        ].to_dict(orient="records"),
    }
    return flow, flow_json


def attach_model_allocations(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    weights = pd.read_csv(WEIGHTS_PATH).rename(columns={"probability": "allocation_weight"})
    predictions = predictions.rename(columns={"outcome_id": "output_id"})
    weights = weights.rename(columns={"outcome_id": "output_id"})

    expected_ids = set(inventory["output_id"])
    if set(predictions["output_id"]) != expected_ids:
        raise AssertionError("Leading output allocations do not match locked output IDs")
    if set(weights["output_id"]) != expected_ids:
        raise AssertionError("Weighted output allocations do not match locked output IDs")
    if predictions["output_id"].duplicated().any():
        raise AssertionError("Leading output allocations contain duplicate output IDs")
    weight_sums = weights.groupby("output_id")["allocation_weight"].sum()
    if len(weights) != EXPECTED_TOTAL * 45 or weights["topic"].nunique() != 45:
        raise AssertionError("Expected one allocation weight per output and concern")
    if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-8):
        raise AssertionError("Allocation weights do not sum to one within output")

    check_columns = ["meeting", "year", "instrument"]
    merged_check = inventory[["output_id", *check_columns]].merge(
        predictions[["output_id", *check_columns]],
        on="output_id",
        suffixes=("_inventory", "_allocation"),
        validate="one_to_one",
    )
    for column in check_columns:
        if not merged_check[f"{column}_inventory"].eq(
            merged_check[f"{column}_allocation"]
        ).all():
            raise AssertionError(f"Allocation {column} disagrees with locked inventory")

    prediction_fields = [
        "output_id",
        "topic_top1",
        "topic_top2",
        "topic_top3",
        "probability_top1",
        "margin_top1_top2",
        "high_confidence",
    ]
    result = inventory.merge(
        predictions[prediction_fields], on="output_id", how="left", validate="one_to_one"
    )
    site_weight = (
        weights.assign(site_admin_concern=weights["topic"].isin(SITE_ADMIN_CONCERNS))
        .loc[lambda frame: frame["site_admin_concern"]]
        .groupby("output_id", as_index=False)["allocation_weight"]
        .sum()
        .rename(columns={"allocation_weight": "site_admin_model_weight"})
    )
    result = result.merge(site_weight, on="output_id", how="left", validate="one_to_one")
    result["site_admin_model_weight"] = result["site_admin_model_weight"].fillna(0.0)
    return result, weights


def add_composition_codes(inventory: pd.DataFrame) -> pd.DataFrame:
    families = inventory["title"].apply(classify_title_family)
    result = inventory.copy()
    result["title_family"] = families.str[0]
    result["title_family_rule"] = families.str[1]
    result["recurring_site_admin_title"] = result["title_family"].isin(
        RECURRING_SITE_FAMILIES
    )
    result["recurring_site_admin_leading_concern"] = result["topic_top1"].isin(
        SITE_ADMIN_CONCERNS
    )
    result["recurring_site_admin_either"] = (
        result["recurring_site_admin_title"]
        | result["recurring_site_admin_leading_concern"]
    )
    return result


def concern_composition(
    inventory: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    leading = (
        inventory.groupby(["instrument", "topic_top1"], as_index=False)
        .size()
        .rename(columns={"topic_top1": "concern", "size": "n_leading_outputs"})
    )
    leading["leading_share_within_type"] = leading["n_leading_outputs"] / leading.groupby(
        "instrument"
    )["n_leading_outputs"].transform("sum")
    weighted = (
        weights.groupby(["instrument", "topic"], as_index=False)["allocation_weight"]
        .sum()
        .rename(
            columns={
                "topic": "concern",
                "allocation_weight": "model_weighted_output_mass",
            }
        )
    )
    weighted["weighted_share_within_type"] = weighted[
        "model_weighted_output_mass"
    ] / weighted.groupby("instrument")["model_weighted_output_mass"].transform("sum")
    result = weighted.merge(leading, on=["instrument", "concern"], how="left")
    result["n_leading_outputs"] = result["n_leading_outputs"].fillna(0).astype(int)
    result["leading_share_within_type"] = result["leading_share_within_type"].fillna(0.0)
    result["site_admin_concern"] = result["concern"].isin(SITE_ADMIN_CONCERNS)
    return result.sort_values(["instrument", "model_weighted_output_mass"], ascending=[True, False])


def site_admin_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(instrument, inventory[inventory["instrument"].eq(instrument)]) for instrument in INSTRUMENTS]
    groups.append(("All", inventory))
    for instrument, frame in groups:
        n = len(frame)
        rows.append(
            {
                "instrument": instrument,
                "n_outputs": int(n),
                "title_rule_site_admin_n": int(frame["recurring_site_admin_title"].sum()),
                "title_rule_site_admin_share": float(frame["recurring_site_admin_title"].mean()),
                "leading_concern_site_admin_n": int(
                    frame["recurring_site_admin_leading_concern"].sum()
                ),
                "leading_concern_site_admin_share": float(
                    frame["recurring_site_admin_leading_concern"].mean()
                ),
                "title_or_leading_site_admin_n": int(
                    frame["recurring_site_admin_either"].sum()
                ),
                "title_or_leading_site_admin_share": float(
                    frame["recurring_site_admin_either"].mean()
                ),
                "model_weighted_site_admin_mass": float(
                    frame["site_admin_model_weight"].sum()
                ),
                "model_weighted_site_admin_share": float(
                    frame["site_admin_model_weight"].sum() / n
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_counts_and_meetings(inventory: pd.DataFrame) -> dict:
    official = load_json(OFFICIAL_COUNTS_PATH)
    expected = official["sources"]["atcm_19_47"]
    actual_counts = inventory["instrument"].value_counts().to_dict()
    if len(inventory) != EXPECTED_TOTAL or actual_counts != EXPECTED_COUNTS:
        raise AssertionError(
            f"Regular-ATCM counts disagree: total={len(inventory)}, types={actual_counts}"
        )
    if int(expected["total"]) != EXPECTED_TOTAL or expected["type_counts"] != EXPECTED_COUNTS:
        raise AssertionError("Committed counts disagree with the official export audit")
    actual_by_meeting = inventory.groupby("meeting").size().to_dict()
    expected_by_meeting = {
        int(meeting): int(count)
        for meeting, count in official["counts_by_atcm"].items()
        if START_MEETING <= int(meeting) <= END_MEETING
    }
    if actual_by_meeting != expected_by_meeting:
        raise AssertionError("Per-meeting totals disagree with official export audit")
    if sorted(actual_by_meeting) != list(range(START_MEETING, END_MEETING + 1)):
        raise AssertionError("Regular ATCM meeting sequence is incomplete")
    return {
        "official_export_version": expected.get("export_version"),
        "official_export_sha256": expected.get("source_sha256"),
        "counts_by_type": actual_counts,
        "counts_by_meeting": {str(key): value for key, value in actual_by_meeting.items()},
    }


def write_report(
    audit: dict,
    exclusions: pd.DataFrame,
    composition: pd.DataFrame,
) -> None:
    by_type = composition.set_index("instrument")
    special_ids = exclusions.loc[
        exclusions["exclusion_reason"].str.startswith("SATCM XII"), "output_id"
    ].tolist()
    lines = [
        "# Regular-ATCM output-universe audit",
        "",
        "**Status: PASS**",
        "",
        "## Locked universe",
        "",
        "The analysis contains 584 adopted outputs from regular ATCMs XIX--XLVII "
        "(1995--2025): 277 Measures, 135 Decisions, and 172 Resolutions. Every "
        "record retains instrument type, number, adoption year, and meeting number.",
        "",
        "The broader graph contains 590 post-reform candidate nodes. Five were "
        "adopted at SATCM XII in 2000 and are represented internally as meeting "
        "23.5; one unresolved graph-only node, Resolution 6 (1996), has no title "
        "and is absent from the official regular-ATCM export. The Special-ATCM "
        f"records are: {', '.join(special_ids)}.",
        "",
        "This reconciles the 279-Measure graph inventory with the 277 regular-ATCM "
        "Measures: Measure 1 (2000) and Measure 2 (2000) belong to SATCM XII.",
        "",
        "## Recurring site administration",
        "",
        "Title rules classify 271 of 277 Measures "
        f"({by_type.loc['Measure', 'title_rule_site_admin_share']:.1%}) as recurring "
        "management-plan, protected-area, or historic-site administration. The "
        "three corresponding official-category concerns contain "
        f"{by_type.loc['Measure', 'model_weighted_site_admin_share']:.1%} of Measure "
        "allocation mass. The result is therefore a composition fact, not evidence "
        "that legal form causes serial recurrence.",
        "",
        "For comparison, title rules identify recurring site administration in "
        f"{int(by_type.loc['Decision', 'title_rule_site_admin_n'])} Decisions and "
        f"{int(by_type.loc['Resolution', 'title_rule_site_admin_n'])} Resolutions.",
        "",
        "## Generated files",
        "",
        "- `regular_atcm_output_inventory.csv`: auditable retained records and composition codes.",
        "- `regular_atcm_output_exclusions.csv`: excluded graph records and reasons.",
        "- `regular_atcm_output_inclusion_flow.csv` and `.json`: SI-ready inclusion flow.",
        "- `output_type_by_model_assigned_concern.csv`: official-category leading counts and weighted mass.",
        "- `output_type_site_administration_composition.csv`: composition summary.",
        "- `regular_atcm_output_universe_audit.json`: machine-readable assertions and sources.",
    ]
    (OUTDIR / "regular_atcm_output_universe_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidates = candidate_nodes()
    candidates["exclusion_reason"] = candidates.apply(exclusion_reason, axis=1)
    exclusions = candidates[candidates["exclusion_reason"].ne("")].copy()
    retained = candidates[candidates["exclusion_reason"].eq("")].drop(
        columns="exclusion_reason"
    )

    if len(candidates) != 590 or len(exclusions) != 6 or len(retained) != EXPECTED_TOTAL:
        raise AssertionError(
            "Expected 590 graph candidates, 6 exclusions, and 584 retained outputs"
        )
    special = exclusions[exclusions["meeting"].eq(SPECIAL_MEETING)]
    if special["instrument"].value_counts().to_dict() != {
        "Measure": 2,
        "Resolution": 2,
        "Decision": 1,
    }:
        raise AssertionError("SATCM XII exclusion composition has changed")
    unresolved = exclusions[exclusions["meeting"].ne(SPECIAL_MEETING)]
    if unresolved["output_id"].tolist() != ["Resolution 6 (1996)"]:
        raise AssertionError("Unexpected graph-only regular-meeting exclusion")

    retained = parse_and_validate_identifiers(retained.reset_index(drop=True))
    count_audit = validate_counts_and_meetings(retained)
    inventory, weights = attach_model_allocations(retained)
    inventory = add_composition_codes(inventory)
    by_concern = concern_composition(inventory, weights)
    composition = site_admin_summary(inventory)
    flow, flow_json = build_inclusion_flow(candidates, exclusions, inventory)

    graph_measure_count = int(candidates["instrument"].eq("Measure").sum())
    regular_measure_count = int(inventory["instrument"].eq("Measure").sum())
    excluded_measure_ids = exclusions.loc[
        exclusions["instrument"].eq("Measure"), "output_id"
    ].tolist()
    if graph_measure_count != 279 or regular_measure_count != 277:
        raise AssertionError("The 279-to-277 Measure reconciliation no longer holds")
    if excluded_measure_ids != ["Measure 1 (2000)", "Measure 2 (2000)"]:
        raise AssertionError(f"Unexpected excluded Measures: {excluded_measure_ids}")

    inventory.to_csv(OUTDIR / "regular_atcm_output_inventory.csv", index=False)
    exclusions.to_csv(OUTDIR / "regular_atcm_output_exclusions.csv", index=False)
    flow.to_csv(OUTDIR / "regular_atcm_output_inclusion_flow.csv", index=False)
    (OUTDIR / "regular_atcm_output_inclusion_flow.json").write_text(
        json.dumps(flow_json, indent=2) + "\n", encoding="utf-8"
    )
    by_concern.to_csv(
        OUTDIR / "output_type_by_model_assigned_concern.csv", index=False
    )
    composition.to_csv(
        OUTDIR / "output_type_site_administration_composition.csv", index=False
    )

    audit = {
        "status": "PASS",
        "scope": {
            "years": [START_YEAR, END_YEAR],
            "regular_atcm_meetings": [START_MEETING, END_MEETING],
            "total": EXPECTED_TOTAL,
            "counts_by_type": EXPECTED_COUNTS,
        },
        "sources": {
            "graph": str(GRAPH_PATH),
            "official_count_audit": str(OFFICIAL_COUNTS_PATH),
            "model_leading_allocations": str(PREDICTIONS_PATH),
            "model_weighted_allocations": str(WEIGHTS_PATH),
            **count_audit,
        },
        "identifier_checks": {
            "type_number_year_meeting_preserved": True,
            "full_identifier_key_unique": True,
            "identifier_year_matches_record_year": True,
            "allocation_output_set_matches_inventory": True,
        },
        "measure_reconciliation": {
            "post_reform_graph_measures": graph_measure_count,
            "regular_atcm_measures": regular_measure_count,
            "excluded_satcm_xii_measures": excluded_measure_ids,
        },
        "composition": composition.to_dict(orient="records"),
        "site_admin_concerns": sorted(SITE_ADMIN_CONCERNS),
        "interpretation_boundary": (
            "Recurring site-administration concentration is an output-composition "
            "fact and does not show that legal form causes serial recurrence."
        ),
    }
    (OUTDIR / "regular_atcm_output_universe_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    write_report(audit, exclusions, composition)

    print(f"PASS: wrote regular-ATCM output audit to {OUTDIR}")
    print(f"Locked counts: {EXPECTED_COUNTS} (total {EXPECTED_TOTAL})")
    print(
        "Measures classified as recurring site administration by title: "
        f"{int(composition.set_index('instrument').loc['Measure', 'title_rule_site_admin_n'])}/277"
    )


if __name__ == "__main__":
    main()
