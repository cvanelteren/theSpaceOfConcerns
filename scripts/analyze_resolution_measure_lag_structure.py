#!/usr/bin/env python3
"""Explore heterogeneous delays from Resolutions to later Measures.

The analysis separates two kinds of evidence. Exact, year-qualified citations
in Measure texts are verified formal links. Shared detailed ATS topics identify
candidate substantive links, but do not establish that one instrument produced
the other. Instrument identities always include type, number, and year because
ATS instrument numbers reset at each meeting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from scripts.official_regular_atcm_outputs import load_official_regular_outputs


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "attention_output_signal"
CITATIONS_PATH = OUTDIR / "measure_resolution_citations.csv"
FORMAL_PREDECESSORS_PATH = OUTDIR / "measure_formal_predecessor_citations.csv"
TOPICS_PATH = (
    ROOT.parent
    / "apparent_consensus"
    / "data"
    / "treaty_instrument_topics_long.csv"
)
EVENT_HISTORY_PATH = OUTDIR / "resolution_measure_event_history.csv"
TOPIC_PAIRS_PATH = OUTDIR / "resolution_measure_detailed_topic_pairs_all_lags.csv"
HORIZONS_PATH = OUTDIR / "resolution_measure_link_rates_by_horizon.csv"
SUMMARY_PATH = OUTDIR / "resolution_measure_lag_structure_summary.json"

HORIZONS = (1, 3, 5, 10, 15, 20)
AREA_CATEGORY = "Area protection and management"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson interval for a binomial proportion."""
    if trials == 0:
        return math.nan, math.nan
    proportion = successes / trials
    scale = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / scale
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
        )
        / scale
    )
    return centre - radius, centre + radius


def output_id(instrument: pd.Series, number: pd.Series, year: pd.Series) -> pd.Series:
    """Construct year-qualified ATS instrument identities."""
    return (
        instrument.astype(str)
        + " "
        + number.astype(int).astype(str)
        + " ("
        + year.astype(int).astype(str)
        + ")"
    )


def load_verified_citations(outputs: pd.DataFrame) -> pd.DataFrame:
    """Load exact Resolution citations found in the pinned Measure corpus."""
    if not CITATIONS_PATH.exists():
        raise FileNotFoundError(
            "Run `python -m scripts.audit_resolution_measure_ladders` first"
        )
    citations = pd.read_csv(CITATIONS_PATH)
    citations = citations[
        citations["citation_resolves_to_inventory"]
        & citations["citation_is_prior"]
    ].copy()
    meeting = outputs.set_index("output_id")["meeting"]
    citations["resolution_meeting"] = citations["resolution_id"].map(meeting)
    citations["measure_meeting"] = citations["measure_id"].map(meeting)
    if citations[["resolution_meeting", "measure_meeting"]].isna().any().any():
        raise AssertionError("A verified citation falls outside the output inventory")
    citations["lag_meetings"] = (
        citations["measure_meeting"] - citations["resolution_meeting"]
    ).astype(int)
    if citations["lag_meetings"].le(0).any():
        raise AssertionError("Every verified Measure must follow its cited Resolution")
    return citations


def detailed_topic_pairs(outputs: pd.DataFrame) -> pd.DataFrame:
    """Construct all later Measure candidates sharing a non-generic ATS topic."""
    if not TOPICS_PATH.exists():
        raise FileNotFoundError(f"Detailed ATS topic table is missing: {TOPICS_PATH}")
    topics = pd.read_csv(TOPICS_PATH)
    topics["instrument"] = topics["query_doc_type"].map(
        {2: "Measure", 3: "Decision", 4: "Resolution"}
    )
    topics = topics[
        topics["instrument"].isin(["Measure", "Resolution"])
        & topics["ats_topic_id"].notna()
        & topics["ats_topic_id"].ne(0)
    ].copy()
    topics["instrument_number"] = (
        topics["irec_no"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
    topics["output_id"] = output_id(
        topics["instrument"], topics["instrument_number"], topics["year_meeting"]
    )
    inventory = set(outputs["output_id"])
    topics = topics[topics["output_id"].isin(inventory)].copy()
    topics["ats_topic_id"] = topics["ats_topic_id"].astype(int)
    topics = topics.drop_duplicates(["output_id", "ats_topic_id"])

    resolutions = topics[topics["instrument"].eq("Resolution")]
    measures = topics[topics["instrument"].eq("Measure")]
    pairs = resolutions.merge(
        measures,
        on="ats_topic_id",
        suffixes=("_resolution", "_measure"),
    )
    pairs = pairs[
        pairs["year_meeting_measure"].gt(pairs["year_meeting_resolution"])
    ].copy()
    meeting = outputs.set_index("output_id")["meeting"]
    pairs["resolution_meeting"] = pairs["output_id_resolution"].map(meeting)
    pairs["measure_meeting"] = pairs["output_id_measure"].map(meeting)
    pairs["lag_years"] = (
        pairs["year_meeting_measure"] - pairs["year_meeting_resolution"]
    ).astype(int)
    pairs["lag_meetings"] = (
        pairs["measure_meeting"] - pairs["resolution_meeting"]
    ).astype(int)
    pairs = (
        pairs.groupby(
            ["output_id_resolution", "output_id_measure"], as_index=False
        )
        .agg(
            resolution_year=("year_meeting_resolution", "first"),
            resolution_meeting=("resolution_meeting", "first"),
            resolution_title=("subject_resolution", "first"),
            measure_year=("year_meeting_measure", "first"),
            measure_meeting=("measure_meeting", "first"),
            measure_title=("subject_measure", "first"),
            shared_detailed_topics=("ats_topic_id", "nunique"),
            lag_years=("lag_years", "first"),
            lag_meetings=("lag_meetings", "first"),
        )
        .rename(
            columns={
                "output_id_resolution": "resolution_id",
                "output_id_measure": "measure_id",
            }
        )
    )
    if pairs["lag_meetings"].le(0).any():
        raise AssertionError("Every detailed-topic candidate must be forward in time")
    return pairs


def first_link_table(
    resolutions: pd.DataFrame,
    links: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Attach first-link timing and downstream reuse counts to each Resolution."""
    if links.empty:
        return resolutions.assign(
            **{
                f"{prefix}_link": False,
                f"first_{prefix}_measure_id": pd.NA,
                f"first_{prefix}_lag_meetings": pd.NA,
                f"first_{prefix}_lag_years": pd.NA,
                f"{prefix}_measure_targets": 0,
                f"{prefix}_edges": 0,
            }
        )
    ordered = links.sort_values(
        ["resolution_id", "lag_meetings", "measure_year", "measure_id"]
    )
    first = ordered.groupby("resolution_id", as_index=False).first()
    counts = ordered.groupby("resolution_id", as_index=False).agg(
        measure_targets=("measure_id", "nunique"),
        edges=("measure_id", "size"),
    )
    first = first[
        ["resolution_id", "measure_id", "lag_meetings", "lag_years"]
    ].rename(
        columns={
            "measure_id": f"first_{prefix}_measure_id",
            "lag_meetings": f"first_{prefix}_lag_meetings",
            "lag_years": f"first_{prefix}_lag_years",
        }
    )
    counts = counts.rename(
        columns={
            "measure_targets": f"{prefix}_measure_targets",
            "edges": f"{prefix}_edges",
        }
    )
    result = resolutions.merge(first, on="resolution_id", how="left").merge(
        counts, on="resolution_id", how="left"
    )
    result[f"{prefix}_link"] = result[f"first_{prefix}_measure_id"].notna()
    for column in (f"{prefix}_measure_targets", f"{prefix}_edges"):
        result[column] = result[column].fillna(0).astype(int)
    return result


def horizon_rows(event_history: pd.DataFrame, prefix: str) -> list[dict[str, object]]:
    """Estimate observed link rates at common follow-up horizons."""
    rows: list[dict[str, object]] = []
    event_time = f"first_{prefix}_lag_meetings"
    for horizon in HORIZONS:
        eligible = event_history[event_history["follow_up_meetings"].ge(horizon)]
        linked = int(eligible[event_time].le(horizon).fillna(False).sum())
        lower, upper = wilson_interval(linked, len(eligible))
        rows.append(
            {
                "link_definition": prefix,
                "horizon_meetings": horizon,
                "eligible_resolutions": int(len(eligible)),
                "linked_resolutions": linked,
                "link_rate": linked / len(eligible) if len(eligible) else math.nan,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
            }
        )
    return rows


def quantiles(values: pd.Series) -> dict[str, float | int]:
    """Summarize a non-empty timing distribution."""
    values = values.dropna().astype(float)
    if values.empty:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "minimum": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "maximum": float(values.max()),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outputs = load_official_regular_outputs()
    resolutions = outputs[outputs["instrument"].eq("Resolution")].copy()
    resolutions = resolutions.rename(columns={"output_id": "resolution_id"})
    resolutions["area_protection"] = resolutions["official_categories"].map(
        lambda categories: AREA_CATEGORY in categories
    )
    latest_meeting = int(outputs["meeting"].max())
    latest_year = int(outputs["year"].max())
    resolutions["follow_up_meetings"] = latest_meeting - resolutions["meeting"]
    resolutions["follow_up_years"] = latest_year - resolutions["year"]
    resolutions = resolutions[
        [
            "resolution_id",
            "year",
            "meeting",
            "title",
            "official_categories",
            "area_protection",
            "follow_up_meetings",
            "follow_up_years",
        ]
    ]

    citations = load_verified_citations(outputs)
    topic_pairs = detailed_topic_pairs(outputs)
    event_history = first_link_table(resolutions, citations, "exact_citation")
    event_history = first_link_table(event_history, topic_pairs, "detailed_topic")

    if event_history["resolution_id"].duplicated().any():
        raise AssertionError("Event history must contain one row per Resolution")
    for prefix in ("exact_citation", "detailed_topic"):
        observed = event_history[f"{prefix}_link"]
        if (
            event_history.loc[observed, f"first_{prefix}_lag_meetings"]
            > event_history.loc[observed, "follow_up_meetings"]
        ).any():
            raise AssertionError("An event occurs beyond its observed follow-up")

    topic_pairs.to_csv(TOPIC_PAIRS_PATH, index=False)
    event_history.to_csv(EVENT_HISTORY_PATH, index=False)
    horizon_table = pd.DataFrame(
        horizon_rows(event_history, "exact_citation")
        + horizon_rows(event_history, "detailed_topic")
    )
    horizon_table.to_csv(HORIZONS_PATH, index=False)

    exact_origins = event_history[event_history["exact_citation_link"]]
    exact_targets = set(citations["measure_id"])
    target_categories = outputs.set_index("output_id")["official_categories"]
    area_targets = sum(
        AREA_CATEGORY in target_categories.loc[measure_id]
        for measure_id in exact_targets
    )
    formal_summary: dict[str, object] = {}
    if FORMAL_PREDECESSORS_PATH.exists():
        formal = pd.read_csv(FORMAL_PREDECESSORS_PATH)
        formal = formal[
            formal["citation_resolves_to_inventory"] & formal["citation_is_prior"]
        ]
        formal_summary = {
            instrument: {
                "citation_edges": int(len(group)),
                "unique_predecessors": int(group["predecessor_id"].nunique()),
                "citing_measures": int(group["measure_id"].nunique()),
                "median_lag_years": float(group["lag_years"].median()),
            }
            for instrument, group in formal.groupby("predecessor_type")
        }

    summary = {
        "archive_boundary": {
            "first_year": int(outputs["year"].min()),
            "last_year": latest_year,
            "first_meeting": int(outputs["meeting"].min()),
            "last_meeting": latest_meeting,
            "pre_1995_recommendations_available": False,
        },
        "resolutions": int(len(resolutions)),
        "verified_exact_citation_links": {
            "linked_resolution_origins": int(len(exact_origins)),
            "citing_measure_targets": int(len(exact_targets)),
            "citation_edges": int(len(citations)),
            "first_link_lag_meetings": quantiles(
                exact_origins["first_exact_citation_lag_meetings"]
            ),
            "first_link_lag_years": quantiles(
                exact_origins["first_exact_citation_lag_years"]
            ),
            "all_citation_edge_lag_years": quantiles(citations["lag_years"]),
            "origins_in_area_protection": int(exact_origins["area_protection"].sum()),
            "targets_in_area_protection": int(area_targets),
        },
        "shared_detailed_topic_candidates": {
            "linked_resolution_origins": int(
                event_history["detailed_topic_link"].sum()
            ),
            "candidate_measure_targets": int(topic_pairs["measure_id"].nunique()),
            "candidate_pairs": int(len(topic_pairs)),
            "first_link_lag_meetings": quantiles(
                event_history.loc[
                    event_history["detailed_topic_link"],
                    "first_detailed_topic_lag_meetings",
                ]
            ),
            "all_pair_lag_years": quantiles(topic_pairs["lag_years"]),
        },
        "formal_predecessor_context": formal_summary,
        "interpretation": (
            "Heterogeneous timing attenuates fixed-window models, but it does not "
            "recover a general Resolution-to-Measure pathway. Exact first links "
            "are sparse and concentrated in area-management work. Long citation "
            "lags mostly reflect repeated downstream reuse after the first link. "
            "Shared detailed topics broaden the candidate set but do not establish "
            "instrumental lineage."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nObserved link rates by follow-up horizon")
    print(horizon_table.to_string(index=False))


if __name__ == "__main__":
    main()
