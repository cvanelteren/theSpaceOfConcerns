#!/usr/bin/env python3
"""Transport collective attention on maps estimated at each ATCM.

For every transition from meeting t to t+1, this analysis estimates two concern
maps using information available at t:

* meeting-only: actor--concern specialization using papers at t alone;
* cumulative: actor--concern specialization using all papers through t.

Each map supplies the cost ``1 - phi_t`` for an exact optimal transport plan
between the paper-attention distributions observed at t and t+1. The analysis
also measures how much each map changes between consecutive meetings. It is a
retrospective description of attention movement, not a forecast of t+1.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_attention_to_outcomes import (
    load_paper_training,
    paper_attention_meeting_panel,
)
from scripts.analyze_concern_transport import (
    _transport_constraints,
    costs_from,
    naive_cost,
    transport,
)
from scripts.hazard_conditional_logit import (
    build_window_interaction,
    choose_period_col,
    load_data_with_fallback,
    phi_from_interaction,
    sanitize_periods,
)

OUTDIR = ROOT / "output/dynamic_concern_transport"
EARLY_CUTOFF_YEAR = 2001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTDIR)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def attention_distributions(
    topics: list[str], meetings: list[int]
) -> tuple[pd.DataFrame, pd.Series]:
    training, _ = load_paper_training(topics)
    attention = paper_attention_meeting_panel(training, topics, meetings)
    counts = attention.pivot(index="meeting", columns="topic", values="paper_count")
    counts = counts.reindex(index=meetings, columns=topics, fill_value=0.0)
    totals = counts.sum(axis=1)
    if totals.le(0).any():
        raise AssertionError("Every ATCM must contain paper attention")
    return counts.div(totals, axis=0), totals


def dynamic_maps() -> tuple[
    dict[str, dict[int, pd.DataFrame]],
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[int],
]:
    counts, submitted, members_raw, topics_raw = load_data_with_fallback()
    period_col = choose_period_col(submitted)
    submitted = sanitize_periods(submitted, period_col)
    topics = list(counts.index)
    members = list(counts.columns)
    meetings = sorted(submitted[period_col].unique().astype(int).tolist())
    first_meeting = min(meetings)
    year_by_meeting = (
        submitted.groupby(period_col)["year"].median().round().astype(int)
    )

    maps: dict[str, dict[int, pd.DataFrame]] = {
        "meeting-only": {},
        "cumulative": {},
    }
    map_rows = []
    interactions: dict[tuple[str, int], pd.DataFrame] = {}
    for meeting in meetings:
        specifications = {
            "meeting-only": (meeting, meeting),
            "cumulative": (first_meeting, meeting),
        }
        for variant, (start, end) in specifications.items():
            interaction = build_window_interaction(
                submitted,
                period_col,
                start,
                end,
                members_raw,
                topics_raw,
                topics,
                members,
            )
            phi = pd.DataFrame(
                phi_from_interaction(interaction, topics),
                index=topics,
                columns=topics,
            )
            maps[variant][meeting] = phi
            interactions[(variant, meeting)] = interaction
            offdiag = phi.to_numpy(float)[~np.eye(len(topics), dtype=bool)]
            map_rows.append(
                {
                    "variant": variant,
                    "meeting": meeting,
                    "year": int(year_by_meeting.loc[meeting]),
                    "actor_concern_mass": float(interaction.to_numpy().sum()),
                    "active_actors": int(interaction.sum(axis=0).gt(0).sum()),
                    "active_concerns": int(interaction.sum(axis=1).gt(0).sum()),
                    "positive_pairs": int(np.count_nonzero(offdiag > 0) // 2),
                    "mean_proximity": float(offdiag.mean()),
                }
            )

    stability_rows = []
    mask = ~np.eye(len(topics), dtype=bool)
    for variant, variant_maps in maps.items():
        for current, following in pairwise(meetings):
            left = variant_maps[current].to_numpy(float)[mask]
            right = variant_maps[following].to_numpy(float)[mask]
            correlation = np.nan
            if left.std() > 0 and right.std() > 0:
                correlation = float(np.corrcoef(left, right)[0, 1])
            stability_rows.append(
                {
                    "variant": variant,
                    "from_meeting": current,
                    "to_meeting": following,
                    "from_year": int(year_by_meeting.loc[current]),
                    "to_year": int(year_by_meeting.loc[following]),
                    "proximity_rmse": float(np.sqrt(np.mean((right - left) ** 2))),
                    "proximity_correlation": correlation,
                }
            )
    return maps, pd.DataFrame(map_rows), pd.DataFrame(stability_rows), topics, meetings


def run_transport(
    maps: dict[str, dict[int, pd.DataFrame]],
    topics: list[str],
    meetings: list[int],
    attention: pd.DataFrame,
    totals: pd.Series,
    year_by_meeting: pd.Series,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constraints = _transport_constraints(len(topics))
    no_map = naive_cost(len(topics))
    summary_rows = []
    flow_rows = []
    for current, following in pairwise(meetings):
        source = attention.loc[current, topics].to_numpy(float)
        target = attention.loc[following, topics].to_numpy(float)
        moved, _ = transport(source, target, no_map, constraints)
        for variant, variant_maps in maps.items():
            phi = variant_maps[current].reindex(index=topics, columns=topics)
            cost = costs_from(phi.to_numpy(float))
            map_cost, plan = transport(
                source,
                target,
                cost,
                constraints,
            )
            null = np.empty(permutations, dtype=float)
            for draw in range(permutations):
                order = rng.permutation(len(topics))
                null[draw] = transport(
                    source,
                    target,
                    cost[np.ix_(order, order)],
                    constraints,
                )[0]
            null_mean = float(null.mean())
            summary_rows.append(
                {
                    "variant": variant,
                    "from_meeting": current,
                    "to_meeting": following,
                    "from_year": int(year_by_meeting.loc[current]),
                    "to_year": int(year_by_meeting.loc[following]),
                    "source_papers": float(totals.loc[current]),
                    "target_papers": float(totals.loc[following]),
                    "mass_moved": moved,
                    "map_transport_cost": map_cost,
                    "locality_ratio": map_cost / moved if moved > 0 else np.nan,
                    "shuffled_cost_mean": null_mean,
                    "shuffled_cost_low": float(np.quantile(null, 0.025)),
                    "shuffled_cost_high": float(np.quantile(null, 0.975)),
                    "cost_relative_to_shuffled": (
                        map_cost / null_mean if null_mean > 0 else np.nan
                    ),
                    "shuffle_p_lower": float(
                        (1 + np.count_nonzero(null <= map_cost))
                        / (1 + permutations)
                    ),
                }
            )
            for source_index, target_index in zip(*np.nonzero(plan > 1e-10)):
                flow_rows.append(
                    {
                        "variant": variant,
                        "from_meeting": current,
                        "to_meeting": following,
                        "from_year": int(year_by_meeting.loc[current]),
                        "to_year": int(year_by_meeting.loc[following]),
                        "source_concern": topics[source_index],
                        "target_concern": topics[target_index],
                        "transport_mass": float(plan[source_index, target_index]),
                        "proximity_at_t": float(phi.iat[source_index, target_index]),
                        "same_concern": source_index == target_index,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(flow_rows)


def write_report(
    map_summary: pd.DataFrame,
    stability: pd.DataFrame,
    transport_summary: pd.DataFrame,
    path: Path,
) -> None:
    rows = []
    for variant in ("meeting-only", "cumulative"):
        stable = stability[stability["variant"].eq(variant)]
        transport = transport_summary[transport_summary["variant"].eq(variant)]
        early = stable[stable["to_year"].le(EARLY_CUTOFF_YEAR)]
        later = stable[stable["to_year"].gt(EARLY_CUTOFF_YEAR)]
        rows.append(
            {
                "variant": variant,
                "early_rmse": early["proximity_rmse"].mean(),
                "later_rmse": later["proximity_rmse"].mean(),
                "early_correlation": early["proximity_correlation"].mean(),
                "later_correlation": later["proximity_correlation"].mean(),
                "mean_cost_relative_to_shuffled": transport[
                    "cost_relative_to_shuffled"
                ].mean(),
            }
        )
    comparison = pd.DataFrame(rows)
    latest = map_summary.sort_values("meeting").groupby("variant").tail(1)

    report = [
        "# Dynamic concern maps and optimal transport",
        "",
        "Each transition uses the map estimated at meeting t to transport the",
        "observed paper-attention distribution at t to the distribution at t+1.",
        "The meeting-only map uses t alone; the cumulative map uses all meetings",
        "through t. Lower proximity RMSE means a more stable map. A lower locality",
        "cost relative to shuffled compares the observed map with coherent label",
        "permutations of the same map; values below one indicate more local movement.",
        "",
        f"The early period ends in {EARLY_CUTOFF_YEAR}, forty years after 1961.",
        "",
        comparison.to_csv(index=False, float_format="%.4f").strip(),
        "",
        "Latest-map diagnostics:",
        latest.to_csv(index=False, float_format="%.4f").strip(),
        "",
        "This is a retrospective transport decomposition because both attention",
        "distributions are observed. It does not predict the t+1 distribution.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    maps, map_summary, stability, topics, meetings = dynamic_maps()
    attention, totals = attention_distributions(topics, meetings)
    year_by_meeting = (
        map_summary.drop_duplicates("meeting").set_index("meeting")["year"]
    )
    transport_summary, flows = run_transport(
        maps,
        topics,
        meetings,
        attention,
        totals,
        year_by_meeting,
        args.permutations,
        np.random.default_rng(args.seed),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    map_summary.to_csv(args.out_dir / "map_diagnostics.csv", index=False)
    stability.to_csv(args.out_dir / "map_stability.csv", index=False)
    transport_summary.to_csv(args.out_dir / "transport_summary.csv", index=False)
    flows.to_csv(args.out_dir / "transport_flows.csv", index=False)
    np.savez_compressed(
        args.out_dir / "dynamic_maps.npz",
        topics=np.array(topics, dtype=object),
        meetings=np.array(meetings, dtype=int),
        meeting_only=np.stack([maps["meeting-only"][m].to_numpy() for m in meetings]),
        cumulative=np.stack([maps["cumulative"][m].to_numpy() for m in meetings]),
    )
    write_report(
        map_summary,
        stability,
        transport_summary,
        args.out_dir / "report.md",
    )
    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "meetings": meetings,
                "concerns": len(topics),
                "transitions": len(meetings) - 1,
                "map_variants": ["meeting-only", "cumulative"],
                "early_cutoff_year": EARLY_CUTOFF_YEAR,
                "transport_is_forecast": False,
                "label_permutations_per_transition": args.permutations,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote dynamic concern transport to {args.out_dir}")


if __name__ == "__main__":
    main()
