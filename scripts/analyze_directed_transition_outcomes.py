#!/usr/bin/env python3
"""Explore whether directed portfolio transitions identify consequential links.

The symmetric concern space records recurrent co-specialization.  It cannot say
whether prior work on concern i is a more plausible route into concern j than
the reverse, or distinguish two concerns that actors commonly hold together.

This script learns two directed maps from rolling five-meeting portfolio
states, preserving ATCM order when calendar years are skipped:

1. A transparent empirical-Bayes transition rate P(entry into j | i held),
   shrunk toward j's overall entry rate.
2. A joint ridge-logit map in which all previously held concerns compete to
   predict entry into j.  Its i->j coefficient can distinguish two concerns
   that tend to be held together, subject to regularization and the available
   history.

For every formal output, both maps use only portfolio transitions ending before
the output meeting.  The maps are evaluated against independently coded output
concerns and verified paper--output links.  Exact concern matches are modelled
separately because entry into j is defined only while j is not already held.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.conditional_models import ConditionalLogit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hazard_conditional_logit as hcl
import scripts.analyze_attention_to_outcomes as outcome_base


OUTDIR = ROOT / "output" / "outcome_linkage"
PANEL_PATH = OUTDIR / "space_discrimination_panel_consensus.csv"
CONSENSUS_PATH = OUTDIR / "outcome_consensus_model_comparison.csv"
MAP_PATH = OUTDIR / "directed_transition_maps.csv"
PANEL_OUT_PATH = OUTDIR / "directed_transition_outcome_panel.csv"
AUC_PATH = OUTDIR / "directed_transition_outcome_auc.csv"
MODEL_PATH = OUTDIR / "directed_transition_outcome_models.csv"
MOVEMENT_PATH = OUTDIR / "directed_transition_movement_validation.csv"
DIAGNOSTIC_PATH = OUTDIR / "directed_transition_diagnostics.json"
REPORT_PATH = OUTDIR / "directed_transition_outcome_report.md"

WINDOW_MEETINGS = 5
RPA_THRESHOLD = 1.0
PRIMARY_PRIOR_STRENGTH = 10.0
PRIOR_STRENGTHS = (5.0, 10.0, 20.0)
RIDGE_C = 1.0
MIN_TARGET_ENTRIES_FOR_RIDGE = 5
N_BOOTSTRAP = 5000
SEED = 20260814


def build_transition_history() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild rolling portfolio transitions and sufficient statistics."""
    counts, submitted, members_raw, topics_raw = hcl.load_data_with_fallback()
    meeting_column = "meeting number"
    submitted = hcl.sanitize_years(submitted, meeting_column)
    topics = list(counts.index)
    members = list(counts.columns)
    meetings = sorted(submitted[meeting_column].astype(int).unique())
    periods = [
        (int(meetings[index - WINDOW_MEETINGS + 1]), int(meetings[index]))
        for index in range(WINDOW_MEETINGS - 1, len(meetings))
    ]
    active_states = []
    for start, end in periods:
        interaction = hcl.build_window_interaction(
            submitted_df=submitted,
            year_col=meeting_column,
            year_start=start,
            year_end=end,
            all_members_raw=set(members_raw),
            all_topics_raw=set(topics_raw),
            topics_order=topics,
            members_order=members,
        )
        active_states.append(
            (hcl.get_rca(interaction) >= RPA_THRESHOLD)
            .reindex(index=topics, columns=members, fill_value=False)
            .to_numpy(dtype=bool)
        )

    transition_records = []
    sample_meetings: list[int] = []
    sample_targets: list[int] = []
    sample_outcomes: list[int] = []
    sample_held: list[np.ndarray] = []
    for transition_index in range(1, len(periods)):
        end_meeting = int(periods[transition_index][1])
        previous = active_states[transition_index - 1]
        current = active_states[transition_index]
        exposures = np.zeros((len(topics), len(topics)), dtype=float)
        entries = np.zeros_like(exposures)
        target_exposures = np.zeros(len(topics), dtype=float)
        target_entries = np.zeros(len(topics), dtype=float)
        for actor_index in range(len(members)):
            held = previous[:, actor_index]
            if not held.any():
                continue
            at_risk = ~held
            entered = current[:, actor_index] & at_risk
            exposures += np.outer(held, at_risk)
            entries += np.outer(held, entered)
            target_exposures += at_risk
            target_entries += entered
            for target in np.flatnonzero(at_risk):
                sample_meetings.append(end_meeting)
                sample_targets.append(int(target))
                sample_outcomes.append(int(entered[target]))
                sample_held.append(held.copy())
        transition_records.append(
            {
                "end_meeting": end_meeting,
                "exposures": exposures,
                "entries": entries,
                "target_exposures": target_exposures,
                "target_entries": target_entries,
            }
        )

    metadata = {
        "topics": topics,
        "members": members,
        "periods": periods,
        "active_states": active_states,
        "transition_records": transition_records,
        "meeting_column": meeting_column,
        "submitted": submitted,
        "members_raw": set(members_raw),
        "topics_raw": set(topics_raw),
    }
    return (
        metadata,
        np.asarray(sample_meetings, dtype=int),
        np.asarray(sample_targets, dtype=int),
        np.asarray(sample_outcomes, dtype=int),
        np.asarray(sample_held, dtype=np.uint8),
    )


def percentile_by_target(values: np.ndarray) -> np.ndarray:
    """Rank source concerns separately for each target concern."""
    ranks = np.full_like(values, np.nan, dtype=float)
    for target in range(values.shape[1]):
        sources = [source for source in range(values.shape[0]) if source != target]
        series = pd.Series(values[sources, target], index=sources)
        ranks[sources, target] = series.rank(pct=True, method="average").to_numpy()
    return ranks


def directed_maps_for_meeting(
    focal_meeting: int,
    metadata: dict,
    sample_meetings: np.ndarray,
    sample_targets: np.ndarray,
    sample_outcomes: np.ndarray,
    sample_held: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    topics = metadata["topics"]
    n_topics = len(topics)
    prior_records = [
        record
        for record in metadata["transition_records"]
        if record["end_meeting"] < focal_meeting
    ]
    exposures = sum(
        (record["exposures"] for record in prior_records),
        np.zeros((n_topics, n_topics), dtype=float),
    )
    entries = sum(
        (record["entries"] for record in prior_records),
        np.zeros((n_topics, n_topics), dtype=float),
    )
    target_exposures = sum(
        (record["target_exposures"] for record in prior_records),
        np.zeros(n_topics, dtype=float),
    )
    target_entries = sum(
        (record["target_entries"] for record in prior_records),
        np.zeros(n_topics, dtype=float),
    )
    base_rate = (target_entries + 0.5) / (target_exposures + 1.0)

    maps: dict[str, np.ndarray | float | int] = {
        "exposures": exposures,
        "entries": entries,
        "base_rate": base_rate,
        "n_prior_transitions": len(prior_records),
    }
    for prior_strength in PRIOR_STRENGTHS:
        rate = (
            entries + prior_strength * base_rate[np.newaxis, :]
        ) / (exposures + prior_strength)
        np.fill_diagonal(rate, np.nan)
        maps[f"rate_{prior_strength:g}"] = rate
        maps[f"rate_rank_{prior_strength:g}"] = percentile_by_target(rate)

    # Joint target-specific models: coefficients estimate the conditional
    # contribution of each previously held concern while the remaining held
    # concerns enter simultaneously.  These are predictive scores, not causal
    # effects.
    ridge = np.zeros((n_topics, n_topics), dtype=float)
    prior_mask = sample_meetings < focal_meeting
    for target in range(n_topics):
        mask = prior_mask & (sample_targets == target)
        y = sample_outcomes[mask]
        if (
            len(y) == 0
            or int(y.sum()) < MIN_TARGET_ENTRIES_FOR_RIDGE
            or int((1 - y).sum()) < MIN_TARGET_ENTRIES_FOR_RIDGE
        ):
            continue
        x = sample_held[mask].astype(float)
        x[:, target] = 0.0
        fitted = LogisticRegression(
            penalty="l2",
            C=RIDGE_C,
            solver="lbfgs",
            max_iter=3000,
        ).fit(x, y)
        ridge[:, target] = fitted.coef_.ravel()
        ridge[target, target] = np.nan
    maps["ridge"] = ridge
    maps["ridge_rank"] = percentile_by_target(ridge)
    return maps


def build_maps_and_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata, sample_meetings, sample_targets, sample_outcomes, sample_held = (
        build_transition_history()
    )
    topics = metadata["topics"]
    topic_index = {topic: index for index, topic in enumerate(topics)}
    panel = pd.read_csv(PANEL_PATH)
    consensus = pd.read_csv(CONSENSUS_PATH)
    outcome_topic = (
        consensus.loc[consensus["codable"].astype(bool)]
        .drop_duplicates("outcome_id")
        .set_index("outcome_id")["consensus_primary"]
        .to_dict()
    )
    _, paper_categories = outcome_base.load_paper_training(topics)
    paper_topic = {
        paper_id: labels[0]
        for paper_id, labels in paper_categories.items()
        if len(labels) == 1
    }
    panel["paper_topic"] = panel["paper_id"].map(paper_topic)
    panel["outcome_topic"] = panel["outcome_id"].map(outcome_topic)
    panel["working_paper"] = panel["paper_id"].str.contains(r":WP").astype(int)
    if panel[["paper_topic", "outcome_topic"]].isna().any().any():
        raise RuntimeError("Candidate panel contains an unmapped paper or outcome concern")

    meetings = sorted(panel["meeting"].astype(int).unique())
    map_rows = []
    maps_by_meeting = {}
    for meeting in meetings:
        maps = directed_maps_for_meeting(
            meeting,
            metadata,
            sample_meetings,
            sample_targets,
            sample_outcomes,
            sample_held,
        )
        maps_by_meeting[meeting] = maps
        for source, source_topic in enumerate(topics):
            for target, target_topic in enumerate(topics):
                if source == target:
                    continue
                map_rows.append(
                    {
                        "focal_meeting": meeting,
                        "source_topic": source_topic,
                        "target_topic": target_topic,
                        "exposures": maps["exposures"][source, target],
                        "entries": maps["entries"][source, target],
                        "target_base_rate": maps["base_rate"][target],
                        "rate_prior5": maps["rate_5"][source, target],
                        "rate_prior10": maps["rate_10"][source, target],
                        "rate_prior20": maps["rate_20"][source, target],
                        "rate_rank_prior5": maps["rate_rank_5"][source, target],
                        "rate_rank_prior10": maps["rate_rank_10"][source, target],
                        "rate_rank_prior20": maps["rate_rank_20"][source, target],
                        "ridge_coefficient": maps["ridge"][source, target],
                        "ridge_rank": maps["ridge_rank"][source, target],
                    }
                )

    def lookup(row: pd.Series, key: str) -> float:
        source = topic_index[row["paper_topic"]]
        target = topic_index[row["outcome_topic"]]
        if source == target:
            return np.nan
        return float(maps_by_meeting[int(row["meeting"])][key][source, target])

    for prior_strength in PRIOR_STRENGTHS:
        panel[f"directed_rate_rank_{prior_strength:g}"] = panel.apply(
            lookup, axis=1, key=f"rate_rank_{prior_strength:g}"
        )
    panel["directed_ridge_rank"] = panel.apply(
        lookup, axis=1, key="ridge_rank"
    )
    # Directed entry is undefined on the diagonal.  In regression models the
    # exact-match indicator carries that case, while a neutral median rank
    # prevents exact rows from being discarded.
    directed_rank_columns = [
        f"directed_rate_rank_{prior_strength:g}"
        for prior_strength in PRIOR_STRENGTHS
    ] + ["directed_ridge_rank"]
    exact_rows = panel["exact_label_match"].eq(1)
    panel.loc[exact_rows, directed_rank_columns] = 0.5
    # Exact matches are deliberately separated.  For a combined ranking, every
    # exact match is placed above every off-label paper, and the directed map
    # ranks papers only within the off-label block.
    panel["directed_rate_full"] = np.where(
        panel["exact_label_match"].eq(1),
        2.0,
        panel[f"directed_rate_rank_{PRIMARY_PRIOR_STRENGTH:g}"],
    )
    panel["directed_ridge_full"] = np.where(
        panel["exact_label_match"].eq(1),
        2.0,
        panel["directed_ridge_rank"],
    )
    diagnostics = {
        "window_meetings": WINDOW_MEETINGS,
        "rpa_threshold": RPA_THRESHOLD,
        "history_rule": "portfolio transitions with rolling-window end before output meeting",
        "primary_prior_strength": PRIMARY_PRIOR_STRENGTH,
        "ridge_C": RIDGE_C,
        "n_transition_samples": int(len(sample_outcomes)),
        "n_transition_entries": int(sample_outcomes.sum()),
        "n_output_meetings": len(meetings),
        "n_topics": len(topics),
    }
    return pd.DataFrame(map_rows), panel, diagnostics


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def outcome_auc(
    data: pd.DataFrame,
    score: str,
    positive: str,
    label: str,
    paper_type_matched: bool = False,
) -> dict:
    values = []
    for _, group in data.groupby("outcome_id"):
        comparisons = []
        linked = group[group[positive].eq(1)]
        for row in linked.itertuples(index=False):
            controls = group[group[positive].eq(0)]
            if paper_type_matched:
                paper_type = "WP" if ":WP" in row.paper_id else "IP"
                controls = controls[
                    controls["paper_id"].str.contains(f":{paper_type}")
                ]
            control_values = controls[score].dropna().to_numpy(float)
            linked_value = float(getattr(row, score))
            if not len(control_values) or not np.isfinite(linked_value):
                continue
            comparisons.extend(
                (
                    (linked_value > control_values).astype(float)
                    + 0.5 * (linked_value == control_values)
                ).tolist()
            )
        if comparisons:
            values.append(float(np.mean(comparisons)))
    array = np.asarray(values, dtype=float)
    low, high = bootstrap_interval(array)
    return {
        "comparison": label,
        "score": score,
        "positive": positive,
        "paper_type_matched": paper_type_matched,
        "estimate": float(array.mean()),
        "ci_low": low,
        "ci_high": high,
        "n_outputs": int(len(array)),
        "scale": "probability linked paper ranks above control paper",
    }


def group_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return np.nan
    return float(
        (
            (positive[:, None] > negative[None, :]).astype(float)
            + 0.5 * (positive[:, None] == negative[None, :])
        ).mean()
    )


def validate_future_movement(
    metadata: dict,
    sample_meetings: np.ndarray,
    sample_targets: np.ndarray,
    sample_outcomes: np.ndarray,
    sample_held: np.ndarray,
) -> pd.DataFrame:
    """Test maps on actor transitions excluded from their construction."""
    topics = metadata["topics"]
    members = metadata["members"]
    submitted = metadata["submitted"]
    meeting_column = metadata["meeting_column"]
    active_states = metadata["active_states"]
    rows = []
    for transition_index in range(10, len(metadata["periods"])):
        focal_meeting = int(metadata["periods"][transition_index][1])
        maps = directed_maps_for_meeting(
            focal_meeting,
            metadata,
            sample_meetings,
            sample_targets,
            sample_outcomes,
            sample_held,
        )
        historical_interaction = hcl.build_window_interaction(
            submitted_df=submitted,
            year_col=meeting_column,
            year_start=int(submitted[meeting_column].min()),
            year_end=focal_meeting - 1,
            all_members_raw=metadata["members_raw"],
            all_topics_raw=metadata["topics_raw"],
            topics_order=topics,
            members_order=members,
        )
        phi = hcl.phi_from_interaction(historical_interaction, topics)
        available = historical_interaction.sum(axis=1).to_numpy(float) > 0
        previous = active_states[transition_index - 1]
        current = active_states[transition_index]
        popularity = previous.sum(axis=1) / max(len(members), 1)
        for actor_index, actor in enumerate(members):
            held = previous[:, actor_index]
            if not held.any():
                continue
            at_risk = (~held) & available
            adopted = current[:, actor_index] & at_risk
            if not adopted.any() or int(at_risk.sum()) == int(adopted.sum()):
                continue
            source_indices = np.flatnonzero(held)
            target_indices = np.flatnonzero(at_risk)
            labels = adopted[target_indices].astype(int)
            score_vectors = {
                "symmetric_proximity": phi[np.ix_(target_indices, source_indices)].max(axis=1),
                "directed_rate": np.nanmax(
                    maps["rate_rank_10"][np.ix_(source_indices, target_indices)], axis=0
                ),
                "directed_ridge": np.nanmax(
                    maps["ridge_rank"][np.ix_(source_indices, target_indices)], axis=0
                ),
                "prior_popularity": popularity[target_indices],
            }
            for score_name, scores in score_vectors.items():
                rows.append(
                    {
                        "actor": actor,
                        "transition_meeting": focal_meeting,
                        "score": score_name,
                        "auc": group_auc(labels, np.asarray(scores, dtype=float)),
                        "n_available": int(len(labels)),
                        "n_entries": int(labels.sum()),
                    }
                )
    groups = pd.DataFrame(rows).dropna(subset=["auc"])
    summary_rows = []
    rng = np.random.default_rng(SEED)
    for period, subset in (
        ("all_validated_meetings", groups),
        ("verified_lineage_era_atcm16_onward", groups[groups["transition_meeting"] >= 16]),
    ):
        for score, group in subset.groupby("score"):
            values = group["auc"].to_numpy(float)
            actor_totals = (
                group.groupby("actor", sort=False)["auc"]
                .agg(["sum", "count"])
                .reset_index(drop=True)
            )
            actor_sums = actor_totals["sum"].to_numpy(float)
            actor_counts = actor_totals["count"].to_numpy(float)
            bootstrap = np.asarray(
                [
                    (
                        actor_sums[draw].sum()
                        / actor_counts[draw].sum()
                    )
                    for _ in range(N_BOOTSTRAP)
                    for draw in [
                        rng.choice(
                            len(actor_totals),
                            len(actor_totals),
                            replace=True,
                        )
                    ]
                ]
            )
            summary_rows.append(
                {
                    "period": period,
                    "score": score,
                    "mean_actor_period_auc": float(values.mean()),
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                    "n_actor_periods": int(len(values)),
                    "n_actors": int(group["actor"].nunique()),
                    "first_transition_meeting": int(group["transition_meeting"].min()),
                    "last_transition_meeting": int(group["transition_meeting"].max()),
                }
            )
    groups.to_csv(
        OUTDIR / "directed_transition_movement_validation_by_actor_period.csv",
        index=False,
    )
    return pd.DataFrame(summary_rows)


def fit_conditional_model(
    data: pd.DataFrame,
    terms: list[str],
    specification: str,
) -> pd.DataFrame:
    frame = data.dropna(subset=terms).copy()
    usable = frame.groupby("outcome_id")["adoption_linked"].transform(
        lambda values: 0 < values.sum() < len(values)
    )
    frame = frame[usable]
    design = pd.DataFrame(index=frame.index)
    for term in terms:
        values = frame[term].astype(float)
        sd = max(float(values.std(ddof=0)), 1e-12)
        design[term] = (values - float(values.mean())) / sd
    fitted = ConditionalLogit(
        frame["adoption_linked"].astype(int),
        design,
        groups=frame["outcome_id"],
    ).fit(method="bfgs", disp=False, maxiter=1000)
    rows = []
    for term in terms:
        coefficient = float(fitted.params[term])
        standard_error = float(fitted.bse[term])
        rows.append(
            {
                "specification": specification,
                "term": term,
                "coefficient": coefficient,
                "standard_error": standard_error,
                "odds_ratio": math.exp(coefficient),
                "ci_low": math.exp(coefficient - 1.96 * standard_error),
                "ci_high": math.exp(coefficient + 1.96 * standard_error),
                "p_value": float(2 * norm.sf(abs(coefficient / standard_error))),
                "n_outputs": int(frame["outcome_id"].nunique()),
                "n_rows": int(len(frame)),
                "scale": "one sample SD",
            }
        )
    return pd.DataFrame(rows)


def map_diagnostics(maps: pd.DataFrame, diagnostics: dict) -> dict:
    latest = maps[
        maps["focal_meeting"].eq(maps["focal_meeting"].max())
    ].copy()
    reverse = latest.rename(
        columns={
            "source_topic": "target_topic",
            "target_topic": "source_topic",
            "rate_prior10": "reverse_rate",
            "ridge_coefficient": "reverse_ridge",
        }
    )[["source_topic", "target_topic", "reverse_rate", "reverse_ridge"]]
    paired = latest.merge(reverse, on=["source_topic", "target_topic"], how="inner")
    diagnostics.update(
        {
            "latest_meeting": int(latest["focal_meeting"].iloc[0]),
            "rate_forward_reverse_spearman": float(
                spearmanr(paired["rate_prior10"], paired["reverse_rate"]).statistic
            ),
            "ridge_forward_reverse_spearman": float(
                spearmanr(
                    paired["ridge_coefficient"], paired["reverse_ridge"]
                ).statistic
            ),
            "median_pair_exposure_latest": float(latest["exposures"].median()),
            "share_pair_exposure_below_10_latest": float(
                latest["exposures"].lt(10).mean()
            ),
        }
    )
    return diagnostics


def write_report(
    auc: pd.DataFrame,
    models: pd.DataFrame,
    movement: pd.DataFrame,
    diagnostics: dict,
) -> None:
    selected_auc = auc[
        auc["comparison"].isin(
            [
                "symmetric_full",
                "exact_only",
                "symmetric_nearby_only",
                "directed_rate_full",
                "directed_rate_nearby_only",
                "directed_ridge_full",
                "directed_ridge_nearby_only",
                "discussion_directed_rate_full",
            ]
        )
    ]
    joint = models[
        models["specification"].isin(
            ["exact_symmetric_directed_rate", "exact_symmetric_directed_ridge"]
        )
        & models["term"].isin(
            ["related_concern_proximity", "directed_rate_rank_10", "directed_ridge_rank"]
        )
    ]
    lines = [
        "# Directed transitions and formal outputs",
        "",
        "The directed maps are learned from rolling five-meeting actor portfolios. For an output at ATCM t, only portfolio transitions completed before that meeting are used. Exact concern matches are kept separate because entry into a concern is observed only while that concern is not already held.",
        "",
        "## Does direction predict later portfolio movement?",
        "",
        "Each transition is scored using maps estimated from earlier transitions. Values below are actor-period-balanced probabilities that a newly specialized concern ranks above another available concern. Intervals resample complete actor histories.",
        "",
        "| Period | Score | Probability | 95% CI | Actor-periods |",
        "|---|---|---:|---:|---:|",
    ]
    for row in movement.itertuples(index=False):
        lines.append(
            f"| {row.period.replace('_', ' ')} | {row.score.replace('_', ' ')} | "
            f"{row.mean_actor_period_auc:.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{row.n_actor_periods} |"
        )
    lines.extend(
        [
        "",
        "## Paper-ranking results",
        "",
        "| Comparison | Probability linked paper ranks higher | 95% CI | Outputs |",
        "|---|---:|---:|---:|",
        ]
    )
    for row in selected_auc.itertuples(index=False):
        lines.append(
            f"| {row.comparison.replace('_', ' ')} | {row.estimate:.3f} | "
            f"[{row.ci_low:.3f}, {row.ci_high:.3f}] | {row.n_outputs} |"
        )
    lines.extend(
        [
            "",
            "## Conditional comparison",
            "",
            "The models below include exact concern alignment, title overlap, and Working Paper status. The displayed rows show the symmetric or directed off-label term when both are allowed to compete.",
            "",
            "| Model | Term | Odds ratio | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in joint.itertuples(index=False):
        lines.append(
            f"| {row.specification.replace('_', ' ')} | {row.term.replace('_', ' ')} | "
            f"{row.odds_ratio:.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Map diagnostics",
            "",
            f"The primary empirical-Bayes map is built from {diagnostics['n_transition_samples']:,} actor--target transition opportunities containing {diagnostics['n_transition_entries']:,} entries. In the latest map, the median directed pair has {diagnostics['median_pair_exposure_latest']:.0f} observed opportunities; {100 * diagnostics['share_pair_exposure_below_10_latest']:.1f}% have fewer than ten. Forward and reverse scores correlate at {diagnostics['rate_forward_reverse_spearman']:.3f} for the shrunk rate and {diagnostics['ridge_forward_reverse_spearman']:.3f} for the joint ridge map. Lower correlations indicate that direction adds information beyond the symmetric space.",
            "",
            "The transparent rate can still credit several concerns held together for the same transition. The joint ridge map is the direct test of whether one of those concerns carries more conditional predictive information, but its coefficients are regularized and should not be interpreted causally.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    maps, panel, diagnostics = build_maps_and_panel()
    maps.to_csv(MAP_PATH, index=False)
    panel.to_csv(PANEL_OUT_PATH, index=False)

    metadata, sample_meetings, sample_targets, sample_outcomes, sample_held = (
        build_transition_history()
    )
    movement = validate_future_movement(
        metadata, sample_meetings, sample_targets, sample_outcomes, sample_held
    )
    movement.to_csv(MOVEMENT_PATH, index=False)

    off_label = panel[panel["exact_label_match"].eq(0)].copy()
    auc_rows = [
        outcome_auc(panel, "expected_proximity", "adoption_linked", "symmetric_full"),
        outcome_auc(panel, "same_concern_mass", "adoption_linked", "exact_only"),
        outcome_auc(
            off_label,
            "related_concern_proximity",
            "adoption_linked",
            "symmetric_nearby_only",
        ),
        outcome_auc(
            panel,
            "directed_rate_full",
            "adoption_linked",
            "directed_rate_full",
        ),
        outcome_auc(
            off_label,
            "directed_rate_rank_10",
            "adoption_linked",
            "directed_rate_nearby_only",
        ),
        outcome_auc(
            panel,
            "directed_ridge_full",
            "adoption_linked",
            "directed_ridge_full",
        ),
        outcome_auc(
            off_label,
            "directed_ridge_rank",
            "adoption_linked",
            "directed_ridge_nearby_only",
        ),
        outcome_auc(
            panel,
            "directed_rate_full",
            "discussion_linked",
            "discussion_directed_rate_full",
        ),
    ]
    for prior_strength in PRIOR_STRENGTHS:
        auc_rows.append(
            outcome_auc(
                off_label,
                f"directed_rate_rank_{prior_strength:g}",
                "adoption_linked",
                f"directed_rate_nearby_prior_{prior_strength:g}",
            )
        )
    for score, label in (
        ("related_concern_proximity", "symmetric_nearby_type_matched"),
        ("directed_rate_rank_10", "directed_rate_nearby_type_matched"),
        ("directed_ridge_rank", "directed_ridge_nearby_type_matched"),
    ):
        auc_rows.append(
            outcome_auc(
                off_label,
                score,
                "adoption_linked",
                label,
                paper_type_matched=True,
            )
        )
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(AUC_PATH, index=False)

    core = ["same_concern_mass", "title_overlap", "working_paper"]
    model_tables = [
        fit_conditional_model(
            panel,
            core + ["related_concern_proximity"],
            "exact_symmetric",
        ),
        fit_conditional_model(
            panel,
            core + ["directed_rate_rank_10"],
            "exact_directed_rate",
        ),
        fit_conditional_model(
            panel,
            core + ["related_concern_proximity", "directed_rate_rank_10"],
            "exact_symmetric_directed_rate",
        ),
        fit_conditional_model(
            panel,
            core + ["directed_ridge_rank"],
            "exact_directed_ridge",
        ),
        fit_conditional_model(
            panel,
            core + ["related_concern_proximity", "directed_ridge_rank"],
            "exact_symmetric_directed_ridge",
        ),
    ]
    models = pd.concat(model_tables, ignore_index=True)
    models.to_csv(MODEL_PATH, index=False)

    diagnostics = map_diagnostics(maps, diagnostics)
    diagnostics["outputs"] = {
        "map": str(MAP_PATH.relative_to(ROOT)),
        "candidate_panel": str(PANEL_OUT_PATH.relative_to(ROOT)),
        "auc": str(AUC_PATH.relative_to(ROOT)),
        "models": str(MODEL_PATH.relative_to(ROOT)),
        "movement_validation": str(MOVEMENT_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }
    DIAGNOSTIC_PATH.write_text(json.dumps(diagnostics, indent=2))
    write_report(auc, models, movement, diagnostics)


if __name__ == "__main__":
    main()
