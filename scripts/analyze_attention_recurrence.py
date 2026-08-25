#!/usr/bin/env python3
"""Separate accumulated attention from repeated activation before ATS output.

The main attention analysis shows whether the number of papers submitted on a
concern over earlier ATCM meetings is associated with formal output.  This
script asks the narrower follow-up question: holding that paper volume fixed,
does it matter whether attention is distributed across meetings and whether
actors return to the same concern?

All windows use ordered ATCM meetings, never calendar years.  The models are
descriptive concern--meeting comparisons with concern and meeting fixed
effects.  They do not identify a causal conversion mechanism.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import ultraplot as uplt
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle
from analysis.data_loading import load_submitted_with_fallback
from scripts import analyze_attention_accumulation as accumulation
from scripts.analyze_attention_to_outcomes import load_paper_training
from scripts.link_movement_to_outcomes import paper_sponsors


OUTDIR = ROOT / "output" / "outcome_linkage"
FIGDIR = ROOT / "figures"
PANEL_PATH = OUTDIR / "topic_meeting_attention_outcomes.csv"
MODEL_PATH = OUTDIR / "attention_recurrence_models.csv"
PANEL_EXPORT_PATH = OUTDIR / "attention_recurrence_panel.csv"
DIAGNOSTIC_PATH = OUTDIR / "attention_recurrence_diagnostics.json"
REPORT_PATH = OUTDIR / "attention_recurrence_report.md"
FIGURE_STEM = FIGDIR / "exploratory_attention_recurrence"

HORIZONS = (3, 5, 8, 10)
MODEL_PREDICTORS = (
    "papers",
    "effective_meetings",
    "returning_activation_share",
    "nearby",
    "outcomes",
)


def build_actor_activity(
    topics: list[str], meetings: list[int]
) -> tuple[pd.DataFrame, dict]:
    """Return unique actor--concern--meeting activity for single-label papers."""
    training, _ = load_paper_training(topics)
    sponsors = paper_sponsors(load_submitted_with_fallback())
    training = training[training["meeting"].isin(meetings)].copy()
    training = training.explode("topics").rename(columns={"topics": "topic"})
    training["actors"] = training["paper_id"].map(sponsors)
    matched = training["actors"].notna()
    with_actors = training.loc[matched].explode("actors")
    with_actors = with_actors[with_actors["actors"].notna()].copy()
    with_actors["actor"] = with_actors["actors"].astype(str).str.strip()
    with_actors = with_actors[with_actors["actor"].ne("")]
    activity = with_actors[["topic", "meeting", "actor"]].drop_duplicates()
    diagnostics = {
        "single_label_papers": int(len(training)),
        "papers_with_sponsor_mapping": int(matched.sum()),
        "paper_sponsor_mapping_share": float(matched.mean()),
        "actor_concern_meeting_activations": int(len(activity)),
        "actors": int(activity["actor"].nunique()),
    }
    return activity, diagnostics


def effective_count(values: np.ndarray) -> float:
    """Effective number of meetings carrying a fixed total paper volume."""
    total = float(values.sum())
    squared = float(np.square(values).sum())
    return total * total / squared if squared > 0 else 0.0


def add_recurrence_measures(
    panel: pd.DataFrame, actor_activity: pd.DataFrame
) -> pd.DataFrame:
    data = panel.sort_values(["topic", "meeting"]).copy()
    meetings = sorted(data["meeting"].unique().astype(int))
    meeting_index = {meeting: index for index, meeting in enumerate(meetings)}
    paper_lookup = data.set_index(["topic", "meeting"])["paper_count"]
    actor_groups = {
        topic: group[["meeting", "actor"]]
        for topic, group in actor_activity.groupby("topic")
    }

    rows = []
    for row in data[["topic", "meeting"]].itertuples(index=False):
        index = meeting_index[int(row.meeting)]
        record = {"topic": row.topic, "meeting": int(row.meeting)}
        for horizon in HORIZONS:
            if index < horizon:
                for name in (
                    "active_meetings",
                    "effective_meetings",
                    "latest_attention_lag",
                    "paper_weighted_lag",
                    "actor_meeting_activations",
                    "distinct_actors",
                    "returning_actors",
                    "returning_actor_share",
                    "returning_activation_share",
                ):
                    record[f"{name}_prior{horizon}"] = np.nan
                continue

            previous = meetings[index - horizon : index]
            paper_values = np.asarray(
                [paper_lookup.get((row.topic, meeting), 0.0) for meeting in previous],
                dtype=float,
            )
            record[f"active_meetings_prior{horizon}"] = int(
                np.count_nonzero(paper_values)
            )
            record[f"effective_meetings_prior{horizon}"] = effective_count(
                paper_values
            )
            lags = np.arange(horizon, 0, -1, dtype=float)
            active = paper_values > 0
            record[f"latest_attention_lag_prior{horizon}"] = (
                float(lags[active].min()) if active.any() else 0.0
            )
            record[f"paper_weighted_lag_prior{horizon}"] = (
                float(np.average(lags, weights=paper_values))
                if paper_values.sum() > 0
                else 0.0
            )

            topic_activity = actor_groups.get(row.topic)
            if topic_activity is None:
                frequencies = pd.Series(dtype=int)
            else:
                frequencies = (
                    topic_activity[topic_activity["meeting"].isin(previous)]
                    .groupby("actor")["meeting"]
                    .nunique()
                )
            n_actors = int(len(frequencies))
            activations = int(frequencies.sum()) if n_actors else 0
            returning = int(frequencies.ge(2).sum()) if n_actors else 0
            return_activations = (
                int((frequencies - 1).clip(lower=0).sum()) if n_actors else 0
            )
            record[f"actor_meeting_activations_prior{horizon}"] = activations
            record[f"distinct_actors_prior{horizon}"] = n_actors
            record[f"returning_actors_prior{horizon}"] = returning
            record[f"returning_actor_share_prior{horizon}"] = (
                returning / n_actors if n_actors else 0.0
            )
            record[f"returning_activation_share_prior{horizon}"] = (
                return_activations / activations if activations else 0.0
            )
        rows.append(record)

    measures = pd.DataFrame(rows)
    # Nearby and earlier-output stocks come from the main analysis and are
    # merged after recurrence measures are computed from the raw meeting rows.
    result = data.merge(measures, on=["topic", "meeting"], how="left")
    for horizon in HORIZONS:
        papers = result[f"papers_prior{horizon}"]
        complete = papers.notna()
        result[f"any_attention_prior{horizon}"] = np.where(
            complete, papers.gt(0).astype(float), np.nan
        )
        result[f"positive_log_papers_prior{horizon}"] = np.where(
            complete,
            np.where(papers.gt(0), np.log(papers.clip(lower=1.0)), 0.0),
            np.nan,
        )
        effective = result[f"effective_meetings_prior{horizon}"]
        result[f"effective_meetings_excess_prior{horizon}"] = np.where(
            complete, (effective.fillna(0.0) - 1.0).clip(lower=0.0), np.nan
        )
    return result


def transform(values: pd.Series, kind: str) -> np.ndarray:
    raw = values.to_numpy(float)
    if kind in {"papers", "nearby", "outcomes", "distinct_actors"}:
        return np.log1p(raw)
    return raw


def contrast(kind: str) -> tuple[float, str]:
    if kind in {"papers", "nearby", "outcomes", "distinct_actors"}:
        return math.log(2.0), "per doubling of count + 1"
    if kind == "any_attention":
        return 1.0, "first focal paper: none to one"
    if kind == "positive_log_papers":
        return math.log(2.0), "per doubling of papers after the first"
    if kind == "effective_meetings_excess":
        return 1.0, "effective meetings: 1 to 2"
    if kind == "effective_meetings":
        return 1.0, "effective meetings: 1 to 2"
    if kind == "returning_activation_share":
        return 0.25, "returning activation share: +25 percentage points"
    if kind in {"latest_attention_lag", "paper_weighted_lag"}:
        return 1.0, "attention is one meeting farther in the past"
    raise KeyError(kind)


def fit_model(
    panel: pd.DataFrame,
    *,
    outcome: str,
    horizon: int,
    specification: str,
    predictors: tuple[str, ...] = MODEL_PREDICTORS,
    quiet_period: bool = False,
    positive_attention_only: bool = True,
) -> pd.DataFrame:
    columns = [f"{predictor}_prior{horizon}" for predictor in predictors]
    data = panel.dropna(subset=columns).copy()
    if positive_attention_only:
        # Recurrence is only defined after a concern has received attention.
        data = data[data[f"papers_prior{horizon}"].gt(0)].copy()
    if quiet_period:
        outcome_name = f"onset_after_{horizon}_quiet_meetings"
        data[outcome_name] = (
            data[outcome].gt(0) & data[f"outcomes_prior{horizon}"].eq(0)
        ).astype(int)
        data = data[data[f"outcomes_prior{horizon}"].eq(0)].copy()
        data = data[data.groupby("meeting")[outcome_name].transform("sum").gt(0)]
        # A modified Poisson model is more stable than fixed-effect logit in
        # these sparse quiet-to-active transitions; clustered uncertainty is
        # used below.  Coefficients are therefore rate ratios, not odds ratios.
        family = sm.families.Poisson()
    else:
        outcome_name = outcome
        family = sm.families.Poisson()

    model_terms = []
    scales = {}
    for predictor, column in zip(predictors, columns):
        values = transform(data[column], predictor)
        scale = max(float(values.std(ddof=0)), 1e-12)
        term = f"z_{predictor}"
        data[term] = (values - values.mean()) / scale
        scales[predictor] = scale
        model_terms.append(term)

    fitted = smf.glm(
        f"{outcome_name} ~ {' + '.join(model_terms)} + C(topic) + C(meeting)",
        data=data,
        family=family,
    ).fit(maxiter=300)
    covariance, _, _ = cov_cluster_2groups(
        fitted,
        pd.Categorical(data["topic"]).codes,
        pd.Categorical(data["meeting"]).codes,
    )
    names = list(fitted.params.index)
    rows = []
    for predictor, term in zip(predictors, model_terms):
        index = names.index(term)
        estimate = float(fitted.params[term])
        se = math.sqrt(max(float(covariance[index, index]), 0.0))
        raw_estimate = estimate / scales[predictor]
        raw_se = se / scales[predictor]
        change, label = contrast(predictor)
        rows.append(
            {
                "specification": specification,
                "outcome": outcome_name,
                "horizon_meetings": horizon,
                "predictor": predictor,
                "coefficient_per_sd": estimate,
                "se_two_way_cluster": se,
                "rate_or_odds_ratio_per_sd": math.exp(estimate),
                "ci_low_per_sd": math.exp(estimate - 1.96 * se),
                "ci_high_per_sd": math.exp(estimate + 1.96 * se),
                "contrast_ratio": math.exp(raw_estimate * change),
                "contrast_ci_low": math.exp((raw_estimate - 1.96 * raw_se) * change),
                "contrast_ci_high": math.exp((raw_estimate + 1.96 * raw_se) * change),
                "contrast": label,
                "n_topic_meetings": int(len(data)),
                "n_events": int(data[outcome_name].gt(0).sum()),
                "n_topics": int(data["topic"].nunique()),
                "n_meetings": int(data["meeting"].nunique()),
                "fixed_effects": "concern and ATCM meeting",
                "clustered_by": "concern and ATCM meeting",
            }
        )
    return pd.DataFrame(rows)


def fit_all(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["any_hard_output"] = panel["outcome_count_hard"].gt(0).astype(int)
    tables = []
    for horizon in HORIZONS:
        hurdle_predictors = (
            "any_attention",
            "positive_log_papers",
            "nearby",
            "outcomes",
        )
        for outcome, label in (
            ("outcome_count_hard", "hard_output"),
            ("outcome_count_high_confidence", "high_confidence_hard_output"),
            ("outcome_mass", "soft_output"),
            ("any_hard_output", "any_hard_output"),
        ):
            tables.append(
                fit_model(
                    panel,
                    outcome=outcome,
                    horizon=horizon,
                    specification=f"attention_hurdle_{label}",
                    predictors=hurdle_predictors,
                    positive_attention_only=False,
                )
            )
        tables.append(
            fit_model(
                panel[~panel["topic"].isin(accumulation.SITE_ADMIN_TOPICS)],
                outcome="outcome_count_hard",
                horizon=horizon,
                specification="attention_hurdle_excluding_site_administration",
                predictors=hurdle_predictors,
                positive_attention_only=False,
            )
        )
        for outcome, label in (
            ("outcome_count_hard", "hard_output"),
            ("outcome_mass", "soft_output"),
        ):
            tables.append(
                fit_model(
                    panel,
                    outcome=outcome,
                    horizon=horizon,
                    specification=f"volume_temporal_actor_{label}",
                )
            )
        recency_predictors = (
            "papers",
            "effective_meetings",
            "latest_attention_lag",
            "returning_activation_share",
            "nearby",
            "outcomes",
        )
        for outcome, label in (
            ("outcome_count_hard", "hard_output"),
            ("outcome_mass", "soft_output"),
            ("any_hard_output", "any_hard_output"),
        ):
            tables.append(
                fit_model(
                    panel,
                    outcome=outcome,
                    horizon=horizon,
                    specification=f"recency_adjusted_{label}",
                    predictors=recency_predictors,
                )
            )
        tables.append(
            fit_model(
                panel[~panel["topic"].isin(accumulation.SITE_ADMIN_TOPICS)],
                outcome="outcome_count_hard",
                horizon=horizon,
                specification="recency_adjusted_excluding_site_administration",
                predictors=recency_predictors,
            )
        )
        tables.append(
            fit_model(
                panel,
                outcome="outcome_count_hard",
                horizon=horizon,
                specification="mean_lag_adjusted_hard_output",
                predictors=(
                    "papers",
                    "effective_meetings",
                    "paper_weighted_lag",
                    "returning_activation_share",
                    "nearby",
                    "outcomes",
                ),
            )
        )
        tables.append(
            fit_model(
                panel,
                outcome="outcome_count_hard",
                horizon=horizon,
                specification="quiet_period_onset_hard_output",
                predictors=(
                    "papers",
                    "effective_meetings",
                    "returning_activation_share",
                    "nearby",
                ),
                quiet_period=True,
            )
        )
    return pd.concat(tables, ignore_index=True)


def model_row(
    models: pd.DataFrame, specification: str, horizon: int, predictor: str
) -> pd.Series:
    return models[
        models["specification"].eq(specification)
        & models["horizon_meetings"].eq(horizon)
        & models["predictor"].eq(predictor)
    ].iloc[0]


def format_interval(row: pd.Series) -> str:
    return (
        f"{row.contrast_ratio:.2f} "
        f"[{row.contrast_ci_low:.2f}, {row.contrast_ci_high:.2f}]"
    )


def add_diagnostics(panel: pd.DataFrame, diagnostics: dict) -> dict:
    result = diagnostics.copy()
    result["time_unit"] = "ordered ATCM meetings"
    result["horizons_meetings"] = list(HORIZONS)
    result["analysis_restriction"] = "positive focal-concern attention in the prior window"
    correlations = {}
    for horizon in HORIZONS:
        columns = [
            f"papers_prior{horizon}",
            f"active_meetings_prior{horizon}",
            f"effective_meetings_prior{horizon}",
            f"returning_activation_share_prior{horizon}",
            f"latest_attention_lag_prior{horizon}",
            f"paper_weighted_lag_prior{horizon}",
        ]
        data = panel.dropna(subset=columns)
        data = data[data[f"papers_prior{horizon}"].gt(0)]
        correlations[str(horizon)] = (
            data[columns].corr().round(6).to_dict()
        )
    result["predictor_correlations_positive_attention"] = correlations
    return result


def plot_figure(models: pd.DataFrame) -> None:
    figure, axes = uplt.subplots(
        ncols=3,
        refwidth=3.1,
        refaspect=1.0,
        share=False,
        wspace=4.8,
    )
    panels = [
        (
            "attention_hurdle_hard_output",
            "any_attention",
            "Direct attention marks output",
            "Output rate ratio\nfor one paper vs none",
            figstyle.ADOPTION,
        ),
        (
            "attention_hurdle_hard_output",
            "positive_log_papers",
            "More papers add a smaller signal",
            "Output rate ratio\nper doubling after the first",
            figstyle.FOCAL,
        ),
        (
            "recency_adjusted_hard_output",
            "effective_meetings",
            "Temporal spread adds little",
            "Output rate ratio\nfor 2 vs 1 effective meeting",
            figstyle.CONTRAST,
        ),
    ]
    for axis, (specification, predictor, title, ylabel, color) in zip(axes, panels):
        subset = models[
            models["specification"].eq(specification)
            & models["predictor"].eq(predictor)
        ].sort_values("horizon_meetings")
        x = subset["horizon_meetings"].to_numpy(float)
        y = subset["contrast_ratio"].to_numpy(float)
        low = subset["contrast_ci_low"].to_numpy(float)
        high = subset["contrast_ci_high"].to_numpy(float)
        axis.plot(x, y, color=color, marker="o", ms=6, lw=2.1)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="none",
            ecolor=color,
            elinewidth=1.7,
            capsize=3.5,
        )
        axis.axhline(1, color=figstyle.REFERENCE, lw=1.0, ls="--")
        padding = 0.08 * max(float(high.max() - low.min()), 0.2)
        lower = max(0.0, min(0.92, float(low.min()) - padding))
        upper = max(1.20, float(high.max()) + padding)
        axis.format(
            title=title,
            xlabel="Preceding ATCM meetings included",
            ylabel=ylabel,
            xlocator=list(HORIZONS),
            xlim=(2.5, 10.5),
            ylim=(lower, upper),
            grid=False,
        )
    axes.format(abc="A", abcloc="ul", abcsize=figstyle.FS_PANEL, abcborder=False)
    figstyle.apply_typography(axes)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure.save(f"{FIGURE_STEM}.pdf", transparent=True)
    figure.save(f"{FIGURE_STEM}.png", dpi=260, transparent=False)
    uplt.close(figure)


def write_report(models: pd.DataFrame, diagnostics: dict) -> None:
    hard = "volume_temporal_actor_hard_output"
    soft = "volume_temporal_actor_soft_output"
    onset = "quiet_period_onset_hard_output"
    hurdle_hard = "attention_hurdle_hard_output"
    hurdle_soft = "attention_hurdle_soft_output"
    hurdle_high = "attention_hurdle_high_confidence_hard_output"
    hurdle_any = "attention_hurdle_any_hard_output"
    hurdle_excluding_sites = "attention_hurdle_excluding_site_administration"
    recency_hard = "recency_adjusted_hard_output"
    recency_soft = "recency_adjusted_soft_output"
    recency_any = "recency_adjusted_any_hard_output"
    five_any = model_row(models, hurdle_hard, 5, "any_attention")
    five_volume = model_row(models, hurdle_hard, 5, "positive_log_papers")
    five_nearby = model_row(models, hurdle_hard, 5, "nearby")
    five_prior = model_row(models, hurdle_hard, 5, "outcomes")
    five_any_soft = model_row(models, hurdle_soft, 5, "any_attention")
    five_volume_soft = model_row(models, hurdle_soft, 5, "positive_log_papers")
    five_any_high = model_row(models, hurdle_high, 5, "any_attention")
    five_volume_high = model_row(
        models, hurdle_high, 5, "positive_log_papers"
    )
    five_any_occurrence = model_row(models, hurdle_any, 5, "any_attention")
    five_volume_occurrence = model_row(
        models, hurdle_any, 5, "positive_log_papers"
    )
    five_any_no_sites = model_row(
        models, hurdle_excluding_sites, 5, "any_attention"
    )
    five_volume_no_sites = model_row(
        models, hurdle_excluding_sites, 5, "positive_log_papers"
    )
    lines = [
        "# Does attention have to recur before formal output?",
        "",
        "All windows use ordered ATCM meetings. The models compare each concern with itself over time, account for meeting-wide changes, and hold nearby-concern papers and earlier focal-concern outputs constant.",
        "",
        "## The main decomposition",
        "",
        f"Across the preceding five meetings, moving from no focal paper to one is associated with a hard-output rate ratio of {format_interval(five_any)}. Among concerns already receiving direct attention, each doubling of focal papers is associated with {format_interval(five_volume)} times as many outputs. Nearby-concern attention adds little ({format_interval(five_nearby)}), while earlier focal-concern outputs remain associated with later output ({format_interval(five_prior)}).",
        "",
        f"The soft-assignment estimates point in the same direction: {format_interval(five_any_soft)} for the first focal paper and {format_interval(five_volume_soft)} per additional doubling. Restricting hard assignments to high-confidence titles gives {format_interval(five_any_high)} and {format_interval(five_volume_high)}, respectively. In the binary occurrence model, the first focal paper is strongly associated with whether any output appears ({format_interval(five_any_occurrence)}), while extra paper volume is not ({format_interval(five_volume_occurrence)}). After routine site-administration concerns are removed, the first-paper association remains ({format_interval(five_any_no_sites)}) but the additional-volume estimate becomes imprecise ({format_interval(five_volume_no_sites)}).",
        "",
        "This separates two signals that the original cumulative-paper coefficient combined. Direct activation mainly distinguishes where any formal output appears; additional paper volume is more closely related to how much output appears on concerns that are already active.",
        "",
        "## Does temporal recurrence add information?",
        "",
    ]
    for horizon in HORIZONS:
        temporal = model_row(models, recency_hard, horizon, "effective_meetings")
        temporal_soft = model_row(models, recency_soft, horizon, "effective_meetings")
        temporal_any = model_row(models, recency_any, horizon, "effective_meetings")
        actors = model_row(
            models, recency_hard, horizon, "returning_activation_share"
        )
        lines.append(
            f"Across {horizon} preceding meetings, spreading the same focal-paper volume from one to two effective meetings yields a hard-output ratio of {format_interval(temporal)}, a soft-output ratio of {format_interval(temporal_soft)}, and an any-output ratio of {format_interval(temporal_any)} after also accounting for how recently attention occurred. The hard-output ratio for a 25-percentage-point increase in returning-actor activity is {format_interval(actors)}."
        )
        lines.append("")
    lines.extend(
        [
            "Temporal spread does not add a stable signal beyond volume. Its hard-count estimate approaches a positive association at ten meetings, but this does not reproduce for soft output, output occurrence, or the site-administration exclusion. Returning activity by the same actors is also unstable once volume, timing, nearby attention, and earlier output are held fixed. The record therefore does not support a strong claim that repetition across meetings or by the same actors independently drives output.",
            "",
            "## Quiet-to-active transitions",
            "",
        ]
    )
    for horizon in (5, 10):
        temporal_soft = model_row(models, soft, horizon, "effective_meetings")
        temporal_onset = model_row(models, onset, horizon, "effective_meetings")
        lines.append(
            f"At {horizon} meetings, the temporal-spread ratio is {format_interval(temporal_soft)} for soft-assigned output and {format_interval(temporal_onset)} for the onset of output after a window with no earlier focal-concern output. The onset model contains {int(temporal_onset.n_events)} events."
        )
        lines.append("")
    lines.extend(
        [
            "The onset estimates are exploratory because the number of quiet-to-active transitions is small and the predictors are correlated. They do not establish that repeated attention causes formal action.",
            "",
            "## Narrative implication",
            "",
            "The concern space describes where documentary portfolios expand. Formal action is associated most clearly with direct attention to the focal concern and with earlier formal instruments, not with proximity alone. A precise narrative is therefore: actors explore locally through the concern space; formal output concentrates where explored concerns receive direct documentary attention. More papers accompany more output, but the evidence does not show that simply spreading those papers across meetings, or having the same actors return, independently carries a concern into formal action.",
            "",
            "These are descriptive associations. Output may sustain later papers, and stable issue streams may generate both recurring attention and repeated output.",
            "",
            "## Reproducibility",
            "",
            "Run `micromamba run -n ultraplot-dev python scripts/analyze_attention_recurrence.py` from the manuscript repository.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    base_panel = pd.read_csv(PANEL_PATH)
    panel = accumulation.add_attention_stocks(base_panel)
    topics = sorted(panel["topic"].unique())
    meetings = sorted(panel["meeting"].unique().astype(int))
    actor_activity, diagnostics = build_actor_activity(topics, meetings)
    panel = add_recurrence_measures(panel, actor_activity)
    diagnostics = add_diagnostics(panel, diagnostics)
    models = fit_all(panel)
    PANEL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_EXPORT_PATH, index=False)
    models.to_csv(MODEL_PATH, index=False)
    DIAGNOSTIC_PATH.write_text(json.dumps(diagnostics, indent=2) + "\n")
    plot_figure(models)
    write_report(models, diagnostics)
    print(REPORT_PATH.read_text())


if __name__ == "__main__":
    main()
