#!/usr/bin/env python3
"""Explore how documentary attention accumulates before formal ATS output.

The analysis uses ATCM order throughout. It asks three related questions:

1. Does attention accumulated over several preceding meetings predict output?
2. Does exact-concern attention differ from attention in nearby concerns?
3. Does attention rise before a new output episode, or mainly persist within
   an existing stream of output?
The models are descriptive. Fixed effects absorb stable concern differences
and meeting-wide changes, but they do not turn attention into a causal effect.
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
OUTDIR = ROOT / "output" / "outcome_linkage"
FIGDIR = ROOT / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle


PANEL_PATH = OUTDIR / "topic_meeting_attention_outcomes.csv"
PREDICTIONS_PATH = OUTDIR / "outcome_topic_predictions.csv"
PROBABILITIES_PATH = OUTDIR / "outcome_topic_probabilities.csv"
MODEL_PATH = OUTDIR / "attention_accumulation_models.csv"
EVENT_PATH = OUTDIR / "attention_accumulation_event_trajectories.csv"
DIAGNOSTIC_PATH = OUTDIR / "attention_accumulation_diagnostics.json"
REPORT_PATH = OUTDIR / "attention_accumulation_report.md"
FIGURE_STEM = FIGDIR / "exploratory_attention_accumulation"

HORIZONS = (1, 2, 3, 5, 8, 10)
EVENT_WINDOW = 10
EPISODE_GAP = 5
N_BOOTSTRAP = 5_000
SEED = 20260814
SITE_ADMIN_TOPICS = {
    "Management Plans",
    "Area Protection and Management Plans General",
    "Historic Sites and Monuments",
    "Site Guidelines for Visitors",
}


def rolling_prior(values: pd.Series, horizon: int) -> pd.Series:
    return values.shift(1).rolling(horizon, min_periods=horizon).sum()


def rolling_future(values: pd.Series, horizon: int) -> pd.Series:
    return (
        values.iloc[::-1]
        .shift(1)
        .rolling(horizon, min_periods=horizon)
        .sum()
        .iloc[::-1]
    )


def add_attention_stocks(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["topic", "meeting"]).copy()
    sources = {
        "papers": "paper_count",
        "nearby": "neighbor_papers",
        "outcomes": "outcome_count_hard",
    }
    for horizon in HORIZONS:
        for stem, source in sources.items():
            data[f"{stem}_prior{horizon}"] = data.groupby("topic")[source].transform(
                lambda values, h=horizon: rolling_prior(values, h)
            )
    for stem, source in sources.items():
        data[f"{stem}_future5"] = data.groupby("topic")[source].transform(
            lambda values: rolling_future(values, 5)
        )
    return data


def fit_ppml(
    panel: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    specification: str,
    horizon: int | None = None,
    minimum_meeting: int | None = None,
) -> pd.DataFrame:
    data = panel.dropna(subset=predictors).copy()
    if minimum_meeting is not None:
        data = data[data["meeting"].ge(minimum_meeting)].copy()
    z_terms = []
    standard_deviations = {}
    for predictor in predictors:
        transformed = np.log1p(data[predictor].to_numpy(float))
        standard_deviation = max(float(transformed.std(ddof=0)), 1e-12)
        term = f"z_{predictor}"
        data[term] = (transformed - transformed.mean()) / standard_deviation
        standard_deviations[predictor] = standard_deviation
        z_terms.append(term)

    fitted = smf.glm(
        formula=f"{outcome} ~ {' + '.join(z_terms)} + C(topic) + C(meeting)",
        data=data,
        family=sm.families.Poisson(),
    ).fit()
    covariance, _, _ = cov_cluster_2groups(
        fitted,
        pd.Categorical(data["topic"]).codes,
        pd.Categorical(data["meeting"]).codes,
    )
    names = list(fitted.params.index)
    rows = []
    for predictor, term in zip(predictors, z_terms):
        index = names.index(term)
        coefficient = float(fitted.params[term])
        standard_error = math.sqrt(max(float(covariance[index, index]), 0.0))
        raw_coefficient = coefficient / standard_deviations[predictor]
        raw_standard_error = standard_error / standard_deviations[predictor]
        doubling_log_change = math.log(2.0)
        rows.append(
            {
                "specification": specification,
                "outcome": outcome,
                "horizon_meetings": horizon,
                "predictor": predictor,
                "coefficient": coefficient,
                "se_two_way_cluster": standard_error,
                "incidence_rate_ratio": math.exp(coefficient),
                "ci_low": math.exp(coefficient - 1.96 * standard_error),
                "ci_high": math.exp(coefficient + 1.96 * standard_error),
                "ratio_per_doubling_plus_one": math.exp(
                    raw_coefficient * doubling_log_change
                ),
                "doubling_ci_low": math.exp(
                    (raw_coefficient - 1.96 * raw_standard_error)
                    * doubling_log_change
                ),
                "doubling_ci_high": math.exp(
                    (raw_coefficient + 1.96 * raw_standard_error)
                    * doubling_log_change
                ),
                "n_topic_meetings": int(len(data)),
                "n_events": int(data[outcome].gt(0).sum()),
                "minimum_meeting": int(data["meeting"].min()),
                "scale": (
                    "one sample SD of log1p predictor; SD="
                    f"{standard_deviations[predictor]:.6g}"
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_episode_onset(
    panel: pd.DataFrame,
    horizon: int = EPISODE_GAP,
    output_source: str = "outcome_count_hard",
    specification: str = "new_output_episode_after_five_quiet_meetings",
) -> pd.DataFrame:
    data = panel.copy()
    data["_prior_output"] = data.groupby("topic")[output_source].transform(
        lambda values: values.shift(1).rolling(horizon, min_periods=horizon).sum()
    )
    data = data.dropna(
        subset=[
            f"papers_prior{horizon}",
            f"nearby_prior{horizon}",
            output_source,
            "_prior_output",
        ]
    ).copy()
    outcome = f"onset_after_{horizon}_quiet_meetings"
    data[outcome] = (
        data[output_source].gt(0)
        & data["_prior_output"].eq(0)
    ).astype(int)
    data = data[
        data.groupby("meeting")[outcome].transform("sum").gt(0)
    ].copy()
    predictors = [f"papers_prior{horizon}", f"nearby_prior{horizon}"]
    z_terms = []
    standard_deviations = {}
    for predictor in predictors:
        values = np.log1p(data[predictor].to_numpy(float))
        standard_deviation = max(float(values.std(ddof=0)), 1e-12)
        term = f"z_{predictor}"
        data[term] = (values - values.mean()) / standard_deviation
        standard_deviations[predictor] = standard_deviation
        z_terms.append(term)
    fitted = smf.glm(
        f"{outcome} ~ {' + '.join(z_terms)} + C(topic) + C(meeting)",
        data=data,
        family=sm.families.Binomial(),
    ).fit()
    covariance, _, _ = cov_cluster_2groups(
        fitted,
        pd.Categorical(data["topic"]).codes,
        pd.Categorical(data["meeting"]).codes,
    )
    names = list(fitted.params.index)
    rows = []
    for predictor, term in zip(predictors, z_terms):
        index = names.index(term)
        coefficient = float(fitted.params[term])
        standard_error = math.sqrt(max(float(covariance[index, index]), 0.0))
        raw_coefficient = coefficient / standard_deviations[predictor]
        raw_standard_error = standard_error / standard_deviations[predictor]
        doubling_log_change = math.log(2.0)
        rows.append(
            {
                "specification": specification,
                "outcome": outcome,
                "horizon_meetings": horizon,
                "predictor": predictor,
                "coefficient": coefficient,
                "se_two_way_cluster": standard_error,
                "incidence_rate_ratio": math.exp(coefficient),
                "ci_low": math.exp(coefficient - 1.96 * standard_error),
                "ci_high": math.exp(coefficient + 1.96 * standard_error),
                "ratio_per_doubling_plus_one": math.exp(
                    raw_coefficient * doubling_log_change
                ),
                "doubling_ci_low": math.exp(
                    (raw_coefficient - 1.96 * raw_standard_error)
                    * doubling_log_change
                ),
                "doubling_ci_high": math.exp(
                    (raw_coefficient + 1.96 * raw_standard_error)
                    * doubling_log_change
                ),
                "n_topic_meetings": int(len(data)),
                "n_events": int(data[outcome].sum()),
                "minimum_meeting": int(data["meeting"].min()),
                "scale": "odds ratio per one sample SD of log1p predictor",
            }
        )
    own_index = names.index(z_terms[0])
    nearby_index = names.index(z_terms[1])
    contrast = float(fitted.params[z_terms[0]] - fitted.params[z_terms[1]])
    contrast_variance = float(
        covariance[own_index, own_index]
        + covariance[nearby_index, nearby_index]
        - 2 * covariance[own_index, nearby_index]
    )
    contrast_se = math.sqrt(max(contrast_variance, 0.0))
    rows.append(
        {
            "specification": specification,
            "outcome": outcome,
            "horizon_meetings": horizon,
            "predictor": "papers_prior5_minus_nearby_prior5",
            "coefficient": contrast,
            "se_two_way_cluster": contrast_se,
            "incidence_rate_ratio": math.exp(contrast),
            "ci_low": math.exp(contrast - 1.96 * contrast_se),
            "ci_high": math.exp(contrast + 1.96 * contrast_se),
            "n_topic_meetings": int(len(data)),
            "n_events": int(data[outcome].sum()),
            "minimum_meeting": int(data["meeting"].min()),
            "scale": "ratio of focal-attention OR to nearby-attention OR, each per one SD",
        }
    )
    return pd.DataFrame(rows)


def build_event_trajectories(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["topic", "meeting"]).copy()
    data["log_papers"] = np.log1p(data["paper_count"])
    data["attention_residual"] = (
        data["log_papers"]
        - data.groupby("topic")["log_papers"].transform("mean")
        - data.groupby("meeting")["log_papers"].transform("mean")
        + data["log_papers"].mean()
    )
    data["prior_output_five"] = data.groupby("topic")["outcome_count_hard"].transform(
        lambda values: values.shift(1).rolling(
            EPISODE_GAP, min_periods=EPISODE_GAP
        ).sum()
    )
    events = data[
        data["outcome_count_hard"].gt(0) & data["prior_output_five"].notna()
    ].copy()
    events["event_type"] = np.where(
        events["prior_output_five"].gt(0),
        "Continuing output stream",
        "After five meetings without output",
    )
    lookup = data.set_index(["topic", "meeting"])["attention_residual"]
    rows = []
    for event in events.itertuples(index=False):
        for relative_meeting in range(-EVENT_WINDOW, 1):
            key = (event.topic, int(event.meeting) + relative_meeting)
            if key not in lookup.index:
                continue
            rows.append(
                {
                    "topic": event.topic,
                    "event_meeting": int(event.meeting),
                    "event_type": event.event_type,
                    "relative_meeting": relative_meeting,
                    "attention_residual": float(lookup.loc[key]),
                }
            )
    event_rows = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    summaries = []
    for (event_type, relative_meeting), group in event_rows.groupby(
        ["event_type", "relative_meeting"]
    ):
        topic_totals = group.groupby("topic")["attention_residual"].agg(["sum", "count"])
        sums = topic_totals["sum"].to_numpy(float)
        counts = topic_totals["count"].to_numpy(float)
        draws = np.empty(N_BOOTSTRAP)
        for draw_index in range(N_BOOTSTRAP):
            selected = rng.choice(len(topic_totals), len(topic_totals), replace=True)
            draws[draw_index] = sums[selected].sum() / counts[selected].sum()
        summaries.append(
            {
                "event_type": event_type,
                "relative_meeting": int(relative_meeting),
                "mean_residual_attention": float(group["attention_residual"].mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "n_events": int(len(group)),
                "n_topics": int(group["topic"].nunique()),
            }
        )
    return pd.DataFrame(summaries)


def fit_all_models(panel: pd.DataFrame, diagnostics: dict) -> pd.DataFrame:
    panel = panel.copy()
    panel["any_hard_output"] = panel["outcome_count_hard"].gt(0).astype(int)
    tables = []
    for horizon in HORIZONS:
        predictors = [
            f"papers_prior{horizon}",
            f"nearby_prior{horizon}",
            f"outcomes_prior{horizon}",
        ]
        tables.append(
            fit_ppml(
                panel,
                "outcome_count_hard",
                predictors,
                "accumulated_attention_hard_output",
                horizon,
            )
        )
        tables.append(
            fit_ppml(
                panel,
                "outcome_mass",
                predictors,
                "accumulated_attention_soft_output",
                horizon,
            )
        )

    tables.append(
        fit_ppml(
            panel,
            "outcome_count_hard",
            [
                "papers_prior5",
                "nearby_prior5",
                "outcomes_prior5",
                "papers_future5",
                "nearby_future5",
                "outcomes_future5",
            ],
            "past_and_future_placebo_hard_output",
            5,
        )
    )
    tables.append(
        fit_ppml(
            panel,
            "any_hard_output",
            ["papers_prior5", "nearby_prior5", "outcomes_prior5"],
            "any_output_occurrence_sensitivity",
            5,
        )
    )
    tables.append(
        fit_ppml(
            panel[~panel["topic"].isin(SITE_ADMIN_TOPICS)],
            "outcome_count_hard",
            ["papers_prior5", "nearby_prior5", "outcomes_prior5"],
            "excluding_site_administration_sensitivity",
            5,
        )
    )
    tables.append(fit_episode_onset(panel))
    tables.append(
        fit_episode_onset(
            panel,
            output_source="outcome_count_high_confidence",
            specification="new_high_confidence_output_episode_after_five_quiet_meetings",
        )
    )
    tables.append(
        fit_episode_onset(
            panel[~panel["topic"].isin(SITE_ADMIN_TOPICS)],
            specification="new_output_episode_after_five_quiet_meetings_excluding_site_administration",
        )
    )
    return pd.concat(tables, ignore_index=True, sort=False)


def plot_figure(models: pd.DataFrame, trajectories: pd.DataFrame) -> None:
    fig, axes = uplt.subplots(
        ncols=3,
        refwidth=3.15,
        refaspect=1.0,
        share=False,
        wspace=5.2,
    )
    ax_a, ax_b, ax_c = axes

    horizon_data = models[
        models["specification"].eq("accumulated_attention_hard_output")
    ]
    line_specs = [
        ("papers_prior", "Same concern", figstyle.FOCAL, "o"),
        ("nearby_prior", "Nearby concerns", figstyle.NEARBY, "s"),
        ("outcomes_prior", "Earlier outputs", figstyle.CONTRAST, "^"),
    ]
    for prefix, label, color, marker in line_specs:
        subset = horizon_data[
            horizon_data["predictor"].str.startswith(prefix)
        ].sort_values("horizon_meetings")
        x = subset["horizon_meetings"].to_numpy(float)
        y = subset["ratio_per_doubling_plus_one"].to_numpy(float)
        low = subset["doubling_ci_low"].to_numpy(float)
        high = subset["doubling_ci_high"].to_numpy(float)
        ax_a.plot(x, y, color=color, lw=2.0, marker=marker, ms=5.5)
        ax_a.fill_between(x, low, high, color=color, alpha=0.13)
        ax_a.text(
            10.25,
            y[-1],
            label,
            color=color,
            fontsize=figstyle.FS_LEGEND,
            ha="left",
            va="center",
        )
    ax_a.axhline(1, color=figstyle.REFERENCE, lw=1.0, ls="--")
    ax_a.format(
        title="Attention accumulates within concerns",
        xlabel="Preceding ATCM meetings included",
        ylabel="Output rate ratio per doubling",
        xlocator=list(HORIZONS),
        xlim=(0.7, 13.9),
        ylim=(0.72, 1.86),
        grid=False,
    )

    trajectory_specs = [
        ("Continuing output stream", "Continuing stream", figstyle.ADOPTION),
        (
            "After five meetings without output",
            "New episode after a quiet period",
            figstyle.CYAN,
        ),
    ]
    for event_type, label, color in trajectory_specs:
        subset = trajectories[trajectories["event_type"].eq(event_type)].sort_values(
            "relative_meeting"
        )
        x = subset["relative_meeting"].to_numpy(float)
        y = subset["mean_residual_attention"].to_numpy(float)
        ax_b.plot(x, y, color=color, lw=2.1, marker="o", ms=4.5, label=label)
        ax_b.fill_between(
            x,
            subset["ci_low"].to_numpy(float),
            subset["ci_high"].to_numpy(float),
            color=color,
            alpha=0.14,
        )
    ax_b.axhline(0, color=figstyle.REFERENCE, lw=1.0, ls="--")
    ax_b.axvline(0, color=figstyle.TEXT, lw=0.9)
    ax_b.format(
        title="Lead-up appears in continuing output streams",
        xlabel="Meetings before formal output",
        ylabel="Adjusted paper attention",
        xlim=(-10.25, 0.25),
        ylim=(-0.38, 0.65),
        xlocator=[-10, -8, -6, -4, -2, 0],
        grid=False,
    )
    ax_b.legend(loc="ur", ncols=1, frame=False, fontsize=figstyle.FS_LEGEND)

    onset = models[
        models["specification"].eq("new_output_episode_after_five_quiet_meetings")
    ].set_index("predictor")
    for predictor, label, color, y in (
        ("papers_prior5", "Same-concern papers", figstyle.FOCAL, 1),
        ("nearby_prior5", "Nearby-concern papers", figstyle.NEARBY, 0),
    ):
        row = onset.loc[predictor]
        ax_c.errorbar(
            [row["incidence_rate_ratio"]],
            [y],
            xerr=np.array(
                [[row["incidence_rate_ratio"] - row["ci_low"]],
                 [row["ci_high"] - row["incidence_rate_ratio"]]]
            ),
            fmt="o", color=color, ecolor=color, capsize=3, ms=6, lw=1.8,
        )
        ax_c.text(0.43, y + 0.20, label, fontsize=figstyle.FS_LEGEND, color=color)
    ax_c.axvline(1, color=figstyle.REFERENCE, lw=1.0, ls="--")
    ax_c.format(
        title="New output episodes remain hard to anticipate",
        xlabel="Odds ratio per 1-SD increase",
        ylabel="",
        xscale="log", xlocator=[0.5, 1, 2, 4], xformatter="{x:g}",
        xlim=(0.40, 4.2), yticks=[], ylim=(-0.45, 1.65),
        grid=False,
    )

    axes.format(
        abc="A",
        abcloc="ul",
        abcsize=figstyle.FS_PANEL,
        abcborder=False,
    )
    figstyle.apply_typography(axes)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.save(f"{FIGURE_STEM}.pdf", transparent=True)
    fig.save(f"{FIGURE_STEM}.png", dpi=260, transparent=False)
    uplt.close(fig)


def number(models: pd.DataFrame, specification: str, predictor: str) -> pd.Series:
    return models[
        models["specification"].eq(specification)
        & models["predictor"].eq(predictor)
    ].iloc[0]


def interval(row: pd.Series) -> str:
    return (
        f"{row.incidence_rate_ratio:.2f} "
        f"[{row.ci_low:.2f}, {row.ci_high:.2f}]"
    )


def doubling_interval(row: pd.Series) -> str:
    return (
        f"{row.ratio_per_doubling_plus_one:.2f} "
        f"[{row.doubling_ci_low:.2f}, {row.doubling_ci_high:.2f}]"
    )


def write_report(models: pd.DataFrame, diagnostics: dict) -> None:
    hard_five = "accumulated_attention_hard_output"
    soft_five = "accumulated_attention_soft_output"
    exact_hard = number(models, hard_five, "papers_prior5")
    nearby_hard = number(models, hard_five, "nearby_prior5")
    earlier_hard = number(models, hard_five, "outcomes_prior5")
    exact_soft = number(models, soft_five, "papers_prior5")
    exact_occurrence = number(
        models, "any_output_occurrence_sensitivity", "papers_prior5"
    )
    exact_without_site = number(
        models,
        "excluding_site_administration_sensitivity",
        "papers_prior5",
    )
    onset_exact = number(
        models,
        "new_output_episode_after_five_quiet_meetings",
        "papers_prior5",
    )
    onset_nearby = number(
        models,
        "new_output_episode_after_five_quiet_meetings",
        "nearby_prior5",
    )
    placebo_past = number(
        models, "past_and_future_placebo_hard_output", "papers_prior5"
    )
    placebo_future = number(
        models, "past_and_future_placebo_hard_output", "papers_future5"
    )
    lines = [
        "# How documentary attention relates to formal output",
        "",
        "All time windows are ordered ATCM meetings, not calendar years. The models compare a concern with itself over time and account for changes shared by each meeting. They are descriptive associations, not causal effects.",
        "",
        "## Main result",
        "",
        f"Across the preceding five meetings, doubling same-concern paper attention plus one is associated with later hard-assigned output at a rate ratio of {doubling_interval(exact_hard)} after accounting for nearby-concern papers and earlier output. The soft-assignment estimate is {doubling_interval(exact_soft)}. The corresponding ratios are {doubling_interval(nearby_hard)} for nearby-concern attention and {doubling_interval(earlier_hard)} for earlier formal output. The same-concern result remains when the outcome is only whether any output occurs ({doubling_interval(exact_occurrence)}) and after removing routine site-administration concerns ({doubling_interval(exact_without_site)}).",
        "",
        "The same-concern association grows through roughly five preceding meetings and remains positive through ten. Nearby-concern attention fades as the window widens. A longer institutional horizon therefore reveals sustained attention--output alignment at the concern level rather than through the wider concern network.",
        "",
        "## New episodes versus continuing streams",
        "",
        f"There are {int(onset_exact.get('n_events', 0))} concern--meeting output episodes after five meetings with no output on that concern. Prior same-concern attention does not clearly distinguish those onsets (OR {interval(onset_exact)}), nor does nearby attention (OR {interval(onset_nearby)}). Event-centered trajectories show the clearest lead-up within continuing output streams, not before genuinely quiet-to-active transitions.",
        "",
        "## Temporal caution",
        "",
        f"When five preceding and five subsequent meetings enter together, doubling past same-concern attention plus one is associated with output at {doubling_interval(placebo_past)}. The future estimate is smaller and uncertain at {doubling_interval(placebo_future)}. Anticipated output, later attention, and stable issue streams can still contribute to both sides of the association. The analysis does not identify a one-way causal pipeline.",
        "",
        "## Interpretation",
        "",
        "The concern space predicts where actors shift relative documentary attention. Adopted ATCM outputs are tied more closely to accumulated attention on the focal concern than to activity in nearby concerns; earlier output provides no stable additional association. New output on a previously quiet concern remains difficult to anticipate.",
        "",
        "## Reproducibility",
        "",
        "Run `micromamba run -n ultraplot-dev python scripts/analyze_attention_accumulation.py` from the manuscript repository.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    panel = pd.read_csv(PANEL_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    probabilities = pd.read_csv(PROBABILITIES_PATH)
    panel = add_attention_stocks(panel)
    diagnostics = {
        "classified_outputs": int(len(predictions)),
        "output_scope": "regular ATCMs 19--47 (1995--2025)",
    }
    models = fit_all_models(panel, diagnostics)
    trajectories = build_event_trajectories(panel)
    models.to_csv(MODEL_PATH, index=False)
    trajectories.to_csv(EVENT_PATH, index=False)
    diagnostics.update(
        {
            "time_unit": "ordered ATCM meetings",
            "rolling_horizons_meetings": list(HORIZONS),
            "new_episode_gap_meetings": EPISODE_GAP,
            "model_interpretation": "descriptive within-concern association",
        }
    )
    DIAGNOSTIC_PATH.write_text(json.dumps(diagnostics, indent=2) + "\n")
    plot_figure(models, trajectories)
    write_report(models, diagnostics)
    print(REPORT_PATH.read_text())


if __name__ == "__main__":
    main()
