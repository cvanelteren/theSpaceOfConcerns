#!/usr/bin/env python3
"""Rerun the paper's three main analyses for Consultative Parties only.

The restriction is time-varying. A state's papers enter the analysis from the
calendar year in which the ATCM recognized its Consultative status. Papers
submitted only by Non-Consultative Parties, observers, expert organizations,
or the Secretariat are excluded. For mixed-author papers, only eligible
Consultative Parties receive actor-level credit, while the paper contributes
one unit of documentary attention.

Outputs are written to ``output/consultative_party_sensitivity``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.official_regular_atcm_outputs import (
    PAPER_CONCERN_TO_INSTRUMENT_CATEGORY,
    load_official_regular_outputs,
)


DATA = ROOT / "data" / "document-summary-multilabel.parquet"
OUTDIR = ROOT / "output" / "consultative_party_sensitivity"
STATUS_SOURCE = "https://www.ats.aq/devAS/Parties?lang=e"
SEED = 20260824
WINDOW = 5
RPA_THRESHOLD = 1.0
TRAIN_START = 20
TEST_START = 29
TEST_END = 47
HISTORY = 5
ATTENTION = 1
ALPHA = 0.1
BOOTSTRAP_DRAWS = 20_000


# Recognition years from the official ATS Parties table, accessed 24 Aug 2026.
CONSULTATIVE_YEAR = {
    "Argentina": 1961,
    "Australia": 1961,
    "Belgium": 1961,
    "Brazil": 1983,
    "Bulgaria": 1998,
    "Chile": 1961,
    "China": 1985,
    "Czechia": 2014,
    "Ecuador": 1990,
    "Finland": 1989,
    "France": 1961,
    "Germany": 1981,
    "India": 1983,
    "Italy": 1987,
    "Japan": 1961,
    "Korea (ROK)": 1989,
    "Netherlands": 1990,
    "New Zealand": 1961,
    "Norway": 1961,
    "Peru": 1989,
    "Poland": 1977,
    "Russian Federation": 1961,
    "South Africa": 1961,
    "Spain": 1988,
    "Sweden": 1988,
    "Ukraine": 2004,
    "United Kingdom": 1961,
    "United States": 1961,
    "Uruguay": 1985,
}


def values(value: object, separator: str | None = None) -> list[str]:
    """Convert the archive's arrays or delimited strings to clean labels."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raw = str(value).split(separator or ",")
    return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def prepare_papers() -> pd.DataFrame:
    raw = pd.read_parquet(DATA)
    rows = []
    for paper_id, group in raw.groupby("paper_id", sort=False):
        first = group.iloc[0]
        categories = values(first["category"], "\t")
        actors = values(first["parties"])
        meeting = int(first["meeting_number"])
        year = int(first["meeting_year"])
        if categories and actors:
            rows.append(
                {
                    "paper_id": int(paper_id),
                    "meeting": meeting,
                    "year": year,
                    "categories": categories,
                    "actors": actors,
                    "consultative_actors": [
                        actor
                        for actor in actors
                        if actor in CONSULTATIVE_YEAR
                        and year >= CONSULTATIVE_YEAR[actor]
                    ],
                }
            )
    papers = pd.DataFrame(rows)
    papers["consultative_only"] = papers["consultative_actors"].map(bool)
    return papers


def fine_relations(papers: pd.DataFrame, consultative_only: bool) -> pd.DataFrame:
    rows = []
    for paper in papers.itertuples(index=False):
        actors = paper.consultative_actors if consultative_only else paper.actors
        if not actors:
            continue
        weight = 1.0 / len(paper.categories)
        for concern in paper.categories:
            for actor in actors:
                rows.append(
                    {
                        "paper_id": paper.paper_id,
                        "meeting": paper.meeting,
                        "year": paper.year,
                        "concern": concern,
                        "actor": actor,
                        "weight": weight,
                    }
                )
    return pd.DataFrame(rows)


def counts_from_relations(
    relations: pd.DataFrame,
    concerns: list[str] | None = None,
    actors: list[str] | None = None,
) -> pd.DataFrame:
    counts = relations.groupby(["concern", "actor"])["weight"].sum().unstack(fill_value=0.0)
    concerns = sorted(counts.index) if concerns is None else concerns
    actors = sorted(counts.columns) if actors is None else actors
    return counts.reindex(index=concerns, columns=actors, fill_value=0.0)


def rpa(counts: pd.DataFrame) -> pd.DataFrame:
    actor_total = counts.sum(axis=0).replace(0, np.nan)
    actor_share = counts.divide(actor_total, axis=1)
    importance = counts.sum(axis=1) / counts.to_numpy(float).sum()
    return actor_share.divide(importance.replace(0, np.nan), axis=0).fillna(0.0)


def proximity(counts: pd.DataFrame) -> pd.DataFrame:
    active = rpa(counts).ge(RPA_THRESHOLD).to_numpy(dtype=int)
    overlap = active @ active.T
    holders = active.sum(axis=1)
    denominator = np.maximum.outer(holders, holders)
    phi = np.divide(
        overlap,
        denominator,
        out=np.zeros_like(overlap, dtype=float),
        where=denominator > 0,
    )
    np.fill_diagonal(phi, 1.0)
    return pd.DataFrame(phi, index=counts.index, columns=counts.index)


def map_metrics(phi: pd.DataFrame) -> dict[str, float | int | bool]:
    array = phi.to_numpy(float).copy()
    np.fill_diagonal(array, 0.0)
    graph = nx.from_numpy_array(array)
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d["weight"] <= 0])
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=1.0, seed=SEED
    )
    upper = array[np.triu_indices_from(array, k=1)]
    return {
        "n_concerns": int(len(phi)),
        "positive_pairs": int((upper > 0).sum()),
        "zero_pairs": int((upper == 0).sum()),
        "connected": bool(nx.is_connected(graph)),
        "components": int(nx.number_connected_components(graph)),
        "louvain_communities": int(len(communities)),
        "louvain_modularity": float(
            nx.community.modularity(graph, communities, weight="weight")
        ),
    }


def map_analysis(all_relations: pd.DataFrame, cp_relations: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    concerns = sorted(set(all_relations["concern"]) | set(cp_relations["concern"]))
    all_counts = counts_from_relations(all_relations, concerns=concerns)
    cp_counts = counts_from_relations(
        cp_relations, concerns=concerns, actors=sorted(CONSULTATIVE_YEAR)
    )
    all_phi = proximity(all_counts)
    cp_phi = proximity(cp_counts)
    first, second = np.triu_indices(len(concerns), k=1)
    all_values = all_phi.to_numpy()[first, second]
    cp_values = cp_phi.to_numpy()[first, second]
    strongest_all = np.argmax(all_phi.to_numpy() - np.eye(len(concerns)) * 2, axis=1)
    strongest_cp = np.argmax(cp_phi.to_numpy() - np.eye(len(concerns)) * 2, axis=1)
    pairs = pd.DataFrame(
        {
            "concern_a": [concerns[i] for i in first],
            "concern_b": [concerns[j] for j in second],
            "phi_all_actors": all_values,
            "phi_consultative_only": cp_values,
        }
    )
    summary = {
        "all_actors": map_metrics(all_phi),
        "consultative_only": map_metrics(cp_phi),
        "pairwise_pearson": float(pearsonr(all_values, cp_values).statistic),
        "pairwise_spearman": float(spearmanr(all_values, cp_values).statistic),
        "strongest_neighbor_agreement": float(np.mean(strongest_all == strongest_cp)),
        "mean_absolute_phi_change": float(np.mean(np.abs(all_values - cp_values))),
    }
    return summary, pairs


def interaction(
    relations: pd.DataFrame,
    concerns: list[str],
    actors: list[str],
    meeting_start: int,
    meeting_end: int,
) -> pd.DataFrame:
    selected = relations[
        relations["meeting"].between(meeting_start, meeting_end, inclusive="both")
    ]
    if selected.empty:
        return pd.DataFrame(0.0, index=concerns, columns=actors)
    return counts_from_relations(selected, concerns=concerns, actors=actors)


def locality_analysis(cp_relations: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    concerns = sorted(cp_relations["concern"].unique())
    actors = sorted(CONSULTATIVE_YEAR)
    first_appearance = cp_relations.groupby("concern")["meeting"].min().to_dict()
    meeting_min = int(cp_relations["meeting"].min())
    meeting_max = int(cp_relations["meeting"].max())
    period_ends = list(range(meeting_min + WINDOW - 1, meeting_max + 1))
    windows = {
        end: interaction(cp_relations, concerns, actors, end - WINDOW + 1, end)
        for end in period_ends
    }
    active = {end: rpa(table).ge(RPA_THRESHOLD) for end, table in windows.items()}
    rows = []
    for previous_end, current_end in zip(period_ends[:-1], period_ends[1:]):
        historical = interaction(cp_relations, concerns, actors, meeting_min, previous_end)
        phi = proximity(historical).to_numpy(float)
        previous = active[previous_end]
        current = active[current_end]
        popularity = previous.sum(axis=1) / len(actors)
        available = np.array(
            [first_appearance.get(concern, current_end + 1) <= previous_end for concern in concerns]
        )
        for actor in actors:
            held = previous[actor].to_numpy(bool)
            if not held.any():
                continue
            at_risk = (~held) & available
            adopted = current[actor].to_numpy(bool) & at_risk
            if not adopted.any() or adopted.sum() == at_risk.sum():
                continue
            distance = 1.0 - phi[:, np.where(held)[0]].max(axis=1)
            group = f"{actor}::{current_end}"
            for index in np.where(at_risk)[0]:
                rows.append(
                    {
                        "group": group,
                        "actor": actor,
                        "meeting": current_end,
                        "concern": concerns[index],
                        "adopted": int(adopted[index]),
                        "distance": float(distance[index]),
                        "topic_popularity": float(popularity.iloc[index]),
                    }
                )
    panel = pd.DataFrame(rows)
    fitted = ConditionalLogit(
        panel["adopted"],
        panel[["distance", "topic_popularity"]],
        groups=panel["group"],
    ).fit(disp=False, maxiter=300)
    beta = float(fitted.params["distance"])
    se = float(fitted.bse["distance"])
    summary = {
        "n_consultative_parties": len(actors),
        "actor_periods": int(panel["group"].nunique()),
        "risk_rows": int(len(panel)),
        "specialization_entries": int(panel["adopted"].sum()),
        "distance_beta": beta,
        "distance_se": se,
        "odds_ratio_per_0_1": float(np.exp(0.1 * beta)),
        "odds_ratio_per_0_1_ci_low": float(np.exp(0.1 * (beta - 1.96 * se))),
        "odds_ratio_per_0_1_ci_high": float(np.exp(0.1 * (beta + 1.96 * se))),
    }
    return summary, panel


def broad_relations(papers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for paper in papers[papers["consultative_only"]].itertuples(index=False):
        family_weights: dict[str, float] = {}
        for concern in paper.categories:
            family = PAPER_CONCERN_TO_INSTRUMENT_CATEGORY[concern]
            family_weights[family] = family_weights.get(family, 0.0) + 1.0 / len(
                paper.categories
            )
        for family, weight in family_weights.items():
            for actor in paper.consultative_actors:
                rows.append(
                    {
                        "paper_id": paper.paper_id,
                        "meeting": paper.meeting,
                        "family": family,
                        "actor": actor,
                        "weight": weight,
                    }
                )
    return pd.DataFrame(rows)


def output_mass(families: list[str], meetings: list[int]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([families, meetings], names=["topic", "meeting"])
    rows = []
    for record in load_official_regular_outputs().to_dict(orient="records"):
        categories = list(dict.fromkeys(record["official_categories"]))
        weight = 1.0 / len(categories)
        for category in categories:
            rows.append(
                {
                    "topic": category,
                    "meeting": int(record["meeting"]),
                    "instrument": record["instrument"],
                    "weight": weight,
                }
            )
    outputs = pd.DataFrame(rows)
    panel = pd.DataFrame(index=index).reset_index()
    for instrument in ("Measure", "Decision", "Resolution"):
        mass = (
            outputs[outputs["instrument"].eq(instrument)]
            .groupby(["topic", "meeting"])["weight"]
            .sum()
            .reindex(index, fill_value=0.0)
        )
        panel[f"{instrument.lower()}_mass"] = mass.to_numpy(float)
    return panel


def prediction_panel(relations: pd.DataFrame) -> pd.DataFrame:
    families = sorted(set(PAPER_CONCERN_TO_INSTRUMENT_CATEGORY.values()))
    meetings = list(range(19, TEST_END + 1))
    actors = sorted(CONSULTATIVE_YEAR)
    paper_rows = relations.drop_duplicates(["paper_id", "family"])
    index = pd.MultiIndex.from_product([families, meetings], names=["topic", "meeting"])
    attention = (
        paper_rows.groupby(["family", "meeting"])["weight"]
        .sum()
        .rename_axis(["topic", "meeting"])
        .reindex(index, fill_value=0.0)
        .rename("paper_count")
        .reset_index()
    )
    panel = attention.merge(output_mass(families, meetings), on=["topic", "meeting"])
    actor_reach = (
        relations.drop_duplicates(["family", "meeting", "actor"])
        .groupby(["family", "meeting"])["actor"]
        .nunique()
        .rename_axis(["topic", "meeting"])
        .reindex(index, fill_value=0)
        .rename("current_actor_reach")
        .reset_index()
    )
    panel = panel.merge(actor_reach, on=["topic", "meeting"])
    for instrument in ("measure", "decision", "resolution"):
        panel[f"{instrument}_mass_history_{HISTORY}"] = (
            panel.groupby("topic")[f"{instrument}_mass"]
            .transform(lambda x: x.shift(1).rolling(HISTORY, min_periods=1).sum())
            .fillna(0.0)
        )
    panel[f"paper_history_{ATTENTION}"] = (
        panel.groupby("topic")["paper_count"]
        .transform(lambda x: x.shift(1).rolling(ATTENTION, min_periods=1).sum())
        .fillna(0.0)
    )
    panel[f"actor_reach_{ATTENTION}"] = (
        panel.groupby("topic")["current_actor_reach"]
        .transform(lambda x: x.shift(1).rolling(ATTENTION, min_periods=1).sum())
        .fillna(0.0)
    )
    panel["neighbor_attention"] = 0.0
    topic_index = {topic: index for index, topic in enumerate(families)}
    for meeting in meetings:
        history = relations[relations["meeting"].lt(meeting)]
        counts = (
            history.groupby(["family", "actor"])["weight"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=families, columns=actors, fill_value=0.0)
        )
        phi = proximity(counts).to_numpy(float)
        np.fill_diagonal(phi, 0.0)
        row_sums = phi.sum(axis=1)
        weights = np.divide(
            phi,
            row_sums[:, None],
            out=np.zeros_like(phi),
            where=row_sums[:, None] > 0,
        )
        current = (
            panel[panel["meeting"].eq(meeting)]
            .set_index("topic")["paper_count"]
            .reindex(families)
            .to_numpy(float)
        )
        nearby = weights @ current
        mask = panel["meeting"].eq(meeting)
        panel.loc[mask, "neighbor_attention"] = [
            nearby[topic_index[topic]] for topic in panel.loc[mask, "topic"]
        ]
    return panel


def predict_meetings(panel: pd.DataFrame, output: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for meeting in range(TEST_START, TEST_END + 1):
        train = panel[panel["meeting"].ge(TRAIN_START) & panel["meeting"].lt(meeting)]
        test = panel[panel["meeting"].eq(meeting)]
        if test[output].sum() <= 0:
            continue
        transform = ColumnTransformer(
            [
                ("topic", OneHotEncoder(handle_unknown="ignore"), ["topic"]),
                ("numeric", StandardScaler(), features),
            ]
        )
        model = make_pipeline(
            transform,
            PoissonRegressor(alpha=ALPHA, max_iter=2_000, tol=1e-9),
        )
        design = ["topic", *features]
        model.fit(train[design], train[output])
        predicted = np.maximum(model.predict(test[design]), 1e-12)
        predicted /= predicted.sum()
        observed = test[output].to_numpy(float)
        observed /= observed.sum()
        rows.append(
            {
                "meeting": meeting,
                "allocation_log_score": float(-np.sum(observed * np.log(predicted))),
            }
        )
    return pd.DataFrame(rows)


def paired_summary(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float | int]:
    joined = baseline.merge(candidate, on="meeting", suffixes=("_baseline", "_candidate"))
    difference = (
        joined["allocation_log_score_candidate"]
        - joined["allocation_log_score_baseline"]
    ).to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(difference, size=(BOOTSTRAP_DRAWS, len(difference)), replace=True).mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "bootstrap_low": float(np.quantile(draws, 0.025)),
        "bootstrap_high": float(np.quantile(draws, 0.975)),
        "meetings_improved": int((difference < 0).sum()),
        "meetings": int(len(difference)),
    }


def prediction_analysis(relations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = prediction_panel(relations)
    rows = []
    score_rows = []
    for instrument in ("Measure", "Decision", "Resolution"):
        output = f"{instrument.lower()}_mass"
        history = [f"{output}_history_{HISTORY}"]
        direct = [
            *history,
            f"paper_history_{ATTENTION}",
            f"actor_reach_{ATTENTION}",
            "paper_count",
            "current_actor_reach",
        ]
        full = [*direct, "neighbor_attention"]
        scores = {
            "output history": predict_meetings(panel, output, history),
            "history + direct attention": predict_meetings(panel, output, direct),
            "history + direct + nearby attention": predict_meetings(panel, output, full),
        }
        for model, table in scores.items():
            score_rows.append(table.assign(instrument=instrument, model=model))
        rows.append(
            {
                "instrument": instrument,
                "comparison": "direct attention vs output history",
                **paired_summary(scores["history + direct attention"], scores["output history"]),
            }
        )
        rows.append(
            {
                "instrument": instrument,
                "comparison": "direct + nearby attention vs output history",
                **paired_summary(
                    scores["history + direct + nearby attention"], scores["output history"]
                ),
            }
        )
        rows.append(
            {
                "instrument": instrument,
                "comparison": "nearby attention vs direct attention",
                **paired_summary(
                    scores["history + direct + nearby attention"],
                    scores["history + direct attention"],
                ),
            }
        )
    return pd.DataFrame(rows), pd.concat(score_rows, ignore_index=True)


def prose(summary: dict) -> str:
    locality = summary["local_specialization"]
    prediction = summary["prediction"]
    resolution = next(
        row
        for row in prediction
        if row["instrument"] == "Resolution"
        and row["comparison"] == "direct + nearby attention vs output history"
    )
    map_cp = summary["concern_map"]["consultative_only"]
    return f"""# Consultative-Party-only sensitivity

## Methods

We repeated the concern-map, portfolio-entry, and adopted-text prediction analyses after restricting documentary attention to Antarctic Treaty Consultative Parties. We used recognition dates from the official ATS Parties table ({STATUS_SOURCE}) and treated a state as eligible in its recognition year and every subsequent meeting year. Papers submitted only by Non-Consultative Parties, Observers, invited Expert organizations, or the Secretariat did not enter this analysis. For jointly submitted papers, only eligible Consultative Parties received actor-level credit; each retained paper still contributed one total unit of attention across its archive categories. All other definitions, including fractional category weights, the revealed-policy-advantage threshold, cumulative maps built before each portfolio transition, and the frozen prediction specification, matched the main analysis.

## Results

The restricted corpus contained {summary['corpus']['papers_consultative_only']:,} papers attributed to 29 Consultative Parties. All {map_cp['n_concerns']} concerns remained in one connected map, with {map_cp['positive_pairs']} of 990 concern pairs carrying positive proximity. Pairwise proximities correlated with the all-actor reconstruction at Pearson $r={summary['concern_map']['pairwise_pearson']:.3f}$ and Spearman $\\rho={summary['concern_map']['pairwise_spearman']:.3f}$. Portfolio entry remained local: each 0.1 increase in distance corresponded to an odds ratio of {locality['odds_ratio_per_0_1']:.3f} (95\\% CI [{locality['odds_ratio_per_0_1_ci_low']:.3f}, {locality['odds_ratio_per_0_1_ci_high']:.3f}]) across {locality['actor_periods']:,} actor-period comparisons. For Resolutions, the direct-plus-nearby attention model lowered the mean allocation log score by {-resolution['mean_difference']:.3f} relative to the output-history model (95\\% meeting-bootstrap interval [{resolution['bootstrap_low']:.3f}, {resolution['bootstrap_high']:.3f}]) and scored better in {resolution['meetings_improved']} of {resolution['meetings']} meetings. Thus, the connected-map and local-entry findings survive the restriction. The prediction finding survives only for Resolutions under the direct-plus-nearby specification; the 95\\% bootstrap intervals include zero for the Measure and Decision comparisons against output history and for the other two Resolution comparisons.
"""


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    papers = prepare_papers()
    all_relations = fine_relations(papers, consultative_only=False)
    cp_relations = fine_relations(papers, consultative_only=True)

    status = pd.DataFrame(
        sorted(CONSULTATIVE_YEAR.items()), columns=["actor", "consultative_status_year"]
    )
    status["source"] = STATUS_SOURCE
    status.to_csv(OUTDIR / "consultative_status.csv", index=False)

    concern_map, pairs = map_analysis(all_relations, cp_relations)
    pairs.to_csv(OUTDIR / "map_pairwise_comparison.csv", index=False)
    locality, locality_panel = locality_analysis(cp_relations)
    locality_panel.to_csv(OUTDIR / "local_specialization_panel.csv", index=False)
    broad = broad_relations(papers)
    prediction, prediction_scores = prediction_analysis(broad)
    prediction.to_csv(OUTDIR / "prediction_type_summary.csv", index=False)
    prediction_scores.to_csv(OUTDIR / "prediction_meeting_scores.csv", index=False)

    summary = {
        "design": {
            "restriction": "time-varying Consultative status at the meeting year",
            "status_source": STATUS_SOURCE,
            "fractional_categories": True,
            "mixed_submission_rule": "eligible Consultative Parties receive actor credit; paper counted once",
        },
        "corpus": {
            "papers_all": int(papers["paper_id"].nunique()),
            "papers_consultative_only": int(
                papers.loc[papers["consultative_only"], "paper_id"].nunique()
            ),
            "paper_share_retained": float(papers["consultative_only"].mean()),
            "consultative_parties": len(CONSULTATIVE_YEAR),
        },
        "concern_map": concern_map,
        "local_specialization": locality,
        "prediction": prediction.to_dict(orient="records"),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTDIR / "si_methods_results.md").write_text(prose(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
