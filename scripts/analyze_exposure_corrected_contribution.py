#!/usr/bin/env python3
"""Estimate documented paper-to-output linkage after accounting for exposure.

The earlier actor-level binary model asked whether an actor appeared at least
once in a verified lineage. That outcome grows mechanically with the number of
papers submitted. Here the unit is an actor--output opportunity and the
response is the number of linked papers out of all papers the actor submitted
to the same meeting. A binomial model therefore compares documented-link rates
rather than the chance of at least one appearance.

The paper-level route to an output is represented by (i) exact concern match
and (ii) off-label proximity through nearby concerns. Earlier actor position is
represented by proximity of the portfolio at the immediately preceding ATCM
and the number of concerns covered there. Working-paper share and actor fixed effects account for
document type and stable actor differences; output fixed effects compare
actors around the same formal output. Uncertainty is clustered by actor and
output.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.data_loading import load_submitted_with_fallback
from scripts.link_movement_to_outcomes import paper_sponsors


OUTDIR = ROOT / "output" / "outcome_linkage"
CANDIDATE_CSV = OUTDIR / "space_discrimination_panel.csv"
ACTOR_STATE_CSV = OUTDIR / "actor_outcome_candidate_panel_independent.csv"
OUT_PANEL = OUTDIR / "actor_output_conversion_panel.csv"
OUT_MODEL = OUTDIR / "actor_output_conversion_model.csv"

TERMS = [
    "same_concern_match",
    "nearby_concern_match",
    "prior_portfolio_proximity",
    "log_prior_concerns",
    "working_paper_share",
]


def build_panel(
    candidates: pd.DataFrame | None = None,
    actor_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build actor--output conversion opportunities.

    Optional data frames allow the same exposure-corrected analysis to be
    rerun against independently coded outcome concerns without replacing the
    classifier-based primary inputs.
    """
    if candidates is None:
        candidates = pd.read_csv(CANDIDATE_CSV)
    else:
        candidates = candidates.copy()
    adoption_outcomes = set(
        candidates.loc[candidates["adoption_linked"].eq(1), "outcome_id"]
    )
    candidates = candidates[candidates["outcome_id"].isin(adoption_outcomes)].copy()

    sponsors = paper_sponsors(load_submitted_with_fallback())
    rows = []
    for row in candidates.itertuples(index=False):
        for actor in sponsors.get(row.paper_id, set()):
            rows.append(
                {
                    "actor": actor,
                    "outcome_id": row.outcome_id,
                    "paper_id": row.paper_id,
                    "documented_link": int(row.adoption_linked),
                    "same_concern_match": float(row.same_concern_mass),
                    "nearby_concern_match": float(row.related_concern_proximity),
                    "working_paper": int(":WP " in row.paper_id),
                }
            )
    paper_panel = pd.DataFrame(rows)
    panel = (
        paper_panel.groupby(["actor", "outcome_id"], as_index=False)
        .agg(
            eligible_papers=("paper_id", "nunique"),
            linked_papers=("documented_link", "sum"),
            same_concern_match=("same_concern_match", "mean"),
            nearby_concern_match=("nearby_concern_match", "mean"),
            working_paper_share=("working_paper", "mean"),
        )
    )

    if actor_state is None:
        actor_state = pd.read_csv(ACTOR_STATE_CSV)
    else:
        actor_state = actor_state.copy()
    actor_state = actor_state[
        ["actor", "outcome_id", "expected_proximity", "breadth"]
    ].rename(
        columns={
            "expected_proximity": "prior_portfolio_proximity",
            "breadth": "prior_concerns",
        }
    )
    panel = panel.merge(actor_state, on=["actor", "outcome_id"], how="left")
    # Missing state means that the actor submitted at the meeting but held no
    # qualifying concern at the immediately preceding ATCM.
    panel["prior_portfolio_proximity"] = panel["prior_portfolio_proximity"].fillna(0.0)
    panel["prior_concerns"] = panel["prior_concerns"].fillna(0.0)
    panel["log_prior_concerns"] = np.log1p(panel["prior_concerns"])
    panel["documented_link_rate"] = panel["linked_papers"] / panel["eligible_papers"]
    return panel


def fit_model(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    scales = {}
    for term in TERMS:
        scales[term] = float(data[term].std(ddof=0))
        data[f"z_{term}"] = (data[term] - data[term].mean()) / scales[term]

    # Actor fixed effects keep the comparison within submitters while output
    # fixed effects keep it within each formal output's same-meeting paper set.
    formula = (
        "documented_link_rate ~ "
        + " + ".join(f"z_{term}" for term in TERMS)
        + " + C(outcome_id) + C(actor)"
    )
    fitted = smf.glm(
        formula,
        data=data,
        family=sm.families.Binomial(),
        var_weights=data["eligible_papers"],
    ).fit(maxiter=300)

    actor_groups = pd.factorize(data["actor"])[0]
    outcome_groups = pd.factorize(data["outcome_id"])[0]
    covariance = cov_cluster_2groups(
        fitted, actor_groups, outcome_groups, use_correction=True
    )[0]
    variances = np.diag(covariance)
    names = list(fitted.params.index)

    rows = []
    for term in TERMS:
        index = names.index(f"z_{term}")
        estimate = float(fitted.params.iloc[index])
        variance = float(variances[index])
        if not np.isfinite(variance) or variance < 0:
            raise ValueError(f"Invalid clustered variance for {term}: {variance}")
        se = math.sqrt(variance)
        rows.append(
            {
                "term": term,
                "coefficient": estimate,
                "se_two_way_clustered": se,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * se),
                "ci_high": math.exp(estimate + 1.96 * se),
                "p_value": float(2 * norm.sf(abs(estimate / se))),
                "predictor_sd": scales[term],
                "scale": "one sample SD",
                "n_actor_output_pairs": int(len(data)),
                "n_eligible_papers": int(data["eligible_papers"].sum()),
                "n_linked_actor_papers": int(data["linked_papers"].sum()),
                "n_actors": int(data["actor"].nunique()),
                "n_outputs": int(data["outcome_id"].nunique()),
                "fixed_effects": "actor and output",
                "clustered_by": "actor and output",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    panel = build_panel()
    model = fit_model(panel)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PANEL, index=False)
    model.to_csv(OUT_MODEL, index=False)
    print(model.to_string(index=False))
    print(f"\nWrote {OUT_PANEL}")
    print(f"Wrote {OUT_MODEL}")


if __name__ == "__main__":
    main()
