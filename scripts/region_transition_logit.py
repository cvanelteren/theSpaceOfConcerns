"""Does the region an actor already works in predict where its next
specialization lands, beyond how close that concern is?

The locality result says a new specialization appears near the portfolio an
actor already holds. That is a statement about distance in phi. This asks a
sharper question: holding distance fixed, does the *identity* of the regions an
actor already covers shift which region the next concern comes from?

Design. The choice sets are the prospective ones used throughout the paper:
rolling five-meeting windows, a concern enters the comparison only if the actor
did not already hold it and it had appeared in the archive before the
transition. Within an actor-period, conditional logit compares the concern that
was taken up against the ones that were not, so everything constant within the
actor-period -- the actor's breadth, its volume, the era, and the anchor main
effects themselves -- drops out of the likelihood and cannot confound.

What is estimated. Destination-region fixed effects absorb the fact that some
regions attract entries whoever you are. `distance` (1 - max phi to the held
set) and `popularity` are the covariates the locality model already uses. The
anchor x destination interactions carry the claim: the extra odds that an entry
lands in region Y rather than the procedural baseline, when the actor already
covers region X.

Inference. Rolling windows overlap, so model-based standard errors understate
uncertainty. Intervals come from a cluster bootstrap over actors: complete
actor histories are resampled and the model refit. This is the dependence-aware
interval; it is one step weaker than the paper's headline actor-history
bootstrap, which also rebuilds the map on each draw, and is reported as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests

from fig01_space_of_concerns_topology import (
    CPM_SPECS,
    build_graphs,
    normalize_topic_key,
    topic_region_assignments,
)
from scripts.bootstrap_fractional_locality import RCA_THRESHOLD, source_matrices
from scripts.primary_concern_sensitivity import phi_from_interaction
from utils import get_rca

OUT_CSV = Path("output/region_transition_logit_coefficients.csv")
OUT_JSON = Path("output/region_transition_logit_meta.json")

# Region 1 is the procedural core: 96% of actor-periods already cover it, so it
# has no contrast to offer as an anchor and serves as the destination baseline.
BASELINE_REGION = 1
# Regions 6 and 7 hold two concerns each and see 6 and 14 entries in the whole
# panel. They stay in the model as destination controls but are not interacted:
# there is nothing there to estimate.
INTERACTED = [2, 3, 4, 5]
CONTROL_DESTINATIONS = [6, 7]
N_BOOTSTRAP = 300
SEED = 7


def build_panel() -> pd.DataFrame:
    """One row per available concern per actor-period, as the locality model."""
    topics, meetings, periods, matrices = source_matrices()
    backbone, *_ = build_graphs()
    nodes = list(backbone.nodes())
    region_of = topic_region_assignments(nodes)
    by_key = {normalize_topic_key(n): region_of[n]["id"] for n in nodes}
    region = np.array([by_key.get(normalize_topic_key(t), 0) for t in topics])

    by_meeting = dict(zip(meetings, matrices))
    windows = [
        pd.DataFrame(
            np.sum([by_meeting[m] for m in meetings if start <= m <= end], axis=0),
            index=topics,
        )
        for start, end in periods
    ]
    active = [get_rca(frame).ge(RCA_THRESHOLD).to_numpy() for frame in windows]

    rows: list[dict] = []
    for t in range(1, len(periods)):
        prev_end = periods[t - 1][1]
        history = np.sum([by_meeting[m] for m in meetings if m <= prev_end], axis=0)
        phi = phi_from_interaction(pd.DataFrame(history, index=topics), topics)
        available = history.sum(axis=1) > 0
        previous, current = active[t - 1], active[t]
        popularity = previous.sum(axis=1) / previous.shape[1]
        for actor in range(previous.shape[1]):
            held = previous[:, actor]
            if not held.any():
                continue
            at_risk = (~held) & available
            entered = current[:, actor] & at_risk
            # A period with no entry, or with nothing left un-entered, carries
            # no within-group contrast and drops out of the likelihood anyway.
            if not entered.any() or entered.sum() == at_risk.sum():
                continue
            distance = 1.0 - phi[:, np.flatnonzero(held)].max(axis=1)
            anchors = set(region[held]) - {0}
            for j in np.flatnonzero(at_risk):
                if region[j] == 0:
                    continue
                rows.append(
                    {
                        "actor": actor,
                        "group": f"{actor}::{t}",
                        "entered": int(entered[j]),
                        "destination": int(region[j]),
                        "distance": float(distance[j]),
                        "popularity": float(popularity[j]),
                        **{f"anchor_{r}": int(r in anchors) for r in range(1, 8)},
                    }
                )
    return pd.DataFrame(rows)


def design(panel: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(index=panel.index)
    matrix["distance"] = panel["distance"]
    matrix["popularity"] = panel["popularity"]
    for destination in INTERACTED + CONTROL_DESTINATIONS:
        matrix[f"dest_{destination}"] = panel["destination"].eq(destination).astype(int)
    for anchor in INTERACTED:
        for destination in INTERACTED:
            matrix[f"anchor{anchor}_dest{destination}"] = (
                panel[f"anchor_{anchor}"] * panel["destination"].eq(destination)
            ).astype(int)
    return matrix


def fit(panel: pd.DataFrame):
    return ConditionalLogit(
        panel["entered"].astype(int), design(panel), groups=panel["group"]
    ).fit(disp=False, maxiter=300)


def bootstrap(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Resample complete actor histories, refit, and keep the coefficients."""
    rng = np.random.default_rng(SEED)
    actors = panel["actor"].unique()
    by_actor = {actor: frame for actor, frame in panel.groupby("actor")}
    draws: list[np.ndarray] = []
    for draw in range(N_BOOTSTRAP):
        picked = rng.choice(actors, size=len(actors), replace=True)
        parts = []
        for copy, actor in enumerate(picked):
            frame = by_actor[actor].copy()
            # Distinct group labels per copy, so a twice-drawn actor contributes
            # two independent sets of choice sets rather than one merged set.
            frame["group"] = frame["group"] + f"#{copy}"
            parts.append(frame)
        resampled = pd.concat(parts, ignore_index=True)
        try:
            draws.append(fit(resampled).params.reindex(columns).to_numpy())
        except Exception:
            continue
    return pd.DataFrame(draws, columns=columns)


def main() -> None:
    panel = build_panel()
    result = fit(panel)
    columns = list(result.params.index)
    draws = bootstrap(panel, columns)

    table = pd.DataFrame(
        {
            "term": columns,
            "coefficient": result.params.to_numpy(),
            "odds_ratio": np.exp(result.params.to_numpy()),
            "se_model": result.bse.to_numpy(),
            "p_model": result.pvalues.to_numpy(),
            "boot_low": np.exp(draws[columns].quantile(0.025).to_numpy()),
            "boot_high": np.exp(draws[columns].quantile(0.975).to_numpy()),
            "boot_se": draws[columns].std().to_numpy(),
        }
    )
    interactions = table["term"].str.startswith("anchor")
    # Sixteen interaction terms are tested at once; control the false discovery
    # rate over exactly that family, not over the controls as well.
    table["p_fdr"] = np.nan
    table.loc[interactions, "p_fdr"] = multipletests(
        table.loc[interactions, "p_model"], method="fdr_bh"
    )[1]
    table["boot_excludes_1"] = (table["boot_low"] > 1) | (table["boot_high"] < 1)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    meta = {
        "n_choice_rows": int(len(panel)),
        "n_entries": int(panel["entered"].sum()),
        "n_actor_periods": int(panel["group"].nunique()),
        "n_actors": int(panel["actor"].nunique()),
        "baseline_destination": BASELINE_REGION,
        "interacted_regions": INTERACTED,
        "n_bootstrap": int(len(draws)),
        "region_labels": {s["id"]: s["short"] for s in CPM_SPECS},
        "distance_odds_ratio_per_0.1": float(np.exp(0.1 * result.params["distance"])),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(table.round(3).to_string(index=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
