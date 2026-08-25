"""Entry against distance, prior attention, and the two candidate barriers.

Rebuilds the popularity-adjusted entry model reported in the main text. The
original analysis that produced the previously quoted numbers (holder breadth
OR 0.70, reach OR 1.07, breadth x distance interaction 0.074) was never saved
into this repository; this script is the consolidated replacement, fit on the
paper's own cumulative-lagged conditional-logit infrastructure.

Two panels are produced, sharing the same construction as
scripts/hazard_conditional_logit.py:

  * the conditional-logit panel: 5-year member-period choice sets on the
    cumulative-lagged space, restricted to sets with at least one adoption
    (40,129 observations / 1,055 choice sets), which the locality claim uses;
  * the full at-risk panel: the same construction without the at-least-one
    adoption restriction (79,207 observations).

Predictors per at-risk observation: raw distance 1 - max phi to the member's
prior portfolio (cumulative-lagged), prior attention (share of members
specialized in the topic in the previous window), reach (frozen from
concern_classes.py), holder breadth (frozen from concern_classes.py), and the
holder breadth x distance interaction.

    PYTHONPATH=. micromamba run -n ultraplot-dev python scripts/entry_barrier_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concern_classes import load_classes  # noqa: E402
from utils import (generate_interaction_matrix,  # noqa: E402
                   compute_product_space, get_rca, load_data,
                   standardize_index_labels)

WINDOW_YEARS = 5
RCA_THRESHOLD = 1.0
OUT_CSV = Path("output/entry_barrier_models.csv")


def _phi_from(interaction, topics):
    rca = get_rca(interaction)
    phi = compute_product_space(rca).reindex(index=topics, columns=topics,
                                             fill_value=0.0)
    arr = 0.5 * (phi.to_numpy() + phi.to_numpy().T)
    np.fill_diagonal(arr, 1.0)
    return arr


def build_panels():
    Q = load_classes()
    topics = Q["topics"]
    reach = np.array(Q["reach"], float)
    breadth = np.array(Q["breadth"], float)

    _c, submitted, countries, tset = load_data(
        "antarctic-database-go/data/processed/document-summary.parquet")
    if "year" not in submitted.columns:
        submitted["year"] = submitted["meeting year"]
    submitted["year"] = pd.to_numeric(submitted["year"], errors="coerce")
    submitted = submitted.dropna(subset=["year"]).copy()
    submitted["year"] = submitted["year"].astype(int)

    year_min, year_max = int(submitted["year"].min()), int(submitted["year"].max())
    periods = [(y - WINDOW_YEARS + 1, y)
               for y in range(year_min + WINDOW_YEARS - 1, year_max + 1)]

    def win_interaction(lo, hi):
        sub = submitted[(submitted["year"] >= lo) & (submitted["year"] <= hi)]
        c = standardize_index_labels(generate_interaction_matrix(sub, countries, tset))
        if c.index.has_duplicates:
            c = c.groupby(level=0).sum()
        return c.reindex(index=topics, columns=sorted(c.columns), fill_value=0)

    active = []
    for lo, hi in periods:
        active.append((get_rca(win_interaction(lo, hi)).values >= RCA_THRESHOLD))

    members = sorted(win_interaction(*periods[0]).columns)
    rows_all, rows_cl = [], []
    for t in range(1, len(periods)):
        prev_end = periods[t - 1][1]
        prev_a, curr_a = active[t - 1], active[t]
        phi = _phi_from(win_interaction(year_min, prev_end), topics)
        prev_pop = prev_a.sum(axis=1) / len(members)
        for j, _member in enumerate(members):
            prev_mask = prev_a[:, j]
            if not prev_mask.any():
                continue
            at_risk = ~prev_mask
            if not at_risk.any():
                continue
            adopted = curr_a[:, j] & at_risk
            pi = np.where(prev_mask)[0]
            for i in np.where(at_risk)[0]:
                row = {
                    "adopted": int(adopted[i]),
                    "distance": float(1.0 - phi[i, pi].max()),
                    "topic_popularity": float(prev_pop[i]),
                    "reach": float(reach[i]),
                    "holder_breadth": float(breadth[i]),
                }
                rows_all.append(row)
                if int(adopted.sum()) > 0:
                    rows_cl.append({**row, "group": f"{_member}::{periods[t][1]}"})
    panel_all = pd.DataFrame(rows_all)
    panel_cl = pd.DataFrame(rows_cl)
    return panel_all, panel_cl


def fit_models(panel, label):
    import statsmodels.api as sm
    from statsmodels.discrete.conditional_models import ConditionalLogit

    out = {}
    specs = {
        "barriers": ["distance", "topic_popularity", "reach", "holder_breadth"],
        "barriers_interaction": ["distance", "topic_popularity", "reach",
                                 "holder_breadth", "breadth_x_distance"],
    }
    for name, cols in specs.items():
        base_cols = [c for c in cols if c != "breadth_x_distance"]
        X = panel[base_cols].copy()
        if "breadth_x_distance" in cols:
            X["breadth_x_distance"] = X["holder_breadth"] * X["distance"]
        if "group" in panel.columns:
            model = ConditionalLogit(panel["adopted"].astype(int), X,
                                     groups=panel["group"])
        else:
            X = sm.add_constant(X)
            model = sm.Logit(panel["adopted"], X)
        res = model.fit(disp=False)
        rows = []
        for k in X.columns:
            rows.append({
                "model": f"{label}_{name}",
                "term": k,
                "coef": float(res.params[k]),
                "se": float(res.bse[k]),
                "odds_ratio": float(np.exp(res.params[k])),
                "odds_ratio_ci_low": float(np.exp(res.params[k] - 1.96 * res.bse[k])),
                "odds_ratio_ci_high": float(np.exp(res.params[k] + 1.96 * res.bse[k])),
                "p": float(res.pvalues[k]),
            })
        out[name] = pd.DataFrame(rows)
    return out


def main():
    panel_all, panel_cl = build_panels()
    print(f"full at-risk observations: {len(panel_all):,}")
    print(f"conditional-logit observations: {len(panel_cl):,}")

    for d, lab in ((panel_cl["distance"].quantile(.1), "10th"),
                   (panel_cl["distance"].quantile(.5), "median"),
                   (panel_cl["distance"].quantile(.9), "90th")):
        print(f"  conditional-logit distance {lab} percentile: {d:.3f}")

    results = {}
    # pooled logit on the full at-risk set (no distance controls, descriptive)
    results["full_pooled"] = fit_models(panel_all, "pooled")

    # conditional logit on the paper's choice-set infrastructure
    results["clogit"] = fit_models(panel_cl, "clogit")

    for family, tables in results.items():
        pnl = panel_cl if family == "clogit" else panel_all
        for name, df in tables.items():
            print(f"\n=== {family} / {name}")
            for _, r in df.iterrows():
                print(f"  {r['term']:<22} OR {r['odds_ratio']:.4f} "
                      f"[{r['odds_ratio_ci_low']:.4f}, {r['odds_ratio_ci_high']:.4f}] "
                      f"p={r['p']:.2e}")
            if "breadth_x_distance" in set(df["term"]):
                b = df.set_index("term").loc["holder_breadth"]
                i_ = df.set_index("term").loc["breadth_x_distance"]
                for d, lab in ((pnl["distance"].quantile(.1), "p10"),
                               (pnl["distance"].quantile(.5), "p50"),
                               (pnl["distance"].quantile(.9), "p90")):
                    print(f"  holder OR at {lab} distance ({d:.3f}): "
                          f"{np.exp(b['coef'] + i_['coef'] * d):.3f}")

    combined = pd.concat([v for tables in results.values() for v in tables.values()])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
