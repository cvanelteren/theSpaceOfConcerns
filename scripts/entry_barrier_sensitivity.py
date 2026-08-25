"""Sensitivity of the entry-barrier model to construction choices.

Extends scripts/entry_barrier_regression.py with three robustness variants of
the conditional-logit and pooled entry models reported in the main text:

  * rpa-0.8 / rpa-1.2: the specialization threshold used to build the
    rolling active sets is moved from its baseline 1.0 to 0.8 and 1.2;
  * no-procedural: the two procedural topics (Opening statements, Exchange of
    information) are dropped from the candidate space before building the
    choice sets, so no at-risk observation on those topics can enter.

Only the holder-breadth and reach odds ratios are reported, since those are
the quantities the main-text argument rests on.

    PYTHONPATH=. micromamba run -n ultraplot-dev python scripts/entry_barrier_sensitivity.py
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
PROCEDURAL = ["Opening statements", "Exchange of information"]
OUT_CSV = Path("output/entry_barrier_sensitivity.csv")


def _phi_from(interaction, topics):
    rca = get_rca(interaction)
    phi = compute_product_space(rca).reindex(index=topics, columns=topics,
                                             fill_value=0.0)
    arr = 0.5 * (phi.to_numpy() + phi.to_numpy().T)
    np.fill_diagonal(arr, 1.0)
    return arr


def build_panels(rca_threshold=1.0, drop_topics=()):
    Q = load_classes()
    topics = [t for t in Q["topics"] if t not in drop_topics]
    idx_of = {t: i for i, t in enumerate(topics)}
    reach = {t: float(Q["reach"][i]) for i, t in enumerate(Q["topics"])}
    breadth = {t: float(Q["breadth"][i]) for i, t in enumerate(Q["topics"])}
    reach = np.array([reach[t] for t in topics], float)
    breadth = np.array([breadth[t] for t in topics], float)

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
        active.append((get_rca(win_interaction(lo, hi)).values >= rca_threshold))

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
    return pd.DataFrame(rows_all), pd.DataFrame(rows_cl)


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
                "variant": label,
                "model": name,
                "n_obs": int(len(panel)),
                "n_groups": (int(panel["group"].nunique())
                             if "group" in panel.columns else 0),
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
    variants = {
        "rpa-0.8": dict(rca_threshold=0.8),
        "rpa-1.2": dict(rca_threshold=1.2),
        "no-procedural": dict(drop_topics=PROCEDURAL),
    }
    print("=== base (reproduces entry_barrier_regression.py) ===")
    base_all, base_cl = build_panels()
    print(f"  full at-risk: {len(base_all):,}   "
          f"conditional-logit: {len(base_cl):,}")

    tables = {}
    for label, kw in variants.items():
        print(f"\n=== {label}")
        panel_all, panel_cl = build_panels(**kw)
        print(f"  full at-risk: {len(panel_all):,}   "
              f"conditional-logit: {len(panel_cl):,}")
        for family, pnl in (("pooled", panel_all), ("clogit", panel_cl)):
            for name, df in fit_models(pnl, f"{label}-{family}").items():
                tables[f"{label}-{family}-{name}"] = df
                for _, r in df.iterrows():
                    if r["term"] in ("holder_breadth", "reach",
                                     "breadth_x_distance"):
                        print(f"    {family}/{name} {r['term']:<18} "
                              f"OR {r['odds_ratio']:.3f} "
                              f"[{r['odds_ratio_ci_low']:.3f}, "
                              f"{r['odds_ratio_ci_high']:.3f}] "
                              f"p={r['p']:.2e} (n={r['n_obs']:,})")

    combined = pd.concat(tables.values())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
