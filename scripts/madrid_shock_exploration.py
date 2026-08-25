#!/usr/bin/env python
"""Exploration: the 1991 Madrid Protocol as a quasi-experiment for path
dependence.

The Protocol is the one exogenous agenda shock in the record: it prohibited
mineral activity and created the compliance machinery, boosting a set of
concerns (liability, inspections, environmental management) from outside the
ordinary portfolio dynamics. Path dependence predicts that actors whose
PRE-shock portfolios sat adjacent to the boosted concerns reoriented into
them faster and more strongly than equally active but distant actors.

Design (all inputs pre-1991 where they could be):
  * boosted set   topics whose document share grew most, 1986-90 -> 1992-96
  * pre space     phi estimated on 1961-1990 only
  * adjacency     max phi between an actor's pre-1991 RPA>1 portfolio and the
                  boosted set, in the pre space
  * outcome       share of an actor's papers in a 5-yr window going to the
                  boosted set
  * event study   window dummies x adjacency + actor FE + window FE +
                  window x pre-breadth controls; reference window 1986-90

Parallel pre-trends are the identifying check. This is an exploration, not a
paper result: run, look, decide.

Usage::

    micromamba run -n ultraplot-dev python scripts/madrid_shock_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import (  # noqa: E402
    _normalize_topic_label,
    _split_multi_value,
    compute_product_space,
    get_rca,
    load_data,
)


def _nk(t):
    return " ".join(str(t).strip().lower().replace("_", " ")
                    .replace("-", " ").split())

OUT_PNG = ROOT / "output" / "preview_madrid_event_study.png"
PRE_END = 1990
BOOST_PRE = (1986, 1990)
BOOST_POST = (1992, 1996)
N_BOOST = 8
PROTOCOL_SET = {
    "liability", "inspections", "environmental monitoring and reporting",
    "environmental protection general", "waste management and disposal",
    "comprehensive environmental evaluations",
}


def long_panel(raw):
    rows = []
    for _, r in raw.dropna(subset=["submitted by"]).iterrows():
        year = r.get("year")
        if pd.isna(year):
            continue
        topics = {
            _normalize_topic_label(t)
            for t in _split_multi_value(r["category"], "\t")
        }
        pid = r.get("paper_id")
        for p in _split_multi_value(r["submitted by"], ","):
            for t in topics:
                rows.append((pid, p, int(year), t))
    df = pd.DataFrame(rows, columns=["paper", "actor", "year", "topic"])
    return df.drop_duplicates()


def main() -> int:
    counts, raw, _, _ = load_data(
        str(ROOT / "antarctic-database-go/data/processed/document-summary.parquet")
    )
    df = long_panel(raw)
    df["w"] = (df["year"] - 1961) // 5
    windows = sorted(df["w"].unique())

    # ---- boosted set: biggest share gain across the Protocol ---------------
    def share(y0, y1):
        sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
        c = sub.groupby("topic").size()
        return c / c.sum()

    s_pre, s_post = share(*BOOST_PRE), share(*BOOST_POST)
    common = s_pre.index.intersection(s_post.index)
    # require real pre-presence so mechanically new topics (e.g. CEE, created
    # by the Protocol itself) cannot enter the boosted set
    common = [t for t in common if s_pre[t] >= 0.01]
    gain = (s_post.reindex(common).fillna(0.0) - s_pre.reindex(common).fillna(0.0))
    boosted = gain.sort_values(ascending=False).head(N_BOOST)
    print("boosted topics (share gain 1986-90 -> 1992-96):")
    for t, g in boosted.items():
        print(f"  {g:+.3f}  {t}")
    boosted_set = set(boosted.index)

    # ---- pre space and pre portfolios --------------------------------------
    pre = df[df["year"] <= PRE_END]
    pc = pre.pivot_table(index="topic", columns="actor", aggfunc="size",
                         fill_value=0)
    pc = pc.loc[pc.sum(axis=1) > 0, pc.sum(axis=0) > 0]
    phi = compute_product_space(get_rca(pc))
    held = (get_rca(pc) > 1.0)

    adj, breadth = {}, {}
    for a in pc.columns:
        h = [t for t in pc.index if held.loc[t, a]]
        breadth[a] = len(h)
        if h:
            adj[a] = float(
                phi.reindex(index=h, columns=list(boosted_set))
                .to_numpy(dtype=float).max()
            )
        else:
            adj[a] = 0.0

    # ---- actor-window outcome panel ----------------------------------------
    recs = []
    for (a, w), sub in df.groupby(["actor", "w"]):
        if a not in adj:
            continue
        n = len(sub)
        recs.append(
            dict(actor=a, w=w, share=float(sub["topic"].isin(boosted_set).mean()),
                 n=n, A=adj[a], B=breadth[a])
        )
    panel = pd.DataFrame(recs)
    panel = panel[panel["n"] >= 3]
    panel["w"] = panel["w"].astype(int)
    ref = (BOOST_PRE[0] - 1961) // 5
    panel["wf"] = pd.Categorical(panel["w"], categories=windows)
    panel["Af"] = panel["w"].apply(lambda k: f"w{k}")

    terms = " + ".join(
        [f"A:C(wf, Treatment(reference={ref}))", "B:C(wf)", "C(actor)", "C(wf)"]
    )
    model = smf.ols(f"share ~ {terms}", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["actor"]}
    )

    # second pass: protocol set + relative adjacency + entry outcome
    adj_rel = {}
    nonboost = [t for t in phi.columns if t not in boosted_set]
    for a in pc.columns:
        h = [t for t in pc.index if held.loc[t, a]]
        if h:
            m = phi.reindex(index=h)
            adj_rel[a] = float(m[list(boosted_set)].to_numpy(float).max()) - \
                float(m[nonboost].to_numpy(float).max())
        else:
            adj_rel[a] = 0.0
    panel["AR"] = panel["actor"].map(adj_rel).fillna(0.0)
    proto = {t for t in df["topic"].unique() if _nk(t) in PROTOCOL_SET}
    print(f"protocol set matched: {len(proto)} topics, "
          f"entry rate {panel['n'].sum() and 'na'}")
    recs2 = []
    for (a, w), sub in df.groupby(["actor", "w"]):
        if a not in adj_rel:
            continue
        rca_w = sub.groupby("topic").size()
        tot_w = rca_w.sum()
        share_w = rca_w / tot_w
        nspec = len(rca_w)
        entry = int(any(
            t in proto and (share_w[t] * len(rca_w) > 1.0 if nspec else False)
            for t in rca_w.index
        )) if tot_w else 0
        recs2.append(dict(actor=a, w=w, AR=adj_rel[a], entry=entry,
                          n=int(tot_w)))
    p2 = pd.DataFrame(recs2)
    p2 = p2[p2["n"] >= 3]
    p2["wf"] = pd.Categorical(p2["w"], categories=windows)
    terms2 = "AR + C(wf)"
    m2 = smf.logit(f"entry ~ {terms2}", data=p2).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": p2["actor"]})
    print(f"\nsecond pass (protocol set, entry outcome): n={len(p2)}, "
          f"AR coef {m2.params['AR']:+.3f} (SE {m2.bse['AR']:.3f}), "
          f"p={m2.pvalues['AR']:.3f}")

    coefs, ses, ks = [], [], []
    for k in windows:
        if k == ref:
            continue
        name = f"A:C(wf, Treatment(reference={ref}))[{k}]"
        if name in model.params:
            coefs.append(model.params[name])
            ses.append(model.bse[name])
            ks.append(k)
    pre_ks = [k for k in ks if 1961 + k * 5 + 4 <= PRE_END]
    post_ks = [k for k in ks if k not in pre_ks]
    print(f"\nactors: {panel['actor'].nunique()}  obs: {len(panel)}")
    print(f"adjacency mean {panel['A'].mean():.2f}  corr(A, breadth) "
          f"{np.corrcoef(panel['A'], panel['B'])[0, 1]:.2f}")
    pre_names = [f"A:C(wf, Treatment(reference={ref}))[{k}]" for k in pre_ks]
    R = np.zeros((len(pre_names), len(model.params)))
    for i, nm in enumerate(pre_names):
        R[i, model.params.index.get_loc(nm)] = 1.0
    b = model.params.to_numpy()
    V = model.cov_params().to_numpy()
    Rb = R @ b
    fval = float(Rb @ np.linalg.solve(R @ V @ R.T, Rb) / len(pre_names))
    from scipy import stats as _st
    fp = float(_st.f.sf(fval, len(pre_names), len(panel) - len(model.params)))
    print(f"parallel pre-trends F({len(pre_names)}, .) = {fval:.2f}, p = {fp:.3f}")
    print("\nevent-study coefficients (share of papers in boosted set):")
    for k, b, s in zip(ks, coefs, ses):
        tag = "pre " if k in pre_ks else "post"
        print(f"  {tag} {1961 + k * 5}-{1965 + k * 5}: {b:+.3f} +/- {1.96 * s:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    yrs = [1961 + k * 5 + 2 for k in ks]
    ax.errorbar(yrs, coefs, yerr=[1.96 * s for s in ses], fmt="o-",
                color="#0072B2", capsize=3)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(1991, color="#D55E00", ls="--", lw=1.0)
    ax.text(1991.4, ax.get_ylim()[0], "Madrid Protocol", color="#D55E00",
            va="bottom", fontsize=9)
    ax.set_xlabel("window mid-year")
    ax.set_ylabel("adjacency x window coefficient")
    ax.set_title("Response to the 1991 shock by pre-shock adjacency")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f"\nWrote {OUT_PNG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
