"""Three tests that gate the four-class scheme.

1. Do the class assignments reproduce under actor resampling?  (ARI)
2. Do they survive moving the phi threshold?
3. Does entry favour Gateway *beyond what proximity alone predicts*?
   -- the uniform baseline in panel C partly restates locality, so this
   redraws each observed entry from the actor's own at-risk set, weighted
   by proximity to its existing portfolio.

    PYTHONPATH=. micromamba run -n ultraplot-dev python scratch/validate_quadrants.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from fig01_space_of_concerns_topology import build_graphs
from utils import compute_rolling_rca, get_rca, load_data

N_BOOT = 300
N_DRAW = 200
ORDER = ["gateway", "platform", "margin", "cul-de-sac"]


def classify(M, PHI, phi_step):
    reach = np.nansum(PHI >= phi_step, axis=1)
    div, ubi = M.sum(1), M.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        breadth = (M * div[:, None]).sum(0) / ubi
    mr, mb = np.median(reach), np.nanmedian(breadth)
    q = np.array(["platform" if (b >= mb and r >= mr) else
                  "cul-de-sac" if b >= mb else
                  "gateway" if r >= mr else "margin"
                  for b, r in zip(breadth, reach)])
    return q, reach, breadth


def main():
    backbone, mst, g, counts_df, rca = build_graphs()
    topics = list(counts_df.index)
    M = (rca.reindex(topics).values >= 1.0).astype(float).T
    PHI = nx.to_pandas_adjacency(g, nodelist=topics, weight="weight").values.copy()
    np.fill_diagonal(PHI, np.nan)
    base, reach0, breadth0 = classify(M, PHI, 0.30)

    # ---------- 1. actor bootstrap -----------------------------------------
    print("=" * 70)
    print("1. DO THE CLASSES REPRODUCE UNDER ACTOR RESAMPLING?")
    print("=" * 70)
    rng = np.random.default_rng(17)
    n_act = counts_df.shape[1]
    aris, agree = [], np.zeros(len(topics))
    for _ in range(N_BOOT):
        cols = counts_df.columns[rng.integers(0, n_act, n_act)]
        cb = counts_df[cols]
        cb.columns = [f"{c}_{i}" for i, c in enumerate(cols)]
        rb = get_rca(cb).reindex(topics)
        Mb = (rb.values >= 1.0).astype(float).T
        PHIb = np.zeros_like(PHI)
        # rebuild proximity from the resampled panel
        S = (rb.values >= 1.0).astype(float)          # topics x actors
        co = S @ S.T
        n_i = S.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_ij = co / n_i[:, None]
            p_ji = co / n_i[None, :]
        PHIb = np.fmin(p_ij, p_ji)
        np.fill_diagonal(PHIb, np.nan)
        qb, _, _ = classify(Mb, PHIb, 0.30)
        aris.append(adjusted_rand_score(base, qb))
        agree += (qb == base)
    aris = np.array(aris); agree /= N_BOOT
    print(f"  ARI vs observed: median {np.median(aris):.3f}  "
          f"mean {aris.mean():.3f}  90% [{np.percentile(aris,5):.3f}, "
          f"{np.percentile(aris,95):.3f}]")
    print(f"  topics keeping their class in >=75% of replicates: "
          f"{(agree >= 0.75).mean():.0%}   >=90%: {(agree >= 0.90).mean():.0%}")
    unstable = pd.Series(agree, index=topics).sort_values()
    print("  least stable:")
    for t, v in unstable.head(6).items():
        print(f"    {t[:52]:52s} {v:.0%}  ({base[topics.index(t)]})")

    # ---------- 2. threshold sweep -----------------------------------------
    print("\n" + "=" * 70)
    print("2. DOES THE SCHEME SURVIVE MOVING THE PHI THRESHOLD?")
    print("=" * 70)
    for th in [0.20, 0.25, 0.30, 0.35, 0.40]:
        q, _, _ = classify(M, PHI, th)
        print(f"  phi >= {th:.2f}   ARI vs the 0.30 scheme = "
              f"{adjusted_rand_score(base, q):.3f}   "
              f"sizes " + str({k: int((q == k).sum()) for k in ORDER}))

    # ---------- 3. proximity-weighted null for panel C ----------------------
    print("\n" + "=" * 70)
    print("3. IS THE GATEWAY SKEW MORE THAN PROXIMITY ALONE?")
    print("=" * 70)
    _c, submitted, _co, _t = load_data(
        "antarctic-database-go/data/processed/document-summary.parquet")
    roll = compute_rolling_rca(submitted, window_years=5)
    tset, idx = set(topics), {t: i for i, t in enumerate(topics)}
    P0 = np.nan_to_num(PHI)

    obs_q, null_counts = [], np.zeros((N_DRAW, 4))
    years = sorted(roll["year"].unique())
    events = []
    for y0, y1 in zip(years[:-1], years[1:]):
        cur = roll[(roll["year"] == y0) & (roll["rca"] >= 1.0)]
        nxt = roll[(roll["year"] == y1) & (roll["rca"] >= 1.0)]
        for a, gg in cur.groupby("country"):
            held = {t for t in gg["topic"] if t in tset}
            if not held:
                continue
            after = {t for t in nxt[nxt["country"] == a]["topic"] if t in tset}
            new = after - held
            if not new:
                continue
            hi = [idx[h] for h in held]
            risk = [i for i, t in enumerate(topics) if t not in held]
            w = P0[np.ix_(risk, hi)].max(axis=1)      # proximity to portfolio
            if w.sum() <= 0:
                continue
            events.append((risk, w / w.sum(), len(new)))
            for t in new:
                obs_q.append(base[idx[t]])

    obs = pd.Series(obs_q).value_counts(normalize=True).reindex(ORDER).fillna(0)
    rng2 = np.random.default_rng(5)
    for d in range(N_DRAW):
        cnt = dict.fromkeys(ORDER, 0)
        for risk, p, k in events:
            pick = rng2.choice(risk, size=k, replace=False, p=p) if k <= len(risk) else risk
            for i in np.atleast_1d(pick):
                cnt[base[i]] += 1
        tot = sum(cnt.values())
        null_counts[d] = [cnt[q] / tot for q in ORDER]

    print(f"  {len(obs_q):,} entry events, {N_DRAW} proximity-weighted redraws\n")
    print(f"  {'class':<12}{'observed':>10}{'prox. null':>12}{'ratio':>8}{'p':>9}")
    for j, q in enumerate(ORDER):
        nm = null_counts[:, j].mean()
        p = (null_counts[:, j] >= obs[q]).mean() if obs[q] >= nm else \
            (null_counts[:, j] <= obs[q]).mean()
        print(f"  {q:<12}{obs[q]:>10.3f}{nm:>12.3f}{obs[q]/nm:>8.2f}{p:>9.3f}")


if __name__ == "__main__":
    main()
