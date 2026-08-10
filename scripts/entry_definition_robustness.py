"""Is the locality result an artefact of defining entry as a relative-share crossing?

RPA is a ratio, so an actor can cross the RPA >= 1 threshold on a topic when the
topic's system-wide share falls, without changing its own behaviour. The referee
objection is that the outcome variable is therefore partly mechanical.

Three outcome definitions are compared on the same choice sets:

    relative   RPA crosses from < 1 to >= 1                       (the paper's)
    absolute   the actor files at least MINDOC papers on the topic
               in the new window, having filed none in the prior one
    both       RPA crossing AND at least MINDOC papers

If the distance coefficient survives the absolute definition, the gradient is
not a by-product of the ratio.

    PYTHONPATH=. micromamba run -n ultraplot-dev python scratch/entry_definition_robustness.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from concern_classes import load_classes
from utils import (_split_multi_value, compute_rolling_rca,
                   generate_interaction_matrix, get_rca, load_data,
                   standardize_index_labels)

WIN = 5
MINDOC = 1


def main():
    Q = load_classes()
    topics, P0 = Q["topics"], np.nan_to_num(Q["PHI"])
    i2 = {t: i for i, t in enumerate(topics)}
    tset = set(topics)

    _c, sub, countries, tset_all = load_data(
        "antarctic-database-go/data/processed/document-summary.parquet")
    if "year" not in sub.columns:
        sub["year"] = sub["meeting year"]

    # raw per-window document counts, so entry can be defined on activity
    years = sorted(sub["year"].unique())
    wins = [(y - WIN + 1, y) for y in range(min(years) + WIN - 1, max(years) + 1)]
    counts = {}
    for lo, hi in wins:
        w = sub[(sub["year"] >= lo) & (sub["year"] <= hi)]
        c = standardize_index_labels(
            generate_interaction_matrix(w, countries, tset_all))
        if c.index.has_duplicates:
            c = c.groupby(level=0).sum()
        counts[(lo, hi)] = c.reindex(topics, fill_value=0)

    roll = compute_rolling_rca(sub, window_years=WIN)
    ry = sorted(roll["year"].unique())

    rows = []
    for y0, y1 in zip(ry[:-1], ry[1:]):
        w0, w1 = (y0 - WIN + 1, y0), (y1 - WIN + 1, y1)
        if w0 not in counts or w1 not in counts:
            continue
        C0, C1 = counts[w0], counts[w1]
        cur = roll[(roll["year"] == y0) & (roll["rca"] >= 1.0)]
        nxt = roll[(roll["year"] == y1) & (roll["rca"] >= 1.0)]
        pop = cur["topic"].value_counts()
        n_act = max(cur["country"].nunique(), 1)
        for a, ga in cur.groupby("country"):
            held = {t for t in ga["topic"] if t in tset}
            if not held or a not in C0.columns or a not in C1.columns:
                continue
            new_rel = {t for t in nxt[nxt["country"] == a]["topic"] if t in tset} - held
            hi = sorted(i2[h] for h in held)
            for t in topics:
                if t in held:
                    continue
                j = i2[t]
                d0, d1 = float(C0.loc[t, a]), float(C1.loc[t, a])
                rows.append((
                    a, y0,
                    1.0 - P0[j, hi].max(),
                    float(pop.get(t, 0)) / n_act,
                    1 if t in new_rel else 0,                       # relative
                    1 if (d0 == 0 and d1 >= MINDOC) else 0,         # absolute
                    1 if (t in new_rel and d1 >= MINDOC) else 0,    # both
                ))
    D = pd.DataFrame(rows, columns=["member", "year", "distance", "popularity",
                                    "rel", "abs", "both"])
    print(f"choice sets {D.groupby(['member','year']).ngroups:,}, "
          f"at-risk observations {len(D):,}\n")
    print(f'{"outcome":10s}{"events":>9}{"rate":>9}{"OR per +0.1 distance":>24}{"p":>12}')
    print("-" * 66)
    for col, lab in (("rel", "relative"), ("abs", "absolute"), ("both", "both")):
        y = D[col].astype(float)
        X = sm.add_constant(D[["distance", "popularity"]].astype(float))
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": D["member"]})
        or10 = np.exp(0.1 * m.params["distance"])
        print(f'{lab:10s}{int(y.sum()):9,}{y.mean():9.4f}{or10:24.3f}{m.pvalues["distance"]:12.2g}')

    # how much do the two definitions actually disagree?
    agree = ((D["rel"] == 1) & (D["abs"] == 1)).sum()
    print(f"\noverlap: {agree:,} observations counted as entry under both "
          f"({100*agree/max(D['rel'].sum(),1):.0f}% of relative entries)")
    print(f"relative-only {int((D.rel==1).sum()-agree):,}   absolute-only "
          f"{int((D['abs']==1).sum()-agree):,}")


if __name__ == "__main__":
    main()
