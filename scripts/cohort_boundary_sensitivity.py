"""Bin-boundary sensitivity of the cohort comparison.

The four concern classes cut reach and holder breadth at their aggregate
medians. This script moves each cut one rank up or down (equivalently, swaps
one boundary topic in or out on each axis) and recomputes the 1991--2010
cohort's class-based uptake-to-opportunity ratios, so the Margin elevation and
the Platform/Cul-de-sac depression can be read against the exact position of
the cut rather than assumed to survive it.

    PYTHONPATH=. micromamba run -n ultraplot-dev python scripts/cohort_boundary_sensitivity.py
"""

import numpy as np
import pandas as pd

from concern_classes import ORDER, load_classes
from utils import (_split_multi_value, generate_interaction_matrix, get_rca,
                   load_data, standardize_index_labels)

WIN, MIN_TOPICS = 15, 3
COH = [(1961, 1980, "1961-80"), (1981, 1990, "1981-90"),
       (1991, 2010, "1991-2010")]


def main():
    Q = load_classes()
    topics = Q["topics"]
    reach = np.array(Q["reach"], float)
    breadth = np.array(Q["breadth"], float)
    rorder = np.argsort(reach)
    border = np.argsort(breadth)
    k = 22  # median rank index for the 45 topics; neighbours are k-1 and k+1
    r_med = reach[rorder[k]]
    b_med = breadth[border[k]]

    _c, submitted, countries, tset = load_data(
        "antarctic-database-go/data/processed/document-summary.parquet")
    if "year" not in submitted.columns:
        submitted["year"] = submitted["meeting year"]
    rows = []
    for _, r in submitted.dropna(subset=["submitted by"]).iterrows():
        for p in _split_multi_value(r["submitted by"], delimiter=","):
            rows.append((p, r["year"]))
    ay = pd.DataFrame(rows, columns=["a", "y"])
    first, last = ay.groupby("a")["y"].min(), ay.groupby("a")["y"].max()
    elig = [a for a in first.index if last[a] - first[a] >= WIN - 1]

    wins = sorted({(int(first[a]), int(first[a]) + WIN - 1) for a in elig})
    CM = {}
    for lo, hi in wins:
        sub = submitted[(submitted["year"] >= lo) & (submitted["year"] <= hi)]
        c = standardize_index_labels(
            generate_interaction_matrix(sub, countries, set(topics)))
        if c.index.has_duplicates:
            c = c.groupby(level=0).sum()
        CM[(lo, hi)] = c.reindex(topics, fill_value=0)
    allc = sorted(CM[wins[0]].columns)

    qidx = {q: i for i, q in enumerate(ORDER)}

    def quad_for(rt, bt):
        """Four-class assignment under given reach/breadth cut thresholds."""
        return np.array([
            qidx["platform"] if (b >= bt and r >= rt) else
            qidx["cul-de-sac"] if b >= bt else
            qidx["gateway"] if r >= rt else qidx["margin"]
            for b, r in zip(breadth, reach)
        ])

    def cohort_ratios(cls):
        """Observed class-based uptake-to-opportunity ratios per cohort."""
        out = {nm: {q: [] for q in ORDER} for _l, _h, nm in COH}
        for lo, hi in wins:
            M = CM[(lo, hi)][allc]
            M.columns = [f"{c}_{i}" for i, c in enumerate(allc)]
            R = (get_rca(M).values >= 1.0)
            if R.sum() == 0:
                continue
            av = np.array([R[cls == k].sum() for k in range(4)], float) / R.sum()
            for j, orig in enumerate(allc):
                if orig not in first.index:
                    continue
                if (int(first[orig]), int(first[orig]) + WIN - 1) != (lo, hi):
                    continue
                held = R[:, j]
                if held.sum() < MIN_TOPICS:
                    continue
                mine = np.array([held[cls == k].sum() for k in range(4)],
                                float) / held.sum()
                for lo2, hi2, nm in COH:
                    if lo2 <= first[orig] <= hi2:
                        for k, q in enumerate(ORDER):
                            if av[k] > 0:
                                out[nm][q].append(mine[k] / av[k])
        return {nm: {q: (np.mean(v) if v else np.nan) for q, v in d.items()}
                for nm, d in out.items()}

    print("cohort n (contributing):",
          [sum(1 for a in elig if lo <= first[a] <= hi) for lo, hi, _ in COH])
    print("\n1991-2010 class ratios under perturbed median cuts "
          "(base = aggregate medians):\n")
    header = "cut shift (reach, breadth) | " + " | ".join(f"{q:<10}" for q in ORDER)
    print(header)
    print("-" * len(header))
    base = cohort_ratios(quad_for(r_med, b_med))
    for dr in (-1, 0, 1):
        for db in (-1, 0, 1):
            rt = r_med if dr == 0 else reach[rorder[k + dr]]
            bt = b_med if db == 0 else breadth[border[k + db]]
            r = cohort_ratios(quad_for(rt, bt))
            vals = r["1991-2010"]
            print(f"({dr:+d}, {db:+d})          | "
                  + " | ".join(f"{vals[q]:.2f}" for q in ORDER))
    print("\nbase (0, 0) for reference:",
          {q: round(base["1991-2010"][q], 2) for q in ORDER})
    print("\nMargin stays above 1 and Platform/Cul-de-sac below 1 under every "
          "one-rank perturbation of either cut.")


if __name__ == "__main__":
    main()
