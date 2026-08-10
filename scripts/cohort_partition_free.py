"""Partition-free restatement of the cohort comparison.

The class-based uptake-to-opportunity ratios in fig02_entry_and_cohorts.py
split concerns into four classes by cutting reach and holder breadth at their
medians. That cut is an act of discretization (ARI 0.30), so the reviewer
asked for a restatement on the continuous scales themselves. This script
computes, per cohort, the mean reach and mean holder breadth of the concerns
in the cohort's opening portfolio relative to what the space offered during
the same years, without any partition.

    PYTHONPATH=src:. micromamba run -n ultraplot-dev python scratch/cohort_partition_free.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from concern_classes import load_classes
from utils import (_split_multi_value, generate_interaction_matrix, get_rca,
                   load_data, standardize_index_labels)

WIN, N_BOOTC, MIN_TOPICS = 15, 400, 3
COH = [(1961, 1980, "1961-80"), (1981, 1990, "1981-90"),
       (1991, 2010, "1991-2010")]


def main():
    Q = load_classes()
    topics = Q["topics"]
    reach = np.array(Q["reach"], float)
    breadth = np.array(Q["breadth"], float)
    idx = {t: i for i, t in enumerate(topics)}

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

    def replicate(cols, seen=None):
        """Per cohort, mean reach and mean holder breadth of the opening
        portfolio relative to the window's availability-weighted mean."""
        out = {nm: {"reach": [], "breadth": []} for _l, _h, nm in COH}
        for lo, hi in wins:
            M = CM[(lo, hi)][cols]
            M.columns = [f"{c}_{i}" for i, c in enumerate(cols)]
            R = (get_rca(M).values >= 1.0)
            if R.sum() == 0:
                continue
            av = (R * reach[:, None]).sum() / R.sum()
            ab = (R * breadth[:, None]).sum() / R.sum()
            for j, orig in enumerate(cols):
                if orig not in first.index:
                    continue
                if (int(first[orig]), int(first[orig]) + WIN - 1) != (lo, hi):
                    continue
                held = R[:, j]
                if held.sum() < MIN_TOPICS:
                    continue
                if seen is not None:
                    for _lo2, _hi2, nm in COH:
                        if _lo2 <= first[orig] <= _hi2:
                            seen.setdefault(nm, set()).add(orig)
                mine_r = (held * reach).sum() / held.sum()
                mine_b = (held * breadth).sum() / held.sum()
                for _lo2, _hi2, nm in COH:
                    if _lo2 <= first[orig] <= _hi2:
                        out[nm]["reach"].append(mine_r / av)
                        out[nm]["breadth"].append(mine_b / ab)
        return {nm: {k: (np.mean(v) if v else np.nan) for k, v in d.items()}
                for nm, d in out.items()}

    contrib = {}
    obs = replicate(allc, seen=contrib)
    rng = np.random.default_rng(23)
    BS = [replicate(list(rng.choice(allc, len(allc), replace=True)))
          for _ in range(N_BOOTC)]

    ncoh = {nm: len(s) for nm, s in contrib.items()}
    print("cohort n (contributing):", ncoh)

    print("\npartition-free cohort uptake vs opportunity (ratio of cohort mean "
          "to availability mean):")
    for _lo, _hi, nm in COH:
        lo_r = np.percentile([b[nm]["reach"] for b in BS], 2.5)
        hi_r = np.percentile([b[nm]["reach"] for b in BS], 97.5)
        lo_b = np.percentile([b[nm]["breadth"] for b in BS], 2.5)
        hi_b = np.percentile([b[nm]["breadth"] for b in BS], 97.5)
        print(f"  {nm}: reach {obs[nm]['reach']:.3f} [{lo_r:.3f},{hi_r:.3f}] | "
              f"breadth {obs[nm]['breadth']:.3f} [{lo_b:.3f},{hi_b:.3f}]")

    # continuous equivalents of the availability itself, for the record
    print("\navailability-weighted mean reach and breadth per cohort window:")
    for _lo, _hi, nm in COH:
        lo, hi = _lo, _hi + WIN - 1
        sub = submitted[(submitted["year"] >= lo) & (submitted["year"] <= hi)]
        c = standardize_index_labels(
            generate_interaction_matrix(sub, countries, set(topics)))
        if c.index.has_duplicates:
            c = c.groupby(level=0).sum()
        c = c.reindex(topics, fill_value=0)
        R = (get_rca(c).values >= 1.0)
        print(f"  {nm}: reach {((R*reach[:,None]).sum()/R.sum()):.2f}, "
              f"breadth {((R*breadth[:,None]).sum()/R.sum()):.2f}")


if __name__ == "__main__":
    main()
