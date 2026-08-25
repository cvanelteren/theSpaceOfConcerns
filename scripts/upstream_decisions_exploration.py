#!/usr/bin/env python
"""Three ATS decisions and when the agenda actually moved.

For each of three formal decisions, plot the share of the archive devoted to
the decision's concern set per 5-yr window, with the decision date as a
vertical line. If formal outcomes ratify upstream reorientations (the reading
supported by the Madrid event study), the engagement rise should begin well
before the line in every case:

  Madrid Protocol   1991  liability, inspections, EMR, env. protection
                            general, waste management, CEE
  Annex VI (liab.)  2005  liability
  Measure 15 (tour) 2009  tourism & NG activities, site guidelines for
                          visitors, search and rescue

Also prints, per set, the window in which the share first exceeds half of its
pre-decision-or-earlier peak, as a crude onset date.

Usage::

    micromamba run -n ultraplot-dev python scripts/upstream_decisions_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from madrid_shock_exploration import long_panel  # noqa: E402
from utils import load_data  # noqa: E402

OUT_PNG = ROOT / "output" / "preview_upstream_decisions.png"

CASES = [
    ("Madrid Protocol", 1991, {
        "liability", "inspections", "environmental monitoring and reporting",
        "environmental protection general", "waste management and disposal",
        "comprehensive environmental evaluations",
    }),
    ("Annex VI (liability)", 2005, {"liability"}),
    ("Measure 15 (tourism)", 2009, {
        "tourism and ng activities", "site guidelines for visitors",
        "search and rescue",
    }),
]


def main() -> int:
    counts, raw, _, _ = load_data(
        str(ROOT / "antarctic-database-go/data/processed/document-summary.parquet")
    )
    df = long_panel(raw)
    df["w"] = (df["year"] - 1961) // 5
    windows = sorted(df["w"].unique())
    tot = df.groupby("w").size()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=False)
    for ax, (name, year, topics) in zip(axes, CASES):
        nk = lambda t: " ".join(str(t).strip().lower().replace("_", " ")
                                .replace("-", " ").split())
        sub = df[df["topic"].map(nk).isin(topics)]
        assert len(sub) > 0, name
        share = sub.groupby("w").size().reindex(windows, fill_value=0) / tot
        yrs = [1961 + w * 5 + 2 for w in windows]
        ax.plot(yrs, share * 100, "o-", color="#0072B2", ms=4)
        ax.axvline(year, color="#D55E00", ls="--", lw=1.0)
        ax.set_title(f"{name} ({year})", fontsize=10)
        ax.set_xlabel("year")
        ax.set_ylabel("share of archive (%)")

        s = share.to_numpy()
        peak = s.max()
        onset = windows[int(np.argmax(s >= 0.5 * peak))]
        onset_yr = 1961 + onset * 5
        half_pre = [w for w in windows if 1961 + w * 5 + 4 <= year]
        pre_peak = s[[w in half_pre for w in windows]].max()
        onset_pre = windows[int(np.argmax(
            [s[i] >= 0.5 * pre_peak if w in half_pre else False
             for i, w in enumerate(windows)]
        ))]
        print(f"{name:24s} decision {year}   share onset (>=50% pre-peak): "
              f"{1961 + onset_pre * 5}-{1965 + onset_pre * 5}   "
              f"peak: {1961 + int(np.argmax(s)) * 5}-{1965 + int(np.argmax(s)) * 5}   "
              f"max share {peak * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
