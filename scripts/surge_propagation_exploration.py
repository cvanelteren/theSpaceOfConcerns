#!/usr/bin/env python
"""Does the agenda front propagate through the space of concerns?

The upstream-decisions finding says the documentary agenda reorients before
formal decisions. If the space of concerns is the structure that reorientation
follows, then at each window the topics whose archive share surges should sit
close (in phi) to the previous window's surging topics -- the front walks
through the topology rather than jumping across it. This is the aggregate
analogue of the actor-level hazard result.

For each window transition, front_w = the top-8 topics by share gain over the
previous window (gains > 0). Statistic: mean 1-max(phi) distance from each
front_w topic to front_{w-1}. Null: 2000 random topic sets of the same size.
One-sided p = share of null draws at or below the observed distance.

Usage::

    micromamba run -n ultraplot-dev python scripts/surge_propagation_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from madrid_shock_exploration import long_panel  # noqa: E402
from utils import compute_product_space, get_rca, load_data  # noqa: E402

FRONT_K = 8
N_NULL = 2000
SEED = 20260806


def main() -> int:
    counts, raw, _, _ = load_data(
        str(ROOT / "antarctic-database-go/data/processed/document-summary.parquet")
    )
    phi = compute_product_space(get_rca(counts))
    topics = list(phi.index)
    ti = {t: i for i, t in enumerate(topics)}
    P = phi.to_numpy(dtype=float)

    df = long_panel(raw)
    df["w"] = (df["year"] - 1961) // 5
    windows = sorted(df["w"].unique())
    tot = df.groupby("w").size()
    share = (
        df.groupby(["w", "topic"]).size().unstack(fill_value=0)
        .reindex(columns=topics, fill_value=0)
        .div(tot, axis=0)
    )

    rng = np.random.default_rng(SEED)
    print("window transition   obs dist   null mean   p(one-sided)   pre-1991?")
    obs_all, null_all = [], []
    for w in windows[1:]:
        prev, cur = share.loc[w - 1].to_numpy(), share.loc[w].to_numpy()
        gain = cur - prev
        front = np.argsort(gain)[::-1][:FRONT_K]
        front = [i for i in front if gain[i] > 0]
        prev_front = np.argsort(prev - share.loc[w - 2].to_numpy())[::-1][:FRONT_K] \
            if w - 2 in windows else np.argsort(prev)[::-1][:FRONT_K]
        if len(front) < 3:
            continue
        d = 1.0 - P[np.ix_(front, list(prev_front))].max(axis=1)
        obs = float(d.mean())
        nulls = []
        for _ in range(N_NULL):
            draw = rng.integers(0, len(topics), size=len(front))
            nd = 1.0 - P[np.ix_(list(draw), list(prev_front))].max(axis=1)
            nulls.append(nd.mean())
        nulls = np.asarray(nulls)
        p = float((nulls <= obs).mean())
        pre = 1961 + w * 5 + 4 <= 1990
        obs_all.append(obs)
        null_all.append(nulls.mean())
        print(f"  {1961 + (w - 1) * 5}-{1965 + (w - 1) * 5} -> "
              f"{1961 + w * 5}-{1965 + w * 5}   {obs:.3f}   {nulls.mean():.3f}   "
              f"{p:.3f}   {'yes' if pre else 'no'}")

    obs_all = np.asarray(obs_all)
    null_all = np.asarray(null_all)
    frac_local = float((obs_all < null_all).mean())
    print(f"\nfront closer than null in {frac_local:.0%} of transitions "
          f"(mean obs {obs_all.mean():.3f} vs null {null_all.mean():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
