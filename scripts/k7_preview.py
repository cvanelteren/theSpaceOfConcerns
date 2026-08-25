#!/usr/bin/env python
"""Preview: what Figure 3A/B would look like under the k=7 backbone-modularity
partition instead of the three modes. NOT a paper figure -- a decision aid.

Rebuilds actor-window dominant labels from the raw document panel (RPA>1 per
5-year window, dominant = largest share of max(RPA-1,0), windows with fewer
than three specialized topics dropped, as in Methods), then draws:

  A  the 7-state transition chain between adjacent windows
  B  the 1,895 single moves with one diagonal band per community

Usage::

    micromamba run -n ultraplot-dev python scripts/k7_preview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fig01_space_of_concerns_topology as f1  # noqa: E402
import figstyle  # noqa: E402
from utils import _normalize_topic_label, _split_multi_value  # noqa: E402

OUT = ROOT / "output" / "preview_k7_fig3AB.png"
PALETTE = [
    "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#F0E442",
    "#56B4E9", "#994F00", "#40B0A6", "#E1BE6A", "#7B3294",
]


def load_k7():
    df = pd.read_csv(ROOT / "output" / "modularity_bb_k7_topics.csv")
    lab = {f1.normalize_topic_key(t): int(c) - 1 for c, t in
           zip(df["community"], df["topic"])}
    return lab


def window_dominants(raw, label_of):
    rows = []
    for _, r in raw.dropna(subset=["submitted by"]).iterrows():
        year = r.get("year")
        if pd.isna(year):
            continue
        parties = _split_multi_value(r["submitted by"], ",")
        topics = {
            _normalize_topic_label(t)
            for t in _split_multi_value(r["category"], "\t")
        }
        for p in parties:
            for t in topics:
                rows.append((p, int(year), t))
    panel = pd.DataFrame(rows, columns=["actor", "year", "topic"])
    panel = panel.drop_duplicates()

    dom = {}
    for w0 in range(1961, 2026, 5):
        sub = panel[(panel["year"] >= w0) & (panel["year"] < w0 + 5)]
        if sub.empty:
            continue
        c = sub.pivot_table(index="actor", columns="topic", aggfunc="size",
                            fill_value=0)
        tot = float(c.to_numpy().sum())
        if tot <= 0:
            continue
        actor_share = c.div(c.sum(axis=1), axis=0)
        topic_share = c.sum(axis=0) / tot
        rca = actor_share.div(topic_share.replace(0, np.nan), axis=1)
        spec = rca > 1.0
        signal = rca.where(spec, 0.0).sub(1.0).clip(lower=0.0)
        for actor in c.index:
            topics_spec = [t for t in c.columns if spec.loc[actor, t]]
            if len(topics_spec) < 3:
                continue
            by_lab = {}
            for t in topics_spec:
                k = label_of.get(f1.normalize_topic_key(t))
                if k is None:
                    continue
                by_lab[k] = by_lab.get(k, 0.0) + float(signal.loc[actor, t])
            if by_lab:
                dom[(actor, w0)] = max(by_lab, key=by_lab.get)
    return dom


def main() -> int:
    label_of = load_k7()
    counts, raw, _, _ = f1.load_data_with_fallback()
    dom = window_dominants(raw, label_of)

    order_csv = pd.read_csv(ROOT / "output/fig45_portfolio_space_ridgelines_topic_order.csv")
    x_of = {f1.normalize_topic_key(t): float(x) for t, x in
            zip(order_csv["topic"], order_csv["x_plot"])}
    members = {}
    for t, k in label_of.items():
        members.setdefault(k, []).append(t)
    kmax = max(label_of.values()) + 1
    med_x = {
        k: float(np.median([x_of[t] for t in members[k] if t in x_of]))
        for k in range(kmax)
    }
    axis_order = sorted(range(kmax), key=lambda k: med_x[k])
    rank = {k: i for i, k in enumerate(axis_order)}

    wins = sorted({w for _, w in dom})
    counts_t = np.zeros((kmax, kmax), dtype=float)
    for (actor, w0), d0 in dom.items():
        d1 = dom.get((actor, w0 + 5))
        if d1 is not None:
            counts_t[d0, d1] += 1.0
    P = counts_t / np.maximum(counts_t.sum(axis=1, keepdims=True), 1.0)

    same = float(np.trace(counts_t) / counts_t.sum())
    adj = float(sum(counts_t[i, j] for i in range(kmax) for j in range(kmax)
                    if abs(rank[i] - rank[j]) <= 1) / counts_t.sum())
    print(f"transitions: {int(counts_t.sum())}  same {same:.1%}  "
          f"same-or-adjacent {adj:.1%}")

    moves = pd.read_csv(ROOT / "output/portfolio_displacement_moves.csv")
    mx = {f1.normalize_topic_key(t): float(x) for t, x in
          zip(order_csv["topic"], order_csv["x_plot"])}
    moves["c_from"] = [label_of.get(f1.normalize_topic_key(t), -1)
                       for t in moves["from_topic"]]
    moves["c_to"] = [label_of.get(f1.normalize_topic_key(t), -1)
                     for t in moves["to_topic"]]
    same_zone = float((moves["c_from"] == moves["c_to"]).mean())
    print(f"moves same-community: {same_zone:.1%}")

    fig, (ax_a, ax_b) = uplt.subplots(ncols=2, refwidth=3.4, refaspect=1.1,
                                      share=0, wspace=4.0)

    xs = {k: (i + 0.5) / kmax for i, k in enumerate(axis_order)}
    for i in range(kmax):
        for j in range(kmax):
            if i == j or P[i, j] < 0.015:
                continue
            ax_a.annotate(
                "", xy=(xs[j], 0.0), xytext=(xs[i], 0.0),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=10,
                                color=PALETTE[i % 10], alpha=0.85,
                                lw=0.8 + 9.0 * P[i, j],
                                connectionstyle="arc3,rad=-0.25",
                                shrinkA=14, shrinkB=14),
                zorder=2,
            )
    for i in range(kmax):
        ax_a.scatter([xs[i]], [0.0], s=900, c="white",
                     edgecolor=PALETTE[i % 10],
                     linewidth=1.0 + 9.0 * P[i, i], zorder=3,
                     absolute_size=True)
        ax_a.text(xs[i], 0.0, f"{P[i, i] * 100:.0f}%", ha="center",
                  va="center", fontsize=8.5, fontweight="bold",
                  color=PALETTE[i % 10], zorder=4)
        ax_a.text(xs[i], -0.16, f"C{axis_order[i] + 1}", ha="center",
                  va="top", fontsize=8.5, color=PALETTE[axis_order[i] % 10],
                  fontweight="bold", zorder=4)
    ax_a.set_xlim(-0.05, 1.05)
    ax_a.set_ylim(-0.3, 0.45)
    ax_a.set_title(f"A  dominant-community transitions, k=7\nsame {same:.0%}, "
                   f"same-or-adjacent {adj:.0%}", fontsize=9.5, weight="bold")
    ax_a.axis("off")

    for k in range(kmax):
        vals = [mx[t] for t in members[k] if t in mx]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        ax_b.fill_between([lo, hi], lo, hi, color=PALETTE[k % 10],
                          alpha=0.10, lw=0, zorder=0)
    ax_b.plot([0, 1], [0, 1], color="0.45", lw=1.0, zorder=2)
    grp = moves.groupby(
        [moves["from_topic"].map(mx), moves["to_topic"].map(mx)]
    ).size().reset_index(name="n")
    ax_b.scatter(grp.iloc[:, 0], grp.iloc[:, 1], s=4.0 * grp["n"],
                 color=figstyle.PRIMARY, alpha=0.45, edgecolor="none", zorder=3)
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_title(f"B  single moves, 7 diagonal bands\nsame community "
                   f"{same_zone:.0%}", fontsize=9.5, weight="bold")
    ax_b.format(xlabel="position of nearest topic held",
                ylabel="position of topic entered", grid=False)

    fig.savefig(OUT, dpi=300)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
