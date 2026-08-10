"""Figure 2. How actors adopt within the space of concerns.

(A) ENTRY     chance of taking up a concern against its distance from the
              actor's existing portfolio, split by the kind of ground it is,
              with Wilson 95% intervals
(B) MOVEMENT  given a move steps off ground of one class, where it lands;
              rows sum to 100%

    PYTHONPATH=. micromamba run -n ultraplot-dev python fig02_entry_and_cohorts.py
"""

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.lines import Line2D

import figstyle
from concern_classes import NICE, ORDER, QCOL, load_classes
from utils import (_split_multi_value, compute_rolling_rca,
                   generate_interaction_matrix, get_rca, load_data,
                   standardize_index_labels)

OUT = "figures/fig02_entry_and_cohorts"
WIN, N_DRAW, N_BOOTC = 15, 200, 400
INK, MUTED = figstyle.PRIMARY, figstyle.MUTED
COH = [(1961, 1980, "Joined\n1961–80"), (1981, 1990, "Joined\n1981–90"),
       (1991, 2010, "Joined\n1991–2010")]
MIN_TOPICS = 3


def main():
    Q = load_classes()
    topics, quad_of, PHI = Q["topics"], Q["quad_of"], Q["PHI"]
    idx = {t: i for i, t in enumerate(topics)}
    P0 = np.nan_to_num(PHI)

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

    roll = compute_rolling_rca(submitted, window_years=5)
    tsetT = set(topics)
    years = sorted(roll["year"].unique())

    # ---------- collect every expansion, with its foothold and its actor ----
    ev, obs_to, from_cls, act = [], [], [], []
    RISK = []
    atrisk = {q: 0 for q in ORDER}
    entered = {q: 0 for q in ORDER}
    for y0, y1 in zip(years[:-1], years[1:]):
        cur = roll[(roll["year"] == y0) & (roll["rca"] >= 1.0)]
        nxt = roll[(roll["year"] == y1) & (roll["rca"] >= 1.0)]
        for a, ga in cur.groupby("country"):
            held = {t for t in ga["topic"] if t in tsetT}
            if not held:
                continue
            new = {t for t in nxt[nxt["country"] == a]["topic"] if t in tsetT} - held
            # sorted, not set order: 6.9% of moves have a tied nearest held
            # concern, and unsorted set iteration makes argmax pick a different
            # one per process, wobbling panel B by about a point between runs
            hi = sorted(idx[h] for h in held)
            risk = [i for i, t in enumerate(topics) if t not in held]
            # every actor-period contributes chances, including those that
            # expanded into nothing -- otherwise the rate conditions on
            # having expanded and roughly doubles
            dmin = 1.0 - P0[np.ix_(risk, hi)].max(axis=1)
            for i, dd in zip(risk, dmin):
                tt = topics[i]
                atrisk[quad_of[tt]] += 1
                RISK.append((dd, quad_of[tt], 1 if tt in new else 0))
            for t in new:
                entered[quad_of[t]] += 1
            if not new:
                continue
            w = P0[np.ix_(risk, hi)].max(axis=1)
            if w.sum() > 0:
                ev.append((risk, w / w.sum(), len(new)))
            for t in new:
                j = idx[t]
                near = hi[int(np.argmax(P0[j, hi]))]
                obs_to.append(quad_of[t])
                from_cls.append(quad_of[topics[near]])
                act.append(a)

    # ---------- (A) entry vs a proximity-weighted null ----------------------
    o = pd.Series(obs_to).value_counts(normalize=True).reindex(ORDER).fillna(0)
    rng = np.random.default_rng(5)
    NC = np.zeros((N_DRAW, 4))
    for d in range(N_DRAW):
        c = dict.fromkeys(ORDER, 0)
        for risk, p, k in ev:
            for i in np.atleast_1d(rng.choice(risk, size=min(k, len(risk)),
                                              replace=False, p=p)):
                c[quad_of[topics[i]]] += 1
        tt = sum(c.values())
        NC[d] = [c[q] / tt for q in ORDER]
    entry_ratio = {q: o[q] / NC[:, j].mean() for j, q in enumerate(ORDER)}
    entry_lo = {q: o[q] / np.percentile(NC[:, j], 97.5) for j, q in enumerate(ORDER)}
    entry_hi = {q: o[q] / np.percentile(NC[:, j], 2.5) for j, q in enumerate(ORDER)}
    print("A entry ratios:", {q: round(entry_ratio[q], 2) for q in ORDER})

    # ---------- (B) the fork: destinations of moves that leave Gateway ------
    dfm = pd.DataFrame({"actor": act, "from": from_cls, "to": obs_to})
    dfm["first"] = dfm["actor"].map(first)
    fork = {}
    for lo, hi, lab in COH:
        s = dfm[(dfm["first"] >= lo) & (dfm["first"] <= hi)]
        if len(s) < 20:
            continue
        fork[lab] = (s["to"].value_counts(normalize=True).reindex(ORDER).fillna(0),
                     len(s))
    print("B fork n:", {k: v[1] for k, v in fork.items()})

    # ---------- (C) uptake vs opportunity, full-pipeline bootstrap ---------
    # Resampling actors alone would hold the availability denominator fixed,
    # which it is not -- it is estimated from the same archive. So each
    # replicate resamples actors and re-estimates BOTH the actor portfolios
    # and the availability they are measured against.
    from utils import generate_interaction_matrix, get_rca, standardize_index_labels
    qidx = {q: i for i, q in enumerate(ORDER)}
    cls = np.array([qidx[Q["quad_of"][t]] for t in topics])
    elig = [a for a in first.index if last[a] - first[a] >= WIN - 1]
    wins = sorted({(int(first[a]), int(first[a]) + WIN - 1) for a in elig})
    CM = {}
    for lo, hi in wins:
        sub = submitted[(submitted["year"] >= lo) & (submitted["year"] <= hi)]
        c = standardize_index_labels(
            generate_interaction_matrix(sub, countries, tset))
        if c.index.has_duplicates:
            c = c.groupby(level=0).sum()
        CM[(lo, hi)] = c.reindex(topics, fill_value=0)
    allc = sorted(CM[wins[0]].columns)

    def replicate(cols, seen=None):
        out = {c[2]: {q: [] for q in ORDER} for c in COH}
        for lo, hi in wins:
            M = CM[(lo, hi)][cols]
            M.columns = [f"{c}_{i}" for i, c in enumerate(cols)]
            R = (get_rca(M).values >= 1.0)
            if R.sum() == 0:
                continue
            av = np.array([R[cls == k].sum() for k in range(4)], float) / R.sum()
            for j, orig in enumerate(cols):
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
                        if seen is not None:
                            seen.setdefault(nm, set()).add(orig)
                        for k, q in enumerate(ORDER):
                            if av[k] > 0:
                                out[nm][q].append(mine[k] / av[k])
        return {nm: {q: (np.mean(v) if v else np.nan) for q, v in d.items()}
                for nm, d in out.items()}

    def cont_rep(cols):
        """Per cohort, mean reach and mean holder breadth of the opening
        portfolio relative to the window's availability-weighted mean, on the
        continuous scales behind the class cut (partition-free restatement)."""
        rch = np.array(Q["reach"], float)
        brd = np.array(Q["breadth"], float)
        out = {nm: {"reach": [], "breadth": []} for _l, _h, nm in COH}
        for lo, hi in wins:
            M = CM[(lo, hi)][cols]
            M.columns = [f"{c}_{i}" for i, c in enumerate(cols)]
            R = (get_rca(M).values >= 1.0)
            if R.sum() == 0:
                continue
            av_r = (R * rch[:, None]).sum() / R.sum()
            av_b = (R * brd[:, None]).sum() / R.sum()
            for j, orig in enumerate(cols):
                if orig not in first.index:
                    continue
                if (int(first[orig]), int(first[orig]) + WIN - 1) != (lo, hi):
                    continue
                held = R[:, j]
                if held.sum() < MIN_TOPICS:
                    continue
                mine_r = (held * rch).sum() / held.sum()
                mine_b = (held * brd).sum() / held.sum()
                for lo2, hi2, nm in COH:
                    if lo2 <= first[orig] <= hi2:
                        if av_r > 0:
                            out[nm]["reach"].append(mine_r / av_r)
                        if av_b > 0:
                            out[nm]["breadth"].append(mine_b / av_b)
        return {nm: {k: (np.mean(v) if v else np.nan) for k, v in d.items()}
                for nm, d in out.items()}

    contrib = {}
    obsC = replicate(allc, seen=contrib)
    rngC = np.random.default_rng(23)
    BS = [replicate(list(rngC.choice(allc, len(allc), replace=True)))
          for _ in range(N_BOOTC)]
    obsC_cont = cont_rep(allc)
    rngC2 = np.random.default_rng(23)
    BS_cont = [cont_rep(list(rngC2.choice(allc, len(allc), replace=True)))
               for _ in range(N_BOOTC)]
    print("\nC continuous (partition-free) cohort ratios, breadth | reach:")
    for lo, hi, nm in COH:
        lr = np.percentile([b[nm]["reach"] for b in BS_cont], 2.5)
        hr = np.percentile([b[nm]["reach"] for b in BS_cont], 97.5)
        lb = np.percentile([b[nm]["breadth"] for b in BS_cont], 2.5)
        hb = np.percentile([b[nm]["breadth"] for b in BS_cont], 97.5)
        print(f"   {nm}: breadth {obsC_cont[nm]['breadth']:.3f} "
              f"[{lb:.3f},{hb:.3f}] | reach {obsC_cont[nm]['reach']:.3f} "
              f"[{lr:.3f},{hr:.3f}]")
    # count the actors that actually contribute a ratio, not those that merely
    # clear the tenure test -- a few hold fewer than MIN_TOPICS specialized
    # concerns in their opening window and drop out inside replicate()
    ncoh = [len(contrib.get(nm, ())) for _lo, _hi, nm in COH]
    print("C cohort n (contributing):", ncoh)
    print("C cohort n (tenure-eligible):",
          [sum(1 for a in elig if lo <= first[a] <= hi) for lo, hi, _ in COH])

    # ======================= figure ======================================
    # Main-text Figure 2 keeps the two locality panels (A entry, B movement
    # between kinds of ground); the cohort panel moved out to the SI.
    fig, axs = uplt.subplots(ncols=2, refwidth=3.5, refaspect=1, share=False,
                             wspace=("1.45in",))
    y = np.arange(4)[::-1]

    # (A) entry probability against distance, split by class. Same form as
    # the paper's existing hazard panel, so the locality gradient is visible
    # and the class effect sits on top of it.
    ax = axs[0]
    def wilson(k, n, z=1.96):
        """Wilson score interval -- the paper's convention for these bins."""
        if n == 0:
            return 0.0, 0.0, 0.0
        p = k / n
        den = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
        return p, max(c - h, 0.0), min(c + h, 1.0)

    RK = pd.DataFrame(RISK, columns=["d", "cls", "y"])
    edges = np.quantile(RK["d"], np.linspace(0, 1, 8))
    for q in ORDER:
        sub = RK[RK["cls"] == q]
        xs_, ys_, lo_e, hi_e = [], [], [], []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            bb = sub[(sub["d"] >= a_) & (sub["d"] < b_)]
            if len(bb) < 150:
                continue
            pr, l, h = wilson(bb["y"].sum(), len(bb))
            xs_.append(bb["d"].mean())
            ys_.append(100 * pr)
            lo_e.append(100 * (pr - l))
            hi_e.append(100 * (h - pr))
        ax.errorbar(xs_, ys_, yerr=[lo_e, hi_e], color=QCOL[q], lw=2.4,
                    marker="o", ms=6, markeredgecolor="white",
                    markeredgewidth=0.9, capsize=2.5, elinewidth=1.2,
                    label=NICE[q], zorder=3)
    ax.legend([Line2D([], [], marker="o", ls="-", markersize=6, color=QCOL[q],
                      markeredgecolor="white", label=NICE[q]) for q in ORDER],
              loc="ur", ncols=1, frame=False, fontsize=9)
    ax.format(xlabel="Distance from the actor's existing portfolio",
              ylabel="Chance of being taken up (%)",
              ylim=(-0.5, 10.4),
              title="Nearer concerns are taken up more — and far out, only\n"
                    "ground that narrow portfolios hold is entered at all",
              titlesize=10.5, labelsize=10, ticklabelsize=9.5)

    # (B) given a move steps off ground of one class, where does it land
    ax = axs[1]
    M4 = pd.crosstab(pd.Series(from_cls), pd.Series(obs_to)).reindex(
        index=ORDER, columns=ORDER).fillna(0)
    Mrow = M4.div(M4.sum(axis=1), axis=0)
    # each cell is the blend of the two classes it joins; opacity carries the
    # probability, so the diagonal shows as the pure class colour
    from matplotlib.colors import to_rgb
    RGBA = np.ones((4, 4, 4))
    vmax = float(Mrow.values.max())
    for i, qi in enumerate(ORDER):
        for j, qj in enumerate(ORDER):
            mix = 0.5 * (np.array(to_rgb(QCOL[qi])) + np.array(to_rgb(QCOL[qj])))
            a = 0.10 + 0.90 * (Mrow.values[i, j] / vmax)
            RGBA[i, j, :3] = 1.0 - a * (1.0 - mix)   # composite onto white
    ax.imshow(RGBA, extent=(0, 4, 0, 4), origin="upper", aspect="auto",
              interpolation="nearest")
    for i in range(4):
        for j in range(4):
            v = Mrow.values[i, j]
            lum = float(np.dot(RGBA[i, j, :3], [0.299, 0.587, 0.114]))
            ax.text(j + 0.5, 3.5 - i, f"{v*100:.0f}%", ha="center", va="center",
                    fontsize=11, color="white" if lum < 0.55 else "0.12")
    ax.set_xticks(np.arange(4) + 0.5); ax.set_yticks(np.arange(4) + 0.5)
    ax.set_xticklabels([NICE[q] for q in ORDER], rotation=25, fontsize=9)
    ax.set_yticklabels([NICE[q] for q in ORDER][::-1], fontsize=9)
    for lab, q in zip(ax.get_yticklabels(), ORDER[::-1]):
        lab.set_color(QCOL[q]); lab.set_fontweight("bold")
    for lab, q in zip(ax.get_xticklabels(), ORDER):
        lab.set_color(QCOL[q]); lab.set_fontweight("bold")
    ax.format(xlabel="…it lands on this kind of ground",
              ylabel="If a move steps off this ground…",
              title="Movement between kinds of ground\n"
                    f"(rows sum to 100%, n={len(obs_to):,})",
              titlesize=10.5, labelsize=10)

    axs.format(abc="a", abcloc="ul", abcsize=12, grid=False)
    fig.save(f"{OUT}.pdf")
    fig.save(f"{OUT}.png", dpi=200)
    print("wrote", f"{OUT}.png")

    # ---- numbers quoted in the caption ------------------------------------
    print("\nA  entry vs a proximity-weighted null (200 redraws):")
    for q in ORDER:
        print(f"   {NICE[q]:<11} {entry_ratio[q]:.2f}x "
              f"[{entry_lo[q]:.2f}, {entry_hi[q]:.2f}]")
    # the two barriers do not bite equally at all distances
    RK2 = RK.assign(narrow=RK["cls"].isin(["gateway", "margin"]),
                    hireach=RK["cls"].isin(["gateway", "platform"]))
    qs = np.quantile(RK2["d"], [0, 1 / 7, 6 / 7, 1.0])
    print("\nA  entry rate by barrier, nearest vs farthest seventh:")
    for nm, lo_, hi_ in [("nearest", qs[0], qs[1]), ("farthest", qs[2], qs[3])]:
        s = RK2[(RK2["d"] >= lo_) & (RK2["d"] <= hi_)]
        nb = 100 * s[s.narrow]["y"].mean(), 100 * s[~s.narrow]["y"].mean()
        rr = 100 * s[s.hireach]["y"].mean(), 100 * s[~s.hireach]["y"].mean()
        print(f"   {nm:<9} holder breadth narrow {nb[0]:.2f}% vs broad "
              f"{nb[1]:.2f}% ({nb[0]/nb[1]:.2f}x) | "
              f"reach high {rr[0]:.2f}% vs low {rr[1]:.2f}% ({rr[0]/rr[1]:.2f}x)")

    print(f"\nB  {len(obs_to):,} moves; stay on the same class "
          f"{np.trace(M4.values) / M4.values.sum():.1%}; "
          f"land on Gateway {100 * pd.Series(obs_to).eq('gateway').mean():.0f}%")
    print("\nC  uptake relative to opportunity, 95% full-pipeline bootstrap:")
    for _lo, _hi, nm in COH:
        cells = []
        for q in ORDER:
            v = obsC[nm][q]
            bs = np.array([b[nm][q] for b in BS if np.isfinite(b[nm][q])])
            l, h = np.percentile(bs, [2.5, 97.5])
            star = "" if l <= 1.0 <= h else "  *excludes 1"
            cells.append(f"   {nm.replace(chr(10), ' '):<18} {NICE[q]:<11} "
                         f"{v:.2f} [{l:.2f}, {h:.2f}]{star}")
        print("\n".join(cells))


if __name__ == "__main__":
    main()
