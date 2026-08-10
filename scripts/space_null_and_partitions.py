#!/usr/bin/env python
"""Every number the Results paragraph on connectivity and grouping reports.

One script, one JSON, so the manuscript's claims about how well connected the
space is and how reproducibly it can be cut come from a single tracked source
rather than from scattered exploratory files.

It produces, in order:

  1. degree null      -- share of concern pairs whose co-specialization is
                         within +-2 SD of a curveball null that preserves every
                         actor's portfolio size and every concern's holder
                         count exactly. Answers: how much of phi is just
                         generalists co-occurring everywhere?
  2. twelve schemes   -- bootstrap reproducibility (adjusted Rand index) of
                         twelve multi-way partitions of the same space, spanning
                         interval cuts, modularity, spectral and hierarchical
                         clustering.
  3. chance floor     -- ARI between two random partitions with the observed
                         class sizes, so 0.30 can be read against something.
  4. axis diagnostics -- reach, strength, holder breadth and ubiquity: their
                         rank correlations, the bootstrap interval on the
                         reach/breadth correlation, and distance correlations,
                         which are zero only under independence.

Usage:
    micromamba run -n ultraplot-dev python scripts/space_null_and_partitions.py [n_boot] [n_rand]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mode_partition_bootstrap import load_pipeline  # noqa: E402

SEED = 20260806
PHI_STEP = 0.30
OUT = ROOT / "output" / "space_null_and_partitions.json"


# ------------------------------------------------------------ quantities ---
def quantities(rca, phi, names):
    """Reach, strength, holder breadth and ubiquity for each concern."""
    spec = rca.reindex(index=names).to_numpy(dtype=float)
    m = np.nan_to_num(spec, nan=0.0) >= 1.0            # concerns x actors
    p = phi.reindex(index=names, columns=names).to_numpy(dtype=float).copy()
    np.fill_diagonal(p, np.nan)
    reach = np.nansum(p >= PHI_STEP, axis=1)
    strength = np.nansum(p, axis=1)
    mm = m.T.astype(float)                             # actors x concerns
    div, ubi = mm.sum(1), mm.sum(0)                    # portfolio size, holders
    with np.errstate(invalid="ignore", divide="ignore"):
        breadth = np.nan_to_num(
            (mm * div[:, None]).sum(0) / np.where(ubi == 0, np.nan, ubi), nan=0.0)
    return dict(reach=reach, strength=strength, breadth=breadth, ubiquity=ubi), m


def quadrants(reach, breadth):
    mr, mb = np.median(reach), np.median(breadth)
    return np.array([0 if (b >= mb and r >= mr) else 1 if b >= mb
                     else 2 if r >= mr else 3
                     for b, r in zip(breadth, reach)], dtype=int)


# ------------------------------------------------------------ 1. degree null ---
def curveball(A, rng, n_swaps):
    """Randomise a binary matrix keeping every row and column sum exact."""
    B = [set(np.nonzero(r)[0]) for r in A]
    n = len(B)
    for _ in range(n_swaps):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        s, t = B[i], B[j]
        inter = s & t
        xs, xt = list(s - inter), list(t - inter)
        if not xs or not xt:
            continue
        pool = xs + xt
        rng.shuffle(pool)
        ns = len(xs)
        B[i] = inter | set(pool[:ns])
        B[j] = inter | set(pool[ns:])
    out = np.zeros_like(A)
    for i, s in enumerate(B):
        if s:
            out[i, list(s)] = 1
    return out


def degree_null(m, rng, n_rand):
    """How many concern pairs co-specialize more than degree alone predicts?"""
    M = m.T.astype(int)                                 # actors x concerns
    obs = (M.T @ M).astype(float)
    iu = np.triu_indices(obs.shape[0], 1)
    swaps = max(5 * int(M.sum()), 2000)
    draws = np.empty((n_rand, len(iu[0])))
    for k in range(n_rand):
        R = curveball(M.copy(), rng, swaps)
        assert (R.sum(1) == M.sum(1)).all() and (R.sum(0) == M.sum(0)).all()
        draws[k] = (R.T @ R)[iu]
    o = obs[iu]
    mu, sd = draws.mean(0), draws.std(0)
    z = np.divide(o - mu, sd, out=np.zeros_like(o), where=sd > 0)
    return dict(
        n_rand=int(n_rand), n_pairs=int(len(o)),
        frac_within_2sd=float((np.abs(z) <= 2).mean()),
        n_above=int((z > 2).sum()), n_below=int((z < -2).sum()),
        pct_within_2sd=round(100 * float((np.abs(z) <= 2).mean())),
    )


# --------------------------------------------------- 2. twelve partitions ---
def _phi_graph(phi, names):
    g = nx.Graph()
    g.add_nodes_from(names)
    a = phi.reindex(index=names, columns=names).to_numpy(dtype=float)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if a[i, j] > 0:
                g.add_edge(names[i], names[j], weight=float(a[i, j]))
    return g


def _backbone(g):
    bb = nx.maximum_spanning_tree(g, weight="weight")
    w = [d["weight"] for _, _, d in g.edges(data=True)]
    cut = float(np.quantile(w, 0.95))
    for u, v, d in g.edges(data=True):
        if d["weight"] >= cut:
            bb.add_edge(u, v, weight=d["weight"])
    return bb


def _dist_weighted(g):
    h = g.copy()
    for _, _, d in h.edges(data=True):
        w = min(max(float(d["weight"]), 1e-12), 1.0)
        d["weight"] = 1.0 / max(-np.log(w), 1e-3)
    return h


def _comms(g, names):
    c = nx.community.greedy_modularity_communities(g, weight="weight")
    lab = {n: k for k, s in enumerate(c) for n in s}
    return np.array([lab.get(n, -1) for n in names], dtype=int)


def _mds(distance, ndim):
    d2 = np.asarray(distance, dtype=float) ** 2
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    w, v = np.linalg.eigh(b)
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    return v[:, :ndim] * np.sqrt(np.clip(w[:ndim], 0, None))


def twelve_schemes(counts, ns):
    """Twelve multi-way partitions of one panel, indexed by concern name."""
    rca = ns["get_rca"](counts)
    phi = ns["compute_product_space"](rca)
    names, distance = ns["_build_distance_matrix"](phi)
    if not np.all(np.isfinite(distance)):
        return None
    coords = _mds(distance, 5)
    q, _m = quantities(rca, phi, names)
    g = _phi_graph(phi, names)
    bb = _backbone(g)
    km = lambda x, k: KMeans(k, n_init=10, random_state=0).fit_predict(x)
    ward = lambda k: AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average").fit_predict(distance)

    # weighted k-means on the 1D ordering, as the mode partition is built
    sig = np.clip(rca.reindex(index=names).to_numpy(dtype=float)
                  - ns["RCA_THRESHOLD"], 0.0, None).sum(axis=1)
    order = np.argsort(coords[:, 0])
    x = coords[order, 0]
    x = (x - x.min()) / (np.ptp(x) or 1)
    peak, _ = ns["_select_region_centers"](x, sig[order])
    bounds = 0.5 * (x[peak][:-1] + x[peak][1:]) if len(peak) > 1 else np.array([])
    zo = np.digitize(x, bounds).astype(int)
    mds_intervals = np.empty(len(names), dtype=int)
    mds_intervals[order] = zo

    return {
        "names": names,
        "mds_intervals_weighted_k3": mds_intervals,
        "mds1d_kmeans_k3": km(coords[:, :1], 3),
        "mds3d_kmeans_k4": km(coords[:, :3], 4),
        "mds5d_kmeans_k4": km(coords[:, :5], 4),
        "quadrants_reach_x_breadth_k4": quadrants(q["reach"], q["breadth"]),
        "modularity_full": _comms(g, names),
        "modularity_backbone": _comms(bb, names),
        "modularity_full_distweighted": _comms(_dist_weighted(g), names),
        "modularity_backbone_distweighted": _comms(_dist_weighted(bb), names),
        "ward_average_k4": ward(4),
        "ward_average_k7": ward(7),
        "spectral_k4": SpectralClustering(
            4, affinity="precomputed", random_state=0,
            assign_labels="kmeans").fit_predict(
                phi.reindex(index=names, columns=names).to_numpy(dtype=float)),
    }


# ------------------------------------------------------- 3. chance floor ---
def chance_floor(labels, rng, reps=5000):
    sizes = np.bincount(labels, minlength=labels.max() + 1)
    base = np.repeat(np.arange(len(sizes)), sizes)
    out = []
    for _ in range(reps):
        a, b = base.copy(), base.copy()
        rng.shuffle(a)
        rng.shuffle(b)
        out.append(adjusted_rand_score(a, b))
    a = np.asarray(out)
    return dict(mean=float(a.mean()), lo=float(np.quantile(a, .025)),
                hi=float(np.quantile(a, .975)))


# ---------------------------------------------------- 4. axis diagnostics ---
def dcor(x, y):
    """Distance correlation: zero only under independence."""
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[:, None]
    a, b = np.abs(x - x.T), np.abs(y - y.T)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov = np.sqrt(max((A * B).mean(), 0))
    vx, vy = np.sqrt(max((A * A).mean(), 0)), np.sqrt(max((B * B).mean(), 0))
    return float(dcov / np.sqrt(vx * vy)) if vx * vy > 0 else 0.0


# -------------------------------------------------------------------- main ---
def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_rand = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    ns = load_pipeline()
    counts = ns["_prepare_counts"]()
    rng = np.random.default_rng(SEED)

    obs = twelve_schemes(counts, ns)
    names = obs["names"]
    keys = [k for k in obs if k != "names"]
    rca = ns["get_rca"](counts)
    phi = ns["compute_product_space"](rca)
    q, m = quantities(rca, phi, names)
    n_actors = counts.shape[1]
    print(f"concerns {len(names)}  actors {n_actors}  "
          f"schemes {len(keys)}  bootstrap {n_boot}  randomisations {n_rand}")

    print("\n[1/4] degree-preserving null ...")
    dn = degree_null(m, rng, n_rand)
    print(f"      {dn['pct_within_2sd']}% of {dn['n_pairs']} pairs within +-2 SD "
          f"({dn['n_above']} above, {dn['n_below']} below)")

    print(f"\n[2/4] bootstrap of {len(keys)} multi-way schemes ...")
    aris = {k: [] for k in keys}
    used = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_actors, size=n_actors)
        rs = counts.iloc[:, pick]
        rs.columns = [f"{c}__{i}" for i, c in enumerate(rs.columns)]
        try:
            b = twelve_schemes(rs, ns)
        except Exception:
            b = None
        if b is None:
            continue
        idx = {t: i for i, t in enumerate(b["names"])}
        if not set(names) <= set(idx):
            continue
        sel = [idx[t] for t in names]
        for k in keys:
            aris[k].append(adjusted_rand_score(obs[k], b[k][sel]))
        used += 1
    table = {k: dict(k_groups=int(len(set(obs[k]))),
                     ari_mean=float(np.mean(aris[k])),
                     ari_lo=float(np.quantile(aris[k], .05)),
                     ari_hi=float(np.quantile(aris[k], .95))) for k in keys}
    means = np.array([v["ari_mean"] for v in table.values()])
    print(f"      {used} usable replicates; ARI range "
          f"{means.min():.3f} to {means.max():.3f}")
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]["ari_mean"]):
        print(f"        {k:36s} k={v['k_groups']:2d}  ARI {v['ari_mean']:.3f}")

    print("\n[3/4] chance floor for the four classes ...")
    floor = chance_floor(obs["quadrants_reach_x_breadth_k4"], rng)
    print(f"      {floor['mean']:+.3f}  95% [{floor['lo']:+.3f}, {floor['hi']:+.3f}]")

    print("\n[4/4] axis diagnostics ...")
    rb = []
    for _ in range(300):
        pick = rng.integers(0, n_actors, size=n_actors)
        rs = counts.iloc[:, pick]
        rs.columns = [f"{c}__{i}" for i, c in enumerate(rs.columns)]
        try:
            r2 = ns["get_rca"](rs)
            p2 = ns["compute_product_space"](r2)
            nm2, d2 = ns["_build_distance_matrix"](p2)
            if not np.all(np.isfinite(d2)):
                continue
            q2, _ = quantities(r2, p2, nm2)
        except Exception:
            continue
        rb.append(spearmanr(q2["reach"], q2["breadth"]).statistic)
    rb = np.asarray(rb)
    axes = dict(
        rho_reach_breadth=float(spearmanr(q["reach"], q["breadth"]).statistic),
        rho_reach_breadth_ci=[float(np.quantile(rb, .025)), float(np.quantile(rb, .975))],
        dcor_reach_breadth=dcor(q["reach"], q["breadth"]),
        rho_breadth_ubiquity=float(spearmanr(q["breadth"], q["ubiquity"]).statistic),
        rho_reach_strength=float(spearmanr(q["reach"], q["strength"]).statistic),
    )
    for k, v in axes.items():
        print(f"      {k:28s} {v}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(
        seed=SEED, n_concerns=len(names), n_actors=int(n_actors),
        degree_null=dn, n_schemes=len(keys), n_boot_used=used,
        schemes=table,
        ari_range=[float(means.min()), float(means.max())],
        ari_quadrants=float(table["quadrants_reach_x_breadth_k4"]["ari_mean"]),
        chance_floor=floor, axes=axes,
    ), indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
