#!/usr/bin/env python3
"""Movement of collective attention across the space of concerns, as optimal transport.

Pre-specified design (fixed before any output was inspected):

  distribution  p_t is the share of paper-concern assignments falling in
                non-overlapping 5-year block t.  Multilabel papers contribute
                to each of their concerns.

  cost          d_ij = 1 - phi_ij, with phi the proximity matrix of the space
                of concerns and d_ii = 0.  The naive cost is the phi = 0 limit
                of the same expression: every move between distinct concerns
                costs 1.  So the naive and map costs live on one scale and
                differ only by the discount proximity applies.

  statistic     W_1(p_{t-1}, p_t), solved exactly as a transportation LP, under
                each cost.  The locality ratio L_t = W_map / W_naive is the mean
                proximity-discounted distance per unit of mass that moved.
                L = 1 means attention moved as if the map did not exist;
                L -> 0 means it moved only between close concerns.

  null          the concern labels on the map are permuted while the observed
                distributions are held fixed.  This asks whether the movement
                is local with respect to *this* map rather than any map with
                the same degree structure.

  variants      the raw 45 Secretariat labels, and an aggregated map in which
                Management Plans and Area Protection and Management Plans
                General are summed into one concern before phi is rebuilt,
                because they name one administrative object.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import compute_product_space, get_rca, load_data

PAPERS = ROOT / "data/document-summary-multilabel.parquet"
DEFAULT_OUT = ROOT / "output/concern_transport"
MERGED = "Area protection and management plans"
SOURCES = ["Management Plans", "Area Protection and Management Plans General"]
BLOCK = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


# ---------------------------------------------------------------- data

def block_of(year: np.ndarray) -> np.ndarray:
    return ((year - 1961) // BLOCK) * BLOCK + 1961


def assignments(submitted: pd.DataFrame) -> pd.DataFrame:
    """Long table of one row per (paper, concern, year)."""
    from utils import _clean_category_cell, _split_multi_value

    df = submitted.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    rows = []
    for year, cell in zip(df["year"], df["category"]):
        clean = _clean_category_cell(cell)
        if clean is pd.NA:
            continue
        for concern in _split_multi_value(clean, delimiter="\t"):
            rows.append((year, concern))
    out = pd.DataFrame(rows, columns=["year", "concern"])
    out["block"] = block_of(out["year"].to_numpy())
    return out


def distributions(long: pd.DataFrame, concerns: list[str]) -> pd.DataFrame:
    """Blocks by concerns, each row a probability distribution."""
    table = (
        long.groupby(["block", "concern"]).size().unstack(fill_value=0)
        .reindex(columns=concerns, fill_value=0)
        .astype(float)
    )
    return table.div(table.sum(axis=1), axis=0)


def spaces() -> dict[str, pd.DataFrame]:
    """Proximity matrices for the raw and aggregated concern maps."""
    counts, _, _, _ = load_data(str(PAPERS))
    raw = compute_product_space(get_rca(counts))
    present = [c for c in SOURCES if c in counts.index]
    if len(present) != 2:
        raise SystemExit(f"expected both area labels, found {present}")
    merged_counts = counts.drop(index=present)
    merged_counts.loc[MERGED] = counts.loc[present].sum(axis=0)
    merged = compute_product_space(get_rca(merged_counts))
    return {"raw": raw, "aggregated": merged}


# ---------------------------------------------------------------- transport

def _transport_constraints(n: int):
    """Equality constraints for a balanced n x n transportation problem."""
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            rows.append(i)
            cols.append(i * n + j)
    for j in range(n):
        for i in range(n):
            rows.append(n + j)
            cols.append(i * n + j)
    data = np.ones(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(2 * n, n * n)).tocsr()


def transport(p: np.ndarray, q: np.ndarray, cost: np.ndarray, A) -> tuple[float, np.ndarray]:
    """Exact W_1 and the optimal plan."""
    n = len(p)
    b = np.concatenate([p, q])
    res = linprog(cost.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun), res.x.reshape(n, n)


def costs_from(phi: np.ndarray) -> np.ndarray:
    d = 1.0 - phi
    np.fill_diagonal(d, 0.0)
    return d


def naive_cost(n: int) -> np.ndarray:
    d = np.ones((n, n))
    np.fill_diagonal(d, 0.0)
    return d


# ---------------------------------------------------------------- driver

def run_variant(name: str, phi: pd.DataFrame, long: pd.DataFrame,
                permutations: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    concerns = list(phi.index)
    dist = distributions(long, concerns)
    blocks = list(dist.index)
    n = len(concerns)
    A = _transport_constraints(n)

    d_map = costs_from(phi.to_numpy(float))
    d_naive = naive_cost(n)

    records = []
    plans = {}
    for prev, cur in zip(blocks[:-1], blocks[1:]):
        p = dist.loc[prev].to_numpy()
        q = dist.loc[cur].to_numpy()
        w_naive, plan = transport(p, q, d_naive, A)
        w_map, plan_map = transport(p, q, d_map, A)

        null = np.empty(permutations)
        for k in range(permutations):
            order = rng.permutation(n)
            null[k] = transport(p, q, d_map[np.ix_(order, order)], A)[0]

        records.append({
            "variant": name,
            "from_block": prev,
            "to_block": cur,
            "mass_moved": w_naive,
            "w_map": w_map,
            "locality": w_map / w_naive if w_naive > 0 else np.nan,
            "null_mean": float(null.mean()) / w_naive if w_naive > 0 else np.nan,
            "null_lo": float(np.quantile(null, 0.025)) / w_naive if w_naive > 0 else np.nan,
            "null_hi": float(np.quantile(null, 0.975)) / w_naive if w_naive > 0 else np.nan,
            "p_value": float((null <= w_map).mean()),
        })
        plans[(prev, cur)] = plan_map
    return pd.DataFrame(records), plans


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, submitted, _, _ = load_data(str(PAPERS))
    long = assignments(submitted)
    long_merged = long.copy()
    long_merged["concern"] = long_merged["concern"].replace({s: MERGED for s in SOURCES})

    rng = np.random.default_rng(args.seed)
    frames = []
    for name, phi in spaces().items():
        source = long if name == "raw" else long_merged
        frame, plans = run_variant(name, phi, source, args.permutations, rng)
        frames.append(frame)
        if name == "aggregated":
            np.save(args.out_dir / "plans_aggregated.npy",
                    np.array([plans[k] for k in sorted(plans)]))
            pd.Series(list(phi.index)).to_csv(
                args.out_dir / "concerns_aggregated.csv", index=False, header=["concern"])

    table = pd.concat(frames, ignore_index=True)
    table.to_csv(args.out_dir / "transport_by_block.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    print()
    for name, group in table.groupby("variant"):
        print(f"{name:>11}  mean locality {group['locality'].mean():.3f}   "
              f"mean null {group['null_mean'].mean():.3f}   "
              f"transitions below null 2.5% "
              f"{int((group['locality'] < group['null_lo']).sum())}/{len(group)}")


if __name__ == "__main__":
    main()
