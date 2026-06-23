"""Supplementary Figure S16. Profiles the institutional attributes of complementary Regime 1–3 pairings. Links the pairing result to ATS hierarchy, gateways, and long-tenure anchors rather than generic blocs."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy import sparse
from sknetwork.clustering import Leiden

from utils import _split_multi_value, load_data

ROOT = Path(__file__).resolve().parent
DATA_FP = ROOT / "antarctic-database-go/data/processed/document-summary.parquet"
PAIR_FP = ROOT / "output/fig31_regime_pair_coverage_anchored_pairs.csv"
MEMBERSHIP_FP = ROOT / "membership.csv"
OUT_FIG_PDF = ROOT / "figures/figS16_regime_pair_profiles.pdf"
OUT_FIG_PNG = ROOT / "figures/figS16_regime_pair_profiles.png"
OUT_SUMMARY = ROOT / "output/fig32_regime_pair_institutional_profiles_summary.csv"
OUT_COMMUNITY = ROOT / "output/fig32_cosponsorship_communities_nodes.csv"
OUT_ACTORS = ROOT / "output/fig32_regime_pair_institutional_profiles_actors.csv"
ACTOR_SUMMARY_FP = ROOT / "output/fig45_portfolio_space_ridgelines_actor_summary.csv"
N_PERM = 20000

CLAIMANTS = {
    "Argentina",
    "Australia",
    "Chile",
    "France",
    "New Zealand",
    "Norway",
    "United Kingdom",
}
RESERVING_NONCLAIMANTS = {"United States", "Russian Federation"}
NAME_MAP = {"Russia": "Russian Federation", "Turkey": "Türkiye"}


def build_country_communities() -> dict[str, int]:
    _, submitted, countries, _ = load_data(str(DATA_FP))
    countries = sorted(countries)
    c_map = {c: i for i, c in enumerate(countries)}
    A = np.zeros((len(countries), len(countries)), dtype=float)
    for _, row in submitted.dropna(subset=["parties"]).iterrows():
        cleaned = sorted(set(str(p).strip() for p in row["parties"] if str(p).strip()))
        for i, j in combinations(cleaned, 2):
            if i in c_map and j in c_map:
                A[c_map[i], c_map[j]] += 1
    A += A.T
    A_country = pd.DataFrame(A, index=countries, columns=countries)
    g = nx.from_pandas_adjacency(A_country)
    gc = max(nx.connected_components(g), key=len)
    g = g.subgraph(gc)
    A_country = nx.to_pandas_adjacency(g)
    X = A_country.values
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    S = (X / norm) @ (X / norm).T
    np.fill_diagonal(S, 0)
    fits = Leiden(random_state=0).fit_predict(sparse.csr_matrix(S))
    mapping = {node: int(fits[idx]) for idx, node in enumerate(A_country.index)}
    pd.DataFrame(
        {"actor": list(mapping.keys()), "community": list(mapping.values())}
    ).sort_values("actor").to_csv(OUT_COMMUNITY, index=False)
    return mapping


def load_membership() -> pd.DataFrame:
    m = pd.read_csv(MEMBERSHIP_FP).copy()
    m["country"] = m["country"].replace(NAME_MAP)
    return m.set_index("country")


def annotate_pairs(
    pairs: pd.DataFrame, membership: pd.DataFrame, communities: dict[str, int]
) -> pd.DataFrame:
    def status_sum(a: str, b: str, col: str) -> int:
        return int(membership.loc[a, col]) + int(membership.loc[b, col])

    rows = []
    for _, r in pairs.iterrows():
        a, b = r["actor_a"], r["actor_b"]
        cp_sum = status_sum(a, b, "consultative_party")
        gateway_sum = status_sum(a, b, "gateway_state")
        rows.append(
            {
                "actor_a": a,
                "actor_b": b,
                "regime_pair": r["regime_pair"],
                "mixed_cp_ncp": int(cp_sum == 1),
                "both_cp": int(cp_sum == 2),
                "both_ncp": int(cp_sum == 0),
                "one_gateway": int(gateway_sum == 1),
                "both_gateway": int(gateway_sum == 2),
                "same_community": int(communities.get(a, -1) == communities.get(b, -2)),
                "either_claimant": int((a in CLAIMANTS) or (b in CLAIMANTS)),
                "both_claimant": int((a in CLAIMANTS) and (b in CLAIMANTS)),
                "either_reserving": int(
                    (a in RESERVING_NONCLAIMANTS) or (b in RESERVING_NONCLAIMANTS)
                ),
            }
        )
    return pd.DataFrame(rows)


def actor_tenure_for_13_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    actor_summary = pd.read_csv(ACTOR_SUMMARY_FP)[
        ["actor", "dominant_region", "dominant_region_share"]
    ].drop_duplicates()
    _, submitted, _, _ = load_data(str(DATA_FP))
    year_col = "year" if "year" in submitted.columns else "meeting year"
    actor_rows = []
    source_col = "parties" if "parties" in submitted.columns else "submitted by"
    for _, row in submitted[[source_col, year_col]].dropna().iterrows():
        for actor in _split_multi_value(row[source_col], delimiter=","):
            actor_rows.append((actor, int(row[year_col])))
    actor_years = (
        pd.DataFrame(actor_rows, columns=["actor", "year"])
        .drop_duplicates()
        .groupby("actor")["year"]
        .agg(first_year="min", last_year="max", active_years="nunique")
        .reset_index()
    )
    pair_13 = pairs[pairs["regime_pair"].eq("1-3")]
    actors = sorted(set(pair_13["actor_a"]).union(pair_13["actor_b"]))
    out = actor_summary.merge(actor_years, on="actor", how="left")
    out = out[out["actor"].isin(actors)].copy()
    out["is_claimant"] = out["actor"].isin(CLAIMANTS).astype(int)
    out["is_reserving_nonclaimant"] = (
        out["actor"].isin(RESERVING_NONCLAIMANTS).astype(int)
    )
    out = out.sort_values(["dominant_region", "first_year", "actor"]).reset_index(
        drop=True
    )
    out.to_csv(OUT_ACTORS, index=False)
    return out


def permutation_difference(
    df: pd.DataFrame,
    attr: str,
    select_mask: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> tuple[float, float]:
    obs = df.loc[select_mask, attr].mean() - df.loc[~select_mask, attr].mean()
    idx = np.arange(len(df))
    k = int(select_mask.sum())
    perm = np.empty(n_perm)
    for i in range(n_perm):
        chosen = rng.choice(idx, size=k, replace=False)
        mask = np.zeros(len(df), dtype=bool)
        mask[chosen] = True
        perm[i] = df.loc[mask, attr].mean() - df.loc[~mask, attr].mean()
    p = ((np.abs(perm) >= abs(obs)).sum() + 1) / (n_perm + 1)
    return obs, p


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    pairs = pd.read_csv(PAIR_FP)
    membership = load_membership()
    communities = build_country_communities()
    annotated = annotate_pairs(pairs, membership, communities)
    actor_panel = actor_tenure_for_13_pairs(pairs)
    select_mask = annotated["regime_pair"].eq("1-3").to_numpy()
    rng = np.random.default_rng(0)

    attrs = [
        ("mixed_cp_ncp", "Mixed CP/NCP"),
        ("both_cp", "Both CP"),
        ("one_gateway", "One gateway state"),
        ("same_community", "Same Leiden community"),
        ("either_claimant", "Includes claimant"),
        ("either_reserving", "Includes reserving non-claimant"),
    ]

    records = []
    for attr, label in attrs:
        obs_diff, p = permutation_difference(annotated, attr, select_mask, rng)
        records.append(
            {
                "attribute": attr,
                "label": label,
                "share_1_3": annotated.loc[select_mask, attr].mean(),
                "share_other": annotated.loc[~select_mask, attr].mean(),
                "difference": obs_diff,
                "p_value": p,
                "stars": stars(p),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(OUT_SUMMARY, index=False)

    order = list(range(len(summary)))[::-1]
    y = np.arange(len(summary))

    uplt.rc["grid"] = True
    fig, axs = uplt.subplots(
        ncols=2,
        width_ratios=(1.6, 1.0),
        figsize=(11.0, 4.8),
        share=0,
    )
    ax = axs[0]

    # connector lines
    for yi, idx in enumerate(order):
        row = summary.iloc[idx]
        ax.plot(
            [row["share_other"], row["share_1_3"]],
            [yi, yi],
            color="0.75",
            lw=1.5,
            zorder=1,
        )

    ax.scatter(
        summary.iloc[order]["share_other"],
        y,
        s=46,
        color="0.55",
        label="Other anchored pairs",
        zorder=3,
    )
    ax.scatter(
        summary.iloc[order]["share_1_3"],
        y,
        s=54,
        color="#d62728",
        label="Anchored 1-3 pairs",
        zorder=4,
    )

    for yi, idx in enumerate(order):
        row = summary.iloc[idx]
        mark = row["stars"]
        if mark:
            x = max(row["share_1_3"], row["share_other"]) + 0.03
            ax.text(x, yi, mark, ha="left", va="center", fontsize=10)

    ax.format(
        xlabel="Share of pairs with attribute",
        ylocator=y,
        yticklabels=summary.iloc[order]["label"].tolist(),
        xlim=(0, 1.02),
        xformatter="%.1f",
        title="Institutional profile of anchored 1-3 pairs",
    )
    ax.legend(loc="cr", ncols=1, frame=False)
    ax.text(
        0.0,
        1.05,
        "* p < 0.05, ** p < 0.01, *** p < 0.001\nPermutation test against random anchored pair subsets of equal size",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    ax2 = axs[1]
    actor_panel = actor_panel.sort_values(
        ["dominant_region", "first_year", "actor"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    yy = np.arange(len(actor_panel))
    colors = (
        actor_panel["dominant_region"]
        .map({1: "#1f77b4", 3: "#d62728"})
        .fillna("0.45")
        .to_numpy()
    )

    for yi, row in zip(yy, actor_panel.itertuples(index=False)):
        ax2.plot(
            [row.first_year, row.last_year],
            [yi, yi],
            color=colors[yi],
            lw=2.0,
            alpha=0.85,
            zorder=2,
        )
        ax2.scatter(row.first_year, yi, s=28, color=colors[yi], zorder=3)
        ax2.scatter(row.last_year, yi, s=28, color=colors[yi], zorder=3)
        marker = (
            "*" if row.is_claimant else ("D" if row.is_reserving_nonclaimant else "o")
        )
        ax2.scatter(
            row.last_year + 0.35,
            yi,
            s=54 if marker == "*" else 34,
            marker=marker,
            facecolor="white",
            edgecolor=colors[yi],
            linewidth=1.1,
            zorder=4,
        )

    ax2.format(
        xlabel="Archive years",
        ylocator=yy,
        yticklabels=actor_panel["actor"].tolist(),
        xlim=(1960, 2026),
        title="Actors appearing in 1-3 pairs",
    )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="Mode 1 anchor"),
        Line2D([0], [0], color="#d62728", lw=2, label="Mode 3 anchor"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="white",
            markeredgecolor="0.2",
            markersize=8,
            lw=0,
            label="Claimant",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="0.2",
            markersize=6,
            lw=0,
            label="Reserving non-claimant",
        ),
    ]
    ax2.legend(handles=handles, loc="ll", frame=False, ncols=1)

    fig.save(OUT_FIG_PDF)
    fig.save(OUT_FIG_PNG, transparent=False)


if __name__ == "__main__":
    main()
