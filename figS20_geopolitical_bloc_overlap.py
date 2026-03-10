"""Supplementary Figure S20. Compares overlap within EU, NATO, and BRICS pairs against permutations. Shows that broad geopolitical blocs explain little of the ATS regime structure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import get_rca, load_data, standardize_index_labels

DATA_FP = Path("./antarctic-database-go/data/processed/document-summary.parquet")
GEO_PAIRS_FP = Path("output/step6_geo_portfolio_pairs.csv")

OUT_DATA = Path("output/fig26_geopolitical_bloc_overlap_pairs.csv")
OUT_SUMMARY = Path("output/fig26_geopolitical_bloc_overlap_summary.csv")
OUT_META = Path("output/fig26_geopolitical_bloc_overlap_meta.json")
OUT_PDF = Path("figures/figS20_geopolitical_bloc_overlap.pdf")
OUT_PNG = Path("figures/figS20_geopolitical_bloc_overlap.png")
OUT_PERM = Path("output/fig26_geopolitical_bloc_overlap_permutations.csv")

RPA_THRESHOLD = 1.0
N_PERMUTATIONS = int(os.environ.get("FIG26_N_PERMUTATIONS", "5000"))
RNG_SEED = int(os.environ.get("FIG26_RNG_SEED", "42"))

EU = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
}

NATO = {
    "Albania",
    "Belgium",
    "Bulgaria",
    "Canada",
    "Croatia",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Türkiye",
    "United Kingdom",
    "United States",
}

BRICS = {
    "Brazil",
    "Russia",
    "India",
    "China",
    "South Africa",
    "Egypt",
    "Ethiopia",
    "Indonesia",
    "Iran",
    "Saudi Arabia",
    "United Arab Emirates",
}

NAME_MAP = {
    "Russian Federation": "Russia",
}


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else np.nan


def mean_gap(df: pd.DataFrame, mask: np.ndarray, metric: str) -> tuple[float, int, int]:
    values = df[metric].to_numpy(dtype=float)
    same_vals = values[mask]
    other_vals = values[~mask]
    same_mean = float(np.nanmean(same_vals)) if same_vals.size else np.nan
    other_mean = float(np.nanmean(other_vals)) if other_vals.size else np.nan
    gap = float(same_mean - other_mean) if np.isfinite(same_mean) and np.isfinite(other_mean) else np.nan
    return gap, int(mask.sum()), int((~mask).sum())


def main() -> None:
    counts, _, _, _ = load_data(DATA_FP)
    counts = standardize_index_labels(counts)
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0).sum()

    geo_pairs = pd.read_csv(GEO_PAIRS_FP)
    states = sorted(set(geo_pairs["actor_i"]).union(set(geo_pairs["actor_j"])))
    presence = counts.gt(0).astype(int).T.reindex(states).fillna(0).astype(int)
    rpa = get_rca(counts).gt(RPA_THRESHOLD).astype(int).T.reindex(states).fillna(0).astype(int)

    rows: list[dict[str, object]] = []
    for i, a in enumerate(states):
        aa = NAME_MAP.get(a, a)
        pa = presence.loc[a].to_numpy(dtype=bool)
        ra = rpa.loc[a].to_numpy(dtype=bool)
        for j in range(i + 1, len(states)):
            b = states[j]
            bb = NAME_MAP.get(b, b)
            pb = presence.loc[b].to_numpy(dtype=bool)
            rb = rpa.loc[b].to_numpy(dtype=bool)
            rows.append(
                {
                    "actor_i": a,
                    "actor_j": b,
                    "presence_jaccard": jaccard(pa, pb),
                    "rpa_jaccard": jaccard(ra, rb),
                    "both_eu": aa in EU and bb in EU,
                    "both_nato": aa in NATO and bb in NATO,
                    "both_brics": aa in BRICS and bb in BRICS,
                }
            )

    pair_df = pd.DataFrame(rows)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(OUT_DATA, index=False)

    actors = np.array(states, dtype=object)
    actor_index = {actor: idx for idx, actor in enumerate(actors)}
    actor_i_idx = pair_df["actor_i"].map(actor_index).to_numpy(dtype=int)
    actor_j_idx = pair_df["actor_j"].map(actor_index).to_numpy(dtype=int)

    summary_rows: list[dict[str, object]] = []
    perm_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RNG_SEED)
    for bloc_key, bloc_label in [
        ("both_eu", "EU"),
        ("both_nato", "NATO"),
        ("both_brics", "BRICS"),
    ]:
        member_mask = np.zeros(len(actors), dtype=bool)
        if bloc_label == "EU":
            member_names = {a for a in actors if NAME_MAP.get(a, a) in EU}
        elif bloc_label == "NATO":
            member_names = {a for a in actors if NAME_MAP.get(a, a) in NATO}
        else:
            member_names = {a for a in actors if NAME_MAP.get(a, a) in BRICS}
        for name in member_names:
            member_mask[actor_index[name]] = True
        same_pair_mask = member_mask[actor_i_idx] & member_mask[actor_j_idx]

        for metric in ["rpa_jaccard", "presence_jaccard"]:
            same_mean = float(np.nanmean(pair_df.loc[same_pair_mask, metric])) if same_pair_mask.any() else np.nan
            other_mean = float(np.nanmean(pair_df.loc[~same_pair_mask, metric])) if (~same_pair_mask).any() else np.nan
            same_count = int(same_pair_mask.sum())
            other_count = int((~same_pair_mask).sum())
            observed_gap = float(same_mean - other_mean) if np.isfinite(same_mean) and np.isfinite(other_mean) else np.nan

            perm_gaps = np.empty(N_PERMUTATIONS, dtype=float)
            n_members = int(member_mask.sum())
            for p in range(N_PERMUTATIONS):
                sampled = rng.choice(len(actors), size=n_members, replace=False)
                perm_mask = np.zeros(len(actors), dtype=bool)
                perm_mask[sampled] = True
                pair_mask = perm_mask[actor_i_idx] & perm_mask[actor_j_idx]
                perm_gaps[p], _, _ = mean_gap(pair_df, pair_mask, metric)

            abs_obs = abs(observed_gap)
            abs_perm = np.abs(perm_gaps)
            p_two_sided = float((np.count_nonzero(abs_perm >= abs_obs) + 1) / (N_PERMUTATIONS + 1))
            p_one_sided_high = float((np.count_nonzero(perm_gaps >= observed_gap) + 1) / (N_PERMUTATIONS + 1))
            p_one_sided_low = float((np.count_nonzero(perm_gaps <= observed_gap) + 1) / (N_PERMUTATIONS + 1))
            perm_rows.extend(
                {
                    "bloc": bloc_label,
                    "metric": metric,
                    "permutation": p,
                    "gap": float(g),
                }
                for p, g in enumerate(perm_gaps)
            )
            summary_rows.append(
                {
                    "bloc": bloc_label,
                    "metric": metric,
                    "same_group_mean": same_mean,
                    "other_pairs_mean": other_mean,
                    "same_group_minus_other": observed_gap,
                    "same_group_count": same_count,
                    "other_pairs_count": other_count,
                    "perm_mean": float(np.mean(perm_gaps)),
                    "perm_sd": float(np.std(perm_gaps, ddof=1)),
                    "perm_q05": float(np.quantile(perm_gaps, 0.05)),
                    "perm_q95": float(np.quantile(perm_gaps, 0.95)),
                    "p_two_sided": p_two_sided,
                    "p_one_sided_high": p_one_sided_high,
                    "p_one_sided_low": p_one_sided_low,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(perm_rows).to_csv(OUT_PERM, index=False)
    OUT_META.write_text(
        json.dumps(
            {
                "rpa_threshold": RPA_THRESHOLD,
                "n_permutations": N_PERMUTATIONS,
                "rng_seed": RNG_SEED,
                "note": (
                    "Current EU, NATO, and BRICS memberships were coded from current official membership lists. "
                    "Permutation inference preserves the number of bloc members among ATS sovereign states and "
                    "randomizes the bloc label across actors."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axs = uplt.subplots(ncols=2, refwidth=3.4, refaspect=1.0, share=False)
    axs.format(abc="[A]", grid=False)

    panel_specs = [
        ("rpa_jaccard", "RPA > 1", "blue7"),
        ("presence_jaccard", "Any activity", "orange7"),
    ]
    bloc_order = ["EU", "NATO", "BRICS"]
    x = np.arange(len(bloc_order))

    for ax, (metric, title, color) in zip(axs, panel_specs):
        sub = summary_df[summary_df["metric"] == metric].set_index("bloc").loc[bloc_order]
        ax.bar(
            x,
            sub["same_group_minus_other"].to_numpy(dtype=float),
            color=color,
            edgecolor="black",
            lw=0.6,
            alpha=0.9,
        )
        ax.axhline(0, color="black", lw=1.0, alpha=0.65)
        for xi, bloc in enumerate(bloc_order):
            n_same = int(sub.loc[bloc, "same_group_count"])
            pval = float(sub.loc[bloc, "p_two_sided"])
            ax.text(
                xi,
                sub.loc[bloc, "same_group_minus_other"] + (0.008 if sub.loc[bloc, "same_group_minus_other"] >= 0 else -0.012),
                f"n={n_same}\np={pval:.3f}",
                ha="center",
                va="bottom" if sub.loc[bloc, "same_group_minus_other"] >= 0 else "top",
                fontsize=7,
            )
        ax.format(
            title=title,
            xlabel="Bloc",
            ylabel="Same-group minus other overlap",
            xticks=x,
            xticklabels=bloc_order,
        )
        ax.grid(alpha=0.18, color="black")

    fig.format(suptitle="Do current geopolitical blocs align with ATS portfolio similarity?")
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
