"""Supplementary Figure S07. Re-estimates local adoption under alternative concern-space constructions. Checks that the locality result does not depend on one specific space definition."""

# %%
#!/usr/bin/env python3
from __future__ import annotations

# Hazard sensitivity to concern-space construction.
#
# Compares the topic-adoption hazard distance effect under three distance
# spaces:
# 1) aggregate full-history space;
# 2) instantaneous previous-window space;
# 3) cumulative-lagged space (all data up to t-1).

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

# TODO(meeting-sequence): This sensitivity analysis still rolls over calendar years.
# If the main hazard analysis standardizes on sequential meetings, convert this
# window definition and period builder to meeting-number windows as well.
WINDOW_YEARS = 5
RCA_THRESHOLD = 1.0

DATA_PATHS = [
    Path("antarctic-database-go/data/processed/document-summary.parquet"),
    Path("antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"),
    Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
    Path("document-summary.csv"),
]

MODE_ORDER = ["aggregate", "instantaneous", "cumulative_lagged"]
MODE_LABELS = {
    "aggregate": "Aggregate (full history)",
    "instantaneous": "Instantaneous (window t-1)",
    "cumulative_lagged": "Cumulative-lagged (<= t-1)",
}
MODE_COLORS = {
    "aggregate": "#1f77b4",
    "instantaneous": "#d62728",
    "cumulative_lagged": "#2ca02c",
}

OUT_PDF = Path("figures/figS07_hazard_space_sensitivity.pdf")
OUT_PNG = Path("figures/figS07_hazard_space_sensitivity.png")
OUT_MEMBER_PERIOD = Path("output/fig15_hazard_space_member_period_coefficients.csv")
OUT_PERIOD_SUMMARY = Path("output/fig15_hazard_space_period_summary.csv")
OUT_POOLED_COEF = Path("output/fig15_hazard_space_pooled_coefficients.csv")
OUT_META = Path("output/fig15_hazard_space_meta.json")


def load_data_with_fallback():
    last_err = None
    for path in DATA_PATHS:
        if not path.exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_err = exc
    if last_err is not None:
        raise RuntimeError("Failed to load ATS data from fallback paths.") from last_err
    raise FileNotFoundError("No known ATS data path exists.")


def sanitize_years(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=[year_col]).copy()
    out[year_col] = out[year_col].astype(int)
    return out


def build_periods(year_min: int, year_max: int, window: int) -> list[tuple[int, int]]:
    return [(y - window + 1, y) for y in range(year_min + window - 1, year_max + 1)]


def build_window_interaction(
    submitted_df: pd.DataFrame,
    year_col: str,
    year_start: int,
    year_end: int,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    members_order: list[str],
) -> pd.DataFrame:
    window_df = submitted_df[
        (submitted_df[year_col] >= int(year_start))
        & (submitted_df[year_col] <= int(year_end))
    ]
    interaction = generate_interaction_matrix(
        window_df, all_members_raw, all_topics_raw
    )
    interaction = standardize_index_labels(interaction)
    if interaction.index.has_duplicates:
        interaction = interaction.groupby(level=0).sum()
    return interaction.reindex(index=topics_order, columns=members_order, fill_value=0)


def distance_from_interaction(
    interaction: pd.DataFrame, topics_order: list[str]
) -> np.ndarray:
    rca = get_rca(interaction)
    phi = compute_product_space(rca).reindex(
        index=topics_order, columns=topics_order, fill_value=0.0
    )
    dist = 1.0 - phi.to_numpy(dtype=float)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def fit_member_period_logit(panel_df: pd.DataFrame) -> dict[str, float] | None:
    features = ["distance", "prev_diversity", "topic_popularity"]
    df = panel_df.dropna(subset=features + ["adopted"]).copy()
    if df.empty:
        return None

    n_adopted = int(df["adopted"].sum())
    if n_adopted == 0 or n_adopted == len(df):
        return None

    X = df[features].to_numpy()
    y = df["adopted"].to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=4000, solver="lbfgs")
    model.fit(X_std, y)

    coef = model.coef_[0].astype(float)
    probs = model.predict_proba(X_std)[:, 1].astype(float)
    eps = 1e-12
    ll_model = float(
        np.sum(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    )
    p0 = float(np.mean(y))
    ll_null = float(np.sum(y * np.log(p0 + eps) + (1 - y) * np.log(1 - p0 + eps)))
    pseudo_r2 = float(1.0 - (ll_model / ll_null)) if ll_null < 0 else np.nan

    return {
        "distance_coef": float(coef[0]),
        "prev_diversity_coef": float(coef[1]),
        "topic_popularity_coef": float(coef[2]),
        "intercept": float(model.intercept_[0]),
        "n_obs": int(len(df)),
        "n_adopted": int(n_adopted),
        "adoption_rate": float(np.mean(y)),
        "loglik_model": ll_model,
        "loglik_null": ll_null,
        "mcfadden_r2": pseudo_r2,
    }


def fit_pooled_logit(panel_df: pd.DataFrame) -> pd.DataFrame:
    features = ["distance", "prev_diversity", "topic_popularity"]
    df = panel_df.dropna(subset=features + ["adopted", "member"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "term",
                "coef",
                "std_err",
                "z_value",
                "p_value",
                "ci_low_95",
                "ci_high_95",
                "odds_ratio",
            ]
        )
    if int(df["adopted"].sum()) == 0 or int(df["adopted"].sum()) == len(df):
        return pd.DataFrame(
            columns=[
                "term",
                "coef",
                "std_err",
                "z_value",
                "p_value",
                "ci_low_95",
                "ci_high_95",
                "odds_ratio",
            ]
        )

    X = df[features].copy()
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    X = sm.add_constant(X, has_constant="add")
    y = df["adopted"].astype(float).to_numpy()
    groups = df["member"].astype(str).to_numpy()

    glm = sm.GLM(y, X, family=sm.families.Binomial())
    try:
        res = glm.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        res = glm.fit()

    ci = res.conf_int()
    out = pd.DataFrame(
        {
            "term": list(res.params.index),
            "coef": np.asarray(res.params.values, dtype=float),
            "std_err": np.asarray(res.bse, dtype=float),
            "z_value": np.asarray(res.tvalues, dtype=float),
            "p_value": np.asarray(res.pvalues, dtype=float),
            "ci_low_95": np.asarray(ci.iloc[:, 0], dtype=float),
            "ci_high_95": np.asarray(ci.iloc[:, 1], dtype=float),
        }
    )
    out["odds_ratio"] = np.exp(out["coef"])
    return out


def main() -> None:
    counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback()
    year_col = "meeting year" if "meeting year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns:
        raise KeyError("No meeting year or year column found in source data.")
    submitted_df = sanitize_years(submitted_df, year_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    all_members_raw = set(members_raw)
    all_topics_raw = set(topics_raw)

    year_min = int(submitted_df[year_col].min())
    year_max = int(submitted_df[year_col].max())
    periods = build_periods(year_min, year_max, WINDOW_YEARS)

    interaction_by_period: list[pd.DataFrame] = []
    active_by_period: list[pd.DataFrame] = []
    for start, end in periods:
        interaction = build_window_interaction(
            submitted_df=submitted_df,
            year_col=year_col,
            year_start=int(start),
            year_end=int(end),
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics,
            members_order=members,
        )
        interaction_by_period.append(interaction)
        active_by_period.append((get_rca(interaction) >= RCA_THRESHOLD))

    aggregate_interaction = counts_df.reindex(
        index=topics, columns=members, fill_value=0
    )
    aggregate_dist = distance_from_interaction(aggregate_interaction, topics)

    distance_by_mode_and_transition: dict[str, dict[int, np.ndarray]] = {
        "aggregate": {},
        "instantaneous": {},
        "cumulative_lagged": {},
    }
    for t in range(1, len(periods)):
        prev_end = int(periods[t - 1][1])

        distance_by_mode_and_transition["aggregate"][t] = aggregate_dist
        distance_by_mode_and_transition["instantaneous"][t] = distance_from_interaction(
            interaction_by_period[t - 1], topics
        )

        cumulative_interaction = build_window_interaction(
            submitted_df=submitted_df,
            year_col=year_col,
            year_start=year_min,
            year_end=prev_end,
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics,
            members_order=members,
        )
        distance_by_mode_and_transition["cumulative_lagged"][t] = (
            distance_from_interaction(cumulative_interaction, topics)
        )

    result_rows = []
    panel_rows = []
    for mode in MODE_ORDER:
        for t in range(1, len(periods)):
            period_end = int(periods[t][1])
            prev_active = active_by_period[t - 1]
            curr_active = active_by_period[t]
            dist = distance_by_mode_and_transition[mode][t]

            prev_topic_popularity = prev_active.sum(axis=1) / max(len(members), 1)

            for member in members:
                prev_mask = prev_active[member].to_numpy(dtype=bool)
                curr_mask = curr_active[member].to_numpy(dtype=bool)
                if not prev_mask.any():
                    continue

                at_risk = ~prev_mask
                if not at_risk.any():
                    continue
                adopted = curr_mask & at_risk

                prev_indices = np.where(prev_mask)[0]
                min_dist = dist[:, prev_indices].min(axis=1)
                prev_diversity = float(len(prev_indices))

                rows = []
                for idx, topic in enumerate(topics):
                    if not at_risk[idx]:
                        continue
                    rows.append(
                        {
                            "mode": mode,
                            "period_end": period_end,
                            "member": member,
                            "topic": topic,
                            "adopted": int(adopted[idx]),
                            "distance": float(min_dist[idx]),
                            "prev_diversity": prev_diversity,
                            "topic_popularity": float(prev_topic_popularity.loc[topic]),
                        }
                    )

                if not rows:
                    continue
                member_period_panel = pd.DataFrame(rows)
                panel_rows.append(member_period_panel)

                fit = fit_member_period_logit(member_period_panel)
                if fit is None:
                    continue
                result_rows.append(
                    {"mode": mode, "period_end": period_end, "member": member, **fit}
                )

    results_df = pd.DataFrame(result_rows)
    if results_df.empty:
        raise RuntimeError("No valid member-period hazard models were fitted.")
    panel_df = (
        pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    )

    period_summary = (
        results_df.groupby(["mode", "period_end"])
        .agg(
            n_members=("member", "nunique"),
            n_models=("distance_coef", "size"),
            median_distance_coef=("distance_coef", "median"),
            q25_distance_coef=("distance_coef", lambda s: float(s.quantile(0.25))),
            q75_distance_coef=("distance_coef", lambda s: float(s.quantile(0.75))),
            mean_mcfadden_r2=("mcfadden_r2", "mean"),
            mean_adoption_rate=("adoption_rate", "mean"),
        )
        .reset_index()
    )

    pooled_frames = []
    for mode in MODE_ORDER:
        mode_panel = panel_df[panel_df["mode"] == mode].copy()
        mode_pooled = fit_pooled_logit(mode_panel)
        mode_pooled["mode"] = mode
        pooled_frames.append(mode_pooled)
    pooled_df = (
        pd.concat(pooled_frames, ignore_index=True) if pooled_frames else pd.DataFrame()
    )

    OUT_MEMBER_PERIOD.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_MEMBER_PERIOD, index=False)
    period_summary.to_csv(OUT_PERIOD_SUMMARY, index=False)
    pooled_df.to_csv(OUT_POOLED_COEF, index=False)

    meta = {
        "window_years": WINDOW_YEARS,
        "rca_threshold": RCA_THRESHOLD,
        "n_members": int(len(members)),
        "n_topics": int(len(topics)),
        "n_member_period_models": int(len(results_df)),
        "n_panel_rows": int(len(panel_df)),
        "modes": MODE_ORDER,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    for mode in MODE_ORDER:
        dfi = period_summary[period_summary["mode"] == mode].sort_values("period_end")
        if dfi.empty:
            continue
        x = dfi["period_end"].to_numpy(dtype=int)
        med = dfi["median_distance_coef"].to_numpy(dtype=float)
        q25 = dfi["q25_distance_coef"].to_numpy(dtype=float)
        q75 = dfi["q75_distance_coef"].to_numpy(dtype=float)
        color = MODE_COLORS[mode]
        ax1.fill_between(x, q25, q75, color=color, alpha=0.15, linewidth=0)
        ax1.plot(x, med, color=color, lw=1.8, label=MODE_LABELS[mode])
    ax1.axhline(0.0, color="#666666", ls="--", lw=1.0)
    ax1.set_title("Distance Effect Over Time")
    ax1.set_xlabel("Window end year")
    ax1.set_ylabel("Median distance coefficient")
    ax1.grid(alpha=0.2, linewidth=0.6)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")

    pooled_distance = pooled_df[pooled_df["term"] == "distance"].copy()
    pooled_distance["mode"] = pd.Categorical(
        pooled_distance["mode"], categories=MODE_ORDER, ordered=True
    )
    pooled_distance = pooled_distance.sort_values("mode")
    y_pos = np.arange(len(pooled_distance))
    for i, (_, row) in enumerate(pooled_distance.iterrows()):
        mode = str(row["mode"])
        coef = float(row["coef"])
        ci_low = float(row["ci_low_95"])
        ci_high = float(row["ci_high_95"])
        color = MODE_COLORS.get(mode, "#333333")
        ax2.errorbar(
            x=coef,
            y=i,
            xerr=np.array([[coef - ci_low], [ci_high - coef]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3.0,
            markersize=5.5,
        )
    ax2.axvline(0.0, color="#666666", ls="--", lw=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [MODE_LABELS[m] for m in pooled_distance["mode"].astype(str).tolist()],
        fontsize=8,
    )
    ax2.set_xlabel("Pooled distance coefficient (95% CI)")
    ax2.set_title("Pooled Effect by Space Definition")
    ax2.grid(alpha=0.2, linewidth=0.6, axis="x")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Wrote: {OUT_PDF}")
    print(f"Wrote: {OUT_MEMBER_PERIOD}")
    print(f"Wrote: {OUT_PERIOD_SUMMARY}")
    print(f"Wrote: {OUT_POOLED_COEF}")


if __name__ == "__main__":
    main()
