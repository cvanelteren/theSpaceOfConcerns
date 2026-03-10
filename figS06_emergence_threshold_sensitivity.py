"""Supplementary Figure S06. Varies the emergence threshold used in the pioneering analysis. Checks that pioneer timing is not an artifact of one arbitrary cut-off."""

#!/usr/bin/env python3
from __future__ import annotations

# Sensitivity of topic emergence years to cumulative-volume thresholds.
#
# Recomputes topic emergence years over a grid of cumulative-volume cutoffs and
# writes a compact appendix figure plus tidy outputs.

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_data, standardize_index_labels


DATA_PATH = "./antarctic-database-go/data/processed/document-summary.parquet"
DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30"
DEFAULT_MIN_TOPIC_VOLUME = 10

OUT_PDF = Path("figures/figS06_emergence_threshold_sensitivity.pdf")
OUT_PNG = Path("figures/figS06_emergence_threshold_sensitivity.png")
OUT_TOPIC_YEARS = Path("output/fig14_emergence_threshold_topic_years.csv")
OUT_SUMMARY = Path("output/fig14_emergence_threshold_summary.csv")
OUT_META = Path("output/fig14_emergence_threshold_meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergence-threshold sensitivity figure.")
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help=f"Input ATS summary parquet/csv (default: {DATA_PATH}).",
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help=(
            "Comma-separated cumulative-volume thresholds in (0,1), e.g. "
            "'0.05,0.10,0.15,0.20,0.25,0.30'."
        ),
    )
    parser.add_argument(
        "--min-topic-volume",
        type=int,
        default=DEFAULT_MIN_TOPIC_VOLUME,
        help=f"Minimum topic volume to include (default: {DEFAULT_MIN_TOPIC_VOLUME}).",
    )
    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        val = float(token)
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"Threshold must be in (0,1): {val}")
        vals.append(val)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid thresholds provided.")
    return vals


def sanitize_year_column(submitted_df: pd.DataFrame) -> pd.DataFrame:
    df = submitted_df.copy()
    year_col = "meeting_year" if "meeting_year" in df.columns else "year"
    if year_col not in df.columns:
        raise KeyError("No meeting_year or year column found in source data.")
    df["meeting_year"] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=["meeting_year"]).copy()
    df["meeting_year"] = df["meeting_year"].astype(int)
    return df


def explode_topic_long(submitted_df: pd.DataFrame) -> pd.DataFrame:
    topic_col = "category" if "category" in submitted_df.columns else "topic"
    if topic_col not in submitted_df.columns:
        raise KeyError("No category/topic column found in source data.")

    out = submitted_df[["meeting_year", topic_col]].dropna(subset=[topic_col]).copy()
    out["topic"] = out[topic_col].astype(str).str.split("\t")
    out = out.explode("topic")
    out["topic"] = out["topic"].astype(str).str.strip()
    out = out[out["topic"] != ""]
    out["topic"] = out["topic"].str.replace("envirom", "environ", regex=False)
    out["topic"] = out["topic"].str.replace("_", " ", regex=False)
    out = out[~out["topic"].str.upper().isin(["ALL", "OTHER"])]

    proxy = pd.DataFrame(index=out["topic"].to_numpy())
    proxy = standardize_index_labels(proxy)
    out["topic_std"] = proxy.index.to_numpy()
    return out[["meeting_year", "topic_std"]]


def compute_topic_emergence(
    topic_long_df: pd.DataFrame,
    min_topic_volume: int,
    emergence_percentile: float,
) -> pd.DataFrame:
    yearly = topic_long_df.groupby(["meeting_year", "topic_std"]).size().reset_index(name="count")
    rows = []
    for topic, dfi in yearly.groupby("topic_std"):
        dfi = dfi.sort_values("meeting_year")
        total = int(dfi["count"].sum())
        if total < int(min_topic_volume):
            continue
        dfi["cum"] = dfi["count"].cumsum()
        dfi["pct"] = dfi["cum"] / total
        emergence_year = int(dfi.loc[dfi["pct"] >= emergence_percentile, "meeting_year"].iloc[0])
        rows.append(
            {
                "topic": topic,
                "topic_total_volume": total,
                "emergence_year": emergence_year,
                "emergence_percentile": float(emergence_percentile),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    _, submitted_df, _, _ = load_data(args.data_path)
    submitted_df = sanitize_year_column(submitted_df)
    topic_long = explode_topic_long(submitted_df)

    frames = []
    for thr in thresholds:
        frame = compute_topic_emergence(
            topic_long_df=topic_long,
            min_topic_volume=int(args.min_topic_volume),
            emergence_percentile=float(thr),
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No emergence-year records computed for the selected thresholds.")

    emergence_df = pd.concat(frames, ignore_index=True)
    emergence_df["threshold_pct"] = emergence_df["emergence_percentile"] * 100.0

    summary = (
        emergence_df.groupby("emergence_percentile")["emergence_year"]
        .agg(
            n_topics="size",
            median_year="median",
            q25_year=lambda s: float(s.quantile(0.25)),
            q75_year=lambda s: float(s.quantile(0.75)),
        )
        .reset_index()
        .sort_values("emergence_percentile")
    )
    summary["threshold_pct"] = summary["emergence_percentile"] * 100.0

    pivot = (
        emergence_df.pivot_table(
            index="topic",
            columns="emergence_percentile",
            values="emergence_year",
            aggfunc="first",
        )
        .reindex(columns=thresholds)
        .dropna()
    )
    if pivot.empty:
        raise RuntimeError("No complete topic panel across thresholds for plotting.")

    baseline = min(thresholds, key=lambda t: abs(t - 0.15))
    delta = pivot.subtract(pivot[baseline], axis=0)
    abs_shift = delta.abs()
    abs_shift_summary = pd.DataFrame(
        {
            "emergence_percentile": abs_shift.columns.to_numpy(dtype=float),
            "median_abs_shift_years": [
                float(abs_shift[col].median()) for col in abs_shift.columns
            ],
            "q75_abs_shift_years": [
                float(abs_shift[col].quantile(0.75)) for col in abs_shift.columns
            ],
        }
    )
    abs_shift_summary["threshold_pct"] = abs_shift_summary["emergence_percentile"] * 100.0
    summary = summary.merge(
        abs_shift_summary[["emergence_percentile", "median_abs_shift_years", "q75_abs_shift_years"]],
        on="emergence_percentile",
        how="left",
    )

    topic_order = pivot[baseline].sort_values().index
    pivot = pivot.loc[topic_order]
    delta = delta.loc[topic_order]

    x = np.array([t * 100.0 for t in thresholds], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)

    for _, row in pivot.iterrows():
        ax1.plot(x, row.to_numpy(dtype=float), color="#bdbdbd", lw=0.75, alpha=0.45, zorder=1)
    ax1.fill_between(
        summary["threshold_pct"].to_numpy(dtype=float),
        summary["q25_year"].to_numpy(dtype=float),
        summary["q75_year"].to_numpy(dtype=float),
        color="#4c78a8",
        alpha=0.22,
        linewidth=0,
        zorder=2,
        label="IQR across topics",
    )
    ax1.plot(
        summary["threshold_pct"].to_numpy(dtype=float),
        summary["median_year"].to_numpy(dtype=float),
        color="#1f4e79",
        lw=2.0,
        zorder=3,
        label="Median emergence year",
    )
    ax1.set_title("Topic Emergence Year vs Threshold")
    ax1.set_xlabel("Emergence threshold (% of cumulative topic volume)")
    ax1.set_ylabel("Emergence year")
    ax1.grid(alpha=0.2, linewidth=0.5)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    box_data = [delta[col].dropna().to_numpy(dtype=float) for col in thresholds]
    widths = np.full(len(x), 2.8 if len(x) > 1 else 3.0)
    bp = ax2.boxplot(
        box_data,
        positions=x,
        widths=widths,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor="#9ecae1", edgecolor="#1f4e79", alpha=0.85, linewidth=0.9)
    for med in bp["medians"]:
        med.set(color="#1f4e79", linewidth=1.2)
    for whisk in bp["whiskers"]:
        whisk.set(color="#1f4e79", linewidth=0.9)
    for cap in bp["caps"]:
        cap.set(color="#1f4e79", linewidth=0.9)

    ax2.axhline(0.0, color="#555555", ls="--", lw=0.9)
    ax2.set_title(f"Shift Relative to {int(round(baseline * 100))}% Baseline")
    ax2.set_xlabel("Emergence threshold (% of cumulative topic volume)")
    ax2.set_ylabel("Emergence-year shift (years)")
    ax2.grid(alpha=0.2, linewidth=0.5)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(round(v))}%" for v in x])

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_TOPIC_YEARS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    emergence_df.sort_values(["topic", "emergence_percentile"]).to_csv(OUT_TOPIC_YEARS, index=False)
    summary.sort_values("emergence_percentile").to_csv(OUT_SUMMARY, index=False)
    OUT_META.write_text(
        json.dumps(
            {
                "data_path": args.data_path,
                "min_topic_volume": int(args.min_topic_volume),
                "thresholds": thresholds,
                "baseline_threshold": baseline,
                "n_topics_complete_panel": int(len(pivot)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {OUT_PDF}")
    print(f"Wrote: {OUT_TOPIC_YEARS}")
    print(f"Wrote: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
