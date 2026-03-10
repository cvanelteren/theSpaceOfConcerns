"""Supplementary Figure S02. Summarizes archive coverage across topics and over time. Establishes the uneven but expanding documentary base from which the concern space is inferred."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import ultraplot as uplt

from utils import load_data

PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "figures"
OUT_DIR = PROJECT_ROOT / "output"

OUT_PDF = FIG_DIR / "figS02_archive_coverage.pdf"
OUT_PNG = FIG_DIR / "figS02_archive_coverage.png"
OUT_CSV = OUT_DIR / "fig06_archive_coverage_submissions_yearly_counts.csv"

DATA_PATHS = [
    PROJECT_ROOT / "antarctic-database-go/data/processed/document-summary.parquet",
    PROJECT_ROOT
    / "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet",
    PROJECT_ROOT / "Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv",
    PROJECT_ROOT / "document-summary.csv",
]


def load_data_with_fallback() -> tuple[pd.DataFrame, pd.DataFrame, set[str], set[str]]:
    last_error = None
    for path in DATA_PATHS:
        if not path.exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Failed to load {path}: {exc}")
    raise FileNotFoundError("No usable data file found in DATA_PATHS.") from last_error


def build_yearly_counts(submitted_df: pd.DataFrame) -> pd.DataFrame:
    base = submitted_df[["meeting_year", "paper_id"]].dropna().copy()
    base["meeting_year"] = pd.to_numeric(base["meeting_year"], errors="coerce")
    base = base.dropna(subset=["meeting_year"]).copy()
    base["meeting_year"] = base["meeting_year"].astype(int)

    yearly = (
        base.groupby("meeting_year")["paper_id"]
        .nunique()
        .rename("n_unique_submissions")
        .reset_index()
        .sort_values("meeting_year")
    )
    yearly["rolling_5y_mean"] = (
        yearly["n_unique_submissions"]
        .rolling(window=5, min_periods=1, center=True)
        .mean()
    )
    return yearly


def build_topic_counts(submitted_df: pd.DataFrame) -> pd.DataFrame:
    base = submitted_df[["paper_id", "category"]].dropna().copy()
    base["category"] = base["category"].astype(str).str.split("\t")
    base = base.explode("category").dropna(subset=["category"]).copy()
    base["category"] = base["category"].astype(str).str.strip()
    base = base[base["category"] != ""].copy()
    base = base.drop_duplicates(subset=["paper_id", "category"])

    topic_counts = (
        base.groupby("category")["paper_id"]
        .nunique()
        .rename("n_unique_submissions")
        .reset_index()
        .sort_values(["n_unique_submissions", "category"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return topic_counts


def build_figure():
    _, submitted_df, _, _ = load_data_with_fallback()
    yearly = build_yearly_counts(submitted_df)
    topic_counts = build_topic_counts(submitted_df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(OUT_CSV, index=False)

    fig, axs = uplt.subplots(
        ncols=2,
        # figsize=(7.4, 9.8),
        # hratios=[3.2, 1.4],
        share=False,
    )
    fig.patch.set_facecolor("white")
    axs.format(abc="[A]")

    ax0 = axs[0]
    bars = ax0.barh(
        topic_counts["category"],
        topic_counts["n_unique_submissions"],
        color="#3a7fb0",
        edgecolor="white",
        linewidth=0.35,
        autoformat=False,
    )
    ax0.format(
        title="Unique ATS submissions by topic",
        xlabel="Submissions",
        grid=False,
        yticklabelsize=5,
    )
    ax0.grid(axis="x", alpha=0.18)
    ax0.grid(axis="y", visible=False)
    ax0.tick_params(axis="y", labelsize=6.5)
    ax0.set_xlim(0, int(topic_counts["n_unique_submissions"].max()) + 25)

    for bar, value in zip(bars, topic_counts["n_unique_submissions"]):
        ax0.text(
            float(value) + 1.2,
            bar.get_y() + bar.get_height() / 2.0,
            str(int(value)),
            va="center",
            ha="left",
            fontsize=6.2,
            color="0.25",
        )

    ax = axs[1]
    years = yearly["meeting_year"].to_numpy()
    counts = yearly["n_unique_submissions"].to_numpy()
    trend = yearly["rolling_5y_mean"].to_numpy()

    ax.bar(
        years,
        counts,
        color="#c7d3dd",
        edgecolor="white",
        linewidth=0.35,
        width=0.9,
    )
    ax.plot(
        years,
        trend,
        color="#17324d",
        linewidth=2.1,
    )
    ax.format(
        title="Unique ATS submissions by year",
        xlabel="Year",
        ylabel="Submissions",
        xlim=(years.min() - 1, years.max() + 1),
        grid=False,
    )

    peak_idx = int(yearly["n_unique_submissions"].idxmax())
    peak_year = int(yearly.loc[peak_idx, "meeting_year"])
    peak_count = int(yearly.loc[peak_idx, "n_unique_submissions"])
    ax.annotate(
        f"Peak: {peak_count} ({peak_year})",
        xy=(peak_year, peak_count),
        xytext=(0.98, 0.93),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8.2,
        color="#17324d",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
    )

    fig.canvas.draw()
    fig.savefig(OUT_PDF, facecolor="white")
    fig.savefig(OUT_PNG, dpi=320, facecolor="white")
    uplt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    build_figure()
