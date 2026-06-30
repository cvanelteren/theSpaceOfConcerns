#!/usr/bin/env python3
"""Check whether concern-space portfolio structure aligns with geography and GDP.

Outputs:
- output/step6_geo_portfolio_pairs.csv
- output/step6_gdp_breadth_actor_table.csv
- output/step6_gdp_breadth_coefficients.csv
- output/step6_geo_gdp_portfolio_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
import sys

import cartopy.io.shapereader as shpreader
import country_converter as coco
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import compute_product_space, get_rca, load_data


DATA_PATH = "./antarctic-database-go/data/processed/document-summary.parquet"
WORLD_BANK_GDP_PATH = "./gdp-per-capita-worldbank.csv"
DEFAULT_OUTPUT_DIR = Path("output")
RCA_THRESHOLD = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geography/GDP checks for concern-space portfolios."
    )
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help=f"Input ATS summary parquet/csv (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--gdp-path",
        default=WORLD_BANK_GDP_PATH,
        help=f"World Bank GDP-per-capita CSV (default: {WORLD_BANK_GDP_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--natural-earth-resolution",
        default="110m",
        choices=["110m", "50m", "10m"],
        help="Natural Earth admin_0 resolution for country centroids.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("'", "")
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(r * c)


def actor_iso3_map(actors: list[str]) -> dict[str, str]:
    cc = coco.CountryConverter()
    out: dict[str, str] = {}
    for actor in actors:
        iso = cc.convert(actor, to="ISO3")
        if isinstance(iso, str) and iso != "not found" and len(iso) == 3:
            out[actor] = iso
    return out


def load_country_centroids_by_iso3(resolution: str = "110m") -> dict[str, tuple[float, float]]:
    shp_path = shpreader.natural_earth(
        resolution=resolution,
        category="cultural",
        name="admin_0_countries",
    )
    reader = shpreader.Reader(shp_path)
    out: dict[str, tuple[float, float]] = {}
    for rec in reader.records():
        geom = rec.geometry
        if geom is None or geom.is_empty:
            continue
        rp = geom.representative_point()
        lon = float(rp.x)
        lat = float(rp.y)
        for key in ("ADM0_A3", "ISO_A3", "SU_A3", "GU_A3"):
            iso = rec.attributes.get(key)
            if isinstance(iso, str) and len(iso) == 3 and iso not in {"-99", ""}:
                out.setdefault(iso, (lon, lat))
    return out


def topic_positions_1d(counts_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    topics = counts_df.index.tolist()
    rca = get_rca(counts_df)
    phi = compute_product_space(rca).reindex(index=topics, columns=topics, fill_value=0.0)
    dist = 1.0 - phi.to_numpy(dtype=float)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    n = dist.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dist**2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = int(np.argmax(eigvals))
    lam = float(eigvals[idx])

    if lam <= 1e-12:
        x = np.arange(n, dtype=float)
    else:
        x = eigvecs[:, idx] * np.sqrt(lam)

    order = np.argsort(x, kind="mergesort")
    topics_sorted = [topics[i] for i in order]
    x_sorted = x[order]
    return topics_sorted, x_sorted


def pairwise_portfolio_emd_1d(
    counts_df: pd.DataFrame,
    actors: list[str],
    topics_sorted: list[str],
    x_sorted: np.ndarray,
) -> pd.DataFrame:
    counts = counts_df.loc[topics_sorted, actors].astype(float)
    weights = counts.divide(counts.sum(axis=0), axis=1).fillna(0.0).T.to_numpy(dtype=float)

    pair_i, pair_j = np.triu_indices(len(actors), k=1)
    dx = np.diff(x_sorted)
    if np.allclose(dx, 0.0):
        dx = np.ones(len(x_sorted) - 1, dtype=float)
    else:
        dx = np.abs(dx)

    diffs = weights[pair_i] - weights[pair_j]
    cumdiff = np.cumsum(diffs, axis=1)[:, :-1]
    emd_vals = np.abs(cumdiff) @ dx

    return pd.DataFrame(
        {
            "actor_i": [actors[i] for i in pair_i],
            "actor_j": [actors[j] for j in pair_j],
            "portfolio_emd_1d": emd_vals,
        }
    )


def pairwise_geodesic_km(
    actor_centroids: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    actors = sorted(actor_centroids.keys())
    rows = []
    for i in range(len(actors)):
        a = actors[i]
        lon_a, lat_a = actor_centroids[a]
        for j in range(i + 1, len(actors)):
            b = actors[j]
            lon_b, lat_b = actor_centroids[b]
            rows.append(
                {
                    "actor_i": a,
                    "actor_j": b,
                    "geo_km": haversine_km(lat_a, lon_a, lat_b, lon_b),
                }
            )
    return pd.DataFrame(rows)


def load_world_bank_gdp(gdp_path: str) -> pd.DataFrame:
    wb = pd.read_csv(gdp_path, skiprows=4)
    year_cols = [c for c in wb.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        raise RuntimeError("No year columns found in GDP file.")

    values = wb[year_cols].apply(pd.to_numeric, errors="coerce")
    latest_year = []
    latest_value = []
    for idx in values.index:
        row = values.loc[idx]
        non_null = row.dropna()
        if non_null.empty:
            latest_year.append(np.nan)
            latest_value.append(np.nan)
        else:
            latest_year.append(int(non_null.index[-1]))
            latest_value.append(float(non_null.iloc[-1]))

    out = wb[["Country Name"]].copy()
    out["gdp_year"] = latest_year
    out["gdp_per_capita"] = latest_value
    if "Country Code" in wb.columns:
        out["iso3"] = wb["Country Code"].where(
            wb["Country Code"].astype(str).str.fullmatch(r"[A-Z]{3}"),
            pd.NA,
        )
    else:
        out["iso3"] = pd.NA
    return out[["Country Name", "iso3", "gdp_year", "gdp_per_capita"]]


def run_geo_portfolio_check(
    counts_df: pd.DataFrame,
    actor_to_iso3: dict[str, str],
    iso3_to_centroid: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    actors = [
        a
        for a in counts_df.columns.tolist()
        if a in actor_to_iso3 and actor_to_iso3[a] in iso3_to_centroid
    ]
    if len(actors) < 3:
        raise RuntimeError("Need at least 3 country actors with centroid mappings.")

    topics_sorted, x_sorted = topic_positions_1d(counts_df.loc[:, actors])
    portfolio_pairs = pairwise_portfolio_emd_1d(counts_df, actors, topics_sorted, x_sorted)
    actor_centroids = {a: iso3_to_centroid[actor_to_iso3[a]] for a in actors}
    geo_pairs = pairwise_geodesic_km(actor_centroids)

    pair_df = portfolio_pairs.merge(geo_pairs, on=["actor_i", "actor_j"], how="inner")
    pair_df["log_geo_km"] = np.log(pair_df["geo_km"])

    sp = spearmanr(pair_df["geo_km"], pair_df["portfolio_emd_1d"])
    pr = pearsonr(pair_df["log_geo_km"], pair_df["portfolio_emd_1d"])

    q25 = float(pair_df["geo_km"].quantile(0.25))
    q75 = float(pair_df["geo_km"].quantile(0.75))
    near = pair_df.loc[pair_df["geo_km"] <= q25, "portfolio_emd_1d"]
    far = pair_df.loc[pair_df["geo_km"] >= q75, "portfolio_emd_1d"]

    summary = {
        "n_country_actors": int(len(actors)),
        "n_pairs": int(len(pair_df)),
        "spearman_rho_geo_vs_portfolio_emd": float(sp.statistic),
        "spearman_pvalue": float(sp.pvalue),
        "pearson_r_loggeo_vs_portfolio_emd": float(pr.statistic),
        "pearson_pvalue": float(pr.pvalue),
        "near_q25_geo_km": q25,
        "far_q75_geo_km": q75,
        "mean_emd_near_q25": float(near.mean()),
        "mean_emd_far_q75": float(far.mean()),
    }
    return pair_df, summary


def run_gdp_breadth_check(
    counts_df: pd.DataFrame,
    actor_to_iso3: dict[str, str],
    gdp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    actors = [a for a in counts_df.columns.tolist() if a in actor_to_iso3]
    if len(actors) < 3:
        raise RuntimeError("Need at least 3 country actors for GDP breadth check.")

    sub_counts = counts_df.loc[:, actors]
    rca = get_rca(sub_counts)
    breadth = (rca >= RCA_THRESHOLD).sum(axis=0).rename("breadth_topics")
    totals = sub_counts.sum(axis=0).rename("topic_assignments")

    actor_df = pd.concat([breadth, totals], axis=1).reset_index(names="actor")
    actor_df["iso3"] = actor_df["actor"].map(actor_to_iso3)

    gdp_latest = (
        gdp_df.dropna(subset=["iso3", "gdp_per_capita"])
        .sort_values(["iso3", "gdp_year"])
        .drop_duplicates(subset=["iso3"], keep="last")
    )
    gdp_map = gdp_latest.set_index("iso3")[["gdp_per_capita", "gdp_year"]]
    actor_df["gdp_per_capita"] = actor_df["iso3"].map(gdp_map["gdp_per_capita"])
    actor_df["gdp_year"] = actor_df["iso3"].map(gdp_map["gdp_year"])
    actor_df["log_gdp_per_capita"] = np.log(actor_df["gdp_per_capita"])
    actor_df["log_topic_assignments"] = np.log1p(actor_df["topic_assignments"])

    reg_df = actor_df.dropna(subset=["breadth_topics", "log_gdp_per_capita"]).copy()
    if len(reg_df) < 5:
        raise RuntimeError("Too few actors with GDP to run breadth model.")

    sp = spearmanr(reg_df["log_gdp_per_capita"], reg_df["breadth_topics"])
    pr = pearsonr(reg_df["log_gdp_per_capita"], reg_df["breadth_topics"])

    x = sm.add_constant(reg_df[["log_gdp_per_capita"]])
    y = reg_df["breadth_topics"]
    model = sm.OLS(y, x).fit(cov_type="HC3")

    coeff_df = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err_hc3": model.bse.values,
            "z_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low_95": model.conf_int().iloc[:, 0].values,
            "ci_high_95": model.conf_int().iloc[:, 1].values,
        }
    )

    summary = {
        "n_country_actors": int(len(actors)),
        "n_actors_with_gdp": int(len(reg_df)),
        "spearman_rho_loggdp_vs_breadth": float(sp.statistic),
        "spearman_pvalue": float(sp.pvalue),
        "pearson_r_loggdp_vs_breadth": float(pr.statistic),
        "pearson_pvalue": float(pr.pvalue),
        "ols_r2": float(model.rsquared),
        "ols_coef_log_gdp_per_capita": float(model.params["log_gdp_per_capita"]),
        "ols_p_log_gdp_per_capita": float(model.pvalues["log_gdp_per_capita"]),
    }
    return actor_df, coeff_df, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_df, submitted_df, countries_raw, topics_raw = load_data(args.data_path)
    actor_to_iso3 = actor_iso3_map(counts_df.columns.tolist())
    iso3_to_centroid = load_country_centroids_by_iso3(
        resolution=args.natural_earth_resolution
    )
    gdp_df = load_world_bank_gdp(args.gdp_path)

    geo_pair_df, geo_summary = run_geo_portfolio_check(
        counts_df=counts_df,
        actor_to_iso3=actor_to_iso3,
        iso3_to_centroid=iso3_to_centroid,
    )
    gdp_actor_df, gdp_coef_df, gdp_summary = run_gdp_breadth_check(
        counts_df=counts_df,
        actor_to_iso3=actor_to_iso3,
        gdp_df=gdp_df,
    )

    summary = {
        "data_path": args.data_path,
        "gdp_path": args.gdp_path,
        "natural_earth_resolution": args.natural_earth_resolution,
        "geo_vs_portfolio": geo_summary,
        "gdp_vs_breadth": gdp_summary,
    }

    geo_pair_df.to_csv(out_dir / "step6_geo_portfolio_pairs.csv", index=False)
    gdp_actor_df.to_csv(out_dir / "step6_gdp_breadth_actor_table.csv", index=False)
    gdp_coef_df.to_csv(out_dir / "step6_gdp_breadth_coefficients.csv", index=False)
    (out_dir / "step6_geo_gdp_portfolio_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Step 6 checks complete.")
    print(f"Geo pairs: {out_dir / 'step6_geo_portfolio_pairs.csv'}")
    print(f"GDP breadth table: {out_dir / 'step6_gdp_breadth_actor_table.csv'}")
    print(f"Summary: {out_dir / 'step6_geo_gdp_portfolio_summary.json'}")


if __name__ == "__main__":
    main()
