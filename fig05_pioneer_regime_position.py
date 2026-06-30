import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import ultraplot as uplt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage
from matplotlib.patches import Wedge

from utils import get_rca, load_data, load_flag, standardize_index_labels

REGIME_COLS = ["region_1_share", "region_2_share", "region_3_share"]
REGIME_NAMES = ["Mode 1", "Mode 2", "Mode 3"]
REGIME_COLORS = ["#d94841", "#3f88c5", "#4daf4a"]
N_FLAG_TOP_RIGHT = 7
N_FLAG_ENVELOPE = 4

OUT_FIG = Path("figures/fig05_pioneer_regime_position")
OUT_RESIDUAL_FIG = Path("figures/fig05_pioneer_mode_anchoring_residual_appendix")
OUT_DATA = Path("output/fig05_pioneer_regime_position_data.csv")
OUT_SUMMARY = Path("output/fig05_pioneer_regime_position_summary.json")


def load_inputs():
    pioneer_df = pd.read_csv("output/pioneer_analysis_results.csv")
    topic_df = pd.read_csv("output/fig45_portfolio_space_ridgelines_topic_order.csv")
    region_df = pd.read_csv(
        "output/fig45_portfolio_space_ridgelines_region_summary.csv"
    )

    counts_df, _, _, _ = load_data(
        "./antarctic-database-go/data/processed/document-summary.parquet"
    )
    counts_df = standardize_index_labels(counts_df)
    if counts_df.index.has_duplicates:
        counts_df = counts_df.groupby(level=0).sum()
    return pioneer_df, topic_df, region_df, get_rca(counts_df)


def compute_regime_shares(pioneer_df, topic_df, region_df, rca):
    required_topic = {"topic", "x_plot"}
    required_region = {"region_id", "boundary_left", "boundary_right"}
    if not required_topic.issubset(topic_df.columns):
        missing = sorted(required_topic - set(topic_df.columns))
        raise ValueError(f"topic summary missing columns: {missing}")
    if not required_region.issubset(region_df.columns):
        missing = sorted(required_region - set(region_df.columns))
        raise ValueError(f"region summary missing columns: {missing}")

    ordered_topics = topic_df["topic"].tolist()
    x_plot = topic_df["x_plot"].to_numpy(dtype=float)
    region_boundaries = (
        region_df.sort_values("region_id")["boundary_left"].to_numpy(dtype=float)[1:]
        if len(region_df) > 1
        else np.array([], dtype=float)
    )
    if region_boundaries.size == 0 and len(region_df) > 1:
        region_sorted = region_df.sort_values("region_id")
        region_boundaries = region_sorted["boundary_right"].to_numpy(dtype=float)[:-1]
    topic_region_idx = np.digitize(x_plot, region_boundaries).astype(int)
    n_regions = int(region_df["region_id"].nunique())

    rows = []
    for country in pioneer_df["country"].tolist():
        shares = np.zeros(n_regions, dtype=float)
        if country in rca.columns:
            vals = rca.reindex(index=ordered_topics)[country].to_numpy(dtype=float)
            signal = np.clip(vals - 1.0, 0.0, None)
            weights = np.bincount(
                topic_region_idx, weights=signal, minlength=n_regions
            ).astype(float)
            total = float(weights.sum())
            if total > 0:
                shares = weights / total
        row = {"country": country}
        for idx, share in enumerate(shares, start=1):
            row[f"region_{idx}_share"] = float(share)
        rows.append(row)

    regime_df = pd.DataFrame(rows)
    for col in REGIME_COLS:
        if col not in regime_df.columns:
            regime_df[col] = 0.0
    return regime_df


def prepare_data():
    pioneer_df, topic_df, region_df, rca = load_inputs()
    regime_df = compute_regime_shares(pioneer_df, topic_df, region_df, rca)
    df = pioneer_df.merge(
        regime_df[["country", *REGIME_COLS]], on="country", how="left"
    )
    df[REGIME_COLS] = df[REGIME_COLS].fillna(0.0)

    weights = df[REGIME_COLS].to_numpy(dtype=float)
    weights = np.clip(weights, 0.0, None)
    row_sum = weights.sum(axis=1, keepdims=True)
    nz = row_sum[:, 0] > 0
    weights[nz] = weights[nz] / row_sum[nz]
    df[REGIME_COLS] = weights

    df["dominant_regime"] = weights.argmax(axis=1) + 1
    df["dominant_regime_label"] = df["dominant_regime"].map(
        {idx + 1: label for idx, label in enumerate(REGIME_NAMES)}
    )
    df["max_regime_share"] = weights.max(axis=1)
    return df


def fit_models(df):
    regime_dummies = pd.get_dummies(
        df["dominant_regime"].astype(int),
        prefix="reg",
        drop_first=True,
        dtype=float,
    )
    X_anchor = sm.add_constant(
        pd.concat(
            [
                df[["topics_adopted", "max_regime_share"]].astype(float),
                regime_dummies,
            ],
            axis=1,
        )
    )
    model_anchor = sm.OLS(df["pioneer_index"].astype(float), X_anchor).fit()

    X_simple = sm.add_constant(df[["max_regime_share"]].astype(float))
    model_simple = sm.OLS(df["pioneer_index"].astype(float), X_simple).fit()

    controls = sm.add_constant(
        pd.concat([df[["topics_adopted"]].astype(float), regime_dummies], axis=1)
    )
    y_resid = sm.OLS(df["pioneer_index"].astype(float), controls).fit().resid
    x_resid = sm.OLS(df["max_regime_share"].astype(float), controls).fit().resid
    model_partial = sm.OLS(y_resid, sm.add_constant(x_resid)).fit()
    return model_anchor, model_simple, model_partial, x_resid, y_resid


def _zscore(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    scale = arr.std(ddof=0)
    if scale == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / scale


def select_flag_actors(df: pd.DataFrame, breadth_poly) -> set[str]:
    selected: set[str] = set()

    top_right = pd.Series(
        _zscore(df["topics_adopted"]) + _zscore(df["pioneer_index"]),
        index=df.index,
    )
    selected.update(df.loc[top_right.nlargest(N_FLAG_TOP_RIGHT).index, "country"])

    curve_residual = pd.Series(
        df["pioneer_index"].to_numpy(dtype=float)
        - breadth_poly(df["topics_adopted"].to_numpy(dtype=float)),
        index=df.index,
    )
    selected.update(df.loc[curve_residual.nlargest(N_FLAG_ENVELOPE).index, "country"])
    selected.update(df.loc[curve_residual.nsmallest(N_FLAG_ENVELOPE).index, "country"])
    return selected


def draw_pie(
    ax,
    x,
    y,
    weights,
    radius_px=7,
    *,
    alpha=1.0,
    edgecolor="black",
    edgewidth=0.7,
    zorder=4,
):
    total = float(np.sum(weights))
    if total <= 0:
        return
    weights = np.asarray(weights, dtype=float) / total
    start = 90.0
    da = DrawingArea(2 * radius_px, 2 * radius_px, clip=False)
    center = (radius_px, radius_px)
    for ww, color in zip(weights, REGIME_COLORS):
        if ww <= 0:
            continue
        angle = 360.0 * ww
        da.add_artist(
            Wedge(
                center,
                radius_px,
                start,
                start + angle,
                facecolor=color,
                edgecolor="white",
                lw=0.6,
                alpha=alpha,
            )
        )
        start += angle
    da.add_artist(
        Wedge(
            center,
            radius_px,
            0,
            360,
            facecolor="none",
            edgecolor=edgecolor,
            lw=edgewidth,
            alpha=min(1.0, alpha + 0.15),
        )
    )
    ab = AnnotationBbox(
        da,
        (x, y),
        xycoords="data",
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0.0,
        zorder=zorder,
    )
    ax.add_artist(ab)
    ab.set_clip_on(True)
    ab.set_clip_path(ax.patch)
    ab.set_in_layout(False)


def draw_flag(ax, x, y, flag_img, y_offset_pts=7, zoom=0.08):
    if flag_img is None:
        return
    box = OffsetImage(flag_img, zoom=zoom, cmap=None)
    ab = AnnotationBbox(
        box,
        (x, y),
        xycoords="data",
        xybox=(0, y_offset_pts),
        boxcoords="offset points",
        frameon=False,
        box_alignment=(0.5, 0.0),
        pad=0.0,
        zorder=5,
    )
    ax.add_artist(ab)
    ab.set_clip_on(True)
    ab.set_clip_path(ax.patch)
    ab.set_in_layout(False)


def regime_handles(markersize=7):
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=markersize,
            label=name,
        )
        for name, color in zip(REGIME_NAMES, REGIME_COLORS)
    ]


def add_regime_legend(fig, *, fontsize=10.4, markersize=7.2):
    return fig.legend(
        handles=regime_handles(markersize=markersize),
        loc="t",
        ncols=3,
        frameon=False,
        fontsize=fontsize,
    )


def build_main_figure(df):
    fig, ax = uplt.subplots(figwidth="17.5cm", figheight="7.5cm")
    x_breadth = df["topics_adopted"].to_numpy(dtype=float)
    y_pioneer = df["pioneer_index"].to_numpy(dtype=float)
    x_pad = 5
    y_pad = 5

    breadth_coeffs = np.polyfit(x_breadth, y_pioneer, 2)
    breadth_poly = np.poly1d(breadth_coeffs)
    flagged_actors = select_flag_actors(df, breadth_poly)
    flag_lookup = {
        country: load_flag(
            country,
            save=False,
            base="./assets/flags",
            target_area_px=15000,
        )
        for country in flagged_actors
    }

    draw_order = pd.concat(
        [
            df.loc[~df["country"].isin(flagged_actors)],
            df.loc[df["country"].isin(flagged_actors)],
        ]
    )
    for _, row in draw_order.iterrows():
        is_flagged = row["country"] in flagged_actors
        draw_pie(
            ax,
            row["topics_adopted"],
            row["pioneer_index"],
            row[REGIME_COLS].to_numpy(dtype=float),
            radius_px=6.2 if is_flagged else 4.5,
            alpha=1.0 if is_flagged else 0.5,
            edgecolor="#1f2933" if is_flagged else "0.55",
            edgewidth=0.95 if is_flagged else 0.4,
            zorder=4 if is_flagged else 3,
        )

    for _, row in df.loc[df["country"].isin(flagged_actors)].iterrows():
        draw_flag(
            ax,
            row["topics_adopted"],
            row["pioneer_index"],
            flag_lookup.get(row["country"]),
            y_offset_pts=6.8,
            zoom=0.086,
        )

    x_fit = np.linspace(x_breadth.min(), x_breadth.max(), 240)
    ax.plot(x_fit, breadth_poly(x_fit), color="black", lw=1.4, ls="--", zorder=2)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.format(
        xlabel="Breadth (specialized topics)",
        ylabel="Relative pioneer index",
        xlim=(x_breadth.min() - x_pad, x_breadth.max() + x_pad),
        ylim=(y_pioneer.min() - y_pad, y_pioneer.max() + y_pad),
    )
    ax.set_axisbelow(True)
    return fig, flagged_actors


def build_residual_figure(df, model_partial, x_resid, y_resid):
    fig, ax = uplt.subplots(figsize=(5.2, 4.2), share=False)
    for idx, (name, color) in enumerate(zip(REGIME_NAMES, REGIME_COLORS), start=1):
        mask = df["dominant_regime"] == idx
        ax.scatter(
            x_resid[mask],
            y_resid[mask],
            color=color,
            edgecolor="white",
            lw=0.6,
            s=46,
            alpha=0.9,
            zorder=3,
            label=name,
        )
    x_pad = max(0.03, 0.08 * (x_resid.max() - x_resid.min()))
    y_pad = max(0.8, 0.06 * (y_resid.max() - y_resid.min()))
    x_fit = np.linspace(x_resid.min(), x_resid.max(), 200)
    X_fit = sm.add_constant(pd.DataFrame({0: x_fit}))
    ax.plot(
        x_fit,
        model_partial.predict(X_fit),
        color="black",
        lw=1.4,
        ls="--",
        zorder=2,
    )
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)
    ax.format(
        xlabel="Mode anchoring residual",
        ylabel="Pioneer index residual",
        grid=False,
        xlim=(x_resid.min() - x_pad, x_resid.max() + x_pad),
        ylim=(y_resid.min() - y_pad, y_resid.max() + y_pad),
    )
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.86", linewidth=0.65)
    add_regime_legend(fig, fontsize=10.2, markersize=7.0)
    return fig


def write_summary(df, model_anchor, flagged_actors):
    regime_means = (
        df.groupby("dominant_regime_label")["pioneer_index"]
        .agg(["mean", "median", "count"])
        .round(3)
        .to_dict(orient="index")
    )
    summary = {
        "n_actors": int(len(df)),
        "breadth_coef": float(model_anchor.params["topics_adopted"]),
        "breadth_p": float(model_anchor.pvalues["topics_adopted"]),
        "anchor_coef": float(model_anchor.params["max_regime_share"]),
        "anchor_p": float(model_anchor.pvalues["max_regime_share"]),
        "regime2_coef": float(model_anchor.params.get("reg_2", np.nan)),
        "regime2_p": float(model_anchor.pvalues.get("reg_2", np.nan)),
        "regime3_coef": float(model_anchor.params.get("reg_3", np.nan)),
        "regime3_p": float(model_anchor.pvalues.get("reg_3", np.nan)),
        "r_squared": float(model_anchor.rsquared),
        "regime_means": regime_means,
        "flagged_actors": sorted(flagged_actors),
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    df = prepare_data()
    model_anchor, _, model_partial, x_resid, y_resid = fit_models(df)
    fig, flagged_actors = build_main_figure(df)
    fig_residual = build_residual_figure(df, model_partial, x_resid, y_resid)

    df.to_csv(OUT_DATA, index=False)
    write_summary(df, model_anchor, flagged_actors)
    fig.save(f"{OUT_FIG}.pdf")
    fig.save(f"{OUT_FIG}.png", dpi=250, transparent=True)
    fig_residual.save(f"{OUT_RESIDUAL_FIG}.pdf")
    fig_residual.save(f"{OUT_RESIDUAL_FIG}.png", dpi=250, transparent=True)

    print(f"Wrote {OUT_FIG}.pdf/png")
    print(f"Wrote {OUT_RESIDUAL_FIG}.pdf/png")
    print(f"Wrote {OUT_DATA}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
