"""Supplementary Figure S12. Tracks descriptive network properties of the concern space through time. Provides context for how the ATS concern space matures as the system grows."""

# %%
"""Network metrics over time for the space of concerns."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import ultraplot as uplt
from pandas.core import frame

from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)


def load_data_with_fallback():
    paths = [
        Path("antarctic-database-go/data/processed/document-summary.parquet"),
        Path(
            "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"
        ),
    ]
    for path in paths:
        if path.exists():
            try:
                return load_data(str(path))
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load {path}: {exc}")
    raise FileNotFoundError("No usable data file found for metrics figure.")


def rolling_metrics(window=5):
    counts_df, submitted, countries, topics = load_data_with_fallback()
    total_topics = len(counts_df.index)

    min_year = int(submitted["meeting year"].min())
    max_year = int(submitted["meeting year"].max())

    rows = []
    for end_year in range(min_year + window - 1, max_year + 1):
        start_year = end_year - window + 1
        subset = submitted[
            (submitted["meeting year"] >= start_year)
            & (submitted["meeting year"] <= end_year)
        ]
        if subset.empty:
            continue

        interaction = generate_interaction_matrix(subset, countries, topics)
        interaction = standardize_index_labels(interaction)
        rca_block = get_rca(interaction)
        # Keep only active topics in this window.
        active_topics = rca_block.sum(axis=1) > 0
        rca_block = rca_block.loc[active_topics]
        if rca_block.empty:
            continue
        phi_block = compute_product_space(rca_block, threshold=1.0)

        g = nx.from_pandas_adjacency(phi_block)

        n = max(1, g.number_of_nodes())
        density = nx.density(g)

        if g.number_of_edges() > 0:
            gc_size = max(len(c) for c in nx.connected_components(g)) / max(
                1, total_topics
            )
        else:
            gc_size = 0.0

        try:
            clustering = nx.average_clustering(g, weight="weight")
        except Exception:
            clustering = nx.average_clustering(g)

        active_members = int((rca_block.sum(axis=0) > 0).sum())

        # Treat phi as proximity; use -log(phi) so path addition is consistent
        # with multiplicative proximity along indirect paths.
        for _, _, data in g.edges(data=True):
            w = float(data.get("weight", 1.0))
            data["distance"] = float(-np.log(np.clip(w, 1e-12, 1.0)))

        try:
            bet = nx.betweenness_centrality(g, weight="distance")
        except Exception:
            bet = nx.betweenness_centrality(g)
        try:
            clo = nx.closeness_centrality(g, distance="distance")
        except Exception:
            clo = nx.closeness_centrality(g)
        try:
            eig = nx.eigenvector_centrality(g, weight="weight", max_iter=500)
        except Exception:
            eig = nx.eigenvector_centrality(g, max_iter=500)
        deg = dict(g.degree(weight="weight"))

        def _iqr_stats(values):
            vals = np.array(list(values.values()), dtype=float)
            if vals.size == 0:
                return np.nan, np.nan, np.nan, [], []
            med = float(np.nanmedian(vals))
            q1 = float(np.nanpercentile(vals, 25))
            q3 = float(np.nanpercentile(vals, 75))
            iqr = q3 - q1
            low_fence = q1 - 1.5 * iqr
            high_fence = q3 + 1.5 * iqr
            out_low = vals[vals < low_fence]
            out_high = vals[vals > high_fence]
            return med, q1, q3, out_low.tolist(), out_high.tolist()

        bet_med, bet_q1, bet_q3, bet_out_lo, bet_out_hi = _iqr_stats(bet)
        clo_med, clo_q1, clo_q3, clo_out_lo, clo_out_hi = _iqr_stats(clo)
        eig_med, eig_q1, eig_q3, eig_out_lo, eig_out_hi = _iqr_stats(eig)
        deg_med, deg_q1, deg_q3, deg_out_lo, deg_out_hi = _iqr_stats(deg)

        rows.append(
            {
                "start_year": start_year,
                "end_year": end_year,
                "density": density,
                "gc": gc_size,
                "clustering": clustering,
                "active_members": active_members,
                "bet_med": bet_med,
                "bet_q1": bet_q1,
                "bet_q3": bet_q3,
                "bet_out_lo": bet_out_lo,
                "bet_out_hi": bet_out_hi,
                "clo_med": clo_med,
                "clo_q1": clo_q1,
                "clo_q3": clo_q3,
                "clo_out_lo": clo_out_lo,
                "clo_out_hi": clo_out_hi,
                "eig_med": eig_med,
                "eig_q1": eig_q1,
                "eig_q3": eig_q3,
                "eig_out_lo": eig_out_lo,
                "eig_out_hi": eig_out_hi,
                "deg_med": deg_med,
                "deg_q1": deg_q1,
                "deg_q3": deg_q3,
                "deg_out_lo": deg_out_lo,
                "deg_out_hi": deg_out_hi,
            }
        )

    return pd.DataFrame(rows)


def main():
    df = rolling_metrics(window=5)
    if df.empty:
        raise ValueError("No rolling metrics computed.")

    # Compute baseline metrics on the aggregated (all-years) graph.
    counts_df, submitted, countries, topics = load_data_with_fallback()
    total_topics = len(counts_df.index)
    interaction = generate_interaction_matrix(submitted, countries, topics)
    interaction = standardize_index_labels(interaction)
    rca_block = get_rca(interaction)
    active_topics = rca_block.sum(axis=1) > 0
    rca_block = rca_block.loc[active_topics]
    phi_block = compute_product_space(rca_block, threshold=1.0)
    g_all = nx.from_pandas_adjacency(phi_block)
    density_base = nx.density(g_all)
    if g_all.number_of_edges() > 0:
        gc_base = max(len(c) for c in nx.connected_components(g_all)) / max(
            1, total_topics
        )
    else:
        gc_base = 0.0
    try:
        clustering_base = nx.average_clustering(g_all, weight="weight")
    except Exception:
        clustering_base = nx.average_clustering(g_all)
    active_members_base = int((rca_block.sum(axis=0) > 0).sum())

    # Normalize top-panel metrics relative to the aggregated graph.
    density_base = density_base if density_base > 0 else 1.0
    gc_base = gc_base if gc_base > 0 else 1.0
    clustering_base = clustering_base if clustering_base > 0 else 1.0
    active_members_base = active_members_base if active_members_base > 0 else 1.0

    df["density_norm"] = df["density"] / density_base
    df["gc_norm"] = df["gc"] / gc_base
    df["active_members_norm"] = df["active_members"] / active_members_base
    # Normalize centrality medians, IQRs, and outliers to [0,1] per metric.
    for key in ("bet", "clo", "eig", "deg"):
        series_vals = pd.concat(
            [df[f"{key}_q1"], df[f"{key}_q3"], df[f"{key}_med"]]
        ).to_numpy()
        out_lo_vals = np.concatenate(
            [np.array(v, dtype=float) for v in df[f"{key}_out_lo"] if len(v) > 0]
            or [np.array([], dtype=float)]
        )
        out_hi_vals = np.concatenate(
            [np.array(v, dtype=float) for v in df[f"{key}_out_hi"] if len(v) > 0]
            or [np.array([], dtype=float)]
        )
        all_vals = np.concatenate([series_vals, out_lo_vals, out_hi_vals])
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        denom = (vmax - vmin) if vmax > vmin else 1.0
        df[f"{key}_med"] = (df[f"{key}_med"] - vmin) / denom
        df[f"{key}_q1"] = (df[f"{key}_q1"] - vmin) / denom
        df[f"{key}_q3"] = (df[f"{key}_q3"] - vmin) / denom
        df[f"{key}_out_lo"] = df[f"{key}_out_lo"].apply(
            lambda vals: [(v - vmin) / denom for v in vals]
        )
        df[f"{key}_out_hi"] = df[f"{key}_out_hi"].apply(
            lambda vals: [(v - vmin) / denom for v in vals]
        )

    event_windows = [
        (1988, 1992, "CRAMRA/Madrid"),
        (2002, 2005, "Tourism guidelines"),
        (2014, 2017, "Ross Sea MPA"),
    ]

    cycles = [uplt.Cycle("538"), uplt.Cycle("bmh")]

    fig, ax = uplt.subplots(nrows=2, width="20cm", height="10cm")
    for axi, cycle in zip(ax, cycles):
        axi.set_prop_cycle(cycle)

    ax[0].plot(df["end_year"], df["density_norm"], marker="o", label="Link density")
    ax[0].plot(df["end_year"], df["gc_norm"], marker="o", label="Giant component size")
    ax[0].plot(
        df["end_year"],
        df["clustering"],
        marker="o",
        label="Clustering coefficient",
    )
    ax[0].plot(
        df["end_year"],
        df["active_members_norm"],
        marker="o",
        label="Active members (norm)",
    )

    colors = uplt.Colormap("538")(np.arange(0, len(event_windows)))
    for c, (start, end, label) in zip(colors, event_windows):
        window = end - start
        ax.axvspan(start, end, color=c, alpha=0.18, zorder=-2)

        # ax.axvspan(
        #     end,
        #     end + window,
        #     facecolor=c,
        #     alpha=0.32,
        #     edgecolor=c,
        #     linewidth=1,
        #     hatch="xxxx",
        # )

        # Label event window above the top axis.
        ax[0].text(
            (start + end) / 2,
            1.02,
            label,
            transform=ax[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=c,
        )

    def _plot_metric(axi, years, med, q1, q3, out_hi, out_lo, label, color):
        axi.plot(years, med, label=label, marker="o", color=color)
        axi.fill_between(
            years,
            q1,
            q3,
            alpha=0.12,
            color=color,
            cycle=None,
            label="_nolegend_",
        )
        years_list = list(years)
        out_hi_x = []
        out_hi_y = []
        out_lo_x = []
        out_lo_y = []
        for year, vals_hi, vals_lo in zip(years_list, out_hi, out_lo):
            out_hi_x.extend([year] * len(vals_hi))
            out_hi_y.extend(vals_hi)
            out_lo_x.extend([year] * len(vals_lo))
            out_lo_y.extend(vals_lo)
        if out_hi_x:
            axi.scatter(
                out_hi_x,
                out_hi_y,
                marker="^",
                s=18,
                alpha=0.8,
                color=color,
                edgecolor=color,
                cycle=None,
                label="_nolegend_",
            )
        if out_lo_x:
            axi.scatter(
                out_lo_x,
                out_lo_y,
                marker="v",
                s=18,
                alpha=0.8,
                color=color,
                edgecolor=color,
                cycle=None,
                label="_nolegend_",
            )

    metric_colors = uplt.Colormap("bmh")(np.arange(0, 4))

    _plot_metric(
        ax[1],
        df["end_year"],
        df["bet_med"],
        df["bet_q1"],
        df["bet_q3"],
        df["bet_out_hi"],
        df["bet_out_lo"],
        "Betweenness (median)",
        metric_colors[0],
    )
    _plot_metric(
        ax[1],
        df["end_year"],
        df["clo_med"],
        df["clo_q1"],
        df["clo_q3"],
        df["clo_out_hi"],
        df["clo_out_lo"],
        "Closeness (median)",
        metric_colors[1],
    )
    _plot_metric(
        ax[1],
        df["end_year"],
        df["eig_med"],
        df["eig_q1"],
        df["eig_q3"],
        df["eig_out_hi"],
        df["eig_out_lo"],
        "Eigenvector (median)",
        metric_colors[2],
    )
    _plot_metric(
        ax[1],
        df["end_year"],
        df["deg_med"],
        df["deg_q1"],
        df["deg_q3"],
        df["deg_out_hi"],
        df["deg_out_lo"],
        "Degree (median)",
        metric_colors[3],
    )

    out_hi_vals = []
    for key in ("bet", "clo", "eig", "deg"):
        out_hi_vals.extend([v for vals in df[f"{key}_out_hi"] for v in vals])
    max_out = np.nanmax(out_hi_vals) if out_hi_vals else np.nan
    if not np.isfinite(max_out):
        max_out = 1.05
    ax.format(
        xlabel="Year",
        ylabel="Metric (normalized / centrality)",
        ylim=(-0.075, max(1.075, max_out + 0.05)),
    )

    # Order legends by the last y-value (top-to-bottom on the right).
    def _legend_by_last_y(axi):
        lines = [l for l in axi.get_lines() if not l.get_label().startswith("_")]
        items = [(l, l.get_label(), l.get_ydata()[-1]) for l in lines]
        items.sort(key=lambda t: t[2], reverse=True)
        handles = [t[0] for t in items]
        labels = [t[1] for t in items]
        axi.legend(handles, labels, ncols=1, loc="r", frameon=0, fontsize=9, align="t")

    _legend_by_last_y(ax[0])
    _legend_by_last_y(ax[1])

    ax[1].legend(
        handles=[
            uplt.pyplot.Line2D(
                [0],
                [0],
                color="k",
                marker="^",
                label="1.5 IQR above Q3",
                ls="none",
                markeredgecolor="k",
                markerfacecolor="w",
                markersize=8,
            ),
            uplt.pyplot.Line2D(
                [0],
                [0],
                color="k",
                marker="v",
                label="1.5 IQR below Q1",
                ls="none",
                markeredgecolor="k",
                markerfacecolor="w",
                markersize=8,
            ),
        ],
        loc="r",
        align="bottom",
        frameon=0,
        ncols=1,
    )

    fig.savefig("./figures/figS12_space_network_metrics.png", dpi=600, transparent=True)
    fig.savefig("./figures/figS12_space_network_metrics.pdf")


if __name__ == "__main__":
    main()
