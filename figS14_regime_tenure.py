"""Supplementary Figure S14. Relates dominant regime position to actor tenure in the ATS archive. Supports the interpretation of long-established Regime 3 anchors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import _split_multi_value, load_data

ROOT = Path(__file__).resolve().parent
DATA_FP = ROOT / 'antarctic-database-go/data/processed/document-summary.parquet'
ACTOR_SUMMARY_FP = ROOT / 'output/fig45_portfolio_space_ridgelines_actor_summary.csv'
OUT_DATA = ROOT / 'output/fig35_regime_tenure.csv'
OUT_SUMMARY = ROOT / 'output/fig35_regime_tenure_summary.csv'
OUT_PDF = ROOT / 'figures/figS14_regime_tenure.pdf'
OUT_PNG = ROOT / 'figures/figS14_regime_tenure.png'


def main() -> None:
    actor = pd.read_csv(ACTOR_SUMMARY_FP)[['actor', 'dominant_region', 'dominant_region_share']].drop_duplicates()
    _, raw, _, _ = load_data(DATA_FP)
    rows = []
    for _, row in raw[['year', 'submitted by']].dropna().iterrows():
        for a in _split_multi_value(row['submitted by'], delimiter=','):
            rows.append((a, int(row['year'])))
    sub = pd.DataFrame(rows, columns=['actor', 'year']).drop_duplicates()
    years = sub.groupby('actor')['year'].agg(first_year='min', last_year='max', active_years='nunique').reset_index()
    df = actor.merge(years, on='actor', how='left').dropna(subset=['active_years']).copy()
    df['dominant_region'] = df['dominant_region'].astype(int)
    df.to_csv(OUT_DATA, index=False)

    summary = (
        df.groupby('dominant_region')
        .agg(
            n_actors=('actor', 'size'),
            mean_active_years=('active_years', 'mean'),
            median_active_years=('active_years', 'median'),
            min_active_years=('active_years', 'min'),
            max_active_years=('active_years', 'max'),
            mean_first_year=('first_year', 'mean'),
            median_first_year=('first_year', 'median'),
        )
        .reset_index()
        .sort_values('dominant_region')
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    colors = {1: '#1f77b4', 2: '#2a9d8f', 3: '#d62728'}
    labels = ['Mode 1', 'Mode 2', 'Mode 3']
    xpos = np.arange(3)

    fig, axs = uplt.subplots(ncols=2, refwidth=3.3, refaspect=1.0, share=False)
    axs.format(abc='[A]', grid=True)

    metrics = [
        ('active_years', 'Active years in archive', 'Archive participation length'),
        ('first_year', 'First year in archive', 'Entry timing by mode'),
    ]
    for ax, (metric, ylabel, title) in zip(axs, metrics):
        ymax = -np.inf
        ymin = np.inf
        for idx, regime in enumerate([1, 2, 3]):
            vals = df.loc[df['dominant_region'] == regime, metric].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            local_rng = np.random.default_rng(9000 + idx)
            jitter = local_rng.uniform(-0.15, 0.15, size=vals.size)
            ax.scatter(np.full(vals.size, idx) + jitter, vals, s=12, c=colors[regime], alpha=0.5, edgecolor='none', zorder=2)
            q10, q25, q50, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.5, 0.75, 0.90])
            ymax = max(ymax, q90)
            ymin = min(ymin, q10)
            ax.vlines(idx, q10, q90, color='black', lw=1.0, alpha=0.8, zorder=3)
            ax.vlines(idx, q25, q75, color='black', lw=3.2, zorder=4)
            ax.hlines(q50, idx - 0.16, idx + 0.16, color='black', lw=1.2, zorder=5)
        pad = 2 if metric == 'active_years' else 1
        ax.format(title=title, ylabel=ylabel, xlabel='Dominant mode', xticks=xpos, xticklabels=labels, ylim=(ymin - pad, ymax + pad))

    fig.save(OUT_PDF)
    fig.save(OUT_PNG, transparent=False)


if __name__ == '__main__':
    main()
