"""Main-text Figure 5: who reaches emerging concerns first.

This is the closing beat. Figures 1--4 establish that the space exists, that
actors hold distinct positions in it, that movement is local, and that a
evolving-portfolio rule reproduces the record. The question left is the one the
abstract opens on: if where you can go is bounded by where you already are,
who gets to new concerns early?

  A  earliness against portfolio breadth
  B  earliness against how strongly anchored that portfolio is
  C  the same four terms as one fitted model

The answer the panels have to carry is two-sided, and the figure is built so
that neither side can be read without the other. Breadth and anchoring predict
early entry; which mode an actor occupies does not. Panels A and B therefore
colour every actor by its dominant mode -- if modes mattered, the three colours
would separate vertically, and they visibly do not -- while panel C puts the
two mode contrasts on the same axis as the two portfolio terms, where their
intervals straddle zero and the portfolio terms do not.

Colouring by mode in a figure whose finding is that mode does not matter is
deliberate. A reader who is shown only the breadth relationship has to take the
null on trust; showing the modes intermixed makes the null visible.

The model is cross-sectional and its predictors are entangled among the small
set of long-established actors, so panel C is an association, not a decomposition
of cause. The caption says so, and no panel here implies a time ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import ultraplot as uplt
from matplotlib.lines import Line2D

import figstyle

PIONEER_CSV = Path("output/fig05_pioneer_regime_position_data.csv")

OUT_PNG = Path("figures/fig05_who_enters_first.png")
OUT_PDF = Path("figures/fig05_who_enters_first.pdf")

# Actors worth naming: the extremes on each axis plus the ones the Results text
# discusses by name, so a reader can locate the cases the prose refers to.
LABEL_ACTORS = {
    "SCAR", "ASOC", "United States", "Australia", "New Zealand",
    "Russian Federation", "Ukraine", "Malaysia", "China", "IAATO",
}

TERM_LABELS = [
    ("z_breadth", "Portfolio breadth\n(per SD)"),
    ("z_anchor", "Portfolio anchoring\n(per SD)"),
    ("reg_2", "Compliance\nvs Coordination"),
    ("reg_3", "Strategy\nvs Coordination"),
]


def _load() -> pd.DataFrame:
    df = pd.read_csv(PIONEER_CSV)
    needed = {
        "country", "pioneer_index", "topics_adopted",
        "max_regime_share", "dominant_regime",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"pioneer data missing columns: {missing}")
    return df.dropna(subset=sorted(needed - {"country"}))


def _fit(df: pd.DataFrame):
    """Same specification as the published model, on standardized continuous terms.

    Standardizing breadth and anchoring is what lets all four terms share one
    axis in panel C: the two continuous coefficients become "per SD of the
    predictor" and the two dummies stay contrasts, both in units of the
    outcome. The unstandardized fit is identical in significance and R-squared.
    """
    work = df.copy()
    for source, target in (
        ("topics_adopted", "z_breadth"),
        ("max_regime_share", "z_anchor"),
    ):
        values = work[source].to_numpy(dtype=float)
        work[target] = (values - values.mean()) / values.std(ddof=0)

    dummies = pd.get_dummies(
        work["dominant_regime"].astype(int), prefix="reg", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([work[["z_breadth", "z_anchor"]], dummies], axis=1)
    )
    return sm.OLS(work["pioneer_index"].astype(float), design).fit(), work


def _scatter_panel(
    ax, work: pd.DataFrame, xcol: str, *, label_axis: str, trend: bool = True
) -> None:
    x = work[xcol].to_numpy(dtype=float)
    y = work["pioneer_index"].to_numpy(dtype=float)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)

    for mode_id, color in figstyle.MODE_COLORS.items():
        pick = work["dominant_regime"].astype(int).to_numpy() == mode_id
        ax.scatter(
            x[pick], y[pick], s=30, color=color, alpha=0.8,
            edgecolor="white", linewidth=0.5, zorder=3,
        )

    # One pooled trend, not three. Splitting it by mode would suggest the
    # slopes differ, which is the hypothesis the figure rejects.
    #
    # A LOWESS smoother, not a straight line. A pooled OLS line here asserted a
    # linear marginal relation that neither panel has: against topics entered
    # the linear slope is flat and insignificant (R^2=0.02, p=0.22) while a
    # quadratic term is decisive (R^2=0.30, p<1e-4) and the rank correlation is
    # nil (rho=-0.01), so the straight line drew a trend that does not exist;
    # against dominant-mode share the association is monotonic but weak
    # (rho=0.31, p=0.01) and a quadratic adds nothing. The smoother shows the
    # shape each panel actually has without committing to a functional form,
    # and the model that the paper reports is panel C, not these margins.
    if trend:
        smoothed = lowess(y, x, frac=0.75, return_sorted=True)
        ax.plot(
            smoothed[:, 0], smoothed[:, 1],
            color=figstyle.PRIMARY, lw=1.4, zorder=4,
        )

    # Fix the limits before placing labels: they are laddered upward on
    # collision, so without a known ceiling the last few ran off the top, and
    # actors at the extremes of x had their names centred over the spine.
    x_pad = 0.06 * (x.max() - x.min())
    y_pad = 0.06 * (y.max() - y.min())
    x_min, x_max = x.min() - x_pad, x.max() + x_pad
    y_min, y_max = y.min() - y_pad, y.max() + 2.2 * y_pad
    ax.format(
        xlabel=label_axis, ylabel="Earliness of entry (pioneer index)",
        xlim=(x_min, x_max), ylim=(y_min, y_max),
    )

    # The named actors cluster in the high-breadth corner, so labels have to be
    # laddered off each other. Tolerances are taken from the data range rather
    # than hard-coded, since the two panels have different x units.
    x_tol = 0.10 * (x.max() - x.min())
    y_tol = 0.055 * (y.max() - y.min())
    named = work[work["country"].isin(LABEL_ACTORS)].sort_values(xcol)
    placed: list[tuple[float, float]] = []
    for _, row in named.iterrows():
        lx = float(row[xcol])
        ly = float(row["pioneer_index"]) + 0.35 * y_tol
        while any(abs(lx - px) < x_tol and abs(ly - py) < y_tol for px, py in placed):
            ly += y_tol
        placed.append((lx, ly))
        # Anchor away from whichever edge is close, and cap the ladder below
        # the top spine, so no name leaves the panel.
        edge = 0.16 * (x_max - x_min)
        if lx < x_min + edge:
            ha, tx = "left", max(lx, x_min + 0.02 * (x_max - x_min))
        elif lx > x_max - edge:
            ha, tx = "right", min(lx, x_max - 0.02 * (x_max - x_min))
        else:
            ha, tx = "center", lx
        ty = min(ly, y_max - 0.05 * (y_max - y_min))
        ax.text(
            tx, ty,
            str(row["country"]).replace("Russian Federation", "Russia"),
            ha=ha, va="bottom", fontsize=figstyle.FS_ANNOT,
            color=figstyle.TEXT, zorder=5,
        )


def _coefficient_panel(ax, model) -> None:
    conf = model.conf_int()
    positions = np.arange(len(TERM_LABELS))[::-1].astype(float)

    ax.axvline(0.0, color="0.45", lw=1.0, zorder=2)
    for pos, (term, _) in zip(positions, TERM_LABELS):
        lo, hi = float(conf.loc[term, 0]), float(conf.loc[term, 1])
        # Filled where the interval excludes zero, hollow where it does not:
        # the reader can sort the two portfolio terms from the two mode terms
        # without reading a p-value off the page.
        decisive = lo > 0 or hi < 0
        ax.plot([lo, hi], [pos, pos], color=figstyle.PRIMARY, lw=1.6,
                solid_capstyle="round", zorder=3)
        ax.scatter(
            [float(model.params[term])], [pos], s=58, zorder=4,
            color=figstyle.PRIMARY if decisive else "white",
            edgecolor=figstyle.PRIMARY, linewidth=1.3,
        )

    ax.format(
        yticks=positions,
        yticklabels=[label for _, label in TERM_LABELS],
        xlabel="Change in earliness of entry",
        ylabel="",
        ylim=(positions.min() - 0.6, positions.max() + 0.6),
    )
    for label in ax.get_yticklabels():
        label.set_color("0.25")


def build_figure() -> tuple[uplt.Figure, dict]:
    df = _load()
    model, work = _fit(df)

    fig, axs = uplt.subplots(
        ncols=3, share=0, refwidth=2.7, refaspect=1, wspace=(9.5, 13.0)
    )
    _scatter_panel(axs[0], work, "topics_adopted", label_axis="Topics ever entered")
    # No trend line against anchoring: the marginal association is weak
    # (rho=0.31, linear R^2=0.06) and a quadratic adds nothing (p=0.23), so any
    # curve drawn here would assert more shape than the scatter supports. The
    # anchoring result the paper reports is the adjusted coefficient in panel C.
    _scatter_panel(
        axs[1], work, "max_regime_share",
        label_axis="Share in dominant mode", trend=False,
    )
    _coefficient_panel(axs[2], model)

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   color=figstyle.MODE_COLORS[mode_id])
            for mode_id in (1, 2, 3)
        ],
        labels=[figstyle.MODE_GLOSS[mode_id] for mode_id in (1, 2, 3)],
        loc="b", frame=False, fontsize=figstyle.FS_LEGEND, ncols=3,
    )

    axs.format(abc="a", abcloc="ul", abcsize=figstyle.FS_PANEL, grid=False)
    figstyle.apply_typography(axs)

    summary = {
        "n_actors": int(len(work)),
        "r_squared": float(model.rsquared),
        "coefficients": {
            term: {
                "estimate": float(model.params[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(model.conf_int().loc[term, 0]),
                "ci_high": float(model.conf_int().loc[term, 1]),
            }
            for term, _ in TERM_LABELS
        },
    }
    return fig, summary


def main() -> None:
    fig, summary = build_figure()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PNG, dpi=300)
    fig.save(OUT_PDF)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
