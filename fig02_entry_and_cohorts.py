"""Figure 2. How actors adopt within the space of concerns.

(A) ENTRY     the singular, aggregate adoption probability against raw distance
              to the actor's prior portfolio -- the decay curve of the space.
              One line, no class split: topics near an existing portfolio are
              taken up far more often than distant ones, and adoption falls to
              near zero in the farthest distance bins.
(B) RETENTION decay curves with gap tolerances: Kaplan--Meier curves for how
              long a topic stays active after adoption, under one- to five-year
              gap tolerances. Once adopted, topics persist.

    PYTHONPATH=. micromamba run -n ultraplot-dev python fig02_entry_and_cohorts.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ultraplot as uplt

import fig03_local_portfolio_movement as old


OUT_PNG = Path("figures/fig02_entry_and_cohorts.png")
OUT_PDF = Path("figures/fig02_entry_and_cohorts.pdf")


def main() -> None:
    adoption_df, adoption_meta = old._load_adoption_panel()
    raw_distance = adoption_df["distance"].to_numpy(dtype=float)
    adoption_df = adoption_df.copy()
    adoption_df["plot_distance"] = raw_distance
    agg, event_rate = old._binned_adoption_curve(adoption_df)

    hazard_meta = old._load_json_or_empty(old.HAZARD_META_PATH)
    hazard_time_unit = str(hazard_meta.get("time_unit", "year")).strip().lower()
    hazard_window_size = int(
        hazard_meta.get(
            "window_size",
            hazard_meta.get("window_years", old.RETENTION_WINDOW_SIZE),
        )
    )
    retention_curves, retention_step_label = old._compute_retention_curves(
        time_unit=hazard_time_unit,
        window_size=hazard_window_size,
    )

    fig, axs = uplt.subplots(ncols=2, refwidth=3.5, refaspect=1, share=False,
                             wspace=("1.45in",))

    ax_a, ax_b = axs
    # The singular adoption line takes a distinct colour so it reads as the
    # aggregate decay curve rather than as one class among several.
    lower = agg["p"].to_numpy(dtype=float) - agg["ci_low"].to_numpy(dtype=float)
    upper = agg["ci_high"].to_numpy(dtype=float) - agg["p"].to_numpy(dtype=float)
    import figstyle
    line_color = figstyle.PRIMARY
    ax_a.errorbar(
        agg["x"], agg["p"], yerr=np.vstack([lower, upper]),
        fmt="o-", lw=2.0, ms=5, capsize=2.5, elinewidth=1.2,
        color=line_color, ecolor=line_color,
    )
    ax_a.format(
        xlabel=r"Raw distance to prior portfolio, $1-\max(\phi)$",
        ylabel="Adoption probability (topics not yet held)",
        title="New topics are adopted near prior portfolios",
        titlesize=10.5, labelsize=10, ticklabelsize=9.5,
    )
    old._plot_retention_panel(
        ax_b,
        retention_curves,
        step_label=retention_step_label,
    )

    for ax, label in ((ax_a, "a"), (ax_b, "b")):
        ax.text(
            0.025, 0.96, f"[{label}]",
            transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=11, zorder=10,
        )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PDF)
    fig.save(OUT_PNG, dpi=240)
    print(f"wrote {OUT_PDF}")

    print(f"\nA aggregate entry rate: {event_rate:.4f}")
    print(f"A binned adoption curve (n_bins={len(agg)}):")
    print(agg[["x", "n", "p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
