"""Appendix figure: mode-share trajectories for three exemplar actors.

These panels used to be C--E of the main positioning figure. They were moved
here because the message they carry -- movement is gradual and passes through
adjacent zones -- is made quantitatively by the transition matrix and by the
paired displacement test, so three hand-picked actors should not occupy main
text space. They are kept because the Results text describes these three
histories in words, and a reader should be able to check that description.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import ultraplot as uplt
from matplotlib.patches import Patch

import figstyle

PROFILE_TEMPLATE = "output/fig45_regime_member_window_profiles_year_window{window}.csv"

OUT_PNG = Path("figures/figS22_actor_trajectories.png")
OUT_PDF = Path("figures/figS22_actor_trajectories.pdf")

DEFAULT_ACTORS = ["Australia", "Netherlands", "Ukraine"]
WINDOW_SIZE = 5

REGION_COLORS = figstyle.MODE_COLORS
MODE_GLOSS = figstyle.MODE_GLOSS


def build_figure(actors: list[str], window_size: int) -> uplt.Figure:
    profiles = pd.read_csv(PROFILE_TEMPLATE.format(window=int(window_size)))
    missing = [a for a in actors if a not in set(profiles["member"])]
    if missing:
        raise ValueError(f"Actors absent from rolling profiles: {missing}")

    x_lo = float(profiles["period_end"].min())
    x_hi = float(profiles["period_end"].max())

    fig, axs = uplt.subplots(
        nrows=len(actors), ncols=1, refwidth=6.4, refaspect=4.5, hspace=1.2, share=1
    )
    for idx, (actor, ax) in enumerate(zip(actors, axs)):
        sub = profiles[profiles["member"] == actor].sort_values("period_end")
        ax.stackplot(
            sub["period_end"].to_numpy(dtype=float),
            [sub[f"region_{rid}_share"].to_numpy(dtype=float) for rid in (1, 2, 3)],
            colors=[REGION_COLORS[rid] for rid in (1, 2, 3)],
            alpha=0.85,
            baseline="zero",
        )
        ax.format(
            ylim=(0.0, 1.0),
            xlim=(x_lo, x_hi),
            ylocator=[0.5],
            yformatter=[actor],
            yminorticks=[],
            ylabel="",
            grid=False,
        )
        ax.tick_params(axis="y", length=0, labelsize=9)
        if idx < len(actors) - 1:
            ax.set_xticklabels([])

    axs[-1].format(xlabel="Window end year")
    axs[-1].xaxis.label.set_fontsize(9)
    fig.legend(
        handles=[Patch(facecolor=REGION_COLORS[rid], alpha=0.85) for rid in (1, 2, 3)],
        labels=[MODE_GLOSS[rid] for rid in (1, 2, 3)],
        loc="b",
        frame=False,
        fontsize=9,
        ncols=3,
    )
    axs.format(abc="[A]", abcloc="ul", abcsize=10)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default=",".join(DEFAULT_ACTORS))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    fig = build_figure([a.strip() for a in args.actors.split(",")], args.window_size)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PNG, dpi=300)
    fig.save(OUT_PDF)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
