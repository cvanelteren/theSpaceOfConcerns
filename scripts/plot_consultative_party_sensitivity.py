#!/usr/bin/env python3
"""Plot the Consultative-Party-only sensitivity analysis for the SI."""

from pathlib import Path

import pandas as pd
import ultraplot as uplt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "output" / "consultative_party_sensitivity"
OUTPUT = ROOT / "figures" / "figS_consultative_party_sensitivity.pdf"

TEAL = "#147D78"
NAVY = "#263B59"
OCHRE = "#C9872B"
GRAY = "#879099"


def main() -> None:
    pairs = pd.read_csv(INPUT / "map_pairwise_comparison.csv")
    predictions = pd.read_csv(INPUT / "prediction_type_summary.csv")
    predictions = predictions[
        predictions["comparison"].eq("direct + nearby attention vs output history")
    ].set_index("instrument").loc[["Measure", "Decision", "Resolution"]]

    fig, axs = uplt.subplots(ncols=3, refwidth=1.65, share=False)

    ax = axs[0]
    ax.scatter(
        pairs["phi_all_actors"],
        pairs["phi_consultative_only"],
        s=10,
        color=TEAL,
        alpha=0.38,
        edgecolor="none",
    )
    limit = max(pairs["phi_all_actors"].max(), pairs["phi_consultative_only"].max())
    ax.plot([0, limit], [0, limit], color=GRAY, lw=1, linestyle="--")
    ax.text(0.04, 0.93, r"Pearson $r=0.742$", transform="axes", fontsize=8)
    ax.format(
        title="Concern-map agreement",
        xlabel=r"All actors: proximity $\phi$",
        ylabel=r"Consultative Parties: proximity $\phi$",
        xlim=(0, limit * 1.03),
        ylim=(0, limit * 1.03),
        grid=False,
    )

    ax = axs[1]
    labels = ["All actors", "Consultative Parties"]
    estimates = [0.817, 0.900]
    lows = [0.783, 0.860]
    highs = [0.853, 0.942]
    colors = [NAVY, TEAL]
    for y, estimate, low, high, color in zip(range(2), estimates, lows, highs, colors):
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            markersize=5,
            capsize=3,
            lw=1.5,
        )
    ax.axvline(1, color=GRAY, lw=1, linestyle="--")
    ax.format(
        title="Local portfolio development",
        xlabel=r"Odds ratio per $0.1$ farther",
        yticks=range(2),
        yticklabels=labels,
        ylim=(-0.55, 1.55),
        xlim=(0.75, 1.02),
        grid=False,
    )

    ax = axs[2]
    instrument_colors = [OCHRE, NAVY, TEAL]
    for y, (instrument, row), color in zip(
        range(3), predictions.iterrows(), instrument_colors
    ):
        estimate = row["mean_difference"]
        low = row["bootstrap_low"]
        high = row["bootstrap_high"]
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            markersize=5,
            capsize=3,
            lw=1.5,
        )
    ax.axvline(0, color=GRAY, lw=1, linestyle="--")
    ax.format(
        title="Adopted-text prediction",
        xlabel="Change in prediction score",
        yticks=range(3),
        yticklabels=predictions.index.tolist(),
        ylim=(-0.55, 2.55),
        xlim=(-0.11, 0.06),
        grid=False,
    )

    fig.format(abc="a", abcloc="ul", fontsize=9)
    fig.save(OUTPUT, dpi=600)


if __name__ == "__main__":
    main()
