from pathlib import Path

import pandas as pd
import ultraplot as uplt


def plot_sensitivity_curve(
    results_df: pd.DataFrame,
    output_pdf: str = "./figures/fig8_sensitivity_removal_space_full.pdf",
) -> None:
    """Render and save the fig8 sensitivity curve."""
    fig, ax = uplt.subplots(share=0)
    x = results_df.n / len(results_df)
    ax.plot(
        x,
        results_df.rmse,
        color="#1f77b4",
        lw=2.0,
        marker="o",
        ms=2.4,
        label="RMSE to full space",
    )
    ymax = float(results_df.rmse.max())

    for _, row in results_df.iterrows():
        ax.text(
            row.n / len(results_df),
            row.rmse + 0.02 * max(ymax, 1e-6),
            row.country,
            fontsize=3.5,
            rotation=90,
            ha="left",
            va="bottom",
            clip_on=False,
        )

    ax.format(
        xlabel="Fraction of removals",
        ylabel=r"RMSE to full space over unique topic pairs",
        xlim=(0, 1.05),
        ylim=(0, min(1.02 * (ymax + 0.06), 1.35 * ymax + 0.02)),
    )
    ax.grid(alpha=0.18, color="black")
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_linewidth(1.0)
        ax.spines[spine_name].set_color("black")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_linewidth(0.8)
        ax.spines[spine_name].set_color("#777777")
    ax.tick_params(axis="both", width=1.0, length=4, color="black")
    ax.legend(
        loc="ul",
        bbox_to_anchor=(0.02, 0.98),
        frame=False,
        fontsize=8,
        handlelength=2.0,
    )
    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
