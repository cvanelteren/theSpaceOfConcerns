from pathlib import Path

import ultraplot as uplt
import pandas as pd


def plot_sensitivity_curve(
    results_df: pd.DataFrame,
    output_pdf: str = "./figures/fig8_sensitivity_removal_space_full.pdf",
) -> None:
    """Render and save the fig8 sensitivity curve."""
    fig, ax = uplt.subplots()
    ax.plot(results_df.n / len(results_df), results_df.spectral_distance)

    for _, row in results_df.iterrows():
        ax.text(
            row.n / len(results_df),
            row.spectral_distance + 0.1,
            row.country,
            fontsize=3.5,
            rotation=90,
            ha="left",
            va="bottom",
        )

    ax.format(
        xlabel="Fraction of removals",
        ylabel=r"Spectral distance to full space ($\sqrt{\langle (\phi - \phi_{\mathrm{full}})^2 \rangle}$)",
        xlim=(0, 1.05),
        ylim=(0, 1.55),
    )
    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)

