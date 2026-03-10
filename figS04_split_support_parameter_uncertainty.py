"""Supplementary Figure S04. Reports approximate parameter uncertainty for the retain-and-enter model. Clarifies which parts of the mechanism are stable and which remain weakly identified."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ultraplot as uplt


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = PROJECT_ROOT / "figures"

PARAM_PATH = OUTPUT_DIR / "actor_topic_modeling_starter_split_param_uncertainty.csv"
OUT_PDF = FIG_DIR / "figS04_split_support_parameter_uncertainty.pdf"
OUT_PNG = FIG_DIR / "figS04_split_support_parameter_uncertainty.png"

STAGE_TITLES = {
    "allocation": "Allocation Stage",
    "entry": "Entry Stage",
    "retention": "Retention Stage",
}
STAGE_COLORS = {
    "allocation": "#2a9d55",
    "entry": "#4c72b0",
    "retention": "#c44e52",
}
TERM_LABELS = {
    "Persistence (rho)": r"Persistence ($\rho$)",
    "Local fit (beta)": r"Local fit ($\beta$)",
    "Popularity (gamma)": r"Popularity ($\gamma$)",
    "Entry intercept (delta)": r"Entry intercept ($\delta_{\mathrm{ent}}$)",
    "Local fit (beta_ent)": r"Local fit ($\beta_{\mathrm{ent}}$)",
    "Popularity (gamma_ent)": r"Popularity ($\gamma_{\mathrm{ent}}$)",
    "Retention intercept (delta)": r"Retention intercept ($\delta_{\mathrm{ret}}$)",
    "Prior share (lambda_ret)": r"Prior share ($\lambda_{\mathrm{ret}}$)",
    "Popularity (gamma_ret)": r"Popularity ($\gamma_{\mathrm{ret}}$)",
}


def build_figure() -> plt.Figure:
    df = pd.read_csv(PARAM_PATH)
    stages = ["allocation", "entry", "retention"]
    fig, axs = uplt.subplots(ncols=3, sharex=0, sharey=0, refnum=2)
    axs.format(abc="[A]")

    for ax, stage in zip(axs, stages):
        stage_df = df[df["stage"] == stage].copy()
        y = list(range(len(stage_df), 0, -1))
        color = STAGE_COLORS[stage]
        term_labels = [TERM_LABELS.get(term, term) for term in stage_df["term"].tolist()]

        ax.axvline(0.0, color="#999999", lw=1.0, linestyle="--", zorder=1)
        ax.hlines(
            y,
            stage_df["lower95"],
            stage_df["upper95"],
            color=color,
            lw=2.2,
            zorder=2,
        )
        ax.scatter(
            stage_df["estimate"],
            y,
            color=color,
            edgecolor="#222222",
            s=42,
            zorder=3,
        )
        ax.format(
            title=STAGE_TITLES[stage],
            xlabel="Estimate",
            ylocator=y,
            yticklabels=term_labels,
            ylim=(0.5, len(stage_df) + 0.5),
            grid=True,
        )

    fig.text(
        0.5,
        0.02,
        "Points show MLEs; horizontal lines show approximate 95% Wald intervals from the inverse-Hessian.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Wrote", OUT_PDF)
    print("Wrote", OUT_PNG)


if __name__ == "__main__":
    main()
