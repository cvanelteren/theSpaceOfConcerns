"""Supplementary Figure S10. Removes actors greedily and measures how much the concern space changes. Checks that the space is jointly produced rather than dominated by one actor."""

# %%
"""
Figure 8 pipeline.

Separates analysis from plotting:
- analysis.fig8_sensitivity: data transforms and sensitivity computation
- plots.fig8_sensitivity: final figure rendering
"""

from analysis.data_loading import DEFAULT_DATA_PATHS, load_submitted_with_fallback
from analysis.fig8_sensitivity import build_country_topic_matrix, sensitivity_analysis
from plots.fig8_sensitivity import plot_sensitivity_curve


submitted_df = load_submitted_with_fallback(DEFAULT_DATA_PATHS)
country_topic_matrix = build_country_topic_matrix(submitted_df)
results_df = sensitivity_analysis(country_topic_matrix, threshold=1.0)
results_out = results_df[["country", "n", "rmse"]].copy()
results_out["fraction_removed"] = results_out["n"] / len(results_df)
results_out.to_csv("./output/fig14_space_member_removal_sensitivity.csv", index=False)
plot_sensitivity_curve(results_df, "./figures/figS10_space_member_removal.pdf")
plot_sensitivity_curve(results_df, "./figures/figS10_space_member_removal.png")
