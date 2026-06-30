# %%
import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pathlib import Path
from tqdm import tqdm

from utils import (
    extract_unique_countries,
    extract_unique_topics,
    generate_interaction_matrix,
    get_rca,
    load_data,
    load_flag,
)

# Configuration
DATA_PATH = "./antarctic-database-go/data/processed/document-summary.parquet"
WINDOW_SIZE = 5
RCA_THRESHOLD = 1.0
TOPIC_EMERGENCE_PERCENTILE = 0.15  # Year when 5% of total topic volume is reached
MIN_TOPIC_VOLUME = 10  # Minimum total papers to consider a topic


def calculate_rolling_rca(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Computes RCA for each country-topic pair over a rolling window of years.
    Returns a DataFrame with index (Year, Country) and columns (Topics).
    """
    df = df.dropna(subset=["year"])
    years = sorted(df["year"].unique())
    if not years:
        return pd.DataFrame()

    min_year = int(min(years))
    max_year = int(max(years))

    records = []

    print("Calculating rolling RCA...")
    for year in tqdm(range(min_year + window, max_year + 1)):
        # Define window
        mask = (df["year"] > (year - window)) & (df["year"] <= year)
        df_window = df.loc[mask]

        if df_window.empty:
            continue

        countries = extract_unique_countries(df_window)
        # Use topics active in the window
        topics = extract_unique_topics(df_window)

        if not countries or not topics:
            continue

        counts_df = generate_interaction_matrix(df_window, countries, topics)
        rca_df = get_rca(counts_df)

        # Melt to long format: Year, Country, Topic, RCA
        # RCA matrix has Rows=Topics, Cols=Countries
        rca_df.index.name = "topic"
        rca_df = rca_df.reset_index()
        melted = rca_df.melt(id_vars="topic", var_name="country", value_name="rca")
        melted["year"] = year
        records.append(melted)

    if not records:
        return pd.DataFrame()

    full_rca = pd.concat(records, ignore_index=True)
    return full_rca


# 1. Load Data
counts_df, submitted_df, countries, topics = load_data(DATA_PATH)

if "year" not in submitted_df.columns and "meeting year" in submitted_df.columns:
    submitted_df["year"] = submitted_df["meeting year"]

# Ensure year is numeric
submitted_df["year"] = pd.to_numeric(submitted_df["year"], errors="coerce")
submitted_df = submitted_df.dropna(subset=["year"])

# 2. Determine Topic Emergence Years
# Explode categories to handle multiple topics per paper
# We must replicate the cleaning logic from utils.extract_unique_topics to match keys
exploded_df = submitted_df.dropna(subset=["category"]).copy()
exploded_df["category"] = exploded_df["category"].astype(str)
exploded_df = exploded_df.assign(
    category=exploded_df["category"].str.split("\t")
).explode("category")

# Apply cleaning
exploded_df["category"] = (
    exploded_df["category"].str.replace("envirom", "environ").str.replace("_", " ")
)
exploded_df.loc[exploded_df["category"] == "ALL", "category"] = "Other"

# Count total papers per topic per year
topic_counts = (
    exploded_df.groupby(["year", "category"]).size().reset_index(name="count")
)

topic_stats = []
for topic in topics:
    t_data = topic_counts[topic_counts["category"] == topic].sort_values("year")
    total_vol = t_data["count"].sum()

    if total_vol < MIN_TOPIC_VOLUME:
        continue

    # Calculate cumulative sum
    t_data["cumsum"] = t_data["count"].cumsum()
    t_data["pct"] = t_data["cumsum"] / total_vol

    # Find year where it crosses threshold
    emergence_year = t_data.loc[
        t_data["pct"] >= TOPIC_EMERGENCE_PERCENTILE, "year"
    ].min()

    topic_stats.append(
        {
            "topic": topic,
            "total_volume": total_vol,
            "emergence_year": emergence_year,
        }
    )

topic_stats_df = pd.DataFrame(topic_stats).set_index("topic")
print(f"Identified emergence years for {len(topic_stats_df)} topics.")

# 3. Calculate Country Entry Years via Rolling RCA
rca_long = calculate_rolling_rca(submitted_df, window=WINDOW_SIZE)

if rca_long.empty:
    print("No RCA data calculated. Exiting.")

# Filter for "Specialization" (strictly above threshold)
specialized = rca_long[rca_long["rca"] > RCA_THRESHOLD].copy()

# Find first year of specialization for each country-topic
entry_years = specialized.groupby(["country", "topic"])["year"].min().reset_index()
entry_years.rename(columns={"year": "entry_year"}, inplace=True)

# Calculate Intensity (Mean RCA) per topic
topic_intensity = specialized.groupby(["country", "topic"])["rca"].mean().reset_index()
entry_years = entry_years.merge(topic_intensity, on=["country", "topic"])

# 4. Compute Lags and Pioneer Index
# Merge with topic emergence

# Determine Country Start Year (Accession/First Activity)
parties_exploded = submitted_df[["year", "submitted by"]].explode("submitted by")
country_start_years = (
    parties_exploded.groupby("submitted by")["year"].min().reset_index()
)
country_start_years.columns = ["country", "start_year"]

analysis_df = entry_years.merge(
    topic_stats_df[["emergence_year"]], left_on="topic", right_index=True
)
analysis_df = analysis_df.merge(country_start_years, on="country", how="left")

# Lag = Entry Year - Emergence Year
# Adjusted Lag (Relative) = Entry Year - max(Emergence Year, Start Year)
# This measures responsiveness: did they adopt it as soon as they could?

analysis_df["effective_start"] = analysis_df[["emergence_year", "start_year"]].max(
    axis=1
)
analysis_df["relative_lag"] = analysis_df["entry_year"] - analysis_df["effective_start"]
analysis_df["absolute_lag"] = analysis_df["entry_year"] - analysis_df["emergence_year"]

# Use Relative Lag for the index
analysis_df["lag"] = analysis_df["relative_lag"]

# Compute aggregates per country
country_stats = (
    analysis_df.groupby("country")
    .agg(
        pioneer_score=("lag", "mean"),  # Lower is better (more pioneering)
        topics_adopted=("topic", "count"),
        first_activity=("start_year", "min"),
        abs_pioneer_score=("absolute_lag", "mean"),
        intensity=("rca", "mean"),
    )
    .reset_index()
)

# Exclude ATS from the analysis as it's an institution, not a country
country_stats = country_stats[country_stats["country"] != "ATS"].copy()

# Invert score so Higher = More Pioneering (Negative Lag)
country_stats["pioneer_index"] = -country_stats["pioneer_score"]
country_stats["abs_pioneer_index"] = -country_stats["abs_pioneer_score"]

print("Checking for Switzerland in country_stats:")
print(
    country_stats[country_stats["country"] == "Switzerland"][
        ["country", "pioneer_index", "intensity", "topics_adopted"]
    ]
)

# Filter out countries with very few topics to avoid noise
# country_stats = country_stats[country_stats["topics_adopted"] >= 5]

if country_stats.empty:
    print("Country stats is empty after filtering (topics_adopted >= 5).")

print("Top Pioneers (Highest Pioneer Index):")
print(
    country_stats.sort_values("pioneer_index", ascending=False).head(10)[
        ["country", "pioneer_index", "intensity", "topics_adopted"]
    ]
)

# Identify Niche Pioneers (Above average speed, below average breadth)
mean_breadth = country_stats["topics_adopted"].mean()
mean_pioneer = country_stats["pioneer_index"].mean()

niche_pioneers = country_stats[
    (country_stats["topics_adopted"] < mean_breadth)
    & (country_stats["pioneer_index"] > mean_pioneer)
]

print("\n--- Niche Pioneers (High Speed, Low Breadth) ---")
print(
    niche_pioneers.sort_values("pioneer_index", ascending=False).head(10)[
        ["country", "pioneer_index", "intensity", "topics_adopted"]
    ]
)

# 5. Analyze "Pincer" Hypothesis: What are they pioneering?
print("\n--- Pincer Analysis: Who pioneers what? ---")

# Define Broad Pioneers (High Speed, High Breadth)
broad_pioneers_list = country_stats[
    (country_stats["topics_adopted"] >= mean_breadth)
    & (country_stats["pioneer_index"] > mean_pioneer)
]["country"].tolist()

niche_pioneers_list = niche_pioneers["country"].tolist()

# Get pioneering instances (Lag < 0) for each group
pioneering_events = analysis_df[analysis_df["lag"] < 0].copy()

broad_topics = (
    pioneering_events[pioneering_events["country"].isin(broad_pioneers_list)]["topic"]
    .value_counts()
    .head(10)
)
niche_topics = (
    pioneering_events[pioneering_events["country"].isin(niche_pioneers_list)]["topic"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Topics Pioneered by Broad Pioneers (Hegemons):")
print(broad_topics)

print("\nTop 10 Topics Pioneered by Niche Pioneers (Specialists):")
print(niche_topics)

print("\n--- Specific Pioneering Acts by Niche Pioneers ---")
niche_events = (
    pioneering_events[pioneering_events["country"].isin(niche_pioneers_list)]
    .sort_values(["country", "lag"])
    .loc[:, ["country", "topic", "lag", "entry_year"]]
)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print(niche_events)

# Check specific country: Switzerland
target_country = "Switzerland"
if target_country in country_stats["country"].values:
    print(f"\n--- {target_country} Analysis ---")
    stats = country_stats[country_stats["country"] == target_country].iloc[0]
    print(stats)

    # Show what they pioneered or followed
    print(f"\n{target_country} Topic Portfolio (sorted by Lag):")
    portfolio = analysis_df[analysis_df["country"] == target_country].sort_values("lag")
    print(portfolio[["topic", "lag", "entry_year", "emergence_year"]])
else:
    print(
        f"\n{target_country} not found in country stats (possibly filtered out due to low topic count)."
    )

# %%
# --- Data Merging and Strategic Group Analysis ---

print("\n--- Pre-calculating Strategic Groups & Fitting Curve---")

# First, fit the quadratic curve to the full dataset to get the trendline
x_full = country_stats["topics_adopted"]
y_full = country_stats["pioneer_index"]

# Ensure there's data to fit
if not x_full.empty and not y_full.empty:
    coeffs = np.polyfit(x_full, y_full, 2)
    p = np.poly1d(coeffs)
    # Calculate R-squared for the full model
    y_pred_full = p(x_full)
    r_squared = y_full.corr(pd.Series(y_pred_full, index=y_full.index)) ** 2
    print(f"R-squared for quadratic fit on full data: {r_squared:.3f}")


# Load GDP per capita data
gdp_fp = "gdp-per-capita-worldbank.csv"
color_map = {
    "Over-performer": "blue",
    "Under-performer": "red",
    "On-trend": "gray7",
}

if Path(gdp_fp).exists():
    try:
        gdp_df = pd.read_csv(gdp_fp)

        # Rename columns
        gdp_df.rename(
            columns={
                "Entity": "country",
                "GDP per capita, PPP (constant 2021 international $)": "gdp_per_capita",
            },
            inplace=True,
        )
        required_cols = {"country", "gdp_per_capita", "Year"}
        if not required_cols.issubset(gdp_df.columns):
            missing = sorted(required_cols - set(gdp_df.columns))
            raise ValueError(f"GDP file missing expected columns: {missing}")

        # Get most recent GDP
        gdp_df = gdp_df.sort_values("Year").drop_duplicates("country", keep="last")
        gdp_recent = gdp_df[["country", "gdp_per_capita"]]

        # Merge with Country Stats
        merged_df = country_stats.merge(gdp_recent, on="country", how="left")

        # Calculate residuals only for countries with GDP data
        merged_df_gdp = merged_df.dropna(subset=["gdp_per_capita"]).copy()
        x_for_prediction = merged_df_gdp["topics_adopted"]
        merged_df_gdp["predicted_pioneer_index"] = p(x_for_prediction)
        merged_df_gdp["residual"] = (
            merged_df_gdp["pioneer_index"] - merged_df_gdp["predicted_pioneer_index"]
        )

        # Define Strategic Groups based on residuals
        upper_thresh = merged_df_gdp["residual"].quantile(0.67)
        lower_thresh = merged_df_gdp["residual"].quantile(0.33)

        def get_strategic_group(residual):
            if residual > upper_thresh:
                return "Over-performer"
            elif residual < lower_thresh:
                return "Under-performer"
            else:
                return "On-trend"

        merged_df_gdp["strategic_group"] = merged_df_gdp["residual"].apply(
            get_strategic_group
        )

        # Add this info back to the main dataframe
        merged_df = merged_df.merge(
            merged_df_gdp[["country", "strategic_group", "residual"]],
            on="country",
            how="left",
        )
        merged_df["color"] = merged_df["strategic_group"].map(color_map).fillna(
            "gray7"
        )
    except Exception as exc:
        print(f"Warning: failed to parse GDP file {gdp_fp} ({exc}). Skipping GDP-based residuals.")
        merged_df = country_stats.copy()
        merged_df["gdp_per_capita"] = np.nan
        merged_df["residual"] = np.nan
        merged_df["strategic_group"] = "On-trend"
        merged_df["color"] = color_map["On-trend"]
else:
    print(f"Warning: GDP file not found at {gdp_fp}. Skipping GDP-based residuals.")
    merged_df = country_stats.copy()
    merged_df["gdp_per_capita"] = np.nan
    merged_df["residual"] = np.nan
    merged_df["strategic_group"] = "On-trend"
    merged_df["color"] = color_map["On-trend"]


# %%
# 6. Visualization (with Strategic Group Coloring)
fig, ax = uplt.subplots()

x = merged_df["topics_adopted"]
y = merged_df["pioneer_index"]

# Plot the fitted curve
if "p" in locals():
    x_fit = np.linspace(
        country_stats["topics_adopted"].min(),
        country_stats["topics_adopted"].max(),
        200,
    )
    y_fit = p(x_fit)
    ax.plot(x_fit, y_fit, color="k", lw=1.5, ls="--")


# Add flags/text labels with colored frames or text
for _, row in merged_df.iterrows():
    flag = load_flag(row["country"])
    if flag is not None:
        zoom = 9 / max(flag.shape[:2])
        imagebox = OffsetImage(flag, zoom=zoom)
        # Add a colored bounding box instead of a scatter point
        bboxprops = dict(
            edgecolor=row["color"],
            boxstyle="square,pad=0.0",
            lw=1.5,
        )
        ab = AnnotationBbox(
            imagebox,
            (row["topics_adopted"], row["pioneer_index"]),
            frameon=True,
            bboxprops=bboxprops,
            zorder=100,
        )
        ax.add_artist(ab)
    else:
        # For text labels, color the text directly
        ax.text(
            row["topics_adopted"],
            row["pioneer_index"],
            row["country"],
            fontsize=7,
            color=row["color"],
            weight="bold",
            zorder=101,
        )


# Formatting, mean lines, and quadrant labels (from user's current version)
ax.axhline(y_full.mean(), color="gray5", linestyle="--", lw=1)
ax.axvline(x_full.mean(), color="gray5", linestyle="--", lw=1)
kw = dict(
    xy=(x_full.mean(), y_full.mean()),
    xycoords="data",
    textcoords="data",
    color="gray5",
    fontsize=5.2,
)
ax.annotate(
    "Broad Pioneers\n(Established Powers)",
    xytext=(25, 10),
    ha="center",
    va="center",
    weight="bold",
    **kw,
)
ax.annotate(
    "Niche Pioneers\n(Specialized Innovators)",
    xytext=(7.5, 10),
    ha="center",
    va="center",
    weight="bold",
    **kw,
)
ax.annotate(
    "Mainstream Adopters\n(Late Entry)",
    xytext=(25, -14.25),
    ha="center",
    va="center",
    weight="bold",
    **kw,
)
ax.annotate(
    "Peripheral\n(Selective Engagement)",
    xytext=(7.5, -14.25),
    ha="center",
    va="center",
    weight="bold",
    **kw,
)

ax.format(
    xlabel="Breadth (Number of Specialized Topics)",
    ylabel="Relative Pioneer Index (Years vs. Opportunity)",
    grid=True,
)
ax.set_xlim(-2, 45)
ax.set_ylim(-20, 14)


# Custom Legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Over-performer",
        markerfacecolor="blue",
        markersize=8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Under-performer",
        markerfacecolor="red",
        markersize=8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"On-trend",
        markerfacecolor="gray7",
        markersize=8,
    ),
    Line2D(
        [0], [0], color="k", lw=1.5, ls="--", label=f"Trendline (R²={r_squared:.2f})"
    ),
]
ax.legend(
    handles=legend_elements,
    loc="t",
    frameon=0,
    ncols=2,
)


try:
    fig.save("figures/pioneer_index_strategic.pdf")
    fig.save("figures/pioneer_index_strategic.png", transparent=True)
    print("\nStrategic plot saved to figures/pioneer_index_strategic.png")
except Exception as e:
    print(f"Error saving strategic figure: {e}")


# %%
# 7. Residuals vs. GDP Per Capita Analysis (with consistent coloring)
df_res_plot = merged_df.dropna(subset=["gdp_per_capita", "residual"])
if not df_res_plot.empty:
    fig_res, ax_res = uplt.subplots()

    x_res = df_res_plot["gdp_per_capita"]
    y_res = df_res_plot["residual"]

    ax_res.scatter(x_res, y_res)
    ax_res.axhline(0, color="gray5", linestyle="--", lw=1)
    ax_res.format(
        xlabel="GDP per Capita (PPP, constant 2021 international $)",
        ylabel="Pioneer Index Residual (Actual - Predicted)",
        grid=True,
        # xscale="log",
    )

    # Use flags with colored borders, similar to the main plot
    for _, row in df_res_plot.iterrows():
        flag = load_flag(row["country"])
        if flag is not None:
            zoom = 12 / max(flag.shape[:2])
            imagebox = OffsetImage(flag, zoom=zoom)
            bboxprops = dict(
                edgecolor=row["color"],
                boxstyle="square,pad=0.0",
                lw=1.5,
            )
            ab = AnnotationBbox(
                imagebox,
                (row["gdp_per_capita"], row["residual"]),
                frameon=True,
                bboxprops=bboxprops,
                zorder=100,
            )
            ax_res.add_artist(ab)
        else:
            # For text labels, color the text directly
            ax_res.text(
                row["gdp_per_capita"],
                row["residual"],
                row["country"],
                fontsize=7,
                color=row["color"],
                weight="bold",
                zorder=101,
            )

    try:
        fig_res.save("figures/fig10_pioneer_residuals_vs_gdp.pdf")
        fig_res.save("figures/fig10_pioneer_residuals_vs_gdp.png", transparent=True)
        print(
            "Color-coded residuals plot saved to figures/fig10_pioneer_residuals_vs_gdp.png"
        )
    except Exception as e:
        print(f"Error saving residuals figure: {e}")


# %%
# 8. Save Final Pioneer Analysis Data

print("\n--- Saving final pioneer analysis data ---")
if "merged_df" in locals():
    pioneer_output_path = "output/pioneer_analysis_results.csv"
    merged_df.to_csv(pioneer_output_path, index=False)
    print(f"✓ Pioneer analysis results saved to {pioneer_output_path}")
else:
    print("Could not find 'merged_df' to save.")
