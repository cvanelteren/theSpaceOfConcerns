from pathlib import Path

import pandas as pd


def save_mean_archetype_weights(
    temporal_df_clean_best: pd.DataFrame, output_path: str = "output/archetype_mean_weights.csv"
) -> pd.DataFrame:
    """Persist mean archetype weights per actor to a CSV."""
    archetype_cols = ["archetype_weight_1", "archetype_weight_2", "archetype_weight_3"]
    mean_archetype_weights = (
        temporal_df_clean_best.groupby("agent")[archetype_cols].mean().reset_index()
    )
    mean_archetype_weights.rename(columns={"agent": "country"}, inplace=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mean_archetype_weights.to_csv(out, index=False)
    return mean_archetype_weights

