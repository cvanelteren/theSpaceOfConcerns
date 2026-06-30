import pandas as pd

from analysis.fig4_weights import save_mean_archetype_weights


def test_save_mean_archetype_weights_aggregates_and_writes_csv(tmp_path):
    temporal_df_clean_best = pd.DataFrame(
        {
            "agent": ["Australia", "Australia", "Chile"],
            "archetype_weight_1": [0.2, 0.4, 0.9],
            "archetype_weight_2": [0.3, 0.5, 0.1],
            "archetype_weight_3": [0.5, 0.1, 0.0],
        }
    )
    output_path = tmp_path / "nested" / "archetype_mean_weights.csv"

    result = save_mean_archetype_weights(
        temporal_df_clean_best, output_path=str(output_path)
    )

    expected = pd.DataFrame(
        {
            "country": ["Australia", "Chile"],
            "archetype_weight_1": [0.3, 0.9],
            "archetype_weight_2": [0.4, 0.1],
            "archetype_weight_3": [0.3, 0.0],
        }
    )

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(pd.read_csv(output_path), expected)
