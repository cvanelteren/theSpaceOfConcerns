import numpy as np
import pandas as pd

from analysis import fig8_sensitivity


class DummyTqdm:
    def __init__(self, total):
        self.total = total
        self.updated = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount):
        self.updated += amount


def test_build_country_topic_matrix_expands_multivalue_cells_and_deduplicates():
    submitted = pd.DataFrame(
        {
            "paper id": [1, 1, 2],
            "submitted by": [
                "Australia, Chile",
                "Australia, Chile",
                "Chile",
            ],
            "category": [
                "Climate change\tALL",
                "Climate change",
                "envirom monitoring",
            ],
        }
    )

    result = fig8_sensitivity.build_country_topic_matrix(submitted)

    expected = pd.DataFrame(
        {
            "Australia": [1, 0],
            "Chile": [1, 1],
        },
        index=["Climate change", "environ monitoring"],
    )
    expected.columns.name = "submitted by"
    expected.index.name = "category"

    pd.testing.assert_frame_equal(result, expected)


def test_proximity_rmse_only_uses_upper_triangle():
    full = np.array(
        [
            [0.0, 1.0, 2.0],
            [100.0, 0.0, 3.0],
            [200.0, 300.0, 0.0],
        ]
    )
    reduced = np.array(
        [
            [0.0, 4.0, 1.0],
            [-999.0, 0.0, 7.0],
            [-999.0, -999.0, 0.0],
        ]
    )

    result = fig8_sensitivity.proximity_rmse(full, reduced)

    assert np.isclose(result, np.sqrt(26.0 / 3.0))


def test_proximity_rmse_returns_zero_for_single_topic():
    value = fig8_sensitivity.proximity_rmse(np.array([[0.0]]), np.array([[7.0]]))
    assert value == 0.0


def test_sensitivity_analysis_drops_largest_perturbation_each_round(monkeypatch):
    export_data = pd.DataFrame(
        {
            "A": [2, 1],
            "B": [1, 3],
            "C": [4, 5],
        },
        index=["topic_1", "topic_2"],
    )
    weights = {"A": 5.0, "B": 2.0, "C": 1.0}

    def fake_get_rca(df):
        return df.copy()

    def fake_compute_product_space(df, threshold):
        score = sum(weights[column] for column in df.columns)
        matrix = np.array([[0.0, score], [score, 0.0]])
        return pd.DataFrame(matrix, index=df.index, columns=df.index)

    monkeypatch.setattr(fig8_sensitivity, "get_rca", fake_get_rca)
    monkeypatch.setattr(fig8_sensitivity, "compute_product_space", fake_compute_product_space)
    monkeypatch.setattr(fig8_sensitivity, "tqdm", DummyTqdm)

    result = fig8_sensitivity.sensitivity_analysis(export_data)

    assert list(result["country"]) == ["A", "B"]
    assert list(result["n"]) == [1, 2]
    assert list(result["rmse"]) == [5.0, 7.0]
    assert isinstance(result.loc[0, "rca"], pd.DataFrame)
    assert list(result.loc[0, "rca"].columns) == ["B", "C"]
    assert isinstance(result.loc[0, "phi"], pd.DataFrame)
    assert list(result.loc[0, "phi"].index) == ["topic_1", "topic_2"]
