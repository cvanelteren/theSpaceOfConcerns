from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_loading import load_submitted_with_fallback
from analysis.fig8_sensitivity import build_country_topic_matrix, sensitivity_analysis


class DummyTqdm:
    def __init__(self, total):
        self.total = total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount):
        return None


def _write_fixture_csv(tmp_path, rows):
    path = tmp_path / "fixture.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_analysis_pipeline_loads_csv_and_builds_matrix_from_real_utils(tmp_path):
    csv_path = _write_fixture_csv(
        tmp_path,
        [
            {
                "Meeting Year": 2001,
                "Parties": "Australia, Chile",
                "Category": "Climate change\tALL",
                "Paper ID": "P1",
                "Paper Name": "doc1",
            },
            {
                "Meeting Year": 2001,
                "Parties": "Australia, Chile",
                "Category": "Climate change",
                "Paper ID": "P1",
                "Paper Name": "doc1",
            },
            {
                "Meeting Year": 2002,
                "Parties": "Chile",
                "Category": "envirom monitoring",
                "Paper ID": "P2",
                "Paper Name": "doc2",
            },
            {
                "Meeting Year": 2003,
                "Parties": "Argentina",
                "Category": "other",
                "Paper ID": "P3",
                "Paper Name": "doc3",
            },
        ],
    )

    submitted = load_submitted_with_fallback(paths=(csv_path,))

    assert list(submitted["paper id"]) == ["P1", "P2", "P3"]
    assert list(submitted["category"].astype("string")) == [
        "Climate change",
        "environ monitoring",
        pd.NA,
    ]

    matrix = build_country_topic_matrix(submitted)
    expected = pd.DataFrame(
        {
            "Argentina": [0, 0],
            "Australia": [1, 0],
            "Chile": [1, 1],
        },
        index=["Climate change", "environ monitoring"],
    )
    expected.columns.name = "submitted by"
    expected.index.name = "category"

    pd.testing.assert_frame_equal(matrix, expected)


def test_analysis_pipeline_runs_real_sensitivity_from_csv_fixture(
    tmp_path, monkeypatch
):
    csv_path = _write_fixture_csv(
        tmp_path,
        [
            {
                "Meeting Year": 2001,
                "Parties": "A",
                "Category": "Topic 1",
                "Paper ID": "A1",
            },
            {
                "Meeting Year": 2001,
                "Parties": "A",
                "Category": "Topic 1",
                "Paper ID": "A2",
            },
            {
                "Meeting Year": 2001,
                "Parties": "A",
                "Category": "Topic 1",
                "Paper ID": "A3",
            },
            {
                "Meeting Year": 2001,
                "Parties": "B",
                "Category": "Topic 2",
                "Paper ID": "B1",
            },
            {
                "Meeting Year": 2001,
                "Parties": "B",
                "Category": "Topic 2",
                "Paper ID": "B2",
            },
            {
                "Meeting Year": 2001,
                "Parties": "B",
                "Category": "Topic 2",
                "Paper ID": "B3",
            },
            {
                "Meeting Year": 2001,
                "Parties": "C",
                "Category": "Topic 1",
                "Paper ID": "C1",
            },
            {
                "Meeting Year": 2001,
                "Parties": "C",
                "Category": "Topic 2",
                "Paper ID": "C2",
            },
            {
                "Meeting Year": 2001,
                "Parties": "C",
                "Category": "Topic 3",
                "Paper ID": "C3",
            },
            {
                "Meeting Year": 2001,
                "Parties": "C",
                "Category": "Topic 3",
                "Paper ID": "C4",
            },
            {
                "Meeting Year": 2001,
                "Parties": "C",
                "Category": "Topic 3",
                "Paper ID": "C5",
            },
        ],
    )
    monkeypatch.setattr("analysis.fig8_sensitivity.tqdm", DummyTqdm)

    submitted = load_submitted_with_fallback(paths=(csv_path,))
    matrix = build_country_topic_matrix(submitted)
    result = sensitivity_analysis(matrix)

    expected_matrix = pd.DataFrame(
        {
            "A": [3, 0, 0],
            "B": [0, 3, 0],
            "C": [1, 1, 3],
        },
        index=["Topic 1", "Topic 2", "Topic 3"],
    )
    expected_matrix.columns.name = "submitted by"
    expected_matrix.index.name = "category"

    pd.testing.assert_frame_equal(matrix, expected_matrix)
    assert list(result["country"]) == ["A", "B"]
    assert np.isclose(result.loc[0, "rmse"], 1 / np.sqrt(3))
    assert result.loc[1, "rmse"] == 0.0
    assert list(result["n"]) == [1, 2]
    assert list(result.loc[0, "phi"].index) == ["Topic 1", "Topic 2", "Topic 3"]
