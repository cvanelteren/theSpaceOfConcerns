from pathlib import Path

import pandas as pd
import pytest

from analysis import data_loading


def test_load_submitted_with_fallback_returns_first_successful_dataframe(
    tmp_path, monkeypatch
):
    missing_path = tmp_path / "missing.parquet"
    broken_path = tmp_path / "broken.parquet"
    working_path = tmp_path / "working.parquet"
    broken_path.touch()
    working_path.touch()

    expected = pd.DataFrame(
        {"submitted by": ["Australia"], "category": ["Climate change"]}
    )
    calls = []

    def fake_load_data(path):
        path_obj = Path(path)
        calls.append(path_obj.name)
        if path_obj == broken_path:
            raise ValueError("broken file")
        if path_obj == working_path:
            return pd.DataFrame(), expected, set(), set()
        raise AssertionError(f"Unexpected path {path_obj}")

    monkeypatch.setattr(data_loading, "load_data", fake_load_data)

    result = data_loading.load_submitted_with_fallback(
        paths=(missing_path, broken_path, working_path)
    )

    pd.testing.assert_frame_equal(result, expected)
    assert calls == ["broken.parquet", "working.parquet"]


def test_load_submitted_with_fallback_raises_runtime_error_after_existing_paths_fail(
    tmp_path, monkeypatch
):
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    first_path.touch()
    second_path.touch()

    def fake_load_data(path):
        raise ValueError(f"could not read {Path(path).name}")

    monkeypatch.setattr(data_loading, "load_data", fake_load_data)

    with pytest.raises(
        RuntimeError, match="Could not load submitted dataframe from fallback paths."
    ) as excinfo:
        data_loading.load_submitted_with_fallback(paths=(first_path, second_path))

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert str(excinfo.value.__cause__) == "could not read second.parquet"


def test_load_submitted_with_fallback_raises_when_no_paths_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="No known data path was found."):
        data_loading.load_submitted_with_fallback(
            paths=(tmp_path / "missing-a.parquet", tmp_path / "missing-b.parquet")
        )
