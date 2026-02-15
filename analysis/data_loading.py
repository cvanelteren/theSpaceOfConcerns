from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from utils import load_data


DEFAULT_DATA_PATHS = (
    Path("antarctic-database-go/data/processed/document-summary.parquet"),
    Path("antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"),
    Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
    Path("document-summary.csv"),
)


def load_submitted_with_fallback(paths: Sequence[Path] = DEFAULT_DATA_PATHS) -> pd.DataFrame:
    """Load the canonical submitted dataframe from the first valid data source."""
    last_error = None
    for path in paths:
        if not path.exists():
            continue
        try:
            _, submitted_df, _, _ = load_data(str(path))
            return submitted_df
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Failed to load {path}: {exc}")
    if last_error is not None:
        raise RuntimeError("Could not load submitted dataframe from fallback paths.") from last_error
    raise FileNotFoundError("No known data path was found.")

