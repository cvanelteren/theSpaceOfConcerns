from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_directed_attention_adoption import (
    TransitionHistory,
    path_kernel,
    row_normalize,
    strength_preserving_rewire,
    transition_operator,
)


def test_row_normalize_handles_empty_rows() -> None:
    values = np.array([[1.0, 1.0], [0.0, 0.0]])
    normalized = row_normalize(values)
    np.testing.assert_allclose(normalized[0], [0.5, 0.5])
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])


def test_transition_operator_excludes_focal_cutoff() -> None:
    record = {
        "end_meeting": 10,
        "exposures": np.array([[0.0, 2.0], [2.0, 0.0]]),
        "entries": np.array([[0.0, 2.0], [0.0, 0.0]]),
        "target_exposures": np.array([2.0, 2.0]),
        "target_entries": np.array([0.0, 2.0]),
    }
    history = TransitionHistory(("a", "b"), (9, 10), (record,))
    before, before_prevalence = transition_operator(history, 10)
    after, after_prevalence = transition_operator(history, 11)
    np.testing.assert_allclose(before, 0.0)
    assert after[0, 1] == 1.0
    assert not np.allclose(before_prevalence, after_prevalence)


def test_path_kernel_excludes_self_exposure() -> None:
    operator = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]]).to_numpy()
    kernel = path_kernel(operator, 0.5)
    np.testing.assert_allclose(np.diag(kernel), 0.0)
    assert (kernel >= 0).all()


def test_strength_null_preserves_margins_and_diagonal() -> None:
    operator = row_normalize(
        np.array(
            [
                [0.0, 2.0, 1.0, 3.0],
                [1.0, 0.0, 4.0, 2.0],
                [3.0, 1.0, 0.0, 2.0],
                [2.0, 4.0, 1.0, 0.0],
            ]
        )
    )
    rewired = strength_preserving_rewire(
        operator, np.random.default_rng(123), steps=100
    )
    np.testing.assert_allclose(rewired.sum(axis=0), operator.sum(axis=0))
    np.testing.assert_allclose(rewired.sum(axis=1), operator.sum(axis=1))
    np.testing.assert_allclose(np.diag(rewired), 0.0)
    assert not np.allclose(rewired, operator)
