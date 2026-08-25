from __future__ import annotations

import numpy as np

from scripts.analyze_proximity_pathways_to_outputs import (
    maximum_product_paths,
    reconstruct_path,
)


def test_indirect_path_can_exceed_direct_probability() -> None:
    proximity = np.array(
        [
            [0.0, 0.9, 0.2],
            [0.9, 0.0, 0.8],
            [0.2, 0.8, 0.0],
        ]
    )
    best, predecessors = maximum_product_paths(proximity)
    np.testing.assert_allclose(best[0, 2], 0.72)
    assert reconstruct_path(0, 2, predecessors) == [0, 1, 2]


def test_direct_path_remains_when_it_is_stronger() -> None:
    proximity = np.array(
        [
            [0.0, 0.5, 0.8],
            [0.5, 0.0, 0.5],
            [0.8, 0.5, 0.0],
        ]
    )
    best, predecessors = maximum_product_paths(proximity)
    np.testing.assert_allclose(best[0, 2], 0.8)
    assert reconstruct_path(0, 2, predecessors) == [0, 2]
