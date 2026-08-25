"""Checks for meeting-to-meeting concern transport."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_dynamic_concern_transport import run_transport


def test_transport_uses_map_at_t_and_records_directed_flow() -> None:
    topics = ["A", "B", "C"]
    phi = pd.DataFrame(
        [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]],
        index=topics,
        columns=topics,
    )
    maps = {
        "meeting-only": {1: phi},
        "cumulative": {1: phi},
    }
    attention = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        index=[1, 2],
        columns=topics,
    )
    totals = pd.Series({1: 10.0, 2: 12.0})
    years = pd.Series({1: 2000, 2: 2001})

    summary, flows = run_transport(
        maps,
        topics,
        [1, 2],
        attention,
        totals,
        years,
        permutations=20,
        rng=np.random.default_rng(7),
    )

    assert len(summary) == 2
    assert np.allclose(summary["mass_moved"], 1.0)
    assert np.allclose(summary["map_transport_cost"], 0.1)
    assert set(flows["source_concern"]) == {"A"}
    assert set(flows["target_concern"]) == {"B"}
    assert np.allclose(flows["transport_mass"], 1.0)
    assert np.allclose(flows["proximity_at_t"], 0.9)
    assert not flows["same_concern"].any()
