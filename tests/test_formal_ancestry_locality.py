from collections import defaultdict

import pandas as pd

from scripts.analyze_formal_ancestry_locality import minimum_depths, pair_metrics


def test_minimum_depths_uses_shortest_route_and_terminates_on_cycle():
    incoming = defaultdict(list, {"focal": ["a", "b"], "a": ["b"], "b": ["a", "c"]})

    assert minimum_depths("focal", incoming, max_depth=4) == {"a": 1, "b": 1, "c": 2}


def test_pair_metrics_separates_exact_overlap_from_nearby_concerns():
    phi = pd.DataFrame(
        [[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    regions = {"a": 1, "b": 1, "c": 2}

    nearby = pair_metrics(frozenset({"a"}), frozenset({"b"}), phi, regions)
    overlap = pair_metrics(frozenset({"a"}), frozenset({"a"}), phi, regions)

    assert nearby["exact_overlap"] == 0.0
    assert nearby["mean_nearest_phi"] == 0.8
    assert nearby["ancestor_topic_share_same_region"] == 1.0
    assert overlap["exact_overlap"] == 1.0
    assert overlap["mean_nearest_phi"] == 1.0
