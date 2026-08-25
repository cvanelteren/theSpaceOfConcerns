import networkx as nx
import pandas as pd

from scripts.audit_documented_pathway_coverage import administrative_title_flag
from scripts.forecast_dual_channel_outcomes import (
    ancestor_topic_evidence,
    formal_frontier_nodes,
    formal_memory_scores,
    official_target_topics,
    recent_formal_activity_scores,
)


def test_frontier_includes_sink_cycle_members():
    graph = nx.DiGraph(
        [
            ("root", "middle"),
            ("middle", "latest-a"),
            ("latest-a", "latest-b"),
            ("latest-b", "latest-a"),
        ]
    )

    assert formal_frontier_nodes(graph) == ["latest-a", "latest-b"]


def test_ancestor_evidence_uses_minimum_depth_in_cycle():
    graph = nx.DiGraph(
        [("root", "a"), ("a", "b"), ("b", "a")]
    )
    topics = {"root": frozenset({"science"}), "a": frozenset({"climate"})}

    assert ancestor_topic_evidence(graph, "b", topics, max_depth=3) == {
        "climate": 1,
        "science": 2,
    }


def test_formal_memory_excludes_focal_meeting_nodes():
    graph = nx.DiGraph()
    graph.add_node("past", meeting=9)
    graph.add_node("focal", meeting=10)
    graph.add_edge("past", "focal")
    phi = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["science", "climate"],
        columns=["science", "climate"],
    )
    topics = {
        "past": frozenset({"science"}),
        "focal": frozenset({"climate"}),
    }

    scores = formal_memory_scores(
        graph,
        topics,
        phi,
        cutoff_meeting=10,
        max_depth=0,
    )

    assert scores["science"] > scores["climate"]


def test_official_target_topics_excludes_nonregular_and_historical_outputs():
    topics = {
        "Resolution 6 (1996)": frozenset({"science"}),
        "Resolution 1 (2025)": frozenset({"tourism"}),
        "Recommendation 1 (1990)": frozenset({"operations"}),
    }
    official = pd.DataFrame({"output_id": ["Resolution 1 (2025)"]})

    assert official_target_topics(topics, official) == {
        "Resolution 1 (2025)": frozenset({"tourism"})
    }


def test_administrative_title_flag_is_narrow_and_review_only():
    assert administrative_title_flag("Appointment of the Executive Secretary")
    assert administrative_title_flag(
        "Revised Rules of Procedure for the Antarctic Treaty Consultative Meeting"
    )
    assert not administrative_title_flag(
        "Consideration of Mitigation Measures in Environmental Impact Assessment"
    )


def test_recent_formal_activity_separates_same_and_nearby_concerns():
    graph = nx.DiGraph()
    graph.add_node("past", meeting=8)
    phi = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["science", "climate"],
        columns=["science", "climate"],
    )

    scores = recent_formal_activity_scores(
        graph,
        {"past": frozenset({"science"})},
        phi,
        cutoff_meeting=10,
        horizon=5,
    )

    assert scores.loc["science", "same"] > 0
    assert scores.loc["science", "nearby"] == 0
    assert scores.loc["climate", "same"] == 0
    assert scores.loc["climate", "nearby"] > 0
