from scripts.build_outcome_concern_browser import (
    classify_paper_edge,
    display_outcome_id,
    paper_node_id,
)


def test_display_outcome_id_adds_year_to_historical_identifier():
    node = {"id": "Recommendation 16-1", "year": 1991}
    assert display_outcome_id(node) == "Recommendation 16-1 (1991)"


def test_display_outcome_id_does_not_duplicate_year():
    node = {"id": "Resolution 2 (1995)", "year": 1995}
    assert display_outcome_id(node) == "Resolution 2 (1995)"


def test_paper_node_id_uses_meeting_type_and_number():
    row = {
        "meeting number": 47,
        "paper number": 17,
        "paper url": "https://documents.ats.aq/ATCM47/wp/ATCM47_wp017_e.pdf",
    }
    assert paper_node_id(row) == "ATCM47:WP 17"


def test_direct_reference_to_earlier_year_is_rejected():
    edge = {
        "channel": "direct_citation",
        "relation": "direct_proposal_or_discussion",
        "evidence": "The paper discussed Resolution 4 (1998).",
    }
    outcome = {"id": "Resolution 4 (2025)"}
    level, reason = classify_paper_edge(edge, outcome)
    assert level == "rejected"
    assert "1998" in reason


def test_strong_same_meeting_title_path_is_corroborated():
    edge = {"channel": "wp_title_window", "jaccard": 0.8, "lag_meetings": 0}
    level, _ = classify_paper_edge(edge, {"id": "Resolution 5 (2025)"})
    assert level == "corroborated"


def test_official_paragraph_relationship_is_confirmed():
    edge = {"channel": "official_paragraph"}
    level, _ = classify_paper_edge(edge, {"id": "Resolution 5 (2025)"})
    assert level == "confirmed"
