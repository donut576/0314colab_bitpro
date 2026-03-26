"""Property-based tests for the Graph Engine.

# Feature: aml-advanced-features, Property 3: graph neighborhood elevation
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.services.graph_engine import GraphEngine


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=32,
)

_neighbor_risk_levels = st.sampled_from(["HIGH", "LOW"])

_neighbor_lists = st.lists(
    st.tuples(
        _user_ids,  # neighbor id
        _neighbor_risk_levels,  # risk level
    ),
    min_size=1,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Property 3: Graph neighborhood elevation
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    user_id=_user_ids,
    neighbors=_neighbor_lists,
)
def test_property_3_graph_neighborhood_elevation(
    user_id: str,
    neighbors: list[tuple[str, str]],
):
    """Property 3: Graph neighborhood elevation — generate random neighborhoods
    with varying HIGH fractions, assert elevation logic applies iff
    fraction > threshold.

    # Feature: aml-advanced-features, Property 3: graph neighborhood elevation

    **Validates: Requirements REQ-A2.8**
    """
    # Filter out neighbors that collide with the user_id itself
    neighbors = [(nid, level) for nid, level in neighbors if nid != user_id]

    engine = GraphEngine()
    threshold = engine._high_fraction_threshold  # default 0.3

    if len(neighbors) == 0:
        # No neighbors → elevated must be False
        score = engine.get_score(user_id)
        assert score.elevated is False, (
            f"Expected elevated=False when user has no neighbors, got {score.elevated}"
        )
        return

    # Build graph: add edges from user to each neighbor
    for nid, _ in neighbors:
        engine._graph.add_edge(user_id, nid)

    # Set risk levels for each neighbor
    for nid, level in neighbors:
        engine._risk_levels[nid] = level

    # Call get_score
    score = engine.get_score(user_id)

    # Compute expected elevation
    # Deduplicate neighbors by id (successors + predecessors gives unique set)
    unique_neighbors = {nid for nid, _ in neighbors}
    total = len(unique_neighbors)
    high_count = sum(
        1 for nid in unique_neighbors
        if engine._risk_levels.get(nid) == "HIGH"
    )
    high_fraction = high_count / total
    expected_elevated = high_fraction > threshold

    assert score.elevated == expected_elevated, (
        f"Expected elevated={expected_elevated} "
        f"(high_fraction={high_fraction}, threshold={threshold}, "
        f"high_count={high_count}, total={total}), "
        f"got elevated={score.elevated}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(user_id=_user_ids)
def test_property_3_no_neighbors_not_elevated(user_id: str):
    """Property 3 (edge case): A user with no neighbors must never be elevated.

    # Feature: aml-advanced-features, Property 3: graph neighborhood elevation

    **Validates: Requirements REQ-A2.8**
    """
    engine = GraphEngine()

    # User not in graph at all
    score = engine.get_score(user_id)
    assert score.elevated is False, (
        f"Expected elevated=False for user not in graph, got {score.elevated}"
    )

    # User in graph but with no edges
    engine._graph.add_node(user_id)
    score = engine.get_score(user_id)
    assert score.elevated is False, (
        f"Expected elevated=False for user with no neighbors, got {score.elevated}"
    )
