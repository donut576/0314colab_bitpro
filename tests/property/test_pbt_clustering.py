"""Property-based tests for the Identity Clusterer.

# Feature: aml-advanced-features, Property 5: cluster merge transitivity
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.services.identity_clusterer import IdentityClusterer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_user_id = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=16,
)

_user_id_set = st.lists(_user_id, min_size=1, max_size=5, unique=True)

_ip_addr = st.from_regex(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}", fullmatch=True)
_wallet = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=4,
    max_size=16,
).map(lambda s: f"wallet_{s}")


# ---------------------------------------------------------------------------
# Property 5: Cluster merge transitivity
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    group_a=_user_id_set,
    group_b=_user_id_set,
    group_c=_user_id_set,
    shared_ip_ab=_ip_addr,
    shared_wallet_bc=_wallet,
)
def test_property_5_cluster_merge_transitivity(
    group_a: list[str],
    group_b: list[str],
    group_c: list[str],
    shared_ip_ab: str,
    shared_wallet_bc: str,
):
    """Property 5: Cluster merge transitivity — generate random overlapping
    cluster sets, assert union-find produces correct merged clusters
    containing all members.

    If cluster A and B share a signal (IP), and B and C share a signal
    (wallet), then after recompute_clusters() all members of A, B, and C
    must end up in the same cluster (transitivity).

    # Feature: aml-advanced-features, Property 5: cluster merge transitivity

    **Validates: Requirements REQ-A3.3**
    """
    # Ensure groups are disjoint by prefixing
    a_users = [f"a_{u}" for u in group_a]
    b_users = [f"b_{u}" for u in group_b]
    c_users = [f"c_{u}" for u in group_c]

    all_users = set(a_users + b_users + c_users)

    clusterer = IdentityClusterer()

    # Register signals for group A — they share shared_ip_ab
    for uid in a_users:
        clusterer.register_signals(uid, ips=[shared_ip_ab])

    # Register signals for group B — shares shared_ip_ab (link to A)
    # and shared_wallet_bc (link to C)
    for uid in b_users:
        clusterer.register_signals(uid, ips=[shared_ip_ab], wallets=[shared_wallet_bc])

    # Register signals for group C — they share shared_wallet_bc
    for uid in c_users:
        clusterer.register_signals(uid, wallets=[shared_wallet_bc])

    # Recompute clusters
    clusterer.recompute_clusters()

    # All users should be in the same cluster (transitivity)
    # Pick any user and find their cluster
    reference_user = a_users[0]
    cluster = clusterer.get_cluster_for_account(reference_user)

    assert cluster is not None, (
        f"Expected user {reference_user} to be in a cluster, but got None"
    )

    cluster_members = set(cluster.member_user_ids)

    # Every user from A, B, C must be in this same cluster
    for uid in all_users:
        assert uid in cluster_members, (
            f"Transitivity violated: user {uid} not in the merged cluster. "
            f"Cluster members: {cluster_members}, "
            f"Expected all of: {all_users}"
        )


# ---------------------------------------------------------------------------
# Strategies (Property 6)
# ---------------------------------------------------------------------------

_risk_score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_user_with_score = st.tuples(_user_id, _risk_score)

_members_with_scores = st.lists(
    _user_with_score,
    min_size=2,
    max_size=8,
    unique_by=lambda x: x[0],
)


# ---------------------------------------------------------------------------
# Property 6: Cluster risk score is max member
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    members=_members_with_scores,
    shared_ip=_ip_addr,
    extra_user=_user_id,
    extra_score=_risk_score,
)
def test_property_6_cluster_risk_score_is_max_member(
    members: list[tuple[str, float]],
    shared_ip: str,
    extra_user: str,
    extra_score: float,
):
    """Property 6: Cluster risk score is max member — generate random member
    risk scores, assert cluster_risk_score = max(scores); adding a
    higher-scored member updates the cluster score.

    # Feature: aml-advanced-features, Property 6: cluster risk score is max member

    **Validates: Requirements REQ-A3.4**
    """
    # Prefix member user_ids to avoid collision with extra_user
    prefixed = [(f"m_{uid}", score) for uid, score in members]
    extra_uid = f"x_{extra_user}"

    clusterer = IdentityClusterer()

    # Register all members with the same shared IP so they form one cluster
    for uid, score in prefixed:
        clusterer.register_signals(uid, ips=[shared_ip])
        clusterer.set_risk_score(uid, score)

    # Recompute and verify cluster_risk_score == max of member scores
    clusterer.recompute_clusters()

    reference_uid = prefixed[0][0]
    cluster = clusterer.get_cluster_for_account(reference_uid)
    assert cluster is not None, (
        f"Expected user {reference_uid} to be in a cluster"
    )

    expected_max = max(score for _, score in prefixed)
    assert cluster.cluster_risk_score == expected_max, (
        f"cluster_risk_score {cluster.cluster_risk_score} != "
        f"max member score {expected_max}"
    )

    # --- Add a new member with a higher score and verify update ---
    higher_score = min(expected_max + 0.1, 1.0)
    # Use the extra_score only if it's strictly higher; otherwise force higher
    new_score = max(extra_score, higher_score)

    clusterer.register_signals(extra_uid, ips=[shared_ip])
    clusterer.set_risk_score(extra_uid, new_score)
    clusterer.recompute_clusters()

    updated_cluster = clusterer.get_cluster_for_account(reference_uid)
    assert updated_cluster is not None, (
        f"Expected user {reference_uid} to still be in a cluster after adding member"
    )

    new_expected_max = max(expected_max, new_score)
    assert updated_cluster.cluster_risk_score == new_expected_max, (
        f"After adding member with score {new_score}, "
        f"cluster_risk_score {updated_cluster.cluster_risk_score} != "
        f"expected max {new_expected_max}"
    )
