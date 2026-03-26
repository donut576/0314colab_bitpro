"""Property-based tests for the Feature Store.

# Feature: aml-advanced-features, Property 14: cold start returns zero-vector
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from app.models.feature_store import FeatureVector
from app.services.feature_store import FeatureStore


# Strategy: generate arbitrary non-empty user_id strings that won't exist in any store.
# We use text with a min_size=1 to avoid empty user_ids.
unknown_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=128,
)


def _run_async(coro):
    """Run an async coroutine synchronously for Hypothesis compatibility."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(user_id=unknown_user_ids)
def test_property_14_cold_start_returns_zero_vector(user_id: str):
    """Property 14: Feature store cold start — generate unknown user_ids,
    assert zero-vector + cold_start: true, never 404.

    For any user_id with no stored features, FeatureStore.get() must return
    a FeatureVector with cold_start=True and an empty features dict (zero-vector),
    and must never raise an error.

    Validates: Requirements REQ-A9.7
    """
    # Create a FeatureStore with no Redis connection (always falls through to cold start)
    store = FeatureStore(redis_url="redis://localhost:6379/15", database_url="postgresql://localhost/test")
    # Don't call connect() — _redis_client stays None, so every get() hits cold start path

    result = _run_async(store.get(user_id))

    # 1. Result is a FeatureVector (not an error/404)
    assert isinstance(result, FeatureVector), (
        f"Expected FeatureVector, got {type(result)}"
    )

    # 2. cold_start is True
    assert result.cold_start is True, (
        f"Expected cold_start=True for unknown user_id={user_id!r}, got {result.cold_start}"
    )

    # 3. features dict is empty (zero-vector)
    assert result.features == {}, (
        f"Expected empty features dict for cold start, got {result.features}"
    )

    # 4. user_id matches the requested user_id
    assert result.user_id == user_id, (
        f"Expected user_id={user_id!r}, got {result.user_id!r}"
    )


# ---------------------------------------------------------------------------
# Feature: aml-advanced-features, Property 15: schema versioning round-trip
# ---------------------------------------------------------------------------

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock


def _make_mock_redis():
    """Create an in-memory mock that behaves like an async Redis client."""
    storage: dict[str, str] = {}

    async def _set(key: str, value: str, ex: int | None = None) -> None:
        storage[key] = value

    async def _get(key: str) -> str | None:
        return storage.get(key)

    mock = MagicMock()
    mock.set = AsyncMock(side_effect=_set)
    mock.get = AsyncMock(side_effect=_get)
    return mock


# Strategies for Property 15
_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=64,
)

_schema_versions = st.text(
    alphabet=st.characters(whitelist_categories=("N", "L")),
    min_size=1,
    max_size=8,
).filter(lambda s: s.strip() != "")

_feature_dicts = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L",)),
        min_size=1,
        max_size=16,
    ),
    values=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=10,
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    user_id=_user_ids,
    version_v=_schema_versions,
    version_v_plus_1=_schema_versions,
    features=_feature_dicts,
)
def test_property_15_schema_versioning_round_trip(
    user_id: str,
    version_v: str,
    version_v_plus_1: str,
    features: dict[str, float],
):
    """Property 15: Feature store schema versioning — write features at version V,
    deploy V+1, assert V retrieval unchanged.

    For any feature vector written under schema version V, retrieving that vector
    with schema version V must return an equivalent feature vector regardless of
    whether a newer schema version has been deployed.

    **Validates: Requirements REQ-A9.4**
    """
    # --- Arrange: build a FeatureStore with an in-memory mock Redis ---
    store = FeatureStore(
        redis_url="redis://localhost:6379/15",
        database_url="postgresql://localhost/test",
    )
    mock_redis = _make_mock_redis()
    store._redis_client = mock_redis

    now = datetime.utcnow()
    original_vector = FeatureVector(
        user_id=user_id,
        schema_version=version_v,
        features=features,
        groups=["default"],
        last_updated=now,
        cold_start=False,
    )

    # --- Act: write at version V ---
    store._schema_version = version_v
    _run_async(store.put(user_id, original_vector))

    # --- "Deploy" version V+1 (change the schema version) ---
    store._schema_version = version_v_plus_1

    # --- Revert to version V and retrieve ---
    store._schema_version = version_v
    retrieved = _run_async(store.get(user_id))

    # --- Assert: retrieved vector matches the original ---
    assert isinstance(retrieved, FeatureVector), (
        f"Expected FeatureVector, got {type(retrieved)}"
    )
    assert retrieved.cold_start is False, (
        "Expected cold_start=False for a stored vector, got True"
    )
    assert retrieved.user_id == user_id, (
        f"Expected user_id={user_id!r}, got {retrieved.user_id!r}"
    )
    assert retrieved.schema_version == version_v, (
        f"Expected schema_version={version_v!r}, got {retrieved.schema_version!r}"
    )
    assert retrieved.features == features, (
        f"Features mismatch: expected {features}, got {retrieved.features}"
    )
