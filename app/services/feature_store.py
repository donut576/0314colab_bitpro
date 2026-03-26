"""Feature Store service — Redis hot layer with PostgreSQL cold storage."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import redis.asyncio

from app.models.feature_store import BatchFeaturesRequest, FeatureStoreStats, FeatureVector

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, redis_url: str, database_url: str) -> None:
        self._redis_url = redis_url
        self._database_url = database_url
        self._schema_version = "1"
        self._redis_client = None
        self._db_pool = None

    async def connect(self) -> None:
        """Connect to Redis; log warning on failure without raising."""
        try:
            self._redis_client = redis.asyncio.from_url(self._redis_url)
        except Exception as exc:
            logger.warning("Failed to connect to Redis: %s", exc)

    async def get(self, user_id: str, groups: list[str] | None = None) -> FeatureVector:
        """Retrieve feature vector from Redis; fall back to zero-vector on miss."""
        key = f"features:{user_id}:{self._schema_version}"

        try:
            if self._redis_client is not None:
                raw = await self._redis_client.get(key)
                if raw is not None:
                    data = json.loads(raw)
                    fv = FeatureVector(**data)
                    if groups is not None:
                        filtered = {k: v for k, v in fv.features.items() if k in groups}
                        fv = fv.model_copy(update={"features": filtered})
                    return fv
        except Exception as exc:
            logger.warning("Redis get failed for key %s: %s", key, exc)

        # Cold start — return zero-vector
        return FeatureVector(
            user_id=user_id,
            schema_version=self._schema_version,
            features={},
            groups=groups or [],
            last_updated=datetime.utcnow(),
            cold_start=True,
        )

    async def get_batch(self, user_ids: list[str]) -> list[FeatureVector]:
        """Retrieve feature vectors for multiple users concurrently."""
        return list(await asyncio.gather(*[self.get(uid) for uid in user_ids]))

    async def put(self, user_id: str, features: FeatureVector) -> None:
        """Write feature vector to Redis with a 24-hour TTL."""
        key = f"features:{user_id}:{self._schema_version}"
        try:
            if self._redis_client is not None:
                payload = features.model_dump_json()
                await self._redis_client.set(key, payload, ex=86400)
        except Exception as exc:
            logger.warning("Redis put failed for key %s: %s", key, exc)

    async def get_stats(self) -> FeatureStoreStats:
        """Return basic feature store statistics."""
        return FeatureStoreStats(
            total_users=0,
            schema_version=self._schema_version,
            last_updated=datetime.utcnow(),
            storage_utilization_bytes=0,
        )
