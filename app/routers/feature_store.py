"""Feature Store router — /features endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.feature_store import BatchFeaturesRequest, FeatureStoreStats, FeatureVector

router = APIRouter(prefix="/features", tags=["Feature Store"])


@router.get("/{user_id}", response_model=FeatureVector)
async def get_feature_vector(user_id: str, request: Request) -> FeatureVector:
    feature_store = request.app.state.feature_store
    return await feature_store.get(user_id)


@router.post("/batch", response_model=list[FeatureVector])
async def get_batch_features(body: BatchFeaturesRequest, request: Request) -> list[FeatureVector]:
    feature_store = request.app.state.feature_store
    return await feature_store.get_batch(body.user_ids)


@router.get("/stats", response_model=FeatureStoreStats)
async def get_stats(request: Request) -> FeatureStoreStats:
    feature_store = request.app.state.feature_store
    return await feature_store.get_stats()
