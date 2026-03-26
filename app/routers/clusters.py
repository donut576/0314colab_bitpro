from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.cluster import ClusterStats, IdentityCluster

router = APIRouter(prefix="/clusters", tags=["Clusters"])


@router.get("/stats", response_model=ClusterStats)
async def get_stats(request: Request) -> ClusterStats:
    return request.app.state.identity_clusterer.get_stats()


@router.get("/account/{user_id}", response_model=IdentityCluster)
async def get_cluster_for_account(user_id: str, request: Request) -> IdentityCluster:
    cluster = request.app.state.identity_clusterer.get_cluster_for_account(user_id)
    if cluster is None:
        raise HTTPException(
            status_code=404, detail=f"No cluster found for user {user_id}"
        )
    return cluster


@router.get("/{cluster_id}", response_model=IdentityCluster)
async def get_cluster(cluster_id: str, request: Request) -> IdentityCluster:
    cluster = request.app.state.identity_clusterer.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=404, detail=f"Cluster {cluster_id} not found"
        )
    return cluster
