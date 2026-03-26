from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime

from app.models.cluster import ClusterDiff, ClusterStats, IdentityCluster

logger = logging.getLogger(__name__)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        self._parent[self.find(x)] = self.find(y)

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            result[self.find(node)].append(node)
        return dict(result)


class IdentityClusterer:
    def __init__(self, high_threshold: float = 0.7) -> None:
        self._high_threshold = high_threshold
        self._clusters: dict[str, IdentityCluster] = {}   # cluster_id -> cluster
        self._user_to_cluster: dict[str, str] = {}        # user_id -> cluster_id
        self._user_risk_scores: dict[str, float] = {}
        # Signal registries: signal_value -> list[user_id]
        self._ip_registry: dict[str, list[str]] = defaultdict(list)
        self._wallet_registry: dict[str, list[str]] = defaultdict(list)
        self._device_registry: dict[str, list[str]] = defaultdict(list)

    def register_signals(
        self,
        user_id: str,
        ips: list[str] = (),
        wallets: list[str] = (),
        devices: list[str] = (),
    ) -> None:
        for ip in ips:
            self._ip_registry[ip].append(user_id)
        for w in wallets:
            self._wallet_registry[w].append(user_id)
        for d in devices:
            self._device_registry[d].append(user_id)

    def set_risk_score(self, user_id: str, score: float) -> None:
        self._user_risk_scores[user_id] = score

    def recompute_clusters(self) -> ClusterDiff:
        uf = _UnionFind()
        shared_signals: dict[str, dict] = defaultdict(
            lambda: {"ips": [], "wallets": [], "devices": []}
        )

        for ip, users in self._ip_registry.items():
            for i in range(1, len(users)):
                uf.union(users[0], users[i])
            for u in users:
                shared_signals[uf.find(u)]["ips"].append(ip)

        for wallet, users in self._wallet_registry.items():
            for i in range(1, len(users)):
                uf.union(users[0], users[i])
            for u in users:
                shared_signals[uf.find(u)]["wallets"].append(wallet)

        for device, users in self._device_registry.items():
            for i in range(1, len(users)):
                uf.union(users[0], users[i])
            for u in users:
                shared_signals[uf.find(u)]["devices"].append(device)

        groups = uf.groups()
        old_ids = set(self._clusters.keys())
        new_clusters: list[str] = []
        merged_clusters: list[str] = []
        now = datetime.utcnow()

        new_state: dict[str, IdentityCluster] = {}
        new_user_map: dict[str, str] = {}

        for root, members in groups.items():
            if len(members) < 2:
                continue
            cluster_risk = max(
                (self._user_risk_scores.get(m, 0.0) for m in members), default=0.0
            )
            cid = str(uuid.uuid4())
            cluster = IdentityCluster(
                cluster_id=cid,
                member_user_ids=members,
                shared_signals=shared_signals.get(root, {}),
                cluster_risk_score=cluster_risk,
                created_at=now,
                updated_at=now,
            )
            new_state[cid] = cluster
            for m in members:
                new_user_map[m] = cid
            new_clusters.append(cid)

        dissolved = list(old_ids - set(new_state.keys()))
        self._clusters = new_state
        self._user_to_cluster = new_user_map

        logger.info("Clusters recomputed: %d clusters", len(self._clusters))
        return ClusterDiff(
            new_clusters=new_clusters,
            merged_clusters=merged_clusters,
            dissolved_clusters=dissolved,
            computed_at=now,
        )

    def get_cluster(self, cluster_id: str) -> IdentityCluster | None:
        return self._clusters.get(cluster_id)

    def get_cluster_for_account(self, user_id: str) -> IdentityCluster | None:
        cid = self._user_to_cluster.get(user_id)
        return self._clusters.get(cid) if cid else None

    def get_stats(self) -> ClusterStats:
        clusters = list(self._clusters.values())
        total = len(clusters)
        avg_size = sum(len(c.member_user_ids) for c in clusters) / max(total, 1)
        high_risk = sum(
            1 for c in clusters if c.cluster_risk_score >= self._high_threshold
        )
        return ClusterStats(
            total_clusters=total,
            average_cluster_size=avg_size,
            high_risk_cluster_count=high_risk,
            computed_at=datetime.utcnow(),
        )
