from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING

import networkx as nx

from app.models.graph import GraphScore, SubgraphResult, TransactionEdge

logger = logging.getLogger(__name__)


class GraphEngine:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._embeddings: dict[str, list[float]] = {}
        self._risk_scores: dict[str, float] = {}
        self._risk_levels: dict[str, str] = {}  # user_id -> "HIGH"/"MEDIUM"/"LOW"
        self._high_fraction_threshold = float(
            os.getenv("GRAPH_HIGH_FRACTION_THRESHOLD", "0.3")
        )
        self._hidden_dim = int(os.getenv("GRAPH_HIDDEN_DIM", "64"))

    def update_graph(self, transactions: list[TransactionEdge]) -> None:
        """Add nodes and edges incrementally."""
        for tx in transactions:
            self._graph.add_node(tx.sender_user_id)
            self._graph.add_node(tx.receiver_user_id)
            self._graph.add_edge(
                tx.sender_user_id,
                tx.receiver_user_id,
                amount=tx.amount,
                timestamp=tx.timestamp.isoformat(),
                channel=tx.channel,
            )
        logger.info(
            "Graph updated: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def recompute_embeddings(self) -> None:
        """Stub: in production runs GraphSAGE/GAT via PyTorch Geometric."""
        # Production implementation would call PyG's SAGEConv or GATConv here.
        for node in self._graph.nodes():
            self._embeddings[node] = [0.0] * self._hidden_dim
        logger.info("Embeddings recomputed for %d nodes", len(self._embeddings))

    def get_score(self, user_id: str) -> GraphScore:
        """Return graph-based risk score for a user."""
        if user_id not in self._graph:
            return GraphScore(
                user_id=user_id,
                graph_risk_score=0.0,
                embedding=self._embeddings.get(user_id, [0.0] * self._hidden_dim),
                hop1_count=0,
                hop2_count=0,
                betweenness_centrality=0.0,
                elevated=False,
            )

        neighbors_1 = set(self._graph.successors(user_id)) | set(
            self._graph.predecessors(user_id)
        )
        neighbors_2: set = set()
        for n in neighbors_1:
            neighbors_2 |= set(self._graph.successors(n)) | set(
                self._graph.predecessors(n)
            )
        neighbors_2 -= {user_id}

        # Neighborhood elevation check
        high_count = sum(
            1 for n in neighbors_1 if self._risk_levels.get(n) == "HIGH"
        )
        elevated = (len(neighbors_1) > 0) and (
            high_count / len(neighbors_1) > self._high_fraction_threshold
        )

        try:
            bc = nx.betweenness_centrality(self._graph).get(user_id, 0.0)
        except Exception:
            bc = 0.0

        base_score = self._risk_scores.get(user_id, 0.0)
        graph_risk_score = min(1.0, base_score + (0.2 if elevated else 0.0))

        return GraphScore(
            user_id=user_id,
            graph_risk_score=graph_risk_score,
            embedding=self._embeddings.get(user_id, [0.0] * self._hidden_dim),
            hop1_count=len(neighbors_1),
            hop2_count=len(neighbors_2),
            betweenness_centrality=bc,
            elevated=elevated,
        )

    def get_subgraph(self, user_id: str, hops: int = 2) -> SubgraphResult:
        """Return ego network as node-link structure."""
        if user_id not in self._graph:
            return SubgraphResult(user_id=user_id, nodes=[], edges=[], hops=hops)

        ego = nx.ego_graph(self._graph, user_id, radius=hops, undirected=True)
        nodes = [
            {"id": n, "risk_level": self._risk_levels.get(n, "LOW")}
            for n in ego.nodes()
        ]
        edges = [
            {"source": u, "target": v, **ego.edges[u, v]}
            for u, v in ego.edges()
        ]
        return SubgraphResult(user_id=user_id, nodes=nodes, edges=edges, hops=hops)
