from __future__ import annotations

import math

from genesis.manifold_utils import build_citation_graph, cosine_distance, hash_embedding
from genesis.models import Idea
from genesis.storage.manifold import ManifoldIndex


class PollinationSearch:
    def __init__(self, manifold: ManifoldIndex):
        self.manifold = manifold

    def sample_distant_point(self, task_embedding: list[float], theta_jump: float = 2.0) -> list[float]:
        candidates = self.manifold.all_papers()
        if not candidates:
            return task_embedding
        ranked = sorted(
            candidates,
            key=lambda paper: cosine_distance(task_embedding, paper.get("latent_z", [])),
            reverse=True,
        )
        return ranked[0].get("latent_z", task_embedding)

    def find_return_path(self, start: list[float], target: list[float], max_length: int = 6) -> list[list[float]]:
        papers = self.manifold.all_papers()
        if not papers:
            return [start, target]
        graph = build_citation_graph(papers)
        if graph.number_of_nodes() == 0:
            return [start, target]
        start_node = max(
            papers,
            key=lambda paper: -cosine_distance(start, paper.get("latent_z", [])),
        )["paper_id"]
        target_node = min(
            papers,
            key=lambda paper: cosine_distance(target, paper.get("latent_z", [])),
        )["paper_id"]
        try:
            node_path = __import__("networkx").astar_path(
                graph,
                start_node,
                target_node,
                heuristic=lambda left, right: cosine_distance(
                    graph.nodes[left]["paper"].get("latent_z", []),
                    graph.nodes[right]["paper"].get("latent_z", []),
                ),
            )
        except Exception:  # noqa: BLE001
            return [start, target]
        vectors = [
            graph.nodes[node]["paper"].get("latent_z", [])
            for node in node_path[:max_length]
        ]
        return vectors or [start, target]

    def propose_pollination(self, task_description: str) -> Idea:
        papers = self.manifold.all_papers()
        latent_dim = len(papers[0].get("latent_z", [])) if papers else 32
        seed = hash_embedding(task_description, dim=max(32, latent_dim))[:latent_dim].tolist()
        landing = self.sample_distant_point(seed)
        path = self.find_return_path(landing, seed)
        novelty = cosine_distance(seed, landing)
        return Idea(
            title="Pollinated direction",
            summary=f"Novel direction with path length {len(path)} and novelty {novelty:.2f}",
            source="pollination",
            landing_point=landing,
            metadata={"path": path},
        )
