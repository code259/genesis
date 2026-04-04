from __future__ import annotations

from genesis.models import Idea
from genesis.storage.manifold import ManifoldIndex


class GreedyAdjacencySearch:
    def __init__(self, manifold: ManifoldIndex):
        self.manifold = manifold

    def search(self, task_description: str, k: int = 5, exclude_attempted: bool = True) -> list[Idea]:
        query = [float((ord(character) % 31) / 31.0) for character in task_description[:16]]
        results = self.manifold.search_nearest(query, k=k)
        return [
            Idea(
                title=result.get("title", result.get("paper_id", "idea")),
                summary=result.get("abstract", result.get("summary", "")),
                source="greedy",
                landing_point=result.get("latent_z", []),
                metadata=result,
            )
            for result in results
        ]
