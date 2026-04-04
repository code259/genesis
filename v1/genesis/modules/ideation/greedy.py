from __future__ import annotations

from genesis.manifold_utils import hash_embedding
from genesis.models import Idea
from genesis.storage.manifold import ManifoldIndex


class GreedyAdjacencySearch:
    def __init__(self, manifold: ManifoldIndex):
        self.manifold = manifold

    def search(self, task_description: str, k: int = 5, exclude_attempted: bool = True) -> list[Idea]:
        papers = self.manifold.all_papers()
        latent_dim = len(papers[0].get("latent_z", [])) if papers else 32
        query = hash_embedding(task_description, dim=max(32, latent_dim))[:latent_dim].tolist()
        exclude_ids = self.manifold.attempted_sources() if exclude_attempted else set()
        results = self.manifold.search_nearest(query, k=k, collection="papers", exclude_ids=exclude_ids)
        return [
            Idea(
                title=result.get("title", result.get("paper_id", "idea")),
                summary=result.get("abstract", result.get("summary", "")),
                source="greedy",
                landing_point=result.get("latent_z", []),
                metadata={
                    **result,
                    "source_paper_id": result.get("paper_id"),
                },
            )
            for result in results
        ]
