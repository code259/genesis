from __future__ import annotations

from genesis.models import Idea
from genesis.storage.manifold import ManifoldIndex


class LowDensityExplorer:
    def __init__(self, manifold: ManifoldIndex):
        self.manifold = manifold

    def find_low_density_points(self, n: int = 5) -> list[dict]:
        papers = self.manifold.all_papers()
        if not papers:
            return []
        stats = self.manifold.get_density_statistics(collection="papers")
        threshold = stats["threshold"]
        ranked = sorted(papers, key=lambda paper: float(paper.get("density_score", 0.0)), reverse=True)
        candidates = [paper for paper in ranked if float(paper.get("density_score", 0.0)) >= threshold]
        return (candidates or ranked)[:n]

    def propose_exploration(self, task_description: str, low_density_points: list[dict]) -> list[Idea]:
        return [
            Idea(
                title=point.get("title", "Low-density idea"),
                summary=f"Explore sparse-region concept for task: {task_description}",
                source="low_density",
                landing_point=point.get("latent_z", []),
                metadata={**point, "source_paper_id": point.get("paper_id")},
            )
            for point in low_density_points
        ]
