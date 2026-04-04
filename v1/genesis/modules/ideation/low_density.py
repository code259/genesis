from __future__ import annotations

from statistics import mean, pstdev

from genesis.models import Idea
from genesis.storage.manifold import ManifoldIndex


class LowDensityExplorer:
    def __init__(self, manifold: ManifoldIndex):
        self.manifold = manifold

    def find_low_density_points(self, n: int = 5) -> list[dict]:
        papers = self.manifold.all_papers()
        if not papers:
            return []
        scores = [float(paper.get("density_score", 0.0)) for paper in papers]
        threshold = mean(scores) + 2 * pstdev(scores) if len(scores) > 1 else scores[0]
        return [paper for paper in papers if float(paper.get("density_score", 0.0)) >= threshold][:n]

    def propose_exploration(self, task_description: str, low_density_points: list[dict]) -> list[Idea]:
        return [
            Idea(
                title=point.get("title", "Low-density idea"),
                summary=f"Explore sparse-region concept for task: {task_description}",
                source="low_density",
                landing_point=point.get("latent_z", []),
                metadata=point,
            )
            for point in low_density_points
        ]
