from __future__ import annotations

import math

from genesis.models import Idea
from genesis.storage.manifold import ManifoldIndex


class PollinationSearch:
    def __init__(self, manifold: ManifoldIndex):
        self.manifold = manifold

    def sample_distant_point(self, task_embedding: list[float], theta_jump: float = 2.0) -> list[float]:
        return [value + theta_jump for value in task_embedding]

    def find_return_path(self, start: list[float], target: list[float], max_length: int = 6) -> list[list[float]]:
        path: list[list[float]] = []
        for step in range(1, max_length + 1):
            alpha = step / max_length
            path.append([(1 - alpha) * source + alpha * destination for source, destination in zip(start, target)])
        return path

    def propose_pollination(self, task_description: str) -> Idea:
        seed = [float((ord(character) % 17) / 17.0) for character in task_description[:16]]
        landing = self.sample_distant_point(seed)
        path = self.find_return_path(landing, seed)
        novelty = math.sqrt(sum(value * value for value in landing))
        return Idea(
            title="Pollinated direction",
            summary=f"Novel direction with path length {len(path)} and novelty {novelty:.2f}",
            source="pollination",
            landing_point=landing,
            metadata={"path": path},
        )
