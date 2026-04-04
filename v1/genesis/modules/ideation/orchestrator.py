from __future__ import annotations

from genesis.models import ScoredIdea

from .greedy import GreedyAdjacencySearch
from .low_density import LowDensityExplorer
from .pollination import PollinationSearch
from .scorer import IdeaScorer


class IdeationOrchestrator:
    def __init__(
        self,
        *,
        greedy: GreedyAdjacencySearch,
        pollination: PollinationSearch,
        low_density: LowDensityExplorer,
        scorer: Optional[IdeaScorer] = None,
    ):
        self.greedy = greedy
        self.pollination = pollination
        self.low_density = low_density
        self.scorer = scorer or IdeaScorer()

    def run(self, task_description: str, n_failed_iterations: int) -> list[ScoredIdea]:
        ideas = self.greedy.search(task_description, k=3)
        if not ideas and n_failed_iterations >= 5:
            ideas.append(self.pollination.propose_pollination(task_description))
        if n_failed_iterations >= 5:
            ideas.extend(self.low_density.propose_exploration(task_description, self.low_density.find_low_density_points()))
        return [ScoredIdea(idea=idea, score=self.scorer.score(idea, task_description)) for idea in ideas]
from typing import Optional
