from __future__ import annotations

from typing import Optional

from genesis.models import ScoredIdea
from genesis.taste.gp_model import TasteGP

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

    def run(
        self,
        task_description: str,
        n_failed_iterations: int,
        *,
        taste_model: TasteGP | None = None,
    ) -> list[ScoredIdea]:
        ideas = self.greedy.search(task_description, k=3)
        if n_failed_iterations >= 5:
            ideas.append(self.pollination.propose_pollination(task_description))
        if n_failed_iterations >= 5:
            ideas.extend(
                self.low_density.propose_exploration(
                    task_description, self.low_density.find_low_density_points()
                )
            )
        scored = [
            ScoredIdea(
                idea=idea,
                score=self.scorer.score(idea, task_description, taste_model=taste_model),
            )
            for idea in ideas
        ]
        deduped: dict[tuple[str, str], ScoredIdea] = {}
        for candidate in scored:
            key = (candidate.idea.source, candidate.idea.title)
            existing = deduped.get(key)
            if existing is None or candidate.score.composite_score > existing.score.composite_score:
                deduped[key] = candidate
        return sorted(deduped.values(), key=lambda item: item.score.composite_score, reverse=True)
