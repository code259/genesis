from __future__ import annotations

from genesis.models import Idea, IdeaScore


class IdeaScorer:
    def score(self, idea: Idea, task_description: str, taste_prediction: float = 0.5) -> IdeaScore:
        novelty = min(1.0, float(idea.metadata.get("density_score", 0.5)) if idea.metadata else 0.5)
        tractability = 1.0 / max(1.0, len(idea.summary.split()) / 25.0)
        connection_quality = 1.0 if task_description.lower().split()[0] in idea.summary.lower() else 0.6
        composite = 0.35 * novelty + 0.25 * tractability + 0.2 * connection_quality + 0.2 * taste_prediction
        return IdeaScore(
            novelty=novelty,
            tractability=tractability,
            connection_quality=connection_quality,
            taste_prediction=taste_prediction,
            composite_score=composite,
        )
