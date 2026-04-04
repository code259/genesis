from __future__ import annotations

from genesis.models import ExperimentProposal, Idea, IdeaScore
from genesis.taste.features import ExperimentFeatureExtractor
from genesis.taste.gp_model import TasteGP


class IdeaScorer:
    def __init__(self) -> None:
        self.feature_extractor = ExperimentFeatureExtractor()

    def score(
        self,
        idea: Idea,
        task_description: str,
        *,
        taste_model: TasteGP | None = None,
        taste_prediction: float = 0.5,
    ) -> IdeaScore:
        density_score = float(idea.metadata.get("density_score", 0.5)) if idea.metadata else 0.5
        novelty = max(0.0, min(1.0, density_score))
        path_length = int(idea.metadata.get("path_length", 1)) if idea.metadata else 1
        summary_words = len(idea.summary.split())
        tractability = max(0.1, 1.0 - min(0.8, summary_words / 80.0) - min(0.3, (path_length - 1) * 0.05))
        task_tokens = set(task_description.lower().split())
        idea_tokens = set((idea.title + " " + idea.summary).lower().split())
        overlap = len(task_tokens & idea_tokens)
        connection_quality = max(0.2, min(1.0, 0.4 + overlap / max(1, len(task_tokens))))
        if taste_model and taste_model.training_targets:
            proposal_like = self.feature_extractor.extract(
                ExperimentProposal(
                    description=idea.summary,
                    code_diff=idea.source,
                    expected_metric=0.0,
                    expected_trajectory=[novelty, tractability, connection_quality],
                    compute_budget="local_gpu",
                    model_parameter_count=0,
                )
            )
            predicted, _ = taste_model.predict([proposal_like])
            taste_prediction = max(0.0, min(1.0, predicted[0]))
        composite = 0.35 * novelty + 0.25 * connection_quality + 0.2 * tractability + 0.2 * taste_prediction
        return IdeaScore(
            novelty=round(novelty, 6),
            tractability=round(tractability, 6),
            connection_quality=round(connection_quality, 6),
            taste_prediction=round(taste_prediction, 6),
            composite_score=round(composite, 6),
        )
