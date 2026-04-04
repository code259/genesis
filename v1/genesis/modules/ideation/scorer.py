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
        novelty = min(1.0, float(idea.metadata.get("density_score", 0.5)) if idea.metadata else 0.5)
        tractability = 1.0 / max(1.0, len(idea.summary.split()) / 25.0)
        connection_quality = 1.0 if task_description.lower().split()[0] in idea.summary.lower() else 0.6
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
            taste_prediction = predicted[0]
        composite = 0.35 * novelty + 0.25 * tractability + 0.2 * connection_quality + 0.2 * taste_prediction
        return IdeaScore(
            novelty=novelty,
            tractability=tractability,
            connection_quality=connection_quality,
            taste_prediction=taste_prediction,
            composite_score=composite,
        )
