from __future__ import annotations

from genesis.models import ExperimentProposal


class ExperimentProposer:
    def propose_next(
        self,
        task_id: str,
        n: int = 3,
        *,
        prior_metric: float = 0.0,
        compute_budget: str = "local_gpu",
    ) -> list[ExperimentProposal]:
        proposals: list[ExperimentProposal] = []
        for index in range(n):
            expected_metric = max(prior_metric, 0.35) + 0.05 * (index + 1)
            proposals.append(
                ExperimentProposal(
                    description=f"Experiment variant {index + 1} for task {task_id}",
                    code_diff=f"tune variant {index + 1}",
                    expected_metric=round(expected_metric, 4),
                    expected_trajectory=[
                        round(expected_metric * 0.35, 4),
                        round(expected_metric * 0.7, 4),
                        round(expected_metric, 4),
                    ],
                    compute_budget=compute_budget,
                    model_parameter_count=1_000_000 * (index + 1),
                )
            )
        return proposals
