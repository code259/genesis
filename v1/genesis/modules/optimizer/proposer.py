from __future__ import annotations

from genesis.models import ExperimentProposal


class ExperimentProposer:
    def propose_next(self, task_id: str, n: int = 3) -> list[ExperimentProposal]:
        proposals: list[ExperimentProposal] = []
        for index in range(n):
            proposals.append(
                ExperimentProposal(
                    description=f"Experiment variant {index + 1} for task {task_id}",
                    code_diff=f"tune variant {index + 1}",
                    expected_metric=0.5 + 0.1 * index,
                    expected_trajectory=[0.1, 0.2 + 0.1 * index, 0.3 + 0.1 * index],
                    compute_budget="local_gpu",
                    model_parameter_count=1_000_000 * (index + 1),
                )
            )
        return proposals
