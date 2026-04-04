from __future__ import annotations

from genesis.models import ExperimentProposal
from genesis.storage.ledger import ExperimentLedger


class ExperimentProposer:
    def propose_next(
        self,
        task_id: str,
        n: int = 3,
        *,
        prior_metric: float = 0.0,
        compute_budget: str = "local_gpu",
        ledger: ExperimentLedger | None = None,
    ) -> list[ExperimentProposal]:
        history = ledger.get_by_task(task_id)[:3] if ledger else []
        prior_metric = max(
            [prior_metric] + [float(item.get("primary_metric", 0.0)) for item in history]
        )
        proposals: list[ExperimentProposal] = []
        for index in range(n):
            expected_metric = max(prior_metric, 0.35) + 0.05 * (index + 1)
            warmup_ratio = round(0.2 + 0.1 * index, 2)
            proposals.append(
                ExperimentProposal(
                    description=f"Experiment variant {index + 1} for task {task_id}",
                    code_diff=f"learning_rate=0.{index + 2}; warmup_ratio={warmup_ratio}",
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
