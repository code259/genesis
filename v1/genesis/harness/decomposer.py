from __future__ import annotations

from hashlib import sha1
from typing import Iterable

from genesis.config import ProjectConfig
from genesis.models import TaskNode, TaskTree


class TaskDecomposer:
    def decompose(self, config: ProjectConfig) -> TaskTree:
        tasks: list[TaskNode] = []
        root_id = self._task_id(config.research_question, "root")

        literature = self._task(
            config,
            description=f"Survey prior work relevant to: {config.research_question}",
            suffix="literature",
            dependencies=[],
            success_metric="produce grounded literature map",
            requires_ml_optimizer=False,
        )
        oracle = self._task(
            config,
            description=f"Generate and validate an oracle for domain {config.domain}",
            suffix="oracle",
            dependencies=[literature.task_id],
            success_metric="validated oracle coverage",
            requires_ml_optimizer=False,
        )
        experiment = self._task(
            config,
            description=f"Run controlled experiments for: {config.research_question}",
            suffix="experiments",
            dependencies=[literature.task_id],
            success_metric=config.success_criteria[0] if config.success_criteria else "improve primary metric",
            requires_ml_optimizer=config.domain in {"ml_efficiency", "astrophysics"} or bool(config.success_criteria),
        )
        verification = self._task(
            config,
            description="Verify experiment outputs against oracle and literature",
            suffix="verification",
            dependencies=[oracle.task_id, experiment.task_id],
            success_metric="verification passed",
            requires_ml_optimizer=False,
        )
        synthesis = self._task(
            config,
            description="Synthesize the final paper and result package",
            suffix="paper",
            dependencies=[verification.task_id, literature.task_id],
            success_metric="paper artifact generated",
            requires_ml_optimizer=False,
        )
        tasks.extend([literature, oracle, experiment, verification, synthesis])
        return TaskTree(root_id=root_id, tasks=tasks)

    def amend(self, tree: TaskTree, rationale: str) -> TaskTree:
        if not tree.tasks:
            return tree
        amended = list(tree.tasks)
        amended.append(
            TaskNode(
                task_id=self._task_id(rationale, f"amend-{len(amended) + 1}"),
                description=f"Amendment task: {rationale}",
                acceptance_criteria=[f"Amendment rationale recorded: {rationale}"],
                oracle_checks=[],
                estimated_compute_budget=amended[-1].estimated_compute_budget,
                dependencies=[amended[-1].task_id],
                success_metric="amendment incorporated",
                requires_ml_optimizer=False,
            )
        )
        return TaskTree(root_id=tree.root_id, tasks=amended)

    def _task(
        self,
        config: ProjectConfig,
        *,
        description: str,
        suffix: str,
        dependencies: Iterable[str],
        success_metric: str,
        requires_ml_optimizer: bool,
    ) -> TaskNode:
        return TaskNode(
            task_id=self._task_id(config.research_question, suffix),
            description=description,
            acceptance_criteria=self._acceptance_criteria(config, description, success_metric),
            oracle_checks=config.oracle_hints.copy(),
            estimated_compute_budget=config.compute_budget,
            dependencies=list(dependencies),
            success_metric=success_metric,
            requires_ml_optimizer=requires_ml_optimizer,
        )

    def _task_id(self, question: str, suffix: str) -> str:
        return sha1(f"{question}:{suffix}".encode("utf-8")).hexdigest()[:8]

    def _acceptance_criteria(
        self, config: ProjectConfig, description: str, success_metric: str
    ) -> list[str]:
        criteria = [
            description,
            f"Track success using: {success_metric}",
            f"Stay within compute budget: {config.compute_budget}",
        ]
        criteria.extend(config.success_criteria[:2])
        return criteria
