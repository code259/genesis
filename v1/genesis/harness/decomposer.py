from __future__ import annotations

from hashlib import sha1

from genesis.config import ProjectConfig
from genesis.models import TaskNode, TaskTree


class TaskDecomposer:
    def decompose(self, config: ProjectConfig) -> TaskTree:
        criteria = config.success_criteria or [
            "Establish a baseline plan",
            "Produce a verified experiment loop",
            "Synthesize a reproducible paper artifact",
        ]
        tasks: list[TaskNode] = []
        previous_id = None
        for index, criterion in enumerate(criteria, start=1):
            task_id = sha1(f"{config.research_question}:{index}".encode("utf-8")).hexdigest()[:8]
            tasks.append(
                TaskNode(
                    task_id=task_id,
                    description=criterion,
                    acceptance_criteria=[criterion],
                    oracle_checks=config.oracle_hints.copy(),
                    estimated_compute_budget=config.compute_budget,
                    dependencies=[previous_id] if previous_id else [],
                    success_metric=criterion,
                    requires_ml_optimizer="experiment" in criterion.lower() or config.domain == "ml_efficiency",
                )
            )
            previous_id = task_id
        return TaskTree(root_id=tasks[0].task_id if tasks else "root", tasks=tasks)

    def amend(self, tree: TaskTree, rationale: str) -> TaskTree:
        if not tree.tasks:
            return tree
        amended = list(tree.tasks)
        amended[-1].acceptance_criteria.append(f"Amendment rationale: {rationale}")
        return TaskTree(root_id=tree.root_id, tasks=amended)
