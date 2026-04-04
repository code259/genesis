from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from genesis.config import ProjectConfig
from genesis.models import TaskNode


@dataclass
class OracleSpec:
    metric_name: str
    threshold: Optional[float]
    direction: str


class OracleResolver:
    def resolve_oracle(self, task_node: TaskNode, project_config: ProjectConfig) -> OracleSpec:
        if project_config.success_criteria:
            return OracleSpec(metric_name=project_config.success_criteria[0], threshold=None, direction="maximize")
        if task_node.success_metric:
            return OracleSpec(metric_name=task_node.success_metric, threshold=None, direction="maximize")
        return OracleSpec(metric_name="plausibility_score", threshold=0.5, direction="maximize")
