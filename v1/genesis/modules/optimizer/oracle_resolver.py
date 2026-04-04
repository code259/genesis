from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from genesis.config import ProjectConfig
from genesis.models import TaskNode


@dataclass
class OracleSpec:
    metric_name: str
    threshold: Optional[float]
    direction: str
    metric_function: Callable[[dict[str, Any]], float]

    def evaluate(self, payload: dict[str, Any]) -> float:
        return self.metric_function(payload)


class OracleResolver:
    def resolve_oracle(self, task_node: TaskNode, project_config: ProjectConfig) -> OracleSpec:
        human_spec = self._parse_metric_spec(project_config.success_criteria[0]) if project_config.success_criteria else None
        if human_spec:
            return human_spec
        task_spec = self._parse_metric_spec(task_node.success_metric) if task_node.success_metric else None
        if task_spec:
            return task_spec
        if project_config.domain == "ml_efficiency":
            return OracleSpec(
                metric_name="primary_metric",
                threshold=0.5,
                direction="maximize",
                metric_function=lambda payload: float(payload.get("primary_metric", 0.0)),
            )
        return OracleSpec(
            metric_name="plausibility_score",
            threshold=0.5,
            direction="maximize",
            metric_function=lambda payload: float(payload.get("primary_metric", payload.get("plausibility_score", 0.0))),
        )

    def _parse_metric_spec(self, raw: str) -> Optional[OracleSpec]:
        value = raw.strip()
        if not value:
            return None
        numeric_match = re.search(r"([<>]=?)\s*([0-9]*\.?[0-9]+)", value)
        threshold = float(numeric_match.group(2)) if numeric_match else None
        comparator = numeric_match.group(1) if numeric_match else ""
        direction = "minimize" if comparator.startswith("<") or "loss" in value.lower() else "maximize"
        metric_name = self._extract_metric_name(value)
        return OracleSpec(
            metric_name=metric_name,
            threshold=threshold,
            direction=direction,
            metric_function=self._metric_getter(metric_name),
        )

    def _extract_metric_name(self, value: str) -> str:
        normalized = re.sub(r"track success using:\s*", "", value, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"(must|should|be|achieve|reach|keep|at|least|most|more|less|than|within)\b", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"[<>]=?\s*[0-9]*\.?[0-9]+", " ", normalized)
        normalized = re.sub(r"[^a-z0-9_]+", "_", normalized.lower()).strip("_")
        return normalized or "primary_metric"

    def _metric_getter(self, metric_name: str) -> Callable[[dict[str, Any]], float]:
        aliases = [metric_name]
        if metric_name not in {"primary_metric", "plausibility_score"}:
            aliases.extend(["primary_metric", "score"])

        def _getter(payload: dict[str, Any]) -> float:
            secondary = payload.get("secondary_metrics", {})
            for key in aliases:
                if key in payload:
                    return float(payload[key])
                if isinstance(secondary, dict) and key in secondary:
                    return float(secondary[key])
            return float(payload.get("primary_metric", 0.0))

        return _getter
