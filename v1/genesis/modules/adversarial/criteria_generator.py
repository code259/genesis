from __future__ import annotations

from genesis.config import ProjectConfig


class AcceptanceCriteriaGenerator:
    def generate(self, config: ProjectConfig) -> dict[str, list[str]]:
        criteria = config.success_criteria.copy()
        if not criteria:
            criteria = [
                "Demonstrate progress against the research question.",
                "Provide at least one grounded empirical claim.",
                "Produce artifacts that can be verified automatically.",
            ]
        return {"criteria": criteria}
