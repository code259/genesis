from __future__ import annotations

from genesis.config import ProjectConfig


class AcceptanceCriteriaGenerator:
    def generate(self, config: ProjectConfig) -> dict[str, list[str]]:
        criteria = [criterion.strip() for criterion in config.success_criteria if criterion.strip()]
        if not criteria:
            criteria = [
                f"Address the research question: {config.research_question}",
                "Provide at least one grounded empirical claim.",
                "Produce artifacts that can be verified automatically.",
            ]
        for oracle_hint in config.oracle_hints:
            if oracle_hint.strip():
                criteria.append(f"Honor oracle check: {oracle_hint.strip()}")
        deduped: list[str] = []
        seen: set[str] = set()
        for criterion in criteria:
            normalized = criterion.lower()
            if normalized not in seen:
                deduped.append(criterion)
                seen.add(normalized)
        return {"criteria": deduped}
