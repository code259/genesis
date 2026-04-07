from __future__ import annotations

from genesis.config import ProjectConfig
from genesis.modules.adversarial.runtime import AdversarialRuntime


class AcceptanceCriteriaGenerator:
    def __init__(self, runtime: AdversarialRuntime | None = None) -> None:
        self.runtime = runtime or AdversarialRuntime()

    def generate(self, config: ProjectConfig) -> dict[str, list[str]]:
        criteria: list[str] = []
        try:
            criteria = self.runtime.generate_acceptance_criteria(config)
        except Exception:  # noqa: BLE001
            criteria = []
        user_criteria = [criterion.strip() for criterion in config.success_criteria if criterion.strip()]
        criteria.extend(user_criteria)
        if not criteria:
            criteria = [
                f"Address the research question: {config.research_question}",
                "Provide at least one grounded empirical or implementation claim tied to a produced artifact.",
                "Produce artifacts that can be verified automatically.",
            ]
        if not any("artifact" in criterion.lower() for criterion in criteria):
            criteria.append("Produce at least one substantive artifact or executable result relevant to the task.")
        if not any("verification" in criterion.lower() or "oracle" in criterion.lower() for criterion in criteria):
            criteria.append("Satisfy verification and oracle checks for the produced result package.")
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
