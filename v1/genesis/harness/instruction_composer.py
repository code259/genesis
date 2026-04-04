from __future__ import annotations

from genesis.config import ProjectConfig


class InstructionComposer:
    def compose(
        self,
        *,
        config: ProjectConfig,
        belief_summary: str,
        retrieved_history: str,
        domain_context: str = "",
        current_task_context: str = "",
    ) -> str:
        sections = [
            "# Objective",
            config.research_question,
            "",
            "# Domain",
            config.domain,
            "",
            "# Current Task Context",
            current_task_context or "Continue the next highest-priority unresolved task.",
            "",
            "# Taste Model Belief Summary",
            belief_summary or "No prior belief summary available.",
            "",
            "# Retrieved History",
            retrieved_history or "No prior history selected.",
            "",
            "# Domain Context",
            domain_context or "No domain context injected.",
            "",
            "# Validation Expectations",
            "- Produce auditable artifacts.",
            "- Preserve traceability to the current task.",
            "- Surface blockers explicitly.",
        ]
        return "\n".join(sections).strip() + "\n"

