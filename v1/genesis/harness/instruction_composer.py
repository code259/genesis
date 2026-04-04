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
        budget_allocations: dict[str, int] | None = None,
        requested_modules: list[str] | None = None,
    ) -> str:
        budget_lines = [
            f"- {name}: {tokens}"
            for name, tokens in sorted((budget_allocations or {}).items())
        ] or ["- No explicit token budget provided."]
        module_lines = [
            f"- {module}"
            for module in (requested_modules or [])
        ] or ["- Let the coding agent choose the minimal required modules."]
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
            "# Budget",
            *budget_lines,
            "",
            "# Retrieved History",
            retrieved_history or "No prior history selected.",
            "",
            "# Domain Context",
            domain_context or "No domain context injected.",
            "",
            "# Requested Modules",
            *module_lines,
            "",
            "# Explicit Next Action",
            current_task_context or "Continue the next highest-priority unresolved task.",
            "",
            "# Validation Expectations",
            "- Produce auditable artifacts.",
            "- Preserve traceability to the current task.",
            "- Surface blockers explicitly.",
        ]
        return "\n".join(sections).strip() + "\n"
