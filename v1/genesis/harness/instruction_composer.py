from __future__ import annotations

import re

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
        reasoning = self._reason_about_state(
            current_task_context=current_task_context,
            retrieved_history=retrieved_history,
            requested_modules=requested_modules or [],
            belief_summary=belief_summary,
            domain_context=domain_context,
        )
        sections = [
            "# Objective",
            config.research_question,
            "",
            "# Domain",
            config.domain,
            "",
            "# Current Task Context",
            reasoning["current_task_summary"],
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
            "# Risk Focus",
            *reasoning["risk_focus"],
            "",
            "# Blocking Findings",
            *reasoning["blocker_lines"],
            "",
            "# Adaptive Constraints",
            *reasoning["adaptive_constraints"],
            "",
            "# Explicit Next Action",
            reasoning["explicit_next_action"],
            "",
            "# Execution Contract",
            "- Return actionable work, not prose-only completion claims.",
            "- Emit at least one of: artifact_plan, command_plan, or experiment_plan.",
            "- Commands must be literal executable shell commands.",
            "- File paths must be concrete and relative to the run workspace unless explicitly stated otherwise.",
            "- Do not propose publication, submission, or finalization unless substantive verified artifacts already exist.",
            "",
            "# Validation Expectations",
            "- Produce auditable artifacts.",
            "- Preserve traceability to the current task.",
            "- Surface blockers explicitly.",
        ]
        return "\n".join(sections).strip() + "\n"

    def _reason_about_state(
        self,
        *,
        current_task_context: str,
        retrieved_history: str,
        requested_modules: list[str],
        belief_summary: str,
        domain_context: str,
    ) -> dict[str, object]:
        adaptive_constraints = ["- Prefer the smallest truthful next step that advances the active stage."]
        blocker_lines: list[str] = ["- No explicit blockers surfaced in recent history."]
        risk_focus: list[str] = ["- Default risk focus: preserve forward progress without inventing work."]
        lowered_history = retrieved_history.lower()
        lowered_context = current_task_context.lower()
        current_task_summary = current_task_context or "Continue the next highest-priority unresolved task."
        explicit_next_action = current_task_summary

        blocker_matches = re.findall(r"adversarial blockers run \d+:\s*(.+)", retrieved_history, re.IGNORECASE)
        if blocker_matches:
            blocker_lines = [f"- {match.strip()}" for match in blocker_matches[:3] if match.strip()]
            explicit_next_action = (
                current_task_summary
                + "\nResolve the latest adversarial blockers directly before doing downstream work."
            )
            risk_focus = ["- Primary risk: unresolved adversarial blockers will cause another blocked iteration."]

        verification_matches = re.findall(r"verification_failures[=:]\s*(.+)", retrieved_history, re.IGNORECASE)
        if verification_matches or "verification failed" in lowered_history:
            adaptive_constraints.append("- Address the latest verification failures directly and explain why the next action clears them.")
            risk_focus.append("- Verification is failing; prefer repairs that change the verification outcome, not surface-level output changes.")
            if "Repair the failing verification path before expanding scope." not in explicit_next_action:
                explicit_next_action += "\nRepair the failing verification path before expanding scope."

        if "repeated failure signature" in lowered_history or "escalation" in lowered_context:
            adaptive_constraints.append("- Diagnose repeated failures before retrying the same tactic or command.")
            risk_focus.append("- Repeated failure signatures are present; require a materially different repair strategy.")

        if "ideation_available" in lowered_history and "false" in lowered_history:
            adaptive_constraints.append("- Do not request ideation-dependent pivots until manifold availability is restored.")
            risk_focus.append("- Manifold availability is degraded; avoid ideation as a dependency.")

        if "oracle" in requested_modules:
            adaptive_constraints.append("- Do not rely on an oracle until it passes synthetic validation.")
        if "ideation" in requested_modules:
            adaptive_constraints.append("- Only propose ideation-driven pivots if the current path has materially stalled.")
        if "optimizer" in requested_modules:
            adaptive_constraints.append("- Use runnable experiment commands and concrete artifacts; do not substitute synthetic placeholder runs.")
        if belief_summary.strip():
            risk_focus.append(f"- Taste/DAG context: {belief_summary.strip()}")
        if domain_context.strip():
            risk_focus.append("- Domain context is available; use it only when it materially changes the task decision.")

        return {
            "current_task_summary": current_task_summary,
            "explicit_next_action": explicit_next_action,
            "adaptive_constraints": adaptive_constraints,
            "blocker_lines": blocker_lines,
            "risk_focus": risk_focus,
        }
