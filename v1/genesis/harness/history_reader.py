from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from genesis.storage.filesystem import ProjectFilesystem


class SelectiveHistoryReader:
    def __init__(self, filesystem: ProjectFilesystem, token_budget: Any):
        self.filesystem = filesystem
        self.token_budget = token_budget

    def grep_traces(self, project_id: str, pattern: str, max_results: int = 10) -> list[str]:
        compiled = re.compile(pattern, re.IGNORECASE)
        matches: list[str] = []
        for trace_path in sorted(self.filesystem.get_project_dir(project_id).glob("runs/*/trace.json")):
            text = Path(trace_path).read_text(encoding="utf-8")
            if compiled.search(text):
                matches.append(
                    self.token_budget.trim_to_budget(
                        text,
                        layer_budget=600,
                    )
                )
            if len(matches) >= max_results:
                break
        return matches

    def get_top_k_results(self, project_id: str, k: int = 5) -> list[dict[str, Any]]:
        return self.filesystem.list_all_results(project_id)[:k]

    def get_recent_errors(self, project_id: str, n: int = 3) -> list[str]:
        errors: list[str] = []
        for result_path in sorted(self.filesystem.get_project_dir(project_id).glob("runs/*/result.json"), reverse=True):
            payload = self.filesystem.read_json(result_path)
            if payload.get("errors"):
                errors.extend(str(error) for error in payload["errors"])
            if len(errors) >= n:
                break
        return [
            self.token_budget.trim_to_budget(error, layer_budget=200)
            for error in errors[:n]
        ]

    def get_adversarial_summary(self, project_id: str, run_n: int) -> dict[str, Any]:
        report_path = self.filesystem.get_run_dir(project_id, run_n) / "adversarial_report.json"
        if not report_path.exists():
            return {}
        return self.filesystem.read_json(report_path)

    def summarize_experiment_history(self, project_id: str) -> str:
        results = self.get_top_k_results(project_id, k=5)
        progress_lines = []
        failure_lines = []
        for result in results:
            task_id = result.get("task_id", "unknown")
            metric = result.get("primary_metric", 0)
            summary = str(result.get("summary", "")).strip()
            classification = str(result.get("classification", "")).strip()
            generated_artifacts = result.get("generated_artifacts", [])
            executed_commands = result.get("executed_commands", [])
            failure_summary = str(result.get("failure_summary", "")).strip()
            if generated_artifacts or executed_commands or classification == "success":
                line = f"- {task_id}: metric={metric}"
                if summary:
                    line += f" | {summary}"
                progress_lines.append(line)
            else:
                line = f"- failed {task_id}: {classification or 'unknown_failure'}"
                if failure_summary:
                    line += f" | {failure_summary}"
                failure_lines.append(line)
        for report_path in sorted(self.filesystem.get_project_dir(project_id).glob("runs/*/adversarial_report.json"), reverse=True)[:3]:
            payload = self.filesystem.read_json(report_path)
            blockers = payload.get("critical_blockers", [])
            if blockers:
                failure_lines.append(
                    f"- adversarial blockers run {report_path.parent.name}: {', '.join(str(blocker) for blocker in blockers[:3])}"
                )
        lines = progress_lines + failure_lines
        return self.token_budget.trim_to_budget("\n".join(lines), layer_budget=1200)
