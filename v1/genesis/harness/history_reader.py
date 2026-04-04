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
                matches.append(text)
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
                errors.extend(payload["errors"])
            if len(errors) >= n:
                break
        return errors[:n]

    def get_adversarial_summary(self, project_id: str, run_n: int) -> dict[str, Any]:
        report_path = self.filesystem.get_run_dir(project_id, run_n) / "adversarial_report.json"
        if not report_path.exists():
            return {}
        return self.filesystem.read_json(report_path)

    def summarize_experiment_history(self, project_id: str) -> str:
        results = self.get_top_k_results(project_id, k=5)
        lines = [f"- {result.get('task_id', 'unknown')}: {result.get('primary_metric', 0)}" for result in results]
        return "\n".join(lines)

