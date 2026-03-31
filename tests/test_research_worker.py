import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import research_worker


def test_research_worker_completes_and_persists_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / "project_config.json").write_text("{}")
    (project_path / "global_state.md").write_text("# state")
    (project_path / "conventions.md").write_text("# conventions")
    (project_path / "project_spec.md").write_text("## Research Goal\nGoal")

    task_spec = {
        "id": "S1T1",
        "description": "Fetch a resource and summarize it.",
        "verification_criteria": ["Summary exists"],
        "stage": 1,
    }

    monkeypatch.setattr(
        research_worker,
        "_planner_action",
        lambda *args, **kwargs: {"action": "complete", "summary_markdown": "done", "completion_evidence": ["artifact.txt"], "artifact_paths": []},
    )
    monkeypatch.setattr(research_worker, "_synthesize_observation", lambda *args, **kwargs: "summary")
    monkeypatch.setattr(
        research_worker,
        "_completion_check",
        lambda *args, **kwargs: {"complete": True, "status": "IN_PROGRESS", "reason": "done", "summary_markdown": "done", "completion_evidence": ["artifact.txt"]},
    )

    result = research_worker.run_task(task_spec, project_path)
    state = json.loads(research_worker.worker_state_path(project_path, "S1T1").read_text())
    assert result.success is True
    assert state["ready_for_verification"] is True
    assert state["attempt_count"] == 1


def test_research_worker_marks_verified(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    state_path = research_worker.worker_state_path(project_path, "S1T1")
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"task_status": "IN_PROGRESS", "attempt_count": 1}))

    research_worker.mark_task_verified(project_path, "S1T1", "ACCEPT")
    state = json.loads(state_path.read_text())
    assert state["task_status"] == "VERIFIED"
