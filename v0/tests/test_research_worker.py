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


def test_research_worker_handles_relative_project_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "projects" / "demo"
    project_path.mkdir(parents=True)
    (project_path / "project_config.json").write_text("{}")
    (project_path / "global_state.md").write_text("# state")
    (project_path / "conventions.md").write_text("# conventions")
    (project_path / "project_spec.md").write_text("## Title\nDemo\n\n## Research Goal\nGoal\n\n## Domain Context\nContext")
    monkeypatch.chdir(tmp_path)

    task_spec = {
        "id": "S1T1",
        "description": "Read a project file.",
        "verification_criteria": ["Read succeeds"],
        "stage": 1,
    }

    monkeypatch.setattr(
        research_worker,
        "_planner_action",
        lambda *args, **kwargs: {"action": "read_local_file", "path": "project_config.json"},
    )
    monkeypatch.setattr(research_worker, "_synthesize_observation", lambda *args, **kwargs: "read complete")
    monkeypatch.setattr(
        research_worker,
        "_completion_check",
        lambda *args, **kwargs: {"complete": True, "status": "IN_PROGRESS", "reason": "done", "summary_markdown": "done", "completion_evidence": ["project_config.json"]},
    )

    result = research_worker.run_task(task_spec, Path("projects") / "demo")

    assert result.success is True
    assert "project_config.json" in result.completion_evidence


def test_parse_json_block_tolerates_control_characters():
    raw = """```json
{"action":"complete","summary_markdown":"Line one
Line two","completion_evidence":["artifact.txt"]}
```"""

    parsed = research_worker._parse_json_block(raw)

    assert parsed["action"] == "complete"
    assert "Line two" in parsed["summary_markdown"]


def test_execute_tool_action_missing_local_file_returns_observation(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()

    observation = research_worker.execute_tool_action(
        {"id": "S1T1", "stage": 1},
        project_path,
        {"action": "read_local_file", "path": "project_context/project_spec.md"},
    )

    assert observation["error"] is True
    assert observation["missing_path"] == "project_context/project_spec.md"
    assert "project_context" in observation["text"].lower()


def test_execute_tool_action_denies_project_context_path(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()

    observation = research_worker.execute_tool_action(
        {"id": "S1T1", "stage": 1},
        project_path,
        {"action": "read_local_file", "path": "project_context/project_config.json"},
    )

    assert observation["error"] is True
    assert "denied" in observation["text"].lower()


def test_execute_tool_action_marks_missing_artifact_issue_deferrable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()

    class FakeResult:
        success = False
        artifact_paths = []
        stdout = "installed packages successfully"
        stderr = ""
        results_path = "stages/stage_1/S1T2_artifacts/results.json"
        missing_artifacts = ["install_log.txt"]

    monkeypatch.setattr(research_worker, "run_python_task", lambda *args, **kwargs: FakeResult())
    observation = research_worker.execute_tool_action(
        {"id": "S1T2", "stage": 1},
        project_path,
        {"action": "run_python", "python_code": "print('ok')", "expected_artifacts": [{"path": "install_log.txt", "kind": "txt"}]},
    )

    assert observation["issue_class"] == "deferrable_issue"
    assert observation["blocks_task"] is False
