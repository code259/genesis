import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.supervisor import Decision, SupervisorDecision
from scripts import run_stage as run_stage_module


def _write_project(tmp_path: Path) -> Path:
    project_path = tmp_path / "projects" / "demo"
    project_path.mkdir(parents=True)
    (project_path / "conventions.md").write_text("# Conventions\n")
    (project_path / "global_state.md").write_text("# Global State\n")
    (project_path / "project_config.json").write_text(json.dumps({"oracle_module": None}))
    tasks = [
        {
            "id": "S1T1",
            "description": "Task",
            "dependencies": [],
            "stage": 1,
            "verification_criteria": ["Output exists"],
            "complexity": "STANDARD",
            "foundational": True,
        }
    ]
    (project_path / "tasks.json").write_text(json.dumps(tasks))
    return project_path


def test_run_stage_requires_verifier_and_can_close(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_execute_task(spec, project_path, revision_context=""):
        stage_dir = project_path / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "S1T1.md").write_text("task output")
        return "task output"

    monkeypatch.setattr(run_stage_module, "execute_task", fake_execute_task)
    monkeypatch.setattr(
        run_stage_module,
        "evaluate_output",
        lambda task_id, output, task_spec: SupervisorDecision(Decision.ACCEPT, ["ok"], task_id),
    )
    monkeypatch.setattr(
        run_stage_module,
        "run_oracle_checks",
        lambda *args, **kwargs: type("OracleResult", (), {"configured": False, "applicable": False, "oracle_pass": None, "summary": {}, "failures": [], "warnings": [], "notes": ["No oracle configured"], "report_path": None})(),
    )
    monkeypatch.setattr(
        run_stage_module,
        "verify",
        lambda *args, **kwargs: {"status": "ACCEPT", "checks": [], "open_items": [], "raw_text": "RECOMMENDATION: ACCEPT", "oracle_summary": {}},
    )
    monkeypatch.setattr(run_stage_module, "build_stage_summary", lambda project_path, stage: project_path / "stages" / "stage_1" / "summary.md")
    monkeypatch.setattr(run_stage_module, "build_paper_package", lambda project_path: project_path / "paper")
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    run_stage_module.run_stage("demo", 1)

    verify_json = project_path / "stages" / "stage_1" / "S1T1_verify.json"
    assert verify_json.exists()
    assert json.loads(verify_json.read_text())["status"] == "ACCEPT"


def test_run_stage_blocks_on_revise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_execute_task(spec, project_path, revision_context=""):
        stage_dir = project_path / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "S1T1.md").write_text("task output")
        return "task output"

    monkeypatch.setattr(run_stage_module, "execute_task", fake_execute_task)
    monkeypatch.setattr(
        run_stage_module,
        "evaluate_output",
        lambda task_id, output, task_spec: SupervisorDecision(Decision.ACCEPT, ["ok"], task_id),
    )
    monkeypatch.setattr(
        run_stage_module,
        "run_oracle_checks",
        lambda *args, **kwargs: type("OracleResult", (), {"configured": False, "applicable": False, "oracle_pass": None, "summary": {}, "failures": [], "warnings": [], "notes": ["No oracle configured"], "report_path": None})(),
    )
    monkeypatch.setattr(
        run_stage_module,
        "verify",
        lambda *args, **kwargs: {"status": "REVISE", "checks": [], "open_items": ["fix"], "raw_text": "RECOMMENDATION: REVISE", "oracle_summary": {}},
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    run_stage_module.run_stage("demo", 1)

    gate = run_stage_module.check_stage_gate(project_path, 1, json.loads((project_path / "tasks.json").read_text()))
    assert gate["can_close"] is False


def test_run_stage_skip_human_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GENESIS_SKIP_HUMAN_CHECKPOINTS", "1")

    def fake_execute_task(spec, project_path, revision_context=""):
        stage_dir = project_path / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "S1T1.md").write_text("task output")
        return "task output"

    monkeypatch.setattr(run_stage_module, "execute_task", fake_execute_task)
    monkeypatch.setattr(
        run_stage_module,
        "evaluate_output",
        lambda task_id, output, task_spec: SupervisorDecision(Decision.ACCEPT, ["ok"], task_id),
    )
    monkeypatch.setattr(
        run_stage_module,
        "run_oracle_checks",
        lambda *args, **kwargs: type("OracleResult", (), {"configured": False, "applicable": False, "oracle_pass": None, "summary": {}, "failures": [], "warnings": [], "notes": ["No oracle configured"], "report_path": None})(),
    )
    monkeypatch.setattr(
        run_stage_module,
        "verify",
        lambda *args, **kwargs: {"status": "ACCEPT", "checks": [], "open_items": [], "raw_text": "RECOMMENDATION: ACCEPT", "oracle_summary": {}},
    )
    monkeypatch.setattr(run_stage_module, "build_stage_summary", lambda project_path, stage: project_path / "stages" / "stage_1" / "summary.md")
    monkeypatch.setattr(run_stage_module, "build_paper_package", lambda project_path: project_path / "paper")
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("input should not be called")))

    run_stage_module.run_stage("demo", 1)

    run_state = json.loads((project_path / "run_state.json").read_text())
    assert run_state["phase"] == "stage_closed"
    assert run_state["awaiting_human_review"] is False


def test_run_stage_continues_to_independent_task_after_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    tasks = [
        {
            "id": "S1T1",
            "description": "Partial task",
            "dependencies": [],
            "stage": 1,
            "verification_criteria": ["Output exists"],
            "complexity": "STANDARD",
            "foundational": False,
        },
        {
            "id": "S1T2",
            "description": "Independent task",
            "dependencies": [],
            "stage": 1,
            "verification_criteria": ["Output exists"],
            "complexity": "STANDARD",
            "foundational": False,
        },
    ]
    (project_path / "tasks.json").write_text(json.dumps(tasks))

    def fake_execute_task(spec, project_path, revision_context=""):
        stage_dir = project_path / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / f"{spec['id']}.md").write_text("task output")
        worker_dir = stage_dir / f"{spec['id']}_worker"
        worker_dir.mkdir(parents=True, exist_ok=True)
        if spec["id"] == "S1T1":
            (worker_dir / "worker_state.json").write_text(
                json.dumps(
                    {
                        "task_status": "PARTIAL",
                        "last_reason": "Deferred non-critical issue",
                        "deferred_issues": [{"reason": "Missing optional snapshot", "blocks_dependents": False}],
                        "ready_for_verification": False,
                    }
                )
            )
        else:
            (worker_dir / "worker_state.json").write_text(
                json.dumps({"task_status": "IN_PROGRESS", "last_reason": "done", "ready_for_verification": True})
            )
        return "task output"

    monkeypatch.setattr(run_stage_module, "execute_task", fake_execute_task)
    monkeypatch.setattr(
        run_stage_module,
        "evaluate_output",
        lambda task_id, output, task_spec: SupervisorDecision(Decision.ACCEPT, ["ok"], task_id),
    )
    monkeypatch.setattr(
        run_stage_module,
        "run_oracle_checks",
        lambda *args, **kwargs: type("OracleResult", (), {"configured": False, "applicable": False, "oracle_pass": None, "summary": {}, "failures": [], "warnings": [], "notes": ["No oracle configured"], "report_path": None})(),
    )
    monkeypatch.setattr(
        run_stage_module,
        "verify",
        lambda *args, **kwargs: {"status": "ACCEPT", "checks": [], "open_items": [], "raw_text": "RECOMMENDATION: ACCEPT", "oracle_summary": {}},
    )
    monkeypatch.setattr(run_stage_module, "build_stage_summary", lambda project_path, stage: project_path / "stages" / "stage_1" / "summary.md")
    monkeypatch.setattr(run_stage_module, "build_paper_package", lambda project_path: project_path / "paper")
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    run_stage_module.run_stage("demo", 1)

    verify_json = project_path / "stages" / "stage_1" / "S1T2_verify.json"
    assert verify_json.exists()
    partial_state = json.loads((project_path / "stages" / "stage_1" / "S1T1_worker" / "worker_state.json").read_text())
    assert partial_state["task_status"] == "PARTIAL"
