import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.status_manager import build_project_status, update_run_state, write_dashboard, write_project_status


def test_status_manager_builds_status_and_dashboard(tmp_path: Path):
    project_path = tmp_path / "project"
    stage_dir = project_path / "stages" / "stage_1"
    worker_dir = stage_dir / "S1T1_worker"
    worker_dir.mkdir(parents=True)
    (project_path / "project_config.json").write_text(json.dumps({"title": "Demo", "domain": "general"}))
    (project_path / "tasks.json").write_text(
        json.dumps(
            [
                {"id": "S1T1", "description": "Task", "stage": 1, "verification_criteria": ["exists"], "dependencies": [], "complexity": "STANDARD", "foundational": False}
            ]
        )
    )
    (worker_dir / "worker_state.json").write_text(json.dumps({"task_status": "BLOCKED", "last_reason": "Need API key"}))
    update_run_state(project_path, phase="blocked", next_human_action="Provide API key")

    status = build_project_status(project_path)
    status_path = write_project_status(project_path)
    dashboard_path = write_dashboard(project_path)

    assert status["tasks"][0]["status"] == "BLOCKED"
    assert status["next_required_human_action"] == "Provide API key"
    assert status_path.exists()
    assert dashboard_path.exists()
