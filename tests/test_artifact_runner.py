import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import artifact_runner


def test_run_python_task_produces_expected_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    task_spec = {"id": "S1T3"}
    plan = {
        "python_code": """
from pathlib import Path
import json
import os

artifact_dir = Path(os.environ["GENESIS_ARTIFACT_DIR"])
results_path = Path(os.environ["GENESIS_RESULTS_PATH"])
(artifact_dir / "data.csv").write_text("x,y\\n1,2\\n")
(artifact_dir / "figure_1.png").write_bytes(b"png")
results_path.write_text(json.dumps({"oracle_inputs": []}))
""",
        "expected_artifacts": [
            {"path": "data.csv", "kind": "csv"},
            {"path": "figure_1.png", "kind": "png"},
        ],
    }

    def fake_run_in_runtime(project_path: Path, task_id: str, script_path: Path, env: dict[str, str]):
        artifact_dir = project_path / env["GENESIS_ARTIFACT_DIR"]
        results_path = project_path / env["GENESIS_RESULTS_PATH"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "data.csv").write_text("x,y\n1,2\n")
        (artifact_dir / "figure_1.png").write_bytes(b"png")
        results_path.write_text(json.dumps({"oracle_inputs": []}))
        return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(artifact_runner.project_runtime, "run_in_runtime", fake_run_in_runtime)
    result = artifact_runner.run_python_task(task_spec, project_path, plan)
    assert result.success is True
    assert any(path.endswith("data.csv") for path in result.artifact_paths)
    assert any(path.endswith("figure_1.png") for path in result.artifact_paths)


def test_run_python_task_fails_when_artifact_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    task_spec = {"id": "S1T4"}
    plan = {
        "python_code": "print('no artifacts')",
        "expected_artifacts": [{"path": "missing.csv", "kind": "csv"}],
    }

    monkeypatch.setattr(
        artifact_runner.project_runtime,
        "run_in_runtime",
        lambda *args, **kwargs: type("Result", (), {"stdout": "no artifacts", "stderr": "", "returncode": 0})(),
    )
    result = artifact_runner.run_python_task(task_spec, project_path, plan)
    assert result.success is False
    assert result.missing_artifacts == ["missing.csv"]


def test_run_python_task_uses_container_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    task_spec = {"id": "S1T5"}
    calls = {}

    def fake_run_in_runtime(project_path: Path, task_id: str, script_path: Path, env: dict[str, str]):
        calls["task_id"] = task_id
        calls["script_path"] = script_path
        calls["env"] = env
        results_path = project_path / env["GENESIS_RESULTS_PATH"]
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({"ok": True}))
        return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(artifact_runner.project_runtime, "run_in_runtime", fake_run_in_runtime)

    artifact_runner.run_python_task(
        task_spec,
        project_path,
        {"python_code": "print('ok')", "expected_artifacts": []},
    )

    assert calls["task_id"] == "S1T5"
    assert "GENESIS_ARTIFACT_DIR" in calls["env"]
