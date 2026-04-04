import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from core import project_runtime


def test_ensure_project_runtime_creates_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "projects" / "demo"
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.md").write_text("agent")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "genesis-runtime.Dockerfile").write_text("FROM python:3.9-slim")
    project_path.mkdir(parents=True)
    commands = []

    def fake_docker(command, check=True):
        commands.append(command)
        if command[:4] == ["docker", "version", "--format", "{{.Server.Version}}"]:
            return type("Result", (), {"returncode": 0, "stdout": "27.0.0\n", "stderr": ""})()
        if command[:4] == ["docker", "image", "inspect", config.RUNTIME_IMAGE_TAG]:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "missing"})()
        if command[:2] == ["docker", "build"]:
            return type("Result", (), {"returncode": 0, "stdout": "built", "stderr": ""})()
        if command[:3] == ["docker", "container", "inspect"]:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "missing"})()
        if command[:2] == ["docker", "run"]:
            return type("Result", (), {"returncode": 0, "stdout": "container-id", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(project_runtime, "_docker_command", fake_docker)

    handle = project_runtime.ensure_project_runtime(project_path)

    assert handle.container_name == "genesis-demo"
    state = json.loads(project_runtime.runtime_state_path(project_path).read_text())
    assert state["container_state"] == "ready"
    assert any(command[:2] == ["docker", "build"] for command in commands)


def test_ensure_project_runtime_reuses_running_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "projects" / "demo"
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.md").write_text("agent")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "genesis-runtime.Dockerfile").write_text("FROM python:3.9-slim")
    project_path.mkdir(parents=True)
    commands = []

    def fake_docker(command, check=True):
        commands.append(command)
        if command[:4] == ["docker", "version", "--format", "{{.Server.Version}}"]:
            return type("Result", (), {"returncode": 0, "stdout": "27.0.0\n", "stderr": ""})()
        if command[:4] == ["docker", "image", "inspect", config.RUNTIME_IMAGE_TAG]:
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        if command[:3] == ["docker", "container", "inspect"]:
            return type("Result", (), {"returncode": 0, "stdout": "running\n", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(project_runtime, "_docker_command", fake_docker)

    handle = project_runtime.ensure_project_runtime(project_path)

    assert handle.state == "ready"
    assert not any(command[:2] == ["docker", "run"] for command in commands)


def test_run_in_runtime_executes_python_and_updates_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "projects" / "demo"
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.md").write_text("agent")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "genesis-runtime.Dockerfile").write_text("FROM python:3.9-slim")
    script_path = project_path / "stages" / "stage_1" / "S1T1_artifacts" / "run.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("print('ok')")

    def fake_docker(command, check=True):
        if command[:4] == ["docker", "version", "--format", "{{.Server.Version}}"]:
            return type("Result", (), {"returncode": 0, "stdout": "27.0.0\n", "stderr": ""})()
        if command[:4] == ["docker", "image", "inspect", config.RUNTIME_IMAGE_TAG]:
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        if command[:3] == ["docker", "container", "inspect"]:
            return type("Result", (), {"returncode": 0, "stdout": "running\n", "stderr": ""})()
        if command[:2] == ["docker", "exec"] and "pip" in command:
            return type("Result", (), {"returncode": 0, "stdout": '[{"name":"pip","version":"26.0.1"}]', "stderr": ""})()
        if command[:2] == ["docker", "exec"]:
            return type("Result", (), {"returncode": 0, "stdout": "ok\n", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(project_runtime, "_docker_command", fake_docker)

    result = project_runtime.run_in_runtime(project_path, "S1T1", script_path, {"GENESIS_RESULTS_PATH": "projects/demo/results.json"})

    assert result.returncode == 0
    state = json.loads(project_runtime.runtime_state_path(project_path).read_text())
    assert state["installed_packages_preview"] == ["pip==26.0.1"]


def test_ensure_project_runtime_recovers_name_conflict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "projects" / "demo"
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.md").write_text("agent")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "genesis-runtime.Dockerfile").write_text("FROM python:3.9-slim")
    project_path.mkdir(parents=True)
    commands = []

    def fake_docker(command, check=True):
        commands.append(command)
        if command[:4] == ["docker", "version", "--format", "{{.Server.Version}}"]:
            return type("Result", (), {"returncode": 0, "stdout": "27.0.0\n", "stderr": ""})()
        if command[:4] == ["docker", "image", "inspect", config.RUNTIME_IMAGE_TAG]:
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        if command[:3] == ["docker", "container", "inspect"]:
            inspect_calls = sum(1 for item in commands if item[:3] == ["docker", "container", "inspect"])
            if inspect_calls == 1:
                return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "missing"})()
            return type("Result", (), {"returncode": 0, "stdout": "exited\n", "stderr": ""})()
        if command[:2] == ["docker", "run"]:
            return type(
                "Result",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": 'docker: Error response from daemon: Conflict. The container name "/genesis-demo" is already in use.',
                },
            )()
        if command[:2] == ["docker", "start"]:
            return type("Result", (), {"returncode": 0, "stdout": "genesis-demo\n", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(project_runtime, "_docker_command", fake_docker)

    handle = project_runtime.ensure_project_runtime(project_path)

    assert handle.state == "ready"
    assert any(command[:2] == ["docker", "start"] for command in commands)


def test_ensure_project_runtime_fails_loudly_without_docker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "projects" / "demo"
    project_path.mkdir(parents=True)

    monkeypatch.setattr(
        project_runtime,
        "_docker_command",
        lambda *args, **kwargs: type("Result", (), {"returncode": 1, "stdout": "", "stderr": "docker missing"})(),
    )

    with pytest.raises(RuntimeError, match="Docker is unavailable"):
        project_runtime.ensure_project_runtime(project_path)


def test_teardown_project_runtime_updates_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_path = tmp_path / "projects" / "demo"
    project_path.mkdir(parents=True)
    state_path = project_runtime.runtime_state_path(project_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "backend": "docker",
                "container_name": "genesis-demo",
                "image_tag": config.RUNTIME_IMAGE_TAG,
                "project_id": "demo",
                "container_state": "ready",
                "created_at": "now",
                "last_used_at": "now",
                "last_error": None,
                "installed_packages_preview": [],
            }
        )
    )
    calls = []
    monkeypatch.setattr(
        project_runtime,
        "_docker_command",
        lambda command, check=False: calls.append(command) or type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    project_runtime.teardown_project_runtime(project_path)

    state = json.loads(state_path.read_text())
    assert state["container_state"] == "stopped"
    assert any(command[:3] == ["docker", "rm", "-f"] for command in calls)
