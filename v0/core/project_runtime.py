from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import config  # type: ignore


@dataclass
class RuntimeHandle:
    backend: str
    container_name: str
    image_tag: str
    project_id: str
    state: str


@dataclass
class RuntimeExecResult:
    stdout: str
    stderr: str
    returncode: int


def ensure_project_runtime(project_path: Path) -> RuntimeHandle:
    project_path = project_path.resolve()
    project_id = project_path.name
    container_name = f"genesis-{project_id}"
    image_tag = config.RUNTIME_IMAGE_TAG

    _ensure_docker_available()
    _ensure_runtime_image(project_path)

    container_state = _container_state(container_name)
    if container_state is None:
        container_state = _create_or_recover_container(project_path, container_name, image_tag)
    elif container_state != "running":
        _docker_command(["docker", "start", container_name])
        container_state = "running"

    handle = RuntimeHandle(
        backend="docker",
        container_name=container_name,
        image_tag=image_tag,
        project_id=project_id,
        state="ready" if container_state == "running" else container_state,
    )
    _write_runtime_state(project_path, handle, last_error=None)
    return handle


def run_in_runtime(project_path: Path, task_id: str, script_path: Path, env: dict[str, str]) -> RuntimeExecResult:
    project_path = project_path.resolve()
    handle = ensure_project_runtime(project_path)
    workspace_root = _workspace_root(project_path)
    container_script = _container_path(workspace_root, script_path.resolve())
    env_args = []
    for key, value in env.items():
        env_args.extend(["-e", f"{key}={value}"])

    process = _docker_command(
        ["docker", "exec", *env_args, handle.container_name, "python", container_script],
        check=False,
    )

    _write_runtime_state(
        project_path,
        handle,
        last_error=process.stderr.strip() or None if process.returncode != 0 else None,
        installed_packages_preview=_installed_packages_preview(handle.container_name),
    )
    return RuntimeExecResult(stdout=process.stdout, stderr=process.stderr, returncode=process.returncode)


def teardown_project_runtime(project_path: Path) -> None:
    project_path = project_path.resolve()
    state = runtime_status(project_path)
    container_name = state.get("container_name")
    if not container_name or state.get("container_state") == "not_created":
        return
    _docker_command(["docker", "rm", "-f", container_name], check=False)
    path = runtime_state_path(project_path)
    if path.exists():
        payload = json.loads(path.read_text())
        payload["container_state"] = "stopped"
        payload["last_used_at"] = _timestamp()
        path.write_text(json.dumps(payload, indent=2))


def runtime_status(project_path: Path) -> dict:
    path = runtime_state_path(project_path.resolve())
    if not path.exists():
        return {
            "backend": "docker",
            "container_state": "not_created",
            "container_name": None,
            "image_tag": config.RUNTIME_IMAGE_TAG,
            "last_used_at": None,
            "last_error": None,
            "installed_packages_preview": [],
        }
    return json.loads(path.read_text())


def runtime_state_path(project_path: Path) -> Path:
    return project_path / "runtime" / "runtime_state.json"


def _ensure_docker_available() -> None:
    process = _docker_command(["docker", "version", "--format", "{{.Server.Version}}"], check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Docker is unavailable for Genesis runtime: {process.stderr.strip() or process.stdout.strip()}")


def _ensure_runtime_image(project_path: Path) -> None:
    inspect = _docker_command(["docker", "image", "inspect", config.RUNTIME_IMAGE_TAG], check=False)
    if inspect.returncode == 0:
        return
    workspace_root = _workspace_root(project_path)
    dockerfile = workspace_root / "docker" / "genesis-runtime.Dockerfile"
    if not dockerfile.exists():
        raise RuntimeError(f"Missing runtime Dockerfile: {dockerfile}")
    build = _docker_command(
        ["docker", "build", "-f", str(dockerfile), "-t", config.RUNTIME_IMAGE_TAG, str(workspace_root)],
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError(f"Failed to build Genesis runtime image: {build.stderr.strip() or build.stdout.strip()}")


def _create_container(project_path: Path, container_name: str, image_tag: str) -> None:
    workspace_root = _workspace_root(project_path)
    project_rel = project_path.resolve().relative_to(workspace_root.resolve())
    container_project_path = f"/workspace/{project_rel.as_posix()}"
    process = _docker_command(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            container_project_path,
            "-v",
            f"{workspace_root.resolve()}:/workspace",
            image_tag,
            "sleep",
            "infinity",
        ],
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Failed to create Genesis runtime container: {process.stderr.strip() or process.stdout.strip()}")


def _create_or_recover_container(project_path: Path, container_name: str, image_tag: str) -> str:
    try:
        _create_container(project_path, container_name, image_tag)
        return "running"
    except RuntimeError as exc:
        message = str(exc)
        if "already in use" not in message:
            raise
        recovered_state = _container_state(container_name)
        if recovered_state is None:
            raise RuntimeError(message)
        if recovered_state != "running":
            _docker_command(["docker", "start", container_name])
            return "running"
        return recovered_state


def _container_state(container_name: str) -> str | None:
    process = _docker_command(
        ["docker", "container", "inspect", container_name, "--format", "{{.State.Status}}"],
        check=False,
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def _installed_packages_preview(container_name: str) -> list[str]:
    process = _docker_command(
        ["docker", "exec", container_name, "python", "-m", "pip", "list", "--format=json"],
        check=False,
    )
    if process.returncode != 0:
        return []
    try:
        packages = json.loads(process.stdout)
    except json.JSONDecodeError:
        return []
    return [f"{item['name']}=={item['version']}" for item in packages[:20]]


def _write_runtime_state(project_path: Path, handle: RuntimeHandle, last_error: str | None, installed_packages_preview: list[str] | None = None) -> None:
    path = runtime_state_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": handle.backend,
        "container_name": handle.container_name,
        "image_tag": handle.image_tag,
        "project_id": handle.project_id,
        "container_state": handle.state,
        "created_at": _read_existing_field(path, "created_at") or _timestamp(),
        "last_used_at": _timestamp(),
        "last_error": last_error,
        "installed_packages_preview": installed_packages_preview or _read_existing_field(path, "installed_packages_preview") or [],
    }
    path.write_text(json.dumps(payload, indent=2))


def _read_existing_field(path: Path, field: str):
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(field)


def _workspace_root(project_path: Path) -> Path:
    for candidate in [project_path, *project_path.parents]:
        if (candidate / "agent" / "agent.md").exists():
            return candidate
    raise RuntimeError(f"Unable to locate Genesis workspace root from {project_path}")


def _container_path(workspace_root: Path, host_path: Path) -> str:
    relative = host_path.resolve().relative_to(workspace_root.resolve())
    return f"/workspace/{relative.as_posix()}"


def _docker_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, capture_output=True, text=True)
    if check and process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    return process


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
