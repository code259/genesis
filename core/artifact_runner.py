from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import sys

from core import runtime_policy
from core.task_parser import extract_stage


@dataclass
class ExecutionResult:
    success: bool
    artifact_paths: list[str]
    stdout: str
    stderr: str
    script_path: str
    results_path: str
    missing_artifacts: list[str]


def run_python_task(task_spec: dict, project_path: Path, execution_plan: dict) -> ExecutionResult:
    project_path = project_path.resolve()
    task_dir = task_artifact_dir(project_path, task_spec["id"])
    task_dir.mkdir(parents=True, exist_ok=True)

    policy = runtime_policy.validate_python_code(task_spec, execution_plan["python_code"])
    if not policy.allowed:
        results_path = task_dir / "results.json"
        results_path.write_text(json.dumps({"policy_error": policy.reason}, indent=2))
        return ExecutionResult(
            success=False,
            artifact_paths=[str(results_path.relative_to(project_path))],
            stdout="",
            stderr=policy.reason,
            script_path=str((task_dir / "run.py").relative_to(project_path)),
            results_path=str(results_path.relative_to(project_path)),
            missing_artifacts=[item["path"] for item in execution_plan.get("expected_artifacts", [])],
        )

    script_path = task_dir / "run.py"
    results_path = task_dir / "results.json"
    script_path.write_text(execution_plan["python_code"])

    env = os.environ.copy()
    env["GENESIS_ARTIFACT_DIR"] = str(task_dir)
    env["GENESIS_RESULTS_PATH"] = str(results_path)

    process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        env=env,
    )

    expected = []
    for item in execution_plan.get("expected_artifacts", []):
        expected_path = item["path"]
        expected.append(expected_path)
        write_check = runtime_policy.validate_write_path(project_path, task_spec["id"], expected_path)
        if not write_check.allowed:
            missing = [expected_path]
            results_path.write_text(json.dumps({"policy_error": write_check.reason}, indent=2))
            return ExecutionResult(
                success=False,
                artifact_paths=[str(results_path.relative_to(project_path))],
                stdout=process.stdout,
                stderr=write_check.reason,
                script_path=str(script_path.relative_to(project_path)),
                results_path=str(results_path.relative_to(project_path)),
                missing_artifacts=missing,
            )
    missing = [path for path in expected if not (task_dir / path).exists()]

    artifact_paths = []
    for path in sorted(task_dir.rglob("*")):
        if path.is_file() and path.name != "run.py":
            artifact_paths.append(str(path.relative_to(project_path)))

    if not results_path.exists():
        results_path.write_text(
            json.dumps(
                {
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "returncode": process.returncode,
                },
                indent=2,
            )
        )

    success = process.returncode == 0 and not missing
    return ExecutionResult(
        success=success,
        artifact_paths=artifact_paths,
        stdout=process.stdout,
        stderr=process.stderr,
        script_path=str(script_path.relative_to(project_path)),
        results_path=str(results_path.relative_to(project_path)),
        missing_artifacts=missing,
    )


def task_artifact_dir(project_path: Path, task_id: str) -> Path:
    stage = extract_stage(task_id)
    return project_path / "stages" / f"stage_{stage}" / f"{task_id}_artifacts"
