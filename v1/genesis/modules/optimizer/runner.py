from __future__ import annotations

import hashlib
import json
import shutil
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from genesis.models import ExperimentResult, ensure_parent


class ExperimentRunner:
    def __init__(self, sandbox_root: Union[str, Path]):
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

    def run(
        self, task_id: str, config: dict[str, Any], sandbox_name: Optional[str] = None
    ) -> ExperimentResult:
        experiment_id = sandbox_name or str(uuid.uuid4())
        sandbox_dir = self.sandbox_root / experiment_id
        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        plan_path = sandbox_dir / "plan.json"
        plan = self._normalize_plan(task_id, config, experiment_id)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        artifact_path = sandbox_dir / "result.json"
        command = self._resolve_command(plan, plan_path, artifact_path)
        process = subprocess.run(
            command,
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        runtime_seconds = time.time() - started
        if process.returncode != 0:
            error_path = sandbox_dir / "stderr.log"
            error_path.write_text(process.stderr, encoding="utf-8")
            return ExperimentResult(
                experiment_id=experiment_id,
                task_id=task_id,
                primary_metric=0.0,
                secondary_metrics={
                    "returncode": float(process.returncode),
                    "stdout_chars": float(len(process.stdout)),
                    "stderr_chars": float(len(process.stderr)),
                },
                trajectory=[],
                peak_memory=0.0,
                runtime_seconds=runtime_seconds,
                status="crash",
                code_hash=self._code_hash(plan),
                artifact_path=str(error_path),
                anomaly_score=self._trajectory_anomaly_score([], plan.get("expected_trajectory", [])),
            )
        if not artifact_path.exists():
            missing_output = sandbox_dir / "missing_output.txt"
            missing_output.write_text("experiment command completed without producing result.json", encoding="utf-8")
            return ExperimentResult(
                experiment_id=experiment_id,
                task_id=task_id,
                primary_metric=0.0,
                secondary_metrics={"returncode": 0.0},
                trajectory=[],
                peak_memory=0.0,
                runtime_seconds=runtime_seconds,
                status="crash",
                code_hash=self._code_hash(plan),
                artifact_path=str(missing_output),
                anomaly_score=self._trajectory_anomaly_score([], plan.get("expected_trajectory", [])),
            )
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        trajectory = self._normalize_trajectory(payload.get("trajectory", []))
        metric = float(payload.get("primary_metric", trajectory[-1] if trajectory else 0.0))
        secondary_metrics = dict(payload.get("secondary_metrics", {}))
        if trajectory:
            secondary_metrics.setdefault("start_metric", trajectory[0])
            secondary_metrics.setdefault("end_metric", trajectory[-1])
            secondary_metrics.setdefault("improvement", round(trajectory[-1] - trajectory[0], 6))
            secondary_metrics.setdefault("trajectory_mean", round(sum(trajectory) / len(trajectory), 6))
        payload.update(
            {
                "task_id": task_id,
                "config": plan,
                "runtime_seconds": runtime_seconds,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "experiment_id": experiment_id,
            }
        )
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        trajectory_path = sandbox_dir / "trajectory.npz"
        np.savez(
            trajectory_path,
            trajectory=np.asarray(trajectory, dtype=float),
            runtime_seconds=np.asarray([runtime_seconds], dtype=float),
        )
        code_hash = self._code_hash(plan)
        return ExperimentResult(
            experiment_id=experiment_id,
            task_id=task_id,
            primary_metric=metric,
            secondary_metrics=secondary_metrics,
            trajectory=trajectory,
            peak_memory=float(payload.get("peak_memory", 0.0)),
            runtime_seconds=runtime_seconds,
            status=payload.get("status", "keep" if metric >= float(plan.get("keep_threshold", 0.0)) else "discard"),
            code_hash=code_hash,
            artifact_path=str(artifact_path),
            trajectory_path=str(trajectory_path),
            anomaly_score=self._trajectory_anomaly_score(trajectory, plan.get("expected_trajectory", [])),
        )

    def cleanup(self, sandbox_name: str) -> None:
        shutil.rmtree(self.sandbox_root / sandbox_name, ignore_errors=True)

    def _normalize_plan(self, task_id: str, config: dict[str, Any], experiment_id: str) -> dict[str, Any]:
        plan = dict(config)
        plan.setdefault("task_id", task_id)
        plan.setdefault("experiment_id", experiment_id)
        plan.setdefault("task_type", "ml_efficiency")
        plan.setdefault("steps", max(6, len(plan.get("expected_trajectory", [])) or 8))
        plan.setdefault("learning_rate", float(plan.get("learning_rate", 0.12)))
        plan.setdefault("warmup_ratio", float(plan.get("warmup_ratio", 0.35)))
        plan.setdefault("scale", float(plan.get("scale", 0.8)))
        plan.setdefault("keep_threshold", float(plan.get("keep_threshold", 0.4)))
        return plan

    def _resolve_command(self, plan: dict[str, Any], plan_path: Path, artifact_path: Path) -> list[str]:
        command = plan.get("command")
        if isinstance(command, str) and command.strip():
            return shlex.split(command)
        if isinstance(command, list) and all(isinstance(part, str) for part in command):
            return list(command)
        return self._default_command(plan_path, artifact_path)

    def _default_command(self, plan_path: Path, artifact_path: Path) -> list[str]:
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_experiment.py"
        return ["python3", str(script_path), "--plan", str(plan_path), "--output", str(artifact_path)]

    def _normalize_trajectory(self, values: Any) -> list[float]:
        if not isinstance(values, list):
            return []
        trajectory: list[float] = []
        for value in values:
            try:
                trajectory.append(float(value))
            except (TypeError, ValueError):
                continue
        return trajectory

    def _code_hash(self, plan: dict[str, Any]) -> str:
        return hashlib.sha1(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()

    def _trajectory_anomaly_score(self, actual: list[float], expected: Any) -> float:
        expected_values = self._normalize_trajectory(expected)
        if not actual or not expected_values:
            return 0.0
        length = min(len(actual), len(expected_values))
        error = sum((actual[index] - expected_values[index]) ** 2 for index in range(length))
        return round(error / max(1, length), 6)
