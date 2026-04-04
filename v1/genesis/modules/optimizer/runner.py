from __future__ import annotations

import hashlib
import json
import subprocess
import shutil
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
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        plan_path = sandbox_dir / "plan.json"
        plan_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        artifact_path = sandbox_dir / "result.json"
        command = config.get("command") or self._default_command(plan_path, artifact_path)
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
                secondary_metrics={"returncode": float(process.returncode)},
                trajectory=[],
                peak_memory=0.0,
                runtime_seconds=runtime_seconds,
                status="crash",
                code_hash=hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
                artifact_path=str(error_path),
                anomaly_score=float(config.get("anomaly_score", 1.0)),
            )
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        trajectory = payload.get("trajectory", [])
        metric = float(payload.get("primary_metric", trajectory[-1] if trajectory else 0.0))
        payload.update(
            {
                "task_id": task_id,
                "config": config,
                "runtime_seconds": runtime_seconds,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        trajectory_path = sandbox_dir / "trajectory.npz"
        np.savez(
            trajectory_path,
            trajectory=np.asarray(trajectory, dtype=float),
            runtime_seconds=np.asarray([runtime_seconds], dtype=float),
        )
        code_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        return ExperimentResult(
            experiment_id=experiment_id,
            task_id=task_id,
            primary_metric=metric,
            secondary_metrics=payload.get("secondary_metrics", {"trajectory_end": trajectory[-1] if trajectory else 0.0}),
            trajectory=trajectory,
            peak_memory=float(payload.get("peak_memory", 0.0)),
            runtime_seconds=runtime_seconds,
            status=payload.get("status", "keep" if metric >= float(config.get("keep_threshold", 0.0)) else "discard"),
            code_hash=code_hash,
            artifact_path=str(artifact_path),
            trajectory_path=str(trajectory_path),
            anomaly_score=float(config.get("anomaly_score", 0.0)),
        )

    def cleanup(self, sandbox_name: str) -> None:
        shutil.rmtree(self.sandbox_root / sandbox_name, ignore_errors=True)

    def _default_command(self, plan_path: Path, artifact_path: Path) -> list[str]:
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_experiment.py"
        return ["python3", str(script_path), "--plan", str(plan_path), "--output", str(artifact_path)]
