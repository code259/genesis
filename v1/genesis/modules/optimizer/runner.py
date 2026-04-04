from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

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
        trajectory = config.get("trajectory") or self._derive_trajectory(config)
        metric = float(config.get("primary_metric", trajectory[-1]))
        payload = {
            "task_id": task_id,
            "config": config,
            "trajectory": trajectory,
            "primary_metric": metric,
            "runtime_seconds": time.time() - started,
        }
        artifact_path = sandbox_dir / "result.json"
        ensure_parent(artifact_path)
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (sandbox_dir / "artifact.py").write_text(
            "def run_experiment():\n    return " + repr(metric) + "\n",
            encoding="utf-8",
        )
        code_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        return ExperimentResult(
            experiment_id=experiment_id,
            task_id=task_id,
            primary_metric=metric,
            secondary_metrics={"trajectory_end": trajectory[-1]},
            trajectory=trajectory,
            peak_memory=float(config.get("peak_memory", 0.0)),
            runtime_seconds=time.time() - started,
            status="keep" if metric >= float(config.get("keep_threshold", 0.0)) else "discard",
            code_hash=code_hash,
            artifact_path=str(artifact_path),
            anomaly_score=float(config.get("anomaly_score", 0.0)),
        )

    def cleanup(self, sandbox_name: str) -> None:
        shutil.rmtree(self.sandbox_root / sandbox_name, ignore_errors=True)

    def _derive_trajectory(self, config: dict[str, Any]) -> list[float]:
        baseline = float(config.get("baseline_metric", 0.1))
        step = float(config.get("step_gain", 0.1))
        length = int(config.get("epochs", 4))
        return [round(baseline + step * index, 4) for index in range(1, length + 1)]
