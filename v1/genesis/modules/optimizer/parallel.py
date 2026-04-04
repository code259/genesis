from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Union

from genesis.models import ExperimentResult

from .runner import ExperimentRunner


class ParallelExperimentManager:
    def __init__(self, sandbox_root: Union[str, Path]):
        self.runner = ExperimentRunner(sandbox_root)

    def run_batch(self, experiments: list[dict[str, Any]], n_parallel: int = 3) -> list[ExperimentResult]:
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = [
                executor.submit(self.runner.run, experiment["task_id"], experiment, experiment.get("experiment_id"))
                for experiment in experiments
            ]
        results: list[ExperimentResult] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append(
                    ExperimentResult(
                        experiment_id="failed-experiment",
                        task_id="unknown",
                        primary_metric=0.0,
                        secondary_metrics={"error": 1.0},
                        trajectory=[],
                        peak_memory=0.0,
                        runtime_seconds=0.0,
                        status="crash",
                        code_hash="",
                        artifact_path=str(exc),
                        anomaly_score=1.0,
                    )
                )
        return results
