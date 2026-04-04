from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Union

from genesis.models import ExperimentResult

from .runner import ExperimentRunner


class ParallelExperimentManager:
    def __init__(self, sandbox_root: Union[str, Path]):
        self.runner = ExperimentRunner(sandbox_root)

    def run_batch(
        self,
        experiments: list[dict[str, Any]],
        n_parallel: int = 3,
        *,
        cleanup: bool = False,
    ) -> list[ExperimentResult]:
        if not experiments:
            return []
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = {
                executor.submit(
                    self.runner.run,
                    experiment["task_id"],
                    experiment,
                    experiment.get("experiment_id"),
                ): (index, experiment)
                for index, experiment in enumerate(experiments)
            }
            indexed_results: dict[int, ExperimentResult] = {}
            for future in as_completed(futures):
                index, experiment = futures[future]
                experiment_id = str(experiment.get("experiment_id", f"failed-{index}"))
                task_id = str(experiment.get("task_id", "unknown"))
                sandbox_name = str(experiment.get("experiment_id", experiment_id))
                result: ExperimentResult
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        task_id=task_id,
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
                indexed_results[index] = result
                if cleanup:
                    self.runner.cleanup(sandbox_name)
        results: list[ExperimentResult] = []
        for index in range(len(experiments)):
            try:
                results.append(indexed_results[index])
            except KeyError:
                experiment = experiments[index]
                results.append(
                    ExperimentResult(
                        experiment_id=str(experiment.get("experiment_id", f"missing-{index}")),
                        task_id=str(experiment.get("task_id", "unknown")),
                        primary_metric=0.0,
                        secondary_metrics={"error": 1.0},
                        trajectory=[],
                        peak_memory=0.0,
                        runtime_seconds=0.0,
                        status="crash",
                        code_hash="",
                        artifact_path="missing future result",
                        anomaly_score=1.0,
                    )
                )
        return results
