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
        return [future.result() for future in futures]
