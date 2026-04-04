from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Union

from genesis.models import ensure_parent


class ProjectFilesystem:
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)

    def init_project(self, project_id: str, spec: dict[str, Any]) -> Path:
        project_dir = self.base_dir / project_id
        (project_dir / "runs").mkdir(parents=True, exist_ok=True)
        (project_dir / "knowledge").mkdir(exist_ok=True)
        (project_dir / "outputs" / "paper").mkdir(parents=True, exist_ok=True)
        (project_dir / "outputs" / "code").mkdir(parents=True, exist_ok=True)
        (project_dir / "experiments" / "trajectories").mkdir(parents=True, exist_ok=True)
        self.write_json(project_dir / "spec.json", spec)
        self.write_json(project_dir / "causal_dag.json", {"nodes": [], "edges": []})
        return project_dir

    def get_project_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id

    def get_run_dir(self, project_id: str, run_n: int) -> Path:
        run_dir = self.base_dir / project_id / "runs" / str(run_n)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_instruction(self, project_id: str, run_n: int, content: str) -> Path:
        destination = self.get_run_dir(project_id, run_n) / "instruction.md"
        ensure_parent(destination)
        destination.write_text(content, encoding="utf-8")
        return destination

    def write_json(self, path: Union[str, Path], payload: Union[dict[str, Any], List[Any]]) -> Path:
        destination = Path(path)
        ensure_parent(destination)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def read_json(self, path: Union[str, Path]) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def read_trace(self, project_id: str, run_n: int) -> dict[str, Any]:
        return self.read_json(self.get_run_dir(project_id, run_n) / "trace.json")

    def list_all_results(self, project_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for result_path in sorted((self.base_dir / project_id / "runs").glob("*/result.json")):
            results.append(self.read_json(result_path))
        return sorted(results, key=lambda item: item.get("primary_metric", 0.0), reverse=True)
