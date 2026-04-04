from __future__ import annotations

import json
import os
import tempfile
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
        (project_dir / "outputs" / "paper" / "figures").mkdir(parents=True, exist_ok=True)
        (project_dir / "outputs" / "code").mkdir(parents=True, exist_ok=True)
        (project_dir / "experiments" / "trajectories").mkdir(parents=True, exist_ok=True)
        (project_dir / "runtime" / "sandboxes").mkdir(parents=True, exist_ok=True)
        self.write_json(project_dir / "spec.json", spec)
        if not (project_dir / "causal_dag.json").exists():
            self.write_json(project_dir / "causal_dag.json", {"nodes": [], "edges": []})
        if not (project_dir / "project_state.json").exists():
            self.write_json(
                project_dir / "project_state.json",
                {"status": "initialized", "run_count": 0, "last_run_status": None},
            )
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
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(destination.parent), encoding="utf-8") as handle:
            handle.write(content)
            temp_name = handle.name
        os.replace(temp_name, destination)
        return destination

    def write_json(self, path: Union[str, Path], payload: Union[dict[str, Any], List[Any]]) -> Path:
        destination = Path(path)
        ensure_parent(destination)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(destination.parent), encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
            temp_name = handle.name
        os.replace(temp_name, destination)
        return destination

    def read_json(self, path: Union[str, Path]) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def read_trace(self, project_id: str, run_n: int) -> dict[str, Any]:
        return self.read_json(self.get_run_dir(project_id, run_n) / "trace.json")

    def list_all_results(self, project_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for result_path in sorted((self.base_dir / project_id / "runs").glob("*/result.json")):
            try:
                results.append(self.read_json(result_path))
            except json.JSONDecodeError:
                continue
        return sorted(results, key=lambda item: item.get("primary_metric", 0.0), reverse=True)

    def validate_project(self, project_id: str) -> bool:
        project_dir = self.get_project_dir(project_id)
        required = [
            project_dir / "spec.json",
            project_dir / "runs",
            project_dir / "knowledge",
            project_dir / "outputs" / "paper",
            project_dir / "outputs" / "code",
            project_dir / "experiments" / "trajectories",
            project_dir / "runtime" / "sandboxes",
            project_dir / "causal_dag.json",
        ]
        return all(path.exists() for path in required)

    def read_human_intervention(self, project_id: str) -> dict[str, Any] | None:
        path = self.get_project_dir(project_id) / "human_intervention.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def clear_human_intervention(self, project_id: str) -> None:
        path = self.get_project_dir(project_id) / "human_intervention.json"
        if path.exists():
            path.unlink()

    def write_halt(self, project_id: str, payload: dict[str, Any]) -> Path:
        return self.write_json(self.get_project_dir(project_id) / "HALT.json", payload)

    def write_project_state(self, project_id: str, payload: dict[str, Any]) -> Path:
        return self.write_json(self.get_project_dir(project_id) / "project_state.json", payload)

    def read_project_state(self, project_id: str) -> dict[str, Any]:
        path = self.get_project_dir(project_id) / "project_state.json"
        if not path.exists():
            return {"status": "unknown", "run_count": 0, "last_run_status": None}
        return self.read_json(path)
