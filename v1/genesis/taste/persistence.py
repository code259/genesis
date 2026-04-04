from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from genesis.models import ensure_parent

from .gp_model import TasteGP


class TasteModelPersistence:
    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.model_path = self.root_dir / "taste_model.json"
        self.dataset_path = self.root_dir / "training_data.json"
        ensure_parent(self.model_path)
        if not self.dataset_path.exists():
            self.dataset_path.write_text("[]", encoding="utf-8")

    def save_after_project(self, project_id: str, model: TasteGP) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.model_path)

    def load_for_project(self, project_id: str, snapshot_path: Union[str, Path]) -> TasteGP:
        snapshot = Path(snapshot_path)
        ensure_parent(snapshot)
        if self.model_path.exists():
            snapshot.write_text(self.model_path.read_text(encoding="utf-8"), encoding="utf-8")
            return TasteGP.load(snapshot)
        model = TasteGP()
        model.save(snapshot)
        return model

    def merge_project_data(self, project_id: str, experiments: list[dict[str, Any]]) -> None:
        current = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        indexed = {
            item.get("experiment_id", f"existing-{index}"): item
            for index, item in enumerate(current)
            if isinstance(item, dict)
        }
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            experiment_id = experiment.get("experiment_id")
            if experiment_id:
                indexed[experiment_id] = {
                    **experiment,
                    "project_id": project_id,
                }
        merged = list(indexed.values())
        self.dataset_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
