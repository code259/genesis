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
        current.extend(experiments)
        self.dataset_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
