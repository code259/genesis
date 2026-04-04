from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Union

from genesis.models import ensure_parent


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 1.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - numerator / (left_norm * right_norm)


class ManifoldIndex:
    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.papers_path = self.root_dir / "papers.json"
        self.experiments_path = self.root_dir / "experiments.json"
        ensure_parent(self.papers_path)
        if not self.papers_path.exists():
            self.papers_path.write_text("[]", encoding="utf-8")
        if not self.experiments_path.exists():
            self.experiments_path.write_text("[]", encoding="utf-8")

    def add_paper(self, paper: dict[str, Any]) -> None:
        payload = self._load(self.papers_path)
        payload.append(paper)
        self._save(self.papers_path, payload)

    def add_experiment(self, experiment: dict[str, Any]) -> None:
        payload = self._load(self.experiments_path)
        payload.append(experiment)
        self._save(self.experiments_path, payload)

    def search_nearest(
        self,
        query: list[float],
        *,
        k: int = 5,
        distance_threshold: Optional[float] = None,
        collection: str = "papers",
    ) -> list[dict[str, Any]]:
        items = self._load(self.papers_path if collection == "papers" else self.experiments_path)
        ranked = sorted(
            (
                {
                    **item,
                    "distance": _cosine_distance(query, item.get("latent_z", item.get("embedding", []))),
                }
                for item in items
            ),
            key=lambda item: item["distance"],
        )
        if distance_threshold is not None:
            ranked = [item for item in ranked if item["distance"] <= distance_threshold]
        return ranked[:k]

    def get_density_score(self, point_id: str) -> float:
        for item in self._load(self.papers_path):
            if item.get("paper_id") == point_id:
                return float(item.get("density_score", 0.0))
        for item in self._load(self.experiments_path):
            if item.get("experiment_id") == point_id:
                return float(item.get("density_score", 0.0))
        raise KeyError(point_id)

    def all_papers(self) -> list[dict[str, Any]]:
        return self._load(self.papers_path)

    def all_experiments(self) -> list[dict[str, Any]]:
        return self._load(self.experiments_path)

    def recompute_density_scores(self, *, collection: str = "papers", k: int = 10) -> list[dict[str, Any]]:
        path = self.papers_path if collection == "papers" else self.experiments_path
        items = self._load(path)
        for index, item in enumerate(items):
            vector = item.get("latent_z", item.get("embedding", []))
            distances = sorted(
                _cosine_distance(vector, other.get("latent_z", other.get("embedding", [])))
                for other_index, other in enumerate(items)
                if other_index != index
            )
            neighbors = distances[: min(k, len(distances))]
            item["density_score"] = round(sum(neighbors) / len(neighbors), 6) if neighbors else 0.0
        self._save(path, items)
        return items

    def upsert_collection(self, items: list[dict[str, Any]], *, collection: str = "papers") -> None:
        path = self.papers_path if collection == "papers" else self.experiments_path
        self._save(path, items)

    def _load(self, path: Path) -> list[dict[str, Any]]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, path: Path, payload: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
