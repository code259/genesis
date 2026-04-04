from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from statistics import mean, pstdev
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
            self._save(self.papers_path, [])
        if not self.experiments_path.exists():
            self._save(self.experiments_path, [])

    def add_paper(self, paper: dict[str, Any]) -> None:
        self._upsert_item(paper, collection="papers")

    def add_experiment(self, experiment: dict[str, Any]) -> None:
        self._upsert_item(experiment, collection="experiments")

    def search_nearest(
        self,
        query: list[float],
        *,
        k: int = 5,
        distance_threshold: Optional[float] = None,
        collection: str = "papers",
        exclude_ids: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        items = self._load(self._path_for_collection(collection))
        exclude_ids = exclude_ids or set()
        ranked = sorted(
            (
                {
                    **item,
                    "distance": _cosine_distance(query, self._vector_for(item)),
                }
                for item in items
                if self._item_id(item, collection=collection) not in exclude_ids
            ),
            key=lambda item: (item["distance"], -float(item.get("density_score", 0.0))),
        )
        if distance_threshold is not None:
            ranked = [item for item in ranked if item["distance"] <= distance_threshold]
        return ranked[:k]

    def get_density_score(self, point_id: str) -> float:
        for collection in ("papers", "experiments"):
            for item in self._load(self._path_for_collection(collection)):
                if self._item_id(item, collection=collection) == point_id:
                    return float(item.get("density_score", 0.0))
        raise KeyError(point_id)

    def get_density_statistics(self, *, collection: str = "papers") -> dict[str, float]:
        items = self._load(self._path_for_collection(collection))
        scores = [float(item.get("density_score", 0.0)) for item in items]
        if not scores:
            return {"mean": 0.0, "stdev": 0.0, "threshold": 0.0, "max": 0.0}
        deviation = pstdev(scores) if len(scores) > 1 else 0.0
        center = mean(scores)
        return {
            "mean": round(center, 6),
            "stdev": round(deviation, 6),
            "threshold": round(center + 2 * deviation, 6),
            "max": round(max(scores), 6),
        }

    def all_papers(self) -> list[dict[str, Any]]:
        return self._load(self.papers_path)

    def all_experiments(self) -> list[dict[str, Any]]:
        return self._load(self.experiments_path)

    def attempted_sources(self) -> set[str]:
        attempted: set[str] = set()
        for experiment in self.all_experiments():
            for key in ("source_paper_id", "paper_id", "title"):
                value = experiment.get(key)
                if isinstance(value, str) and value:
                    attempted.add(value)
        return attempted

    def recompute_density_scores(self, *, collection: str = "papers", k: int = 10) -> list[dict[str, Any]]:
        path = self._path_for_collection(collection)
        items = self._load(path)
        for index, item in enumerate(items):
            vector = self._vector_for(item)
            distances = sorted(
                _cosine_distance(vector, self._vector_for(other))
                for other_index, other in enumerate(items)
                if other_index != index
            )
            neighbors = distances[: min(k, len(distances))]
            item["density_score"] = round(sum(neighbors) / len(neighbors), 6) if neighbors else 0.0
        self._save(path, items)
        return items

    def upsert_collection(self, items: list[dict[str, Any]], *, collection: str = "papers") -> None:
        path = self._path_for_collection(collection)
        indexed = {
            self._item_id(item, collection=collection): self._normalize_item(item, collection=collection)
            for item in self._load(path)
        }
        for item in items:
            indexed[self._item_id(item, collection=collection)] = self._normalize_item(item, collection=collection)
        self._save(path, list(indexed.values()))

    def sample_distant(
        self,
        query: list[float],
        *,
        min_distance: float,
        collection: str = "papers",
    ) -> Optional[dict[str, Any]]:
        candidates = self._load(self._path_for_collection(collection))
        ranked = sorted(
            (
                {
                    **item,
                    "distance": _cosine_distance(query, self._vector_for(item)),
                }
                for item in candidates
            ),
            key=lambda item: item["distance"],
            reverse=True,
        )
        for candidate in ranked:
            if candidate["distance"] >= min_distance:
                return candidate
        return ranked[0] if ranked else None

    def _upsert_item(self, item: dict[str, Any], *, collection: str) -> None:
        path = self._path_for_collection(collection)
        payload = self._load(path)
        item_id = self._item_id(item, collection=collection)
        normalized = self._normalize_item(item, collection=collection)
        updated = False
        for index, existing in enumerate(payload):
            if self._item_id(existing, collection=collection) == item_id:
                payload[index] = normalized
                updated = True
                break
        if not updated:
            payload.append(normalized)
        self._save(path, payload)

    def _path_for_collection(self, collection: str) -> Path:
        return self.papers_path if collection == "papers" else self.experiments_path

    def _item_id(self, item: dict[str, Any], *, collection: str) -> str:
        if collection == "papers":
            return str(item.get("paper_id") or item.get("title") or "paper")
        return str(item.get("experiment_id") or item.get("paper_id") or item.get("title") or "experiment")

    def _vector_for(self, item: dict[str, Any]) -> list[float]:
        vector = item.get("latent_z", item.get("embedding", []))
        return [float(value) for value in vector] if isinstance(vector, list) else []

    def _normalize_item(self, item: dict[str, Any], *, collection: str) -> dict[str, Any]:
        normalized = dict(item)
        item_id_key = "paper_id" if collection == "papers" else "experiment_id"
        normalized[item_id_key] = self._item_id(item, collection=collection)
        normalized["density_score"] = float(normalized.get("density_score", 0.0))
        if "latent_z" in normalized and isinstance(normalized["latent_z"], list):
            normalized["latent_z"] = [float(value) for value in normalized["latent_z"]]
        if "embedding" in normalized and isinstance(normalized["embedding"], list):
            normalized["embedding"] = [float(value) for value in normalized["embedding"]]
        return normalized

    def _load(self, path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return payload if isinstance(payload, list) else []

    def _save(self, path: Path, payload: list[dict[str, Any]]) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(path.parent),
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, indent=2)
            temp_name = handle.name
        os.replace(temp_name, path)
