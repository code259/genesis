from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union


class TasteGP:
    def __init__(self) -> None:
        self.training_inputs: list[list[float]] = []
        self.training_targets: list[float] = []
        self.training_trajectories: list[list[float]] = []

    def fit(
        self, X: list[list[float]], y: list[float], trajectories: Optional[list[list[float]]] = None
    ) -> None:
        self.training_inputs = list(X)
        self.training_targets = list(y)
        self.training_trajectories = list(trajectories or [[target] for target in y])

    def predict(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        if not self.training_targets:
            return [0.0 for _ in X], [1.0 for _ in X]
        mean_target = sum(self.training_targets) / len(self.training_targets)
        base_variance = sum((target - mean_target) ** 2 for target in self.training_targets) / max(
            1, len(self.training_targets)
        )
        means: list[float] = []
        variances: list[float] = []
        for features in X:
            nearest_distance = min(
                self._distance(features, candidate) for candidate in self.training_inputs
            ) if self.training_inputs else 1.0
            means.append(mean_target / (1.0 + nearest_distance))
            variances.append(base_variance / max(1, len(self.training_targets)) + nearest_distance)
        return means, variances

    def predict_trajectory(self, X: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        means, variances = self.predict(X)
        mean_trajectory = [self.training_trajectories[-1] if self.training_trajectories else [mean] for mean in means]
        variance_trajectory = [[variance for _ in trajectory] for variance, trajectory in zip(variances, mean_trajectory)]
        return mean_trajectory, variance_trajectory

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "training_inputs": self.training_inputs,
                    "training_targets": self.training_targets,
                    "training_trajectories": self.training_trajectories,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TasteGP":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls()
        model.fit(payload["training_inputs"], payload["training_targets"], payload["training_trajectories"])
        return model

    def _distance(self, left: list[float], right: list[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
