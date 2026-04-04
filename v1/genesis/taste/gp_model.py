from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class TasteGP:
    def __init__(
        self,
        *,
        noise: float = 1e-5,
        rbf_length_scale: float = 1.5,
        matern_length_scale: float = 1.0,
        structured_dims: int = 3,
    ) -> None:
        self.noise = noise
        self.rbf_length_scale = rbf_length_scale
        self.matern_length_scale = matern_length_scale
        self.structured_dims = structured_dims
        self.training_inputs: list[list[float]] = []
        self.training_targets: list[float] = []
        self.training_trajectories: list[list[float]] = []
        self._X: Optional[np.ndarray] = None
        self._target_factor: Optional[tuple[np.ndarray, bool]] = None
        self._target_alpha: Optional[np.ndarray] = None
        self._trajectory_factor: Optional[tuple[np.ndarray, bool]] = None
        self._trajectory_alpha: Optional[np.ndarray] = None

    def fit(
        self, X: list[list[float]], y: list[float], trajectories: Optional[list[list[float]]] = None
    ) -> None:
        self.training_inputs = [list(features) for features in X]
        self.training_targets = list(y)
        self.training_trajectories = list(trajectories or [[target] for target in y])
        if not self.training_inputs:
            self._X = None
            self._target_factor = None
            self._target_alpha = None
            self._trajectory_factor = None
            self._trajectory_alpha = None
            return
        self._X = np.asarray(self.training_inputs, dtype=float)
        target_vector = np.asarray(self.training_targets, dtype=float)
        kernel = self._kernel_matrix(self._X, self._X) + self.noise * np.eye(len(self.training_inputs))
        self._target_factor = cho_factor(kernel, lower=True, check_finite=False)
        self._target_alpha = cho_solve(self._target_factor, target_vector, check_finite=False)

        max_len = max(len(trajectory) for trajectory in self.training_trajectories)
        trajectory_matrix = np.zeros((len(self.training_trajectories), max_len), dtype=float)
        for row_idx, trajectory in enumerate(self.training_trajectories):
            if trajectory:
                trajectory_matrix[row_idx, : len(trajectory)] = trajectory
                if len(trajectory) < max_len:
                    trajectory_matrix[row_idx, len(trajectory) :] = trajectory[-1]
        self._trajectory_factor = cho_factor(kernel, lower=True, check_finite=False)
        self._trajectory_alpha = cho_solve(self._trajectory_factor, trajectory_matrix, check_finite=False)

    def predict(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        if self._X is None or self._target_factor is None or self._target_alpha is None:
            return [0.0 for _ in X], [1.0 for _ in X]
        query = np.asarray(X, dtype=float)
        cross_kernel = self._kernel_matrix(query, self._X)
        means = cross_kernel @ self._target_alpha
        variances: list[float] = []
        for index, features in enumerate(query):
            self_kernel = float(self._kernel_matrix(features[None, :], features[None, :])[0, 0] + self.noise)
            projection = cho_solve(self._target_factor, cross_kernel[index], check_finite=False)
            variance = max(self_kernel - float(cross_kernel[index] @ projection), 1e-9)
            variances.append(variance)
        return means.tolist(), variances

    def predict_trajectory(self, X: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        if self._X is None or self._trajectory_factor is None or self._trajectory_alpha is None:
            return [[0.0] for _ in X], [[1.0] for _ in X]
        query = np.asarray(X, dtype=float)
        cross_kernel = self._kernel_matrix(query, self._X)
        trajectory_means = cross_kernel @ self._trajectory_alpha
        _, scalar_variances = self.predict(X)
        trajectory_variances = [
            [variance for _ in range(trajectory_means.shape[1])] for variance in scalar_variances
        ]
        return trajectory_means.tolist(), trajectory_variances

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "noise": self.noise,
                    "rbf_length_scale": self.rbf_length_scale,
                    "matern_length_scale": self.matern_length_scale,
                    "structured_dims": self.structured_dims,
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
        model = cls(
            noise=payload.get("noise", 1e-5),
            rbf_length_scale=payload.get("rbf_length_scale", 1.5),
            matern_length_scale=payload.get("matern_length_scale", 1.0),
            structured_dims=payload.get("structured_dims", 3),
        )
        model.fit(payload["training_inputs"], payload["training_targets"], payload["training_trajectories"])
        return model

    def _kernel_matrix(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_text, left_structured = self._split_features(left)
        right_text, right_structured = self._split_features(right)
        rbf = self._rbf_kernel(left_text, right_text, self.rbf_length_scale)
        matern = self._matern52_kernel(left_structured, right_structured, self.matern_length_scale)
        return rbf + matern

    def _split_features(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        structured_dims = min(self.structured_dims, matrix.shape[1])
        text_dims = matrix.shape[1] - structured_dims
        return matrix[:, :text_dims], matrix[:, text_dims:]

    def _rbf_kernel(self, left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
        if left.size == 0 or right.size == 0:
            return np.zeros((left.shape[0], right.shape[0]), dtype=float)
        sqdist = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * sqdist / max(length_scale**2, 1e-9))

    def _matern52_kernel(self, left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
        if left.size == 0 or right.size == 0:
            return np.zeros((left.shape[0], right.shape[0]), dtype=float)
        scaled = np.sqrt(np.sum(((left[:, None, :] - right[None, :, :]) / max(length_scale, 1e-9)) ** 2, axis=2))
        sqrt5_r = np.sqrt(5.0) * scaled
        return (1.0 + sqrt5_r + (5.0 / 3.0) * scaled**2) * np.exp(-sqrt5_r)
