from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve
try:
    import gpytorch  # type: ignore
    import torch
except Exception:  # pragma: no cover - optional backend
    gpytorch = None
    torch = None


class TasteGP:
    MIN_POINTS_FOR_GP = 20

    def __init__(
        self,
        *,
        noise: float = 1e-5,
        rbf_length_scale: float = 1.5,
        matern_length_scale: float = 1.0,
        structured_dims: int = 8,
    ) -> None:
        self.noise = noise
        self.rbf_length_scale = rbf_length_scale
        self.matern_length_scale = matern_length_scale
        self.structured_dims = structured_dims
        self.training_inputs: list[list[float]] = []
        self.training_targets: list[float] = []
        self.training_trajectories: list[list[float]] = []
        self._X: Optional[np.ndarray] = None
        self._y_mean = 0.0
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_scale: Optional[np.ndarray] = None
        self._trajectory_mean = np.zeros(0, dtype=float)
        self._target_factor: Optional[tuple[np.ndarray, bool]] = None
        self._target_alpha: Optional[np.ndarray] = None
        self._trajectory_factor: Optional[tuple[np.ndarray, bool]] = None
        self._trajectory_alpha: Optional[np.ndarray] = None
        self.backend = "gpytorch" if gpytorch is not None and torch is not None else "scipy"

    def fit(
        self, X: list[list[float]], y: list[float], trajectories: Optional[list[list[float]]] = None
    ) -> None:
        if self.backend == "gpytorch":
            self._fit_gpytorch(X, y, trajectories)
            return
        self.training_inputs = [list(features) for features in X]
        self.training_targets = list(y)
        self.training_trajectories = list(trajectories or [[target] for target in y])
        if not self.training_inputs:
            self._X = None
            self._feature_mean = None
            self._feature_scale = None
            self._y_mean = 0.0
            self._target_factor = None
            self._target_alpha = None
            self._trajectory_factor = None
            self._trajectory_alpha = None
            return
        raw_features = np.asarray(self.training_inputs, dtype=float)
        self._feature_mean = raw_features.mean(axis=0)
        self._feature_scale = raw_features.std(axis=0)
        self._feature_scale[self._feature_scale == 0.0] = 1.0
        self._X = self._normalize_features(raw_features)
        target_vector = np.asarray(self.training_targets, dtype=float)
        self._y_mean = float(target_vector.mean()) if len(target_vector) else 0.0
        centered_targets = target_vector - self._y_mean
        kernel = self._kernel_matrix(self._X, self._X) + self.noise * np.eye(len(self.training_inputs))
        self._target_factor = cho_factor(kernel, lower=True, check_finite=False)
        self._target_alpha = cho_solve(self._target_factor, centered_targets, check_finite=False)

        max_len = max(len(trajectory) for trajectory in self.training_trajectories)
        trajectory_matrix = np.zeros((len(self.training_trajectories), max_len), dtype=float)
        for row_idx, trajectory in enumerate(self.training_trajectories):
            if trajectory:
                trajectory_matrix[row_idx, : len(trajectory)] = trajectory
                if len(trajectory) < max_len:
                    trajectory_matrix[row_idx, len(trajectory) :] = trajectory[-1]
        self._trajectory_mean = trajectory_matrix.mean(axis=0) if len(trajectory_matrix) else np.zeros(0, dtype=float)
        centered_trajectory_matrix = trajectory_matrix - self._trajectory_mean
        self._trajectory_factor = cho_factor(kernel, lower=True, check_finite=False)
        self._trajectory_alpha = cho_solve(self._trajectory_factor, centered_trajectory_matrix, check_finite=False)

    def predict(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        if len(self.training_targets) < self.MIN_POINTS_FOR_GP:
            return self._predict_nearest_neighbor(X)
        if self.backend == "gpytorch" and hasattr(self, "_gpytorch_state"):
            return self._predict_gpytorch(X)
        if self._X is None or self._target_factor is None or self._target_alpha is None:
            return [0.0 for _ in X], [1.0 for _ in X]
        query = self._normalize_features(np.asarray(X, dtype=float))
        cross_kernel = self._kernel_matrix(query, self._X)
        means = cross_kernel @ self._target_alpha + self._y_mean
        variances: list[float] = []
        for index, features in enumerate(query):
            self_kernel = float(self._kernel_matrix(features[None, :], features[None, :])[0, 0] + self.noise)
            projection = cho_solve(self._target_factor, cross_kernel[index], check_finite=False)
            variance = max(self_kernel - float(cross_kernel[index] @ projection), 1e-9)
            variances.append(variance)
        return means.tolist(), variances

    def predict_trajectory(self, X: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        if len(self.training_targets) < self.MIN_POINTS_FOR_GP:
            means, variances = self._predict_nearest_neighbor(X)
            return [[mean] for mean in means], [[variance] for variance in variances]
        if self.backend == "gpytorch" and hasattr(self, "_gpytorch_state"):
            means, variances = self._predict_gpytorch(X)
            return [[mean] for mean in means], [[variance] for variance in variances]
        if self._X is None or self._trajectory_factor is None or self._trajectory_alpha is None:
            return [[0.0] for _ in X], [[1.0] for _ in X]
        query = self._normalize_features(np.asarray(X, dtype=float))
        cross_kernel = self._kernel_matrix(query, self._X)
        trajectory_means = cross_kernel @ self._trajectory_alpha + self._trajectory_mean
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

    def _normalize_features(self, matrix: np.ndarray) -> np.ndarray:
        if self._feature_mean is None or self._feature_scale is None:
            return matrix
        return (matrix - self._feature_mean) / self._feature_scale

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

    def _fit_gpytorch(
        self, X: list[list[float]], y: list[float], trajectories: Optional[list[list[float]]] = None
    ) -> None:
        # Keep persisted/public behavior aligned with the existing interface even when gpytorch is available.
        self.training_inputs = [list(features) for features in X]
        self.training_targets = list(y)
        self.training_trajectories = list(trajectories or [[target] for target in y])
        if not X:
            self._gpytorch_state = None
            return
        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(y, dtype=torch.float32)
        feature_mean = train_x.mean(dim=0)
        feature_scale = train_x.std(dim=0)
        feature_scale[feature_scale == 0] = 1.0
        normalized_x = (train_x - feature_mean) / feature_scale
        y_mean = train_y.mean()
        centered_y = train_y - y_mean

        class _ExactGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel(nu=2.5)
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = _ExactGP(normalized_x, centered_y, likelihood)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(25):
            optimizer.zero_grad()
            output = model(normalized_x)
            loss = -mll(output, centered_y)
            loss.backward()
            optimizer.step()
        self._gpytorch_state = {
            "model": model.eval(),
            "likelihood": likelihood.eval(),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "y_mean": float(y_mean.item()),
        }

    def _predict_gpytorch(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        if not X or not getattr(self, "_gpytorch_state", None):
            return [0.0 for _ in X], [1.0 for _ in X]
        state = self._gpytorch_state
        query = torch.tensor(X, dtype=torch.float32)
        query = (query - state["feature_mean"]) / state["feature_scale"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = state["likelihood"](state["model"](query))
        means = prediction.mean + state["y_mean"]
        variances = prediction.variance
        return means.tolist(), variances.tolist()

    def _predict_nearest_neighbor(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        if not self.training_inputs or not self.training_targets:
            return [0.0 for _ in X], [1.0 for _ in X]
        train = np.asarray(self.training_inputs, dtype=float)
        query = np.asarray(X, dtype=float)
        means: list[float] = []
        variances: list[float] = []
        for row in query:
            distances = np.linalg.norm(train - row, axis=1)
            nearest_idx = int(np.argmin(distances))
            means.append(float(self.training_targets[nearest_idx]))
            variances.append(float(max(0.01, distances[nearest_idx] ** 2)))
        return means, variances
