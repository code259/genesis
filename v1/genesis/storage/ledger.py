from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from genesis.models import ExperimentResult, ensure_parent


class ExperimentLedger:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        ensure_parent(self.path)
        self._connect().close()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    parent_experiment_id TEXT,
                    task_id TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    config_diff TEXT NOT NULL,
                    primary_metric REAL NOT NULL,
                    secondary_metrics TEXT NOT NULL,
                    trajectory_summary TEXT NOT NULL,
                    attribution_ablations TEXT NOT NULL,
                    status TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    trajectory_path TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "trajectory_path" not in columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN trajectory_path TEXT NOT NULL DEFAULT ''"
                )

    def insert_experiment(
        self,
        result: ExperimentResult,
        *,
        parent_experiment_id: Optional[str] = None,
        config_diff: str = "",
        attribution_ablations: Optional[dict[str, Any]] = None,
        timestamp: str = "",
    ) -> None:
        summary = {
            "length": len(result.trajectory),
            "start": result.trajectory[0] if result.trajectory else None,
            "end": result.trajectory[-1] if result.trajectory else None,
            "max": max(result.trajectory, default=None),
            "min": min(result.trajectory, default=None),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, parent_experiment_id, task_id, code_hash, config_diff,
                    primary_metric, secondary_metrics, trajectory_summary, attribution_ablations,
                    status, anomaly_score, timestamp, artifact_path, trajectory_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_id,
                    parent_experiment_id,
                    result.task_id,
                    result.code_hash,
                    config_diff,
                    result.primary_metric,
                    json.dumps(result.secondary_metrics),
                    json.dumps(summary),
                    json.dumps(attribution_ablations or {}),
                    result.status,
                    result.anomaly_score,
                    timestamp,
                    result.artifact_path,
                    result.trajectory_path,
                ),
            )

    def update_status(self, experiment_id: str, status: str, anomaly_score: Optional[float] = None) -> None:
        with self._connect() as connection:
            if anomaly_score is None:
                connection.execute(
                    "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                    (status, experiment_id),
                )
            else:
                connection.execute(
                    "UPDATE experiments SET status = ?, anomaly_score = ? WHERE experiment_id = ?",
                    (status, anomaly_score, experiment_id),
                )

    def get_by_task(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments WHERE task_id = ? ORDER BY primary_metric DESC",
                (task_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_pareto_frontier(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY primary_metric DESC, anomaly_score ASC"
            ).fetchall()
        best_metric = None
        frontier: list[dict[str, Any]] = []
        for row in rows:
            payload = self._row_to_dict(row)
            metric = payload["primary_metric"]
            anomaly = payload["anomaly_score"]
            if best_metric is None or metric >= best_metric or anomaly < 1.0:
                frontier.append(payload)
                best_metric = metric if best_metric is None else max(best_metric, metric)
        return frontier

    def get_anomalies(self, threshold: float) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments WHERE anomaly_score > ? ORDER BY anomaly_score DESC",
                (threshold,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: Union[sqlite3.Row, Tuple[Any, ...]]) -> dict[str, Any]:
        (
            experiment_id,
            parent_experiment_id,
            task_id,
            code_hash,
            config_diff,
            primary_metric,
            secondary_metrics,
            trajectory_summary,
            attribution_ablations,
            status,
            anomaly_score,
            timestamp,
            artifact_path,
            trajectory_path,
        ) = row
        return {
            "experiment_id": experiment_id,
            "parent_experiment_id": parent_experiment_id,
            "task_id": task_id,
            "code_hash": code_hash,
            "config_diff": config_diff,
            "primary_metric": primary_metric,
            "secondary_metrics": json.loads(secondary_metrics),
            "trajectory_summary": json.loads(trajectory_summary),
            "attribution_ablations": json.loads(attribution_ablations),
            "status": status,
            "anomaly_score": anomaly_score,
            "timestamp": timestamp,
            "artifact_path": artifact_path,
            "trajectory_path": trajectory_path,
        }
