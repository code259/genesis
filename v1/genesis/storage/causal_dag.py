from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

from genesis.models import ensure_parent


class CausalDAG:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        ensure_parent(self.path)
        if not self.path.exists():
            self.path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2), encoding="utf-8")

    def add_edge(
        self,
        source_node: str,
        target_node: str,
        effect_size: float,
        confidence: float,
        experiment_ids: list[str],
        *,
        domain: str = "",
    ) -> None:
        graph = self._load()
        graph["nodes"] = sorted(set(graph["nodes"]) | {source_node, target_node})
        graph["edges"].append(
            {
                "source": source_node,
                "target": target_node,
                "effect_size": effect_size,
                "confidence": confidence,
                "experiment_ids": experiment_ids,
                "domain": domain,
            }
        )
        if self._has_cycle(graph):
            raise ValueError("adding this edge would create a cycle")
        self._save(graph)

    def get_edges_from(self, node: str) -> list[dict[str, Any]]:
        return [edge for edge in self._load()["edges"] if edge["source"] == node]

    def get_high_confidence_edges(self, threshold: float = 0.8) -> list[dict[str, Any]]:
        return [edge for edge in self._load()["edges"] if edge["confidence"] >= threshold]

    def get_high_confidence_edges_for_domain(self, domain: str, threshold: float = 0.8) -> list[dict[str, Any]]:
        normalized = domain.strip().lower()
        return [
            edge
            for edge in self.get_high_confidence_edges(threshold=threshold)
            if not normalized or str(edge.get("domain", "")).strip().lower() in {"", normalized}
        ]

    def merge_global_dag(self, project_dag: dict[str, Any]) -> None:
        graph = self._load()
        graph["nodes"] = sorted(set(graph["nodes"]) | set(project_dag.get("nodes", [])))
        seen = {
            (
                edge["source"],
                edge["target"],
                edge["effect_size"],
                edge["confidence"],
                tuple(edge["experiment_ids"]),
            )
            for edge in graph["edges"]
        }
        for edge in project_dag.get("edges", []):
            signature = (
                edge["source"],
                edge["target"],
                edge["effect_size"],
                edge["confidence"],
                tuple(edge["experiment_ids"]),
            )
            if signature not in seen:
                graph["edges"].append(edge)
                seen.add(signature)
        if self._has_cycle(graph):
            raise ValueError("merged DAG is cyclic")
        self._save(graph)

    def _load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, payload: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(self.path.parent), encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
            temp_name = handle.name
        os.replace(temp_name, self.path)

    def _has_cycle(self, graph: dict[str, Any]) -> bool:
        adjacency: dict[str, list[str]] = {node: [] for node in graph["nodes"]}
        for edge in graph["edges"]:
            adjacency.setdefault(edge["source"], []).append(edge["target"])

        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for neighbor in adjacency.get(node, []):
                if dfs(neighbor):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(dfs(node) for node in graph["nodes"])
