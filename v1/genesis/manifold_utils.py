from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def hash_embedding(text: str, dim: int = 256) -> np.ndarray:
    vector = np.zeros(dim, dtype=float)
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        vector[hash(token) % dim] += float(count)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def build_citation_adjacency(papers: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    paper_index = {paper["paper_id"]: idx for idx, paper in enumerate(papers)}
    adjacency = np.eye(len(papers), dtype=float)
    for idx, paper in enumerate(papers):
        for citation in paper.get("citations", []):
            target_id = citation.get("paper_id")
            if target_id in paper_index:
                adjacency[idx, paper_index[target_id]] = 1.0
                adjacency[paper_index[target_id], idx] = 1.0
    return adjacency, paper_index


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=1)
    degree[degree == 0.0] = 1.0
    inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return inv_sqrt @ adjacency @ inv_sqrt


def train_graph_vae(
    papers: list[dict[str, Any]],
    *,
    embedding_dim: int = 256,
    latent_dim: int = 32,
    epochs: int = 200,
    learning_rate: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack(
        [hash_embedding(f"{paper.get('title', '')} {paper.get('abstract', '')}", dim=embedding_dim) for paper in papers]
    )
    adjacency, _ = build_citation_adjacency(papers)
    normalized = normalize_adjacency(adjacency)
    hidden = normalized @ features
    latent_dim = min(latent_dim, hidden.shape[0], hidden.shape[1])
    if latent_dim == 0:
        return np.zeros((len(papers), 0)), adjacency

    rng = np.random.default_rng(7)
    w_mu = rng.normal(scale=0.05, size=(hidden.shape[1], latent_dim))
    w_logvar = rng.normal(scale=0.05, size=(hidden.shape[1], latent_dim))

    for _ in range(epochs):
        mu = hidden @ w_mu
        logvar = np.clip(hidden @ w_logvar, -4.0, 4.0)
        z = mu
        logits = z @ z.T
        reconstructed = 1.0 / (1.0 + np.exp(-logits))
        reconstruction_error = reconstructed - adjacency
        grad_logits = reconstruction_error / max(1, adjacency.shape[0] ** 2)
        grad_z = (grad_logits + grad_logits.T) @ z
        grad_mu = grad_z + mu / max(1, adjacency.shape[0])
        grad_logvar = 0.5 * (np.exp(logvar) - 1.0) / max(1, adjacency.shape[0])
        grad_w_mu = hidden.T @ grad_mu
        grad_w_logvar = hidden.T @ grad_logvar
        w_mu -= learning_rate * grad_w_mu
        w_logvar -= learning_rate * grad_w_logvar

    mu = hidden @ w_mu
    norms = np.linalg.norm(mu, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mu / norms, adjacency


def compute_density(latent: np.ndarray, k: int = 10) -> list[float]:
    if len(latent) <= 1:
        return [0.0 for _ in range(len(latent))]
    scores: list[float] = []
    for idx, vector in enumerate(latent):
        distances = []
        for other_idx, other in enumerate(latent):
            if idx == other_idx:
                continue
            cosine = float(np.dot(vector, other) / (np.linalg.norm(vector) * np.linalg.norm(other)))
            distances.append(1.0 - cosine)
        distances.sort()
        neighbors = distances[: min(k, len(distances))]
        scores.append(float(sum(neighbors) / len(neighbors)) if neighbors else 0.0)
    return scores


def build_citation_graph(papers: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for paper in papers:
        graph.add_node(paper["paper_id"], paper=paper)
    for paper in papers:
        for citation in paper.get("citations", []):
            target_id = citation.get("paper_id")
            if target_id in graph:
                graph.add_edge(paper["paper_id"], target_id)
    return graph


def cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 1.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - numerator / (left_norm * right_norm)
