from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesis.scholarly import ScholarlyClient
from genesis.storage.manifold import ManifoldIndex

DEFAULT_PAPER_CORPORA = {
    "general": [
        {
            "paper_id": "genesis-general-1",
            "title": "Autonomous research orchestration systems",
            "abstract": "A survey of autonomous research loops, verification layers, and paper synthesis pipelines.",
            "year": 2025,
            "authors": [{"name": "Genesis Team"}],
            "citation_count": 10,
            "domain": "general",
            "arxiv_id": None,
            "citations": [],
            "url": "",
        }
    ],
    "ml_efficiency": [
        {
            "paper_id": "ml-eff-1",
            "title": "Learning rate warmup improves optimization stability",
            "abstract": "Warmup schedules improve convergence stability and reduce optimization shock in deep networks.",
            "year": 2024,
            "authors": [{"name": "A. Researcher"}],
            "citation_count": 25,
            "domain": "ml_efficiency",
            "arxiv_id": "2401.00001",
            "citations": [{"paper_id": "ml-eff-2", "title": "Scaling laws for efficient training"}],
            "url": "",
        },
        {
            "paper_id": "ml-eff-2",
            "title": "Scaling laws for efficient training",
            "abstract": "Training efficiency depends on optimizer schedules, model scale, and compute-aware regularization.",
            "year": 2023,
            "authors": [{"name": "B. Researcher"}],
            "citation_count": 18,
            "domain": "ml_efficiency",
            "arxiv_id": "2301.00002",
            "citations": [{"paper_id": "ml-eff-1", "title": "Learning rate warmup improves optimization stability"}],
            "url": "",
        },
    ],
    "astrophysics": [
        {
            "paper_id": "astro-1",
            "title": "Photometric redshift estimation from survey data",
            "abstract": "Photometric redshift methods rely on spectral priors, calibration, and cross-survey consistency checks.",
            "year": 2024,
            "authors": [{"name": "C. Astronomer"}],
            "citation_count": 12,
            "domain": "astrophysics",
            "arxiv_id": "2402.00003",
            "citations": [],
            "url": "",
        }
    ],
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def _hash_embedding(text: str, dim: int = 256) -> np.ndarray:
    vector = np.zeros(dim, dtype=float)
    counts = Counter(_tokenize(text))
    for token, count in counts.items():
        vector[hash(token) % dim] += float(count)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _build_citation_adjacency(papers: list[dict[str, Any]]) -> np.ndarray:
    paper_index = {paper["paper_id"]: idx for idx, paper in enumerate(papers)}
    adjacency = np.eye(len(papers), dtype=float)
    for idx, paper in enumerate(papers):
        for citation in paper.get("citations", []):
            target_id = citation.get("paper_id")
            if target_id in paper_index:
                adjacency[idx, paper_index[target_id]] = 1.0
                adjacency[paper_index[target_id], idx] = 1.0
    degree = np.sum(adjacency, axis=1)
    degree[degree == 0.0] = 1.0
    inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return inv_sqrt @ adjacency @ inv_sqrt


def _compute_latent_vectors(papers: list[dict[str, Any]], dim: int = 32) -> np.ndarray:
    text_matrix = np.vstack(
        [_hash_embedding(f"{paper.get('title', '')} {paper.get('abstract', '')}") for paper in papers]
    )
    adjacency = _build_citation_adjacency(papers)
    smoothed = adjacency @ text_matrix
    target_dim = min(dim, smoothed.shape[0], smoothed.shape[1])
    if target_dim == 0:
        return np.zeros((len(papers), 0), dtype=float)
    _, _, vh = np.linalg.svd(smoothed, full_matrices=False)
    projection = vh[:target_dim].T
    latent = smoothed @ projection
    norms = np.linalg.norm(latent, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return latent / norms


def _compute_density(latent: np.ndarray, k: int = 10) -> list[float]:
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


def _load_seed_papers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("papers", [])
    papers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            papers.append(json.loads(line))
    return papers


def _fetch_seed_papers(domain: str, limit: int, cache_path: Path) -> list[dict[str, Any]]:
    client = ScholarlyClient(cache_path=cache_path)
    query = f"{domain} research benchmark"
    raw = client.search_semantic_scholar(query, limit=limit)
    papers: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        doi = item.get("externalIds", {}).get("DOI") if isinstance(item.get("externalIds"), dict) else None
        paper_id = doi or item.get("paperId") or item.get("title") or f"{domain}-{idx}"
        papers.append(
            {
                "paper_id": paper_id,
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "year": item.get("year"),
                "authors": item.get("authors", []),
                "citation_count": item.get("citationCount", 0),
                "domain": domain,
                "arxiv_id": item.get("externalIds", {}).get("ArXiv") if isinstance(item.get("externalIds"), dict) else None,
                "citations": [
                    {"paper_id": citation.get("paperId"), "title": citation.get("title", "")}
                    for citation in item.get("citations", []) or []
                    if citation.get("paperId")
                ],
                "url": item.get("url"),
            }
        )
    return [paper for paper in papers if paper.get("title")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="general")
    parser.add_argument("--input", default="")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--latent-dim", type=int, default=32)
    args = parser.parse_args()

    root = Path("manifold_index")
    root.mkdir(exist_ok=True)
    manifold = ManifoldIndex(root)

    source_path = Path(args.input) if args.input else root / f"{args.domain}_papers.json"
    papers = _load_seed_papers(source_path)
    if not papers:
        papers = _fetch_seed_papers(args.domain, args.limit, root / "scholarly_cache.json")
    if not papers:
        papers = DEFAULT_PAPER_CORPORA.get(args.domain, DEFAULT_PAPER_CORPORA["general"])
    if not papers:
        raise SystemExit("No papers available to build the manifold")

    latent = _compute_latent_vectors(papers, dim=args.latent_dim)
    density_scores = _compute_density(latent, k=min(10, max(1, len(papers) - 1)))
    enriched: list[dict[str, Any]] = []
    for paper, latent_vector, density in zip(papers, latent, density_scores):
        enriched.append(
            {
                **paper,
                "latent_z": [round(float(value), 6) for value in latent_vector.tolist()],
                "density_score": round(float(density), 6),
            }
        )
    manifold.upsert_collection(enriched, collection="papers")
    print(
        json.dumps(
            {
                "status": "ok",
                "papers_indexed": len(enriched),
                "domain": args.domain,
                "latent_dim": min(args.latent_dim, len(enriched)),
                "mean_density": round(sum(item["density_score"] for item in enriched) / len(enriched), 6),
            }
        )
    )


if __name__ == "__main__":
    main()
