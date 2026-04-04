from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesis.manifold_utils import compute_density, train_graph_vae
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
    parser.add_argument("--epochs", type=int, default=200)
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

    latent, adjacency = train_graph_vae(
        papers,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
    )
    density_scores = compute_density(latent, k=min(10, max(1, len(papers) - 1)))
    enriched: list[dict[str, Any]] = []
    for index, (paper, latent_vector, density) in enumerate(zip(papers, latent, density_scores)):
        enriched.append(
            {
                **paper,
                "embedding": hash_vector(paper),
                "latent_z": [round(float(value), 6) for value in latent_vector.tolist()],
                "density_score": round(float(density), 6),
                "graph_neighbors": int(adjacency[index].sum() - 1),
            }
        )
    manifold.upsert_collection(enriched, collection="papers")
    manifest_path = root / f"{args.domain}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "domain": args.domain,
                "paper_count": len(enriched),
                "latent_dim": min(args.latent_dim, len(enriched)),
                "mean_density": round(sum(item["density_score"] for item in enriched) / len(enriched), 6),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "papers_indexed": len(enriched),
                "domain": args.domain,
                "latent_dim": min(args.latent_dim, len(enriched)),
                "mean_density": round(sum(item["density_score"] for item in enriched) / len(enriched), 6),
                "manifest_path": str(manifest_path),
            }
        )
    )


def hash_vector(paper: dict[str, Any]) -> list[float]:
    from genesis.manifold_utils import hash_embedding

    return [
        round(float(value), 6)
        for value in hash_embedding(f"{paper.get('title', '')} {paper.get('abstract', '')}", dim=64).tolist()
    ]


if __name__ == "__main__":
    main()
