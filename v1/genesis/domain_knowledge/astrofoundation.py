from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from genesis.storage.manifold import ManifoldIndex
from genesis.scholarly import ScholarlyClient

from .base import DomainKnowledgeProvider


class AstroFoundationProvider(DomainKnowledgeProvider):
    def __init__(self, *, cache_root: str | Path | None = None) -> None:
        super().__init__(cache_root=cache_root)
        self.client = ScholarlyClient(cache_path=self.cache_root / "astrofoundation_cache.json")

    def initialize(self, research_spec: dict[str, Any]) -> str:
        question = str(research_spec.get("research_question", "")).strip()
        provider_summary = self._astrofoundation_summary(question)
        if provider_summary:
            self.summary = provider_summary.strip()
            self.source = "astrofoundation"
            return self.summary

        papers = self.client.search_arxiv(f"cat:astro-ph.* AND all:{question}", limit=3)
        if not papers:
            papers = self._manifold_backed_context(question, limit=3)
        if not papers:
            papers = self.client.search_semantic_scholar(f"astrophysics {question}", limit=3)
        if papers:
            bullet_lines = [
                f"- {paper.get('title', 'Unknown title')} ({paper.get('year', 'n/a')})"
                for paper in papers
            ]
            self.summary = (
                "AstroFoundation unavailable locally; retrieved astrophysics context from external literature.\n"
                + "\n".join(bullet_lines)
            )
            self.source = "retrieval"
            return self.summary

        self.summary = (
            "AstroFoundation unavailable locally and no remote astrophysics context was retrieved. "
            f"Primary focus: {question[:180]}"
        )
        self.source = "fallback"
        return self.summary

    def get_context_summary(self) -> str:
        return self.summary

    def get_relevant_context(self, query: str) -> str:
        query = query.strip()
        if not self.summary:
            return ""
        if not query:
            return self.summary
        papers = self.client.search_arxiv(f"cat:astro-ph.* AND all:{query}", limit=2)
        if not papers:
            papers = self._manifold_backed_context(query, limit=2)
        if not papers:
            papers = self.client.search_semantic_scholar(query, limit=2)
        if not papers:
            return self.summary
        return self.summary + "\n" + "\n".join(
            f"- {paper.get('title', 'Unknown title')}: {paper.get('abstract', '')[:220]}".strip()
            for paper in papers
        )

    def _astrofoundation_summary(self, question: str) -> str:
        if importlib.util.find_spec("astrofoundation") is None:
            return ""
        try:
            import astrofoundation  # type: ignore

            if hasattr(astrofoundation, "summarize"):
                return str(astrofoundation.summarize(question))
        except Exception:
            return ""
        return ""

    def _manifold_backed_context(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        manifold_path = Path.cwd() / "manifold_index"
        if not manifold_path.exists():
            return []
        manifold = ManifoldIndex(manifold_path)
        papers = manifold.all_papers()
        if not papers:
            return []
        query_tokens = set(query.lower().split())
        ranked = sorted(
            papers,
            key=lambda paper: len(query_tokens & set(f"{paper.get('title', '')} {paper.get('abstract', '')}".lower().split())),
            reverse=True,
        )
        return ranked[:limit]
