from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

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

        papers = self.client.search_arxiv(f"astrophysics {question}", limit=3)
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
        papers = self.client.search_arxiv(query, limit=2)
        if not papers:
            papers = self.client.search_semantic_scholar(query, limit=2)
        if not papers:
            return self.summary
        return self.summary + "\n" + "\n".join(
            f"- {paper.get('title', 'Unknown title')}" for paper in papers
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
