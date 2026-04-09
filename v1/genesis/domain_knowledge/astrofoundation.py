from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from genesis.storage.manifold import ManifoldIndex
from genesis.scholarly import ScholarlyClient

from .base import DomainKnowledgeProvider


class AstroFoundationProvider(DomainKnowledgeProvider):
    ASTRO_BOOST_TERMS = {
        "redshift",
        "galaxy",
        "spectroscopic",
        "spectroscopy",
        "photometric",
        "coordinates",
        "ra",
        "dec",
        "survey",
        "catalog",
        "sdss",
        "dr18",
        "imaging",
        "spectrum",
        "spectra",
        "source",
    }

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

        papers = self._retrieve_ranked_context(question, limit=3)
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
        papers = self._retrieve_ranked_context(query, limit=2)
        if not papers:
            return self.summary
        return self.summary + "\n" + "\n".join(
            (
                f"- {paper.get('title', 'Unknown title')}: {paper.get('abstract', '')[:220]}".strip()
                + f" [why: {paper.get('_selection_rationale', 'query-relevant overlap')}]"
            )
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
        query_tokens = self._intent_terms(query)
        ranked = sorted(
            papers,
            key=lambda paper: self._rank_paper(paper, query_tokens),
            reverse=True,
        )
        return ranked[:limit]

    def _search_ranked(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        intent_terms = self._intent_terms(query)
        condensed_query = " ".join(sorted(intent_terms)) or query
        papers = self.client.search_arxiv(f"cat:astro-ph.* AND all:{condensed_query}", limit=max(limit * 3, limit))
        ranked = sorted(papers, key=lambda paper: self._rank_paper(paper, intent_terms), reverse=True)
        return ranked[:limit]

    def _retrieve_ranked_context(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        intent_terms = self._intent_terms(query)
        combined = (
            self._search_ranked(query, limit=max(limit * 2, limit))
            + self._manifold_backed_context(query, limit=max(limit * 2, limit))
            + self.client.search_semantic_scholar(" ".join(sorted(intent_terms)) or query, limit=max(limit * 2, limit))
        )
        deduped: dict[str, dict[str, Any]] = {}
        for paper in combined:
            key = str(paper.get("paper_id") or paper.get("title") or "").strip()
            if key and key not in deduped:
                deduped[key] = paper
        ranked = sorted(deduped.values(), key=lambda paper: self._rank_paper(paper, intent_terms), reverse=True)
        for paper in ranked:
            paper["_selection_rationale"] = self._selection_rationale(paper, intent_terms)
        return ranked[:limit]

    def _intent_terms(self, query: str) -> set[str]:
        raw_tokens = {token.strip(".,:;()[]").lower() for token in query.split() if token.strip()}
        meaningful = {token for token in raw_tokens if len(token) >= 3}
        boosted = {token for token in meaningful if token in self.ASTRO_BOOST_TERMS or any(char.isdigit() for char in token)}
        return boosted or meaningful

    def _rank_paper(self, paper: dict[str, Any], intent_terms: set[str]) -> tuple[int, int]:
        title = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        overlap = sum(1 for token in intent_terms if token in title)
        boosted = sum(1 for token in intent_terms if token in self.ASTRO_BOOST_TERMS and token in title)
        title_overlap = sum(1 for token in intent_terms if token in str(paper.get("title", "")).lower())
        penalties = 0
        if overlap <= 1 and boosted == 0:
            penalties -= 2
        return boosted + title_overlap + penalties, overlap

    def _selection_rationale(self, paper: dict[str, Any], intent_terms: set[str]) -> str:
        haystack = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        matches = [token for token in sorted(intent_terms) if token in haystack]
        if not matches:
            return "retained as fallback astrophysics context"
        return "matched query terms: " + ", ".join(matches[:5])
