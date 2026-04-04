from __future__ import annotations

from .base import DomainKnowledgeProvider


class AstroFoundationProvider(DomainKnowledgeProvider):
    def __init__(self) -> None:
        self.summary = ""

    def initialize(self, research_spec: dict[str, object]) -> str:
        question = str(research_spec.get("research_question", ""))
        self.summary = (
            "AstroFoundation unavailable locally; using a deterministic astrophysics summary fallback. "
            f"Primary focus: {question[:180]}"
        )
        return self.summary

    def get_context_summary(self) -> str:
        return self.summary

    def get_relevant_context(self, query: str) -> str:
        return f"Astrophysics context for query: {query}" if self.summary else ""
