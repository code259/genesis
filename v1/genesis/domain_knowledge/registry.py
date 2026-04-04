from __future__ import annotations

from .astrofoundation import AstroFoundationProvider
from .base import DomainKnowledgeProvider, NullProvider


class DomainKnowledgeRegistry:
    def get_provider(self, domain: str) -> DomainKnowledgeProvider:
        normalized = domain.lower()
        if normalized == "astrophysics":
            return AstroFoundationProvider()
        return NullProvider()
