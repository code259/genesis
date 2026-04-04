from __future__ import annotations

from pathlib import Path

from .astrofoundation import AstroFoundationProvider
from .base import DomainKnowledgeProvider, NullProvider


class DomainKnowledgeRegistry:
    def __init__(self, *, cache_root: str | Path | None = None) -> None:
        self.cache_root = cache_root

    def get_provider(self, domain: str) -> DomainKnowledgeProvider:
        normalized = domain.lower().strip()
        if normalized == "astrophysics":
            return AstroFoundationProvider(cache_root=self.cache_root)
        if normalized in {"general", "ml_efficiency", "none", ""}:
            return NullProvider(cache_root=self.cache_root)
        return NullProvider(reason=f"Unsupported domain provider '{domain}'.", cache_root=self.cache_root)
