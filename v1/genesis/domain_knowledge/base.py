from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DomainKnowledgeProvider(ABC):
    def __init__(self, *, cache_root: str | Path | None = None) -> None:
        env_root = os.getenv("GENESIS_CACHE_ROOT")
        root = Path(cache_root or env_root or (Path(tempfile.gettempdir()) / "genesis-cache"))
        root.mkdir(parents=True, exist_ok=True)
        self.cache_root = root
        self.summary = ""
        self.source = "none"

    @abstractmethod
    def initialize(self, research_spec: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_context_summary(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_relevant_context(self, query: str) -> str:
        raise NotImplementedError


class NullProvider(DomainKnowledgeProvider):
    def __init__(self, *, reason: str = "No domain knowledge provider configured.", cache_root: str | Path | None = None) -> None:
        super().__init__(cache_root=cache_root)
        self.reason = reason

    def initialize(self, research_spec: dict[str, Any]) -> str:
        domain = str(research_spec.get("domain", "general"))
        question = str(research_spec.get("research_question", "")).strip()
        self.source = "null"
        if question:
            self.summary = f"No specialized domain provider configured for '{domain}'. Research focus: {question}"
        else:
            self.summary = self.reason
        return self.summary

    def get_context_summary(self) -> str:
        return self.summary or self.reason

    def get_relevant_context(self, query: str) -> str:
        query = query.strip()
        if not query:
            return self.get_context_summary()
        if not self.summary:
            return f"{self.reason} Query: {query}"
        return f"{self.summary}\nRelevant query: {query}"
