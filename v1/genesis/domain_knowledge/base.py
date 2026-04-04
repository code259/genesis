from __future__ import annotations

from abc import ABC, abstractmethod


class DomainKnowledgeProvider(ABC):
    @abstractmethod
    def initialize(self, research_spec: dict[str, object]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_context_summary(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_relevant_context(self, query: str) -> str:
        raise NotImplementedError


class NullProvider(DomainKnowledgeProvider):
    def initialize(self, research_spec: dict[str, object]) -> str:
        return "No domain knowledge provider configured."

    def get_context_summary(self) -> str:
        return "No domain knowledge available."

    def get_relevant_context(self, query: str) -> str:
        return ""
