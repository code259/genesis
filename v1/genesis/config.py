from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class ProjectConfig:
    research_question: str
    domain: str
    compute_budget: str
    time_budget_hours: int
    domain_knowledge_model: str
    output_dir: str
    success_criteria: list[str] = field(default_factory=list)
    oracle_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return value


def load_project_config(path: Union[str, Path]) -> ProjectConfig:
    payload = json.loads(Path(path).read_text())
    return ProjectConfig(
        research_question=_require_str(payload, "research_question"),
        domain=_require_str(payload, "domain"),
        compute_budget=_require_str(payload, "compute_budget"),
        time_budget_hours=int(payload["time_budget_hours"]),
        domain_knowledge_model=_require_str(payload, "domain_knowledge_model"),
        output_dir=_require_str(payload, "output_dir"),
        success_criteria=_require_list(payload, "success_criteria"),
        oracle_hints=_require_list(payload, "oracle_hints"),
    )


def load_api_config() -> dict[str, Any]:
    semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    groq_keys = [
        value
        for key, value in sorted(os.environ.items())
        if key.startswith("GROQ_API_KEY") and value
    ]
    ollama_keys = [
        value
        for key, value in sorted(os.environ.items())
        if key.startswith("OLLAMA_API_KEY") and value
    ]
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return {
        "semantic_scholar_api_key": semantic_scholar_key,
        "groq_api_keys": groq_keys,
        "ollama_api_keys": ollama_keys,
        "ollama_base_url": ollama_base_url,
    }
