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

    def __post_init__(self) -> None:
        if self.time_budget_hours <= 0:
            raise ValueError("time_budget_hours must be a positive integer")
        self.compute_budget = self.compute_budget.strip()
        self.domain = self.domain.strip()
        self.domain_knowledge_model = self.domain_knowledge_model.strip()
        self.output_dir = self.output_dir.strip()
        if not all([self.compute_budget, self.domain, self.domain_knowledge_model, self.output_dir]):
            raise ValueError("project configuration contains empty required fields")

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
    time_budget = payload.get("time_budget_hours")
    if not isinstance(time_budget, int):
        raise ValueError("time_budget_hours must be an integer")
    return ProjectConfig(
        research_question=_require_str(payload, "research_question"),
        domain=_require_str(payload, "domain"),
        compute_budget=_require_str(payload, "compute_budget"),
        time_budget_hours=time_budget,
        domain_knowledge_model=_require_str(payload, "domain_knowledge_model"),
        output_dir=_require_str(payload, "output_dir"),
        success_criteria=_require_list(payload, "success_criteria"),
        oracle_hints=_require_list(payload, "oracle_hints"),
    )


def load_api_config() -> dict[str, Any]:
    dotenv = _load_dotenv()
    semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or dotenv.get("SEMANTIC_SCHOLAR_API_KEY")
    groq_keys = [
        value
        for key, value in _merged_env(dotenv).items()
        if key.startswith("GROQ_API_KEY") and value
    ]
    ollama_keys = [
        value
        for key, value in _merged_env(dotenv).items()
        if key.startswith("OLLAMA_API_KEY") and value
    ]
    ollama_base_url = os.getenv("OLLAMA_BASE_URL") or dotenv.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
    return {
        "semantic_scholar_api_key": semantic_scholar_key,
        "groq_api_keys": groq_keys,
        "ollama_api_keys": ollama_keys,
        "ollama_base_url": ollama_base_url,
    }


def _load_dotenv() -> dict[str, str]:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path.home() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            values: dict[str, str] = {}
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
            return values
    return {}


def _merged_env(dotenv: dict[str, str]) -> dict[str, str]:
    merged = dict(dotenv)
    merged.update({key: value for key, value in os.environ.items() if value})
    return merged
