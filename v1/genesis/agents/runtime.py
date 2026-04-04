from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


@dataclass
class CategoryConfig:
    provider: str
    model: str
    fallbacks: list[str]
    temperature: float
    max_tokens: int


class ProviderRuntimeError(RuntimeError):
    pass


class CodingAgentRuntime:
    def __init__(
        self,
        config_path: str | Path,
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 120,
    ) -> None:
        self.config_path = Path(config_path)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.categories = self._load_categories()

    def generate_task(
        self,
        *,
        category: str,
        instruction: str,
        context: dict[str, Any],
        budget: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        category_config = self.categories[category]
        prompt = self._build_prompt(instruction, context, budget or {}, category)
        errors: list[str] = []
        for model in [category_config.model] + list(category_config.fallbacks):
            try:
                if category_config.provider == "ollama":
                    content = self._invoke_ollama(model, prompt, category_config)
                elif category_config.provider == "groq":
                    content = self._invoke_groq(model, prompt, category_config)
                else:
                    raise ProviderRuntimeError(f"unsupported provider: {category_config.provider}")
                payload = self._parse_payload(content)
                payload["provider"] = category_config.provider
                payload["model"] = model
                payload["raw_response"] = content
                return payload
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model}: {exc}")
        raise ProviderRuntimeError("; ".join(errors))

    def _load_categories(self) -> dict[str, CategoryConfig]:
        raw = self.config_path.read_text(encoding="utf-8")
        cleaned = re.sub(r"//.*", "", raw)
        payload = json.loads(cleaned)
        categories: dict[str, CategoryConfig] = {}
        for name, config in payload.get("categories", {}).items():
            categories[name] = CategoryConfig(
                provider=config["provider"],
                model=config["model"],
                fallbacks=list(config.get("fallbacks", [])),
                temperature=float(config.get("temperature", 0.2)),
                max_tokens=int(config.get("max_tokens", 1024)),
            )
        return categories

    def _build_prompt(
        self,
        instruction: str,
        context: dict[str, Any],
        budget: dict[str, Any],
        category: str,
    ) -> str:
        return (
            "You are the Genesis coding agent runtime.\n"
            "Return valid JSON only with keys: summary, artifact_plan, experiment_plan, citations, next_action.\n"
            f"Category: {category}\n"
            f"Instruction:\n{instruction}\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            f"Budget:\n{json.dumps(budget, indent=2)}\n"
        )

    def _invoke_ollama(self, model: str, prompt: str, category: CategoryConfig) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        response = self.session.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": category.temperature, "num_predict": category.max_tokens},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "")

    def _invoke_groq(self, model: str, prompt: str, category: CategoryConfig) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderRuntimeError("GROQ_API_KEY is not configured")
        response = self.session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": category.temperature,
                "max_tokens": category.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def _parse_payload(self, content: str) -> dict[str, Any]:
        if not content:
            raise ProviderRuntimeError("empty model response")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        raise ProviderRuntimeError("model response did not contain valid JSON")
