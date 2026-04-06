from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from genesis.config import load_api_config


@dataclass
class CategoryConfig:
    provider: str
    model: str
    fallbacks: list[str]
    temperature: float
    max_tokens: int


class ProviderRuntimeError(RuntimeError):
    def __init__(self, message: str, *, error_class: str = "provider_error", retryable: bool = True):
        super().__init__(message)
        self.error_class = error_class
        self.retryable = retryable


class CodingAgentRuntime:
    DEFAULT_RESPONSE_SCHEMA = {
        "summary": "",
        "artifact_plan": [],
        "command_plan": [],
        "experiment_plan": [],
        "citations": [],
        "next_action": "continue",
    }

    def __init__(
        self,
        config_path: str | Path,
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 600,
    ) -> None:
        self.config_path = Path(config_path)
        self.session = session or requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        self.timeout = timeout
        self.api_config = load_api_config()
        self.categories = self._load_categories()

    def generate_task(
        self,
        *,
        category: str,
        instruction: str,
        context: dict[str, Any],
        budget: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if category not in self.categories:
            raise ProviderRuntimeError(
                f"unknown category: {category}",
                error_class="unknown_category",
                retryable=False,
            )
        category_config = self.categories[category]
        prompt = self._build_prompt(instruction, context, budget or {}, category)
        errors: list[str] = []
        last_error = ProviderRuntimeError("no provider attempts executed")
        for model in [category_config.model] + list(category_config.fallbacks):
            try:
                if category_config.provider == "ollama":
                    content = self._invoke_ollama(model, prompt, category_config)
                elif category_config.provider == "groq":
                    content = self._invoke_groq(model, prompt, category_config)
                else:
                    raise ProviderRuntimeError(f"unsupported provider: {category_config.provider}")
                payload = self._normalize_payload(self._parse_payload(content), category)
                self._validate_execution_payload(payload, category)
                payload["provider"] = category_config.provider
                payload["model"] = model
                payload["primary_model"] = category_config.model
                payload["attempted_models"] = [category_config.model] + list(category_config.fallbacks)
                payload["fallback_used"] = model != category_config.model
                payload["raw_response"] = content
                payload["retryable"] = False
                payload["error_class"] = None
                return payload
            except ProviderRuntimeError as exc:
                errors.append(f"{model}: {exc}")
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                generic = ProviderRuntimeError(str(exc), error_class="unexpected_runtime_error", retryable=False)
                errors.append(f"{model}: {generic}")
                last_error = generic
        raise ProviderRuntimeError(
            "; ".join(errors),
            error_class=last_error.error_class if errors else "provider_error",
            retryable=last_error.retryable if errors else True,
        )

    def _load_categories(self) -> dict[str, CategoryConfig]:
        raw = self.config_path.read_text(encoding="utf-8")
        cleaned = self._strip_jsonc_comments(raw)
        payload = json.loads(cleaned)
        categories: dict[str, CategoryConfig] = {}
        config_block = payload.get("categories") or payload.get("providers") or {}
        for name, config in config_block.items():
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
        schema = {
            "genesis-ideation": [
                "summary",
                "artifact_plan",
                "command_plan",
                "experiment_plan",
                "citations",
                "next_action",
                "task_tree",
            ],
            "genesis-oracle": [
                "summary",
                "artifact_plan",
                "command_plan",
                "experiment_plan",
                "citations",
                "next_action",
                "oracle_rules",
            ],
            "genesis-paper": [
                "summary",
                "artifact_plan",
                "command_plan",
                "experiment_plan",
                "citations",
                "next_action",
                "paper_body",
            ],
        }.get(
            category,
            ["summary", "artifact_plan", "command_plan", "experiment_plan", "citations", "next_action"],
        )
        return (
            "You are the Genesis coding agent runtime.\n"
            f"Return valid JSON only with keys: {', '.join(schema)}.\n"
            "For execution categories, do not claim task completion without emitting actionable work.\n"
            "A valid execution response must include at least one of artifact_plan, command_plan, or experiment_plan.\n"
            "Commands must be literal executable shell commands. Do not invent tool names or pseudocode.\n"
            "Do not suggest publication, submission, or finalization unless substantive verified artifacts already exist.\n"
            f"Category: {category}\n"
            f"Instruction:\n{instruction}\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            f"Budget:\n{json.dumps(budget, indent=2)}\n"
        )

    def _invoke_ollama(self, model: str, prompt: str, category: CategoryConfig) -> str:
        base_url = self.api_config["ollama_base_url"]
        try:
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
        except requests.RequestException as exc:
            raise ProviderRuntimeError(
                str(exc),
                error_class="ollama_connection_error",
                retryable=True,
            ) from exc

    def _invoke_groq(self, model: str, prompt: str, category: CategoryConfig) -> str:
        keys = self.api_config["groq_api_keys"]
        if not keys:
            raise ProviderRuntimeError(
                "GROQ_API_KEY is not configured",
                error_class="groq_credentials_missing",
                retryable=False,
            )
        errors: list[str] = []
        for api_key in keys:
            try:
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
            except requests.RequestException as exc:
                errors.append(str(exc))
        raise ProviderRuntimeError(
            "; ".join(errors),
            error_class="groq_request_failed",
            retryable=True,
        )

    def _parse_payload(self, content: str) -> dict[str, Any]:
        if not content:
            raise ProviderRuntimeError("empty model response", error_class="empty_response", retryable=True)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            extracted = self._extract_json_object(content)
            if extracted is not None:
                return json.loads(extracted)
        raise ProviderRuntimeError(
            "model response did not contain valid JSON",
            error_class="invalid_json_response",
            retryable=True,
        )

    def _normalize_payload(self, payload: dict[str, Any], category: str) -> dict[str, Any]:
        normalized = dict(self.DEFAULT_RESPONSE_SCHEMA)
        if category == "genesis-ideation":
            normalized["task_tree"] = []
        elif category == "genesis-oracle":
            normalized["oracle_rules"] = []
        elif category == "genesis-paper":
            normalized["paper_body"] = ""
        normalized.update(payload)

        for list_key in ("artifact_plan", "command_plan", "experiment_plan", "citations", "task_tree", "oracle_rules"):
            if list_key in normalized and not isinstance(normalized[list_key], list):
                normalized[list_key] = []
        for string_key in ("summary", "next_action", "paper_body"):
            if string_key in normalized and not isinstance(normalized[string_key], str):
                normalized[string_key] = str(normalized[string_key])
        return normalized

    def _validate_execution_payload(self, payload: dict[str, Any], category: str) -> None:
        if category != "sisyphus":
            return
        has_actions = any(
            isinstance(payload.get(key), list) and bool(payload.get(key))
            for key in ("artifact_plan", "command_plan", "experiment_plan")
        )
        if not has_actions:
            raise ProviderRuntimeError(
                "non-actionable execution response: expected artifact_plan, command_plan, or experiment_plan",
                error_class="non_actionable_plan",
                retryable=True,
            )
        artifact_paths = {
            str(item.get("path", "")).strip()
            for item in payload.get("artifact_plan", [])
            if isinstance(item, dict) and str(item.get("path", "")).strip()
        }
        for command_entry in payload.get("command_plan", []):
            command = self._command_tokens(command_entry)
            referenced_file = self._workspace_file_reference(command)
            if referenced_file and referenced_file not in artifact_paths:
                raise ProviderRuntimeError(
                    f"command_plan references workspace file '{referenced_file}' without creating it in artifact_plan",
                    error_class="command_plan_missing_artifact",
                    retryable=True,
                )

    def _command_tokens(self, entry: Any) -> list[str]:
        if isinstance(entry, str) and entry.strip():
            return shlex.split(entry)
        if isinstance(entry, dict):
            command_value = entry.get("command")
            if isinstance(command_value, str) and command_value.strip():
                return shlex.split(command_value)
            if isinstance(command_value, list) and all(isinstance(part, str) for part in command_value):
                return list(command_value)
        return []

    def _workspace_file_reference(self, command: list[str]) -> str:
        for token in command[1:]:
            cleaned = token.strip()
            if not cleaned or cleaned.startswith("-"):
                continue
            if "/" in cleaned or "." in Path(cleaned).name:
                name = Path(cleaned).name
                if name.endswith((".py", ".sh", ".ipynb", ".R", ".jl")):
                    return cleaned
        return ""

    def _extract_json_object(self, content: str) -> Optional[str]:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced:
            return fenced.group(1)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def _strip_jsonc_comments(self, content: str) -> str:
        output: list[str] = []
        in_string = False
        escape = False
        index = 0
        length = len(content)
        while index < length:
            char = content[index]
            nxt = content[index + 1] if index + 1 < length else ""
            if in_string:
                output.append(char)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                output.append(char)
                index += 1
                continue
            if char == "/" and nxt == "/":
                while index < length and content[index] != "\n":
                    index += 1
                continue
            output.append(char)
            index += 1
        return "".join(output)
