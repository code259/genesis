from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

from genesis.agents.runtime import ProviderKeyScheduler, ProviderRuntimeError
from genesis.config import ProjectConfig, load_api_config


@dataclass
class AdversarialRoute:
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 2000


class AdversarialRuntime:
    DEFAULT_ROUTES = [
        AdversarialRoute("ollama-cloud", "glm-5:cloud", temperature=0.1, max_tokens=2500),
        AdversarialRoute("groq", "openai/gpt-oss-120b", temperature=0.1, max_tokens=2500),
        AdversarialRoute("groq", "llama-3.3-70b-versatile", temperature=0.1, max_tokens=2500),
        AdversarialRoute("groq", "qwen/qwen3-32b", temperature=0.1, max_tokens=2500),
        AdversarialRoute("ollama-local", "gemma4:e4b", temperature=0.1, max_tokens=1500),
    ]

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        routes: Optional[list[AdversarialRoute]] = None,
        timeout: int = 120,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.routes = routes or list(self.DEFAULT_ROUTES)
        self.api_config = load_api_config()
        self.groq_scheduler = ProviderKeyScheduler(self.api_config["groq_api_keys"])

    def generate_acceptance_criteria(self, config: ProjectConfig) -> list[str]:
        prompt = (
            "Read the project spec and generate a concise JSON object with key 'criteria'. "
            "Each criterion must be measurable or falsifiable and directly relevant to the project.\n\n"
            f"Spec:\n{json.dumps(config.to_dict(), indent=2)}"
        )
        payload = self._invoke_json(prompt)
        criteria = payload.get("criteria", [])
        return [str(item).strip() for item in criteria if str(item).strip()]

    def analyze_criteria(
        self,
        *,
        criteria: list[str],
        completed_output: dict[str, Any],
        iteration: int,
        blockers: list[str],
        task_context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are the Genesis criteria attacker. Return JSON with keys "
            "'criteria_findings', 'critical_blockers', 'stop_recommendation'. "
            "For each criterion, decide if it is satisfied by the completed task output. "
            "Find holes and blockers. Do not invent evidence.\n\n"
            f"Iteration: {iteration}\n"
            f"Existing blockers: {json.dumps(blockers, indent=2)}\n"
            f"Task context:\n{json.dumps(task_context, indent=2)}\n"
            f"Criteria:\n{json.dumps(criteria, indent=2)}\n"
            f"Completed output:\n{json.dumps(completed_output, indent=2)}"
        )
        return self._invoke_json(prompt)

    def analyze_claims(
        self,
        *,
        claims: list[str],
        evidence_context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are the Genesis Socratic analyzer. Return JSON with key 'claim_findings'. "
            "For each claim classify it as GROUNDED, IMPLICIT_ASSUMPTION, or CONTRADICTED. "
            "Apply five-whys reasoning internally and include rationale and evidence_refs.\n\n"
            f"Claims:\n{json.dumps(claims, indent=2)}\n"
            f"Evidence context:\n{json.dumps(evidence_context, indent=2)}"
        )
        return self._invoke_json(prompt)

    def analyze_literature(
        self,
        *,
        claim: str,
        search_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = (
            "You are the Genesis literature contradiction analyzer. Return JSON with keys "
            "'contradicted', 'rationale', 'evidence_refs'. "
            "Decide whether the retrieved literature contradicts the claim.\n\n"
            f"Claim:\n{claim}\n\nRetrieved literature:\n{json.dumps(search_results, indent=2)}"
        )
        return self._invoke_json(prompt)

    def _invoke_json(self, prompt: str) -> dict[str, Any]:
        errors: list[str] = []
        last_error = ProviderRuntimeError("no adversarial provider attempts executed")
        for route in self.routes:
            try:
                content = self._invoke_route(route, prompt)
                return self._parse_json(content)
            except ProviderRuntimeError as exc:
                errors.append(f"{route.provider}/{route.model}: {exc}")
                last_error = exc
        raise ProviderRuntimeError(
            "; ".join(errors),
            error_class=last_error.error_class if errors else "adversarial_runtime_error",
            retryable=last_error.retryable if errors else True,
        )

    def _invoke_route(self, route: AdversarialRoute, prompt: str) -> str:
        if route.provider == "groq":
            return self._invoke_groq(route, prompt)
        if route.provider == "ollama-cloud":
            return self._invoke_ollama_cloud(route, prompt)
        if route.provider == "ollama-local":
            return self._invoke_ollama_local(route, prompt)
        raise ProviderRuntimeError(f"unsupported adversarial provider: {route.provider}", retryable=False)

    def _invoke_groq(self, route: AdversarialRoute, prompt: str) -> str:
        api_key = self.groq_scheduler.choose_key()
        if not api_key:
            raise ProviderRuntimeError("no Groq API keys available", error_class="groq_credentials_missing", retryable=False)
        try:
            response = self.session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": route.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": route.temperature,
                    "max_tokens": route.max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.groq_scheduler.mark_success(api_key)
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            self.groq_scheduler.mark_failure(api_key, "groq_request_failed", str(exc))
            raise ProviderRuntimeError(str(exc), error_class="groq_request_failed", retryable=True) from exc

    def _invoke_ollama_cloud(self, route: AdversarialRoute, prompt: str) -> str:
        keys = self.api_config["ollama_api_keys"]
        if not keys:
            raise ProviderRuntimeError("OLLAMA_API_KEY is not configured", error_class="ollama_credentials_missing", retryable=False)
        try:
            response = self.session.post(
                "https://ollama.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {keys[0]}"},
                json={
                    "model": route.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": route.temperature,
                    "max_tokens": route.max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            raise ProviderRuntimeError(str(exc), error_class="ollama_cloud_request_failed", retryable=True) from exc

    def _invoke_ollama_local(self, route: AdversarialRoute, prompt: str) -> str:
        base = self.api_config.get("ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
        try:
            response = self.session.post(
                f"{base}/api/chat",
                json={
                    "model": route.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": route.temperature, "num_predict": route.max_tokens},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.RequestException as exc:
            raise ProviderRuntimeError(str(exc), error_class="ollama_local_request_failed", retryable=True) from exc

    def _parse_json(self, content: str) -> dict[str, Any]:
        if not content:
            raise ProviderRuntimeError("empty adversarial runtime response", error_class="empty_response", retryable=True)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        raise ProviderRuntimeError("adversarial runtime response was not valid JSON", error_class="invalid_json_response", retryable=True)
