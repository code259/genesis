from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from genesis.config import load_api_config


@dataclass
class RouteConfig:
    provider: str
    model: str
    kind: str = "cloud"
    temperature: float = 0.2
    max_tokens: int = 2048


@dataclass
class RoleConfig:
    agent: str
    routes: list[RouteConfig]
    prompt_append: str = ""


@dataclass
class CategoryConfig:
    provider: str
    model: str
    fallbacks: list[str]
    temperature: float
    max_tokens: int


@dataclass
class RuntimeConfig:
    command: str
    cwd: Path
    config_path: Optional[Path]
    config_dir: Optional[Path]
    omo_config_path: Optional[Path]
    roles: dict[str, RoleConfig]
    category_map: dict[str, str] = field(default_factory=dict)


@dataclass
class KeyState:
    cooldown_until: float = 0.0
    day_exhausted_until: float = 0.0


class ProviderRuntimeError(RuntimeError):
    def __init__(self, message: str, *, error_class: str = "provider_error", retryable: bool = True):
        super().__init__(message)
        self.error_class = error_class
        self.retryable = retryable


class ProviderKeyScheduler:
    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self._states = {key: KeyState() for key in self.keys}
        self._sticky_key: Optional[str] = self.keys[0] if self.keys else None
        self._rotation_index = 0

    def choose_key(self) -> Optional[str]:
        if not self.keys:
            return None
        now = time.time()
        if self._sticky_key is not None and self._is_available(self._sticky_key, now):
            return self._sticky_key
        available = [key for key in self.keys if self._is_available(key, now)]
        if not available:
            return None
        selected = available[self._rotation_index % len(available)]
        self._rotation_index += 1
        self._sticky_key = selected
        return selected

    def mark_success(self, key: Optional[str]) -> None:
        if key:
            self._sticky_key = key

    def mark_failure(self, key: Optional[str], error_class: str, details: str = "") -> None:
        if not key or key not in self._states:
            return
        now = time.time()
        state = self._states[key]
        lowered = details.lower()
        if error_class in {"groq_daily_quota_exhausted", "groq_key_exhausted"} or "per day" in lowered:
            state.day_exhausted_until = max(state.day_exhausted_until, now + 24 * 60 * 60)
        elif error_class in {"groq_rate_limited", "provider_rate_limited"} or "per minute" in lowered or "rate limit" in lowered:
            state.cooldown_until = max(state.cooldown_until, now + 70)
        else:
            state.cooldown_until = max(state.cooldown_until, now + 10)
        if self._sticky_key == key:
            self._sticky_key = None

    def _is_available(self, key: str, now: float) -> bool:
        state = self._states[key]
        return state.cooldown_until <= now and state.day_exhausted_until <= now


class CodingAgentRuntime:
    DEFAULT_RESPONSE_SCHEMA = {
        "summary": "",
        "artifact_plan": [],
        "command_plan": [],
        "experiment_plan": [],
        "citations": [],
        "next_action": "continue",
    }
    DEFAULT_CATEGORY_MAP = {
        "sisyphus": "deep",
        "genesis-ideation": "quick",
        "genesis-oracle": "oracle",
        "genesis-adversarial": "adversarial",
        "genesis-paper": "writer",
        "genesis-citations": "quick",
        "genesis-plotting": "quick",
    }

    def __init__(
        self,
        config_path: str | Path,
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 600,
        runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.session = session or requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        self.timeout = timeout
        self.runner = runner or subprocess.run
        self.api_config = load_api_config()
        self.groq_scheduler = ProviderKeyScheduler(self.api_config["groq_api_keys"])
        self.runtime = self._load_runtime()
        self._category_configs = self._build_category_configs()
        self._provider_model_cache: dict[str, set[str]] = {}

    @property
    def categories(self) -> dict[str, CategoryConfig]:
        return self._category_configs

    def generate_task(
        self,
        *,
        category: str,
        instruction: str,
        context: dict[str, Any],
        budget: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        role_name = self._resolve_role_name(category)
        if role_name not in self.runtime.roles:
            raise ProviderRuntimeError(
                f"unknown category: {category}",
                error_class="unknown_category",
                retryable=False,
            )
        role = self.runtime.roles[role_name]
        errors: list[str] = []
        attempted_models: list[str] = []
        last_error = ProviderRuntimeError("no provider attempts executed")
        for route in role.routes:
            prompt = self._build_prompt(instruction, context, budget or {}, category, role.prompt_append, route)
            attempted_models.append(route.model)
            active_key = self._active_key_for_route(route)
            content = ""
            try:
                if not isinstance(self.session, requests.Session):
                    content = self._invoke_mock_session(route, prompt)
                else:
                    content = self._invoke_opencode(role.agent, route, prompt, active_key)
                payload = self._normalize_payload(self._parse_payload(content), category)
                payload["validation_mode"] = self._validate_execution_payload(payload, category)
                if route.provider == "groq":
                    self.groq_scheduler.mark_success(active_key)
                payload["provider"] = route.provider
                payload["model"] = route.model
                payload["primary_model"] = role.routes[0].model if role.routes else route.model
                payload["attempted_models"] = attempted_models.copy()
                payload["fallback_used"] = len(attempted_models) > 1
                payload["raw_response"] = content
                payload["retryable"] = False
                payload["error_class"] = None
                return payload
            except ProviderRuntimeError as exc:
                if content.strip() and exc.error_class in {"non_actionable_plan", "command_plan_missing_artifact"}:
                    raise ProviderRuntimeError(
                        f"{exc}. First non-empty response from {route.model}: {content[:800]}",
                        error_class=exc.error_class,
                        retryable=False,
                    ) from exc
                if route.provider == "groq":
                    self.groq_scheduler.mark_failure(active_key, exc.error_class, str(exc))
                errors.append(f"{route.model}: {exc}")
                last_error = exc
        raise ProviderRuntimeError(
            "; ".join(errors),
            error_class=last_error.error_class if errors else "provider_error",
            retryable=last_error.retryable if errors else True,
        )

    def check_health(self, *, probe_models: bool = False) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        command_path = shutil.which(self.runtime.command)
        checks.append(
            {
                "name": "opencode_binary",
                "passed": command_path is not None,
                "evidence": [command_path or f"missing:{self.runtime.command}"],
            }
        )
        for name, path in {
            "runtime_config": self.config_path,
            "opencode_config": self.runtime.config_path,
            "omo_config": self.runtime.omo_config_path,
        }.items():
            if path is None:
                continue
            checks.append(
                {
                    "name": name,
                    "passed": path.exists(),
                    "evidence": [str(path)],
                }
            )
        checks.append(
            {
                "name": "ollama_cloud_auth",
                "passed": bool(self.api_config["ollama_api_keys"]),
                "evidence": [f"ollama_api_keys={len(self.api_config['ollama_api_keys'])}"],
            }
        )
        checks.append(
            {
                "name": "groq_auth",
                "passed": bool(self.api_config["groq_api_keys"]),
                "evidence": [f"groq_api_keys={len(self.api_config['groq_api_keys'])}"],
            }
        )
        role_checks = self._role_presence_checks()
        checks.extend(role_checks)
        provider_health_passed = all(bool(check.get("passed")) for check in checks)
        model_listing_passed = True
        execution_viability_passed = True
        if probe_models and command_path is not None:
            probe_checks = self._probe_routes()
            checks.extend(probe_checks)
            model_listing_checks = [check for check in probe_checks if str(check.get("name", "")).startswith("model_listed:")]
            execution_checks = [check for check in probe_checks if str(check.get("name", "")).startswith("execution_viable:")]
            model_listing_passed = all(bool(check.get("passed")) for check in model_listing_checks) if model_listing_checks else True
            execution_viability_passed = all(bool(check.get("passed")) for check in execution_checks) if execution_checks else True
        return {
            "passed": provider_health_passed and model_listing_passed and execution_viability_passed,
            "provider_health_passed": provider_health_passed,
            "model_listing_passed": model_listing_passed,
            "execution_viability_passed": execution_viability_passed,
            "checks": checks,
        }

    def _load_runtime(self) -> RuntimeConfig:
        raw = self.config_path.read_text(encoding="utf-8")
        payload = json.loads(self._strip_jsonc_comments(raw))
        if "routing" in payload:
            return self._load_routing_config(payload)
        return self._load_legacy_runtime(payload)

    def _load_routing_config(self, payload: dict[str, Any]) -> RuntimeConfig:
        opencode_config = payload.get("opencode", {})
        cwd = self._resolve_path(opencode_config.get("cwd")) or self.config_path.resolve().parents[1]
        runtime = RuntimeConfig(
            command=str(opencode_config.get("command", "opencode")),
            cwd=cwd,
            config_path=self._resolve_path(opencode_config.get("config_path")),
            config_dir=self._resolve_path(opencode_config.get("config_dir")),
            omo_config_path=self._resolve_path(opencode_config.get("omo_config_path")),
            roles={},
            category_map={**self.DEFAULT_CATEGORY_MAP, **payload.get("routing", {}).get("category_map", {})},
        )
        for role_name, config in payload.get("routing", {}).get("roles", {}).items():
            routes = [
                RouteConfig(
                    provider=str(item.get("provider", "")),
                    model=str(item.get("model", "")),
                    kind=str(item.get("kind", "cloud")),
                    temperature=float(item.get("temperature", 0.2)),
                    max_tokens=int(item.get("max_tokens", 2048)),
                )
                for item in config.get("routes", [])
                if isinstance(item, dict) and str(item.get("provider", "")).strip() and str(item.get("model", "")).strip()
            ]
            if routes:
                runtime.roles[role_name] = RoleConfig(
                    agent=str(config.get("agent", "Sisyphus")),
                    routes=routes,
                    prompt_append=str(config.get("prompt_append", "")),
                )
        if not runtime.roles:
            raise ValueError(f"runtime config {self.config_path} did not define any routing roles")
        return runtime

    def _load_legacy_runtime(self, payload: dict[str, Any]) -> RuntimeConfig:
        categories = payload.get("categories") or payload.get("providers") or {}
        cwd = self.config_path.resolve().parents[1]
        runtime = RuntimeConfig(
            command="opencode",
            cwd=cwd,
            config_path=cwd / ".opencode" / "opencode.json",
            config_dir=cwd / ".opencode",
            omo_config_path=cwd / ".opencode" / "oh-my-openagent.jsonc",
            roles={},
            category_map=dict(self.DEFAULT_CATEGORY_MAP),
        )
        for name, config in categories.items():
            primary = RouteConfig(
                provider=str(config.get("provider", "")),
                model=str(config.get("model", "")),
                kind="cloud" if ":cloud" in str(config.get("model", "")) else "local",
                temperature=float(config.get("temperature", 0.2)),
                max_tokens=int(config.get("max_tokens", 1024)),
            )
            fallbacks = [
                RouteConfig(
                    provider=str(config.get("provider", "")),
                    model=str(model),
                    kind="cloud" if ":cloud" in str(model) else "local",
                    temperature=float(config.get("temperature", 0.2)),
                    max_tokens=int(config.get("max_tokens", 1024)),
                )
                for model in config.get("fallbacks", [])
                if str(model).strip()
            ]
            role_name = self.DEFAULT_CATEGORY_MAP.get(name, name)
            runtime.roles[role_name] = RoleConfig(
                agent="Sisyphus" if name == "sisyphus" else self._default_agent_for_category(name),
                routes=[primary] + fallbacks,
            )
            runtime.category_map[name] = role_name
        return runtime

    def _build_category_configs(self) -> dict[str, CategoryConfig]:
        category_configs: dict[str, CategoryConfig] = {}
        for category, role_name in self.runtime.category_map.items():
            role = self.runtime.roles.get(role_name)
            if role is None or not role.routes:
                continue
            primary = role.routes[0]
            category_configs[category] = CategoryConfig(
                provider=primary.provider,
                model=primary.model,
                fallbacks=[route.model for route in role.routes[1:]],
                temperature=primary.temperature,
                max_tokens=primary.max_tokens,
            )
        for role_name, role in self.runtime.roles.items():
            if role_name in category_configs or not role.routes:
                continue
            primary = role.routes[0]
            category_configs[role_name] = CategoryConfig(
                provider=primary.provider,
                model=primary.model,
                fallbacks=[route.model for route in role.routes[1:]],
                temperature=primary.temperature,
                max_tokens=primary.max_tokens,
            )
        return category_configs

    def _resolve_role_name(self, category: str) -> str:
        return self.runtime.category_map.get(category, category)

    def _build_prompt(
        self,
        instruction: str,
        context: dict[str, Any],
        budget: dict[str, Any],
        category: str,
        prompt_append: str,
        route: RouteConfig,
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
        context_payload = context
        budget_payload = budget
        extra_bits: list[str] = []
        if prompt_append.strip():
            extra_bits.append(prompt_append.strip())
        if route.kind == "backup_light":
            context_payload = self._compact_context(context)
            budget_payload = {"mode": "backup_light", **{key: value for key, value in budget.items() if key in {"max_runs", "compute_budget"}}}
            extra_bits.append(
                "Backup-light mode: minimize context usage, prefer a compact actionable response, and avoid verbose justification."
            )
        extra = f"\nAdditional routing guidance:\n{chr(10).join(extra_bits)}\n" if extra_bits else ""
        return (
            "You are the Genesis execution backend running inside OpenCode with oh-my-openagent.\n"
            f"Return valid JSON only with keys: {', '.join(schema)}.\n"
            "For execution categories, do not claim task completion without emitting actionable work.\n"
            "A valid execution response must include at least one of artifact_plan, command_plan, or experiment_plan.\n"
            "Commands must be literal executable shell commands. Do not invent tool names or pseudocode.\n"
            "Do not suggest publication, submission, or finalization unless substantive verified artifacts already exist.\n"
            f"Category: {category}\n"
            f"Instruction:\n{instruction}\n\n"
            f"Context:\n{json.dumps(context_payload, indent=2)}\n\n"
            f"Budget:\n{json.dumps(budget_payload, indent=2)}\n"
            f"{extra}"
        )

    def _compact_context(self, context: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key in ("task_id", "research_question", "domain", "failed_iterations", "debug_focus", "verification_failures"):
            if key in context:
                compact[key] = context[key]
        prior_runs = context.get("prior_runs", [])
        if isinstance(prior_runs, list) and prior_runs:
            compact["prior_runs"] = [
                {
                    "run_n": run.get("run_n"),
                    "classification": run.get("classification", ""),
                    "failure_type": run.get("failure_type", ""),
                    "debug_focus": run.get("debug_focus", ""),
                }
                for run in prior_runs[-2:]
                if isinstance(run, dict)
            ]
        return compact or context

    def _invoke_opencode(
        self,
        agent: str,
        route: RouteConfig,
        prompt: str,
        active_key: Optional[str],
    ) -> str:
        env = self._build_subprocess_env(route, active_key)
        command = [
            self.runtime.command,
            "run",
            "--model",
            route.model,
            "--format",
            "json",
            prompt,
        ]
        if agent.strip():
            command[2:2] = ["--agent", agent]
        try:
            process = self.runner(
                command,
                cwd=str(self.runtime.cwd),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise ProviderRuntimeError(
                f"opencode executable not found: {self.runtime.command}",
                error_class="opencode_missing",
                retryable=False,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderRuntimeError(
                f"opencode timed out after {self.timeout}s",
                error_class="opencode_timeout",
                retryable=True,
            ) from exc

        stdout = process.stdout or ""
        stderr = process.stderr or ""
        if process.returncode != 0:
            raise ProviderRuntimeError(
                stderr.strip() or stdout.strip() or f"opencode exited with code {process.returncode}",
                error_class=self._classify_subprocess_error(route.provider, stderr or stdout),
                retryable=self._is_retryable_subprocess_error(stderr or stdout),
            )
        content = self._extract_opencode_content(stdout)
        if not content.strip():
            raise ProviderRuntimeError("empty model response", error_class="empty_response", retryable=True)
        return content.strip()

    def _invoke_mock_session(self, route: RouteConfig, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions" if route.provider == "groq" else "http://127.0.0.1:11434/api/chat"
        response = self.session.post(
            url,
            json={
                "model": route.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if route.provider == "groq":
            return payload["choices"][0]["message"]["content"]
        return payload.get("message", {}).get("content", "")

    def _build_subprocess_env(self, route: RouteConfig, active_key: Optional[str]) -> dict[str, str]:
        env = dict(os.environ)
        if self.runtime.config_path is not None:
            env["OPENCODE_CONFIG"] = str(self.runtime.config_path)
        if self.runtime.config_dir is not None:
            env["OPENCODE_CONFIG_DIR"] = str(self.runtime.config_dir)
        self._inject_numbered_env(env, "OLLAMA_API_KEY", self.api_config["ollama_api_keys"])
        self._inject_numbered_env(env, "GROQ_API_KEY", self.api_config["groq_api_keys"])
        if route.provider == "groq":
            self._unset_prefixed_keys(env, "GROQ_API_KEY")
            if active_key:
                env["GROQ_API_KEY"] = active_key
        elif route.provider.startswith("ollama"):
            keys = self.api_config["ollama_api_keys"]
            if keys:
                env["OLLAMA_API_KEY"] = keys[0]
            base_url = self.api_config.get("ollama_base_url")
            if base_url and route.provider == "ollama":
                env["OLLAMA_BASE_URL"] = str(base_url)
        return env

    def _unset_prefixed_keys(self, env: dict[str, str], prefix: str) -> None:
        for key in list(env):
            if key == prefix or key.startswith(f"{prefix}_"):
                env.pop(key, None)

    def _inject_numbered_env(self, env: dict[str, str], prefix: str, values: list[str]) -> None:
        for index, value in enumerate(values, start=1):
            key = prefix if index == 1 else f"{prefix}_{index}"
            env[key] = value

    def _active_key_for_route(self, route: RouteConfig) -> Optional[str]:
        if route.provider == "groq":
            return self.groq_scheduler.choose_key()
        return None

    def _role_presence_checks(self) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        for category, role_name in sorted(self.runtime.category_map.items()):
            checks.append(
                {
                    "name": f"role_mapping:{category}",
                    "passed": role_name in self.runtime.roles,
                    "evidence": [role_name],
                }
            )
        return checks

    def _probe_routes(self) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for role_name, role in self.runtime.roles.items():
            for route in role.routes:
                signature = f"{role.agent}|{route.model}"
                if signature in seen:
                    continue
                seen.add(signature)
                available_models = self._list_provider_models(route)
                checks.append(
                    {
                        "name": f"model_listed:{route.model}",
                        "passed": route.model in available_models,
                        "evidence": [route.provider, f"listed_count={len(available_models)}"],
                    }
                )
                active_key = self._active_key_for_route(route)
                try:
                    content = self._invoke_opencode(
                        role.agent,
                        route,
                        "Reply with OK only.",
                        active_key,
                    )
                    checks.append(
                        {
                            "name": f"execution_viable:{route.model}",
                            "passed": bool(content.strip()),
                            "evidence": [role_name, content.strip()[:120]],
                        }
                    )
                    if route.provider == "groq":
                        self.groq_scheduler.mark_success(active_key)
                except ProviderRuntimeError as exc:
                    if route.provider == "groq":
                        self.groq_scheduler.mark_failure(active_key, exc.error_class, str(exc))
                    checks.append(
                        {
                            "name": f"execution_viable:{route.model}",
                            "passed": False,
                            "evidence": [exc.error_class, str(exc)],
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    checks.append(
                        {
                            "name": f"execution_viable:{route.model}",
                            "passed": False,
                            "evidence": ["unexpected_probe_error", str(exc)],
                        }
                    )
        return checks

    def _list_provider_models(self, route: RouteConfig) -> set[str]:
        if route.provider in self._provider_model_cache:
            return self._provider_model_cache[route.provider]
        env = self._build_subprocess_env(route, self._active_key_for_route(route))
        try:
            process = self.runner(
                [self.runtime.command, "models", route.provider],
                cwd=str(self.runtime.cwd),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=min(self.timeout, 120),
            )
        except Exception:
            self._provider_model_cache[route.provider] = set()
            return set()
        if process.returncode != 0:
            self._provider_model_cache[route.provider] = set()
            return set()
        lines = [self._strip_ansi(line).strip() for line in process.stdout.splitlines()]
        models = {line for line in lines if line and "/" in line}
        self._provider_model_cache[route.provider] = models
        return models

    def _extract_opencode_content(self, stdout: str) -> str:
        if not stdout.strip():
            return ""
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        fragments: list[str] = []
        parsed_any = False
        for line in lines:
            try:
                payload = json.loads(line)
                parsed_any = True
            except json.JSONDecodeError:
                continue
            self._collect_text_fragments(payload, fragments)
        if fragments:
            return "\n".join(fragment for fragment in fragments if fragment.strip()).strip()
        return stdout.strip() if not parsed_any else "\n".join(lines).strip()

    def _collect_text_fragments(self, payload: Any, fragments: list[str]) -> None:
        if isinstance(payload, str):
            fragments.append(payload)
            return
        if isinstance(payload, list):
            for item in payload:
                self._collect_text_fragments(item, fragments)
            return
        if not isinstance(payload, dict):
            return
        for key in ("content", "text", "output"):
            value = payload.get(key)
            if isinstance(value, str):
                fragments.append(value)
        for key in ("message", "delta", "data", "result", "final", "part"):
            if key in payload:
                self._collect_text_fragments(payload[key], fragments)

    def _classify_subprocess_error(self, provider: str, content: str) -> str:
        lowered = content.lower()
        if "429" in lowered or "rate limit" in lowered or "too many requests" in lowered:
            return f"{provider}_rate_limited" if provider else "provider_rate_limited"
        if "quota" in lowered or "per day" in lowered:
            return f"{provider}_daily_quota_exhausted" if provider else "provider_quota_exhausted"
        if "permission" in lowered or "not allowed" in lowered or "model not found" in lowered or "unauthorized" in lowered:
            return "model_access_denied"
        if "api key" in lowered or "credentials" in lowered or "authentication" in lowered:
            return "credentials_missing"
        return "opencode_run_failed"

    def _is_retryable_subprocess_error(self, content: str) -> bool:
        lowered = content.lower()
        return not any(token in lowered for token in ("permission", "not allowed", "model not found", "unauthorized"))

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

    def _validate_execution_payload(self, payload: dict[str, Any], category: str) -> str:
        if category != "sisyphus":
            return "schema_optional"
        has_actions = any(
            isinstance(payload.get(key), list) and bool(payload.get(key))
            for key in ("artifact_plan", "command_plan", "experiment_plan")
        )
        if not has_actions and self._payload_has_meaningful_plan(payload):
            return "relaxed_plan_only"
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
            self._validate_command_entry(command_entry)
            command = self._command_tokens(command_entry)
            referenced_file = self._workspace_file_reference(command)
            if referenced_file and referenced_file not in artifact_paths:
                raise ProviderRuntimeError(
                    f"command_plan references workspace file '{referenced_file}' without creating it in artifact_plan",
                    error_class="command_plan_missing_artifact",
                    retryable=True,
                )
        return "actionable"

    def _validate_command_entry(self, entry: Any) -> None:
        if isinstance(entry, str) and entry.strip():
            if self._requires_shell_wrapper(entry) and not self._is_shell_wrapped_string(entry):
                raise ProviderRuntimeError(
                    "command_plan contains shell syntax and must be wrapped explicitly with bash -lc or zsh -lc",
                    error_class="command_plan_requires_shell_wrapper",
                    retryable=True,
                )
            return
        if not isinstance(entry, dict):
            return
        command_value = entry.get("command")
        if isinstance(command_value, str) and command_value.strip():
            if self._requires_shell_wrapper(command_value) and not self._is_shell_wrapped_string(command_value):
                raise ProviderRuntimeError(
                    "structured command_plan contains shell syntax and must be wrapped explicitly with bash -lc or zsh -lc",
                    error_class="command_plan_requires_shell_wrapper",
                    retryable=True,
                )
        elif isinstance(command_value, list) and all(isinstance(part, str) for part in command_value):
            if self._requires_shell_wrapper(" ".join(command_value)) and not self._is_shell_wrapped_list(command_value):
                raise ProviderRuntimeError(
                    "structured command_plan contains shell syntax and must be wrapped explicitly with bash -lc or zsh -lc",
                    error_class="command_plan_requires_shell_wrapper",
                    retryable=True,
                )

    def _requires_shell_wrapper(self, command: str) -> bool:
        return any(token in command for token in ("&&", "||", "|", ";", ">", "<", "$(", "`"))

    def _is_shell_wrapped_string(self, command: str) -> bool:
        lowered = command.strip().lower()
        return lowered.startswith("bash -lc ") or lowered.startswith("zsh -lc ") or lowered.startswith("/bin/bash -lc ") or lowered.startswith("/bin/zsh -lc ")

    def _is_shell_wrapped_list(self, command: list[str]) -> bool:
        if len(command) < 3:
            return False
        executable = command[0].lower()
        return executable in {"bash", "zsh", "/bin/bash", "/bin/zsh"} and command[1] == "-lc"

    def _payload_has_meaningful_plan(self, payload: dict[str, Any]) -> bool:
        summary = str(payload.get("summary", "")).strip()
        extra_keys = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "summary",
                "artifact_plan",
                "command_plan",
                "experiment_plan",
                "citations",
                "next_action",
                "provider",
                "model",
                "primary_model",
                "attempted_models",
                "fallback_used",
                "raw_response",
                "retryable",
                "error_class",
                "validation_mode",
            }
        }
        has_extra_structure = any(
            (isinstance(value, str) and len(value.strip()) > 40)
            or (isinstance(value, list) and bool(value))
            or (isinstance(value, dict) and bool(value))
            for value in extra_keys.values()
        )
        summary_looks_actionable = (
            len(summary) >= 80
            or any(token in summary.lower() for token in ("step", "plan", "write", "create", "run", "generate", "artifact"))
        )
        return bool(summary and (summary_looks_actionable or has_extra_structure))

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
        start = content.find("{")
        while start != -1:
            candidate = self._balanced_json_slice(content, start)
            if candidate is not None:
                return candidate
            start = content.find("{", start + 1)
        return None

    def _balanced_json_slice(self, content: str, start: int) -> Optional[str]:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(content)):
            char = content[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[start : index + 1]
                    try:
                        json.loads(candidate)
                    except Exception:
                        return None
                    return candidate
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

    def _strip_ansi(self, content: str) -> str:
        return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", content)

    def _resolve_path(self, value: Any) -> Optional[Path]:
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.config_path.parent / candidate).resolve()
        return candidate

    def _default_agent_for_category(self, category: str) -> str:
        mapping = {
            "genesis-ideation": "explore",
            "genesis-oracle": "oracle",
            "genesis-adversarial": "oracle",
            "genesis-paper": "document-writer",
            "genesis-citations": "librarian",
            "genesis-plotting": "explore",
        }
        return mapping.get(category, "Sisyphus")
