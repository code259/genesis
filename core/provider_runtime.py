from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import config  # type: ignore


GROQ_ROLE_MODEL_FALLBACKS = {
    "supervisor": ["openai/gpt-oss-20b", "qwen/qwen3-32b"],
    "executor": ["openai/gpt-oss-120b", "qwen/qwen3-32b"],
    "verifier": ["openai/gpt-oss-120b", "qwen/qwen3-32b"],
    "cross_check": ["openai/gpt-oss-120b", "qwen/qwen3-32b"],
    "decomposer": ["openai/gpt-oss-120b", "qwen/qwen3-32b"],
    "decomposition_reviewer": ["openai/gpt-oss-120b", "qwen/qwen3-32b"],
}


def runtime_dir() -> Path:
    configured = os.getenv("GENESIS_RUNTIME_DIR")
    path = Path(configured) if configured else Path(".genesis_runtime")
    path.mkdir(parents=True, exist_ok=True)
    return path


def rate_limit_state_path() -> Path:
    return runtime_dir() / "rate_limit_state.json"


def request_log_path() -> Path:
    return runtime_dir() / "request_log.jsonl"


def read_request_log(limit: int | None = None) -> list[dict]:
    path = request_log_path()
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if limit is not None:
        lines = lines[-limit:]
    return [json.loads(line) for line in lines if line.strip()]


def load_rate_limit_state() -> dict:
    path = rate_limit_state_path()
    if not path.exists():
        return {"keys": {}, "last_error": None}
    return json.loads(path.read_text())


def save_rate_limit_state(state: dict):
    rate_limit_state_path().write_text(json.dumps(state, indent=2))


def groq_key_entries() -> list[dict]:
    keys = config.GROQ_KEYS if getattr(config, "GROQ_KEYS", None) else ([config.GROQ_KEY] if config.GROQ_KEY else [])
    return [{"alias": f"groq_key_{idx+1}", "key": key} for idx, key in enumerate(keys) if key]


def fallback_models(role: str, primary_model: str) -> list[str]:
    return [primary_model] + [model for model in GROQ_ROLE_MODEL_FALLBACKS.get(role, []) if model != primary_model]


def eligible_groq_targets(role: str, primary_model: str, max_attempts: int = 3) -> list[dict]:
    state = load_rate_limit_state()
    now = time.time()
    candidates = []
    for model in fallback_models(role, primary_model):
        for entry in groq_key_entries():
            key_state = state["keys"].get(entry["alias"], {})
            key_blocked_until = key_state.get("blocked_until", 0)
            if key_blocked_until and key_blocked_until > now:
                continue
            model_state = key_state.get(model, {})
            blocked_until = model_state.get("blocked_until", 0)
            if blocked_until and blocked_until > now:
                continue
            candidates.append({"alias": entry["alias"], "key": entry["key"], "model": model})
            if len(candidates) >= max_attempts:
                return candidates
    return candidates


def classify_provider_error(error: Exception) -> str:
    text = str(error).lower()
    if "requests per day" in text or "tokens per day" in text or "rpd" in text or "tpd" in text:
        return "rate_limit_day"
    if "requests per minute" in text or "tokens per minute" in text or "rpm" in text or "tpm" in text:
        return "rate_limit_minute"
    if "rate limit" in text or "429" in text:
        return "rate_limit_minute"
    if "timeout" in text or "connection error" in text or "temporar" in text:
        return "transient"
    return "fatal"


def mark_groq_result(alias: str, model: str, error_kind: str | None = None):
    state = load_rate_limit_state()
    state.setdefault("keys", {})
    state["keys"].setdefault(alias, {})
    key_state = state["keys"][alias]
    model_state = key_state.setdefault(model, {})
    model_state["last_used_at"] = _timestamp()
    if error_kind is None:
        model_state["last_success_at"] = _timestamp()
        model_state["blocked_until"] = 0
        save_rate_limit_state(state)
        return

    state["last_error"] = {"alias": alias, "model": model, "kind": error_kind, "at": _timestamp()}
    if error_kind == "rate_limit_minute":
        model_state["blocked_until"] = time.time() + 60
    elif error_kind == "rate_limit_day":
        blocked_until = time.time() + 24 * 60 * 60
        model_state["blocked_until"] = blocked_until
        key_state["blocked_until"] = blocked_until
    save_rate_limit_state(state)


def log_request(metadata: dict):
    record = dict(metadata)
    record["timestamp"] = _timestamp()
    with open(request_log_path(), "a") as handle:
        handle.write(json.dumps(record) + "\n")


def runtime_summary() -> dict:
    state = load_rate_limit_state()
    logs = read_request_log()
    usage_by_role: dict[str, int] = {}
    usage_by_provider: dict[str, int] = {}
    key_usage: dict[str, int] = {}
    token_estimate = 0
    for entry in logs:
        usage_by_role[entry.get("role", "unknown")] = usage_by_role.get(entry.get("role", "unknown"), 0) + 1
        usage_by_provider[entry.get("provider", "unknown")] = usage_by_provider.get(entry.get("provider", "unknown"), 0) + 1
        if entry.get("key_alias"):
            key_usage[entry["key_alias"]] = key_usage.get(entry["key_alias"], 0) + 1
        token_estimate += int(entry.get("max_tokens") or 0)
    return {
        "configured_groq_keys": len(groq_key_entries()),
        "last_error": state.get("last_error"),
        "tracked_models": sum(len(models) for models in state.get("keys", {}).values()),
        "request_count": len(logs),
        "estimated_max_tokens": token_estimate,
        "usage_by_role": usage_by_role,
        "usage_by_provider": usage_by_provider,
        "usage_by_key": key_usage,
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
