import config  # type: ignore
from core import provider_runtime  # type: ignore

# ─────────────────────────────────────────────
# TIER DEFINITIONS — edit model strings here only
# ─────────────────────────────────────────────

TIERS = {
    0: {
        # Ollama local — OpenAI-compatible endpoint, no key needed
        "supervisor":            {"provider": "ollama", "model": "llama3.1:8b"},
        "executor":              {"provider": "ollama", "model": "llama3.1:8b"},
        "verifier":              {"provider": "ollama", "model": "llama3.1:8b"},
        "cross_check":           {"provider": "ollama", "model": "mistral:7b"},
        "decomposer":            {"provider": "ollama", "model": "llama3.1:8b"},
        "decomposition_reviewer": {"provider": "ollama", "model": "mistral:7b"},
    },
    1: {
        # Cheap cloud — Groq for primary work, OpenAI for cross-provider checks
        "supervisor":            {"provider": "groq",   "model": "llama-3.1-8b-instant"},
        "executor":              {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
        "verifier":              {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
        "cross_check":           {"provider": "openai", "model": "gpt-4o-mini"},
        "decomposer":            {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
        "decomposition_reviewer": {"provider": "groq", "model": "deepseek-r1-distill-llama-70b"},
    },
    2: {
        # Production — Anthropic + OpenAI
        "supervisor":            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "executor":              {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "verifier":              {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "cross_check":           {"provider": "openai",    "model": "gpt-4o"},
        "decomposer":            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "decomposition_reviewer": {"provider": "openai", "model": "gpt-4o-mini"},
    },
}

FALLBACK_SPEC_OVERRIDES = {
    1: {
        "executor": [{"provider": "groq", "model": "openai/gpt-oss-120b"}],
        "verifier": [{"provider": "groq", "model": "openai/gpt-oss-120b"}],
        "decomposer": [{"provider": "groq", "model": "openai/gpt-oss-120b"}],
        "decomposition_reviewer": [{"provider": "groq", "model": "openai/gpt-oss-120b"}],
        "supervisor": [{"provider": "groq", "model": "openai/gpt-oss-20b"}],
    }
}


def _resolve_spec(active_tier: int, role: str) -> dict[str, str]:
    spec = dict(TIERS[active_tier][role])

    # Tier 1 should still work when only a Groq key is configured.
    if active_tier == 1 and role in {"cross_check", "decomposition_reviewer"} and spec["provider"] == "openai":
        if not config.OPENAI_KEY and config.GROQ_KEY:
            return {"provider": "groq", "model": "deepseek-r1-distill-llama-70b"}

    return spec


def _candidate_specs(active_tier: int, role: str) -> list[dict[str, str]]:
    primary = _resolve_spec(active_tier, role)
    fallbacks = FALLBACK_SPEC_OVERRIDES.get(active_tier, {}).get(role, [])
    candidates = [primary]
    for fallback in fallbacks:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates

# ─────────────────────────────────────────────
# PROVIDER CLIENTS
# ─────────────────────────────────────────────

def _get_client(provider: str):
    if provider == "anthropic":
        import anthropic  # type: ignore
        return anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    elif provider == "openai":
        import openai  # type: ignore
        return openai.OpenAI(api_key=config.OPENAI_KEY)
    elif provider == "groq":
        import openai  # type: ignore
        return openai.OpenAI(
            api_key=config.GROQ_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
    elif provider == "together":
        import openai  # type: ignore
        return openai.OpenAI(
            api_key=config.TOGETHER_KEY,
            base_url="https://api.together.xyz/v1"
        )
    elif provider == "ollama":
        import openai  # type: ignore
        return openai.OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ─────────────────────────────────────────────
# UNIFIED CALL INTERFACE
# ─────────────────────────────────────────────

import typing

def call(
    role: str,
    system: str,
    user: str,
    max_tokens: typing.Optional[int] = None,
    tier: typing.Optional[int] = None
) -> str:
    """
    Single entry point for all model calls.

    role: one of 'supervisor', 'executor', 'verifier', 'cross_check', 'decomposer', 'decomposition_reviewer'
    system: system prompt string
    user: user message string
    max_tokens: override default if needed
    tier: override config.MODEL_TIER if needed (useful for tests)
    
    Returns: response text as string.
    Raises: RuntimeError with context on API failure.
    """
    active_tier = tier if tier is not None else config.MODEL_TIER

    # Default max_tokens by role
    if max_tokens is None:
        max_tokens = {
            "supervisor":  config.MAX_TOKENS_SUPERVISOR,
            "executor":    config.MAX_TOKENS_EXECUTOR,
            "verifier":    config.MAX_TOKENS_VERIFIER,
            "cross_check": config.MAX_TOKENS_VERIFIER,
            "decomposer":  config.MAX_TOKENS_EXECUTOR,
            "decomposition_reviewer": config.MAX_TOKENS_VERIFIER,
        }.get(role, 2000)

    errors = []
    for spec in _candidate_specs(active_tier, role):
        provider = spec["provider"]
        model = spec["model"]
        try:
            if provider == "anthropic":
                response_text = _call_anthropic(model, system, user, max_tokens)
            elif provider == "groq":
                response_text = _call_groq(role, model, system, user, max_tokens)
            else:
                response_text = _call_openai_compatible(provider, model, system, user, max_tokens)
            provider_runtime.log_request(
                {
                    "role": role,
                    "provider": provider,
                    "model": model,
                    "max_tokens": max_tokens,
                    "result": "success",
                }
            )
            return response_text
        except Exception as e:
            provider_runtime.log_request(
                {
                    "role": role,
                    "provider": provider,
                    "model": model,
                    "max_tokens": max_tokens,
                    "result": "error",
                    "error_class": type(e).__name__,
                    "error": str(e),
                }
            )
            errors.append(f"{provider}/{model}: {e}")

    raise RuntimeError(
        f"router.call failed: role={role}, tier={active_tier}\n" + "\n".join(errors)
    )


def _call_anthropic(model: str, system: str, user: str, max_tokens: int) -> str:
    client = _get_client("anthropic")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return response.content[0].text


def _call_openai_compatible(provider: str, model: str, system: str, user: str, max_tokens: int) -> str:
    client = _get_client(provider)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.choices[0].message.content


def _call_groq(role: str, model: str, system: str, user: str, max_tokens: int) -> str:
    import openai  # type: ignore

    errors = []
    for candidate in provider_runtime.eligible_groq_targets(role, model, max_attempts=3):
        try:
            client = openai.OpenAI(
                api_key=candidate["key"],
                base_url="https://api.groq.com/openai/v1"
            )
            response = client.chat.completions.create(
                model=candidate["model"],
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            provider_runtime.mark_groq_result(candidate["alias"], candidate["model"], None)
            provider_runtime.log_request(
                {
                    "role": role,
                    "provider": "groq",
                    "model": candidate["model"],
                    "key_alias": candidate["alias"],
                    "result": "success",
                }
            )
            return response.choices[0].message.content
        except Exception as error:
            kind = provider_runtime.classify_provider_error(error)
            provider_runtime.mark_groq_result(candidate["alias"], candidate["model"], kind)
            provider_runtime.log_request(
                {
                    "role": role,
                    "provider": "groq",
                    "model": candidate["model"],
                    "key_alias": candidate["alias"],
                    "result": "error",
                    "error_kind": kind,
                    "error": str(error),
                }
            )
            errors.append(f"{candidate['alias']}:{candidate['model']} ({kind}) {error}")
            if kind == "fatal":
                break

    raise RuntimeError("Groq request exhausted candidates: " + " | ".join(errors))


def current_tier_summary() -> str:
    """Print active model assignments. Call at project init for audit trail."""
    tier = config.MODEL_TIER
    lines = [f"Active tier: {tier}"]
    for role in TIERS[tier]:
        spec = _resolve_spec(tier, role)
        lines.append(f"  {role:12s} → {spec['provider']:10s} / {spec['model']}")
        if spec["provider"] == "groq":
            fallbacks = provider_runtime.fallback_models(role, spec["model"])[1:]
            if fallbacks:
                lines.append(f"    fallbacks    → {', '.join(fallbacks)}")
    runtime = provider_runtime.runtime_summary()
    lines.append(f"Groq keys configured: {runtime['configured_groq_keys']}")
    return "\n".join(lines)
