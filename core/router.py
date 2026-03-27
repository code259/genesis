import os
import anthropic  # type: ignore
import openai  # type: ignore
import config  # type: ignore

# ─────────────────────────────────────────────
# TIER DEFINITIONS — edit model strings here only
# ─────────────────────────────────────────────

TIERS = {
    0: {
        # Ollama local — OpenAI-compatible endpoint, no key needed
        "supervisor":   {"provider": "ollama", "model": "llama3.1:8b"},
        "executor":     {"provider": "ollama", "model": "llama3.1:8b"},
        "verifier":     {"provider": "ollama", "model": "llama3.1:8b"},
        "cross_check":  {"provider": "ollama", "model": "mistral:7b"},
        "decomposer":   {"provider": "ollama", "model": "llama3.1:8b"},
    },
    1: {
        # Cheap cloud — Together AI (OpenAI-compatible) + Google AI
        "supervisor":   {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        "executor":     {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        "verifier":     {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        "cross_check":  {"provider": "together", "model": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
        "decomposer":   {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    },
    2: {
        # Production — Anthropic + OpenAI
        "supervisor":   {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "executor":     {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "verifier":     {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "cross_check":  {"provider": "openai",    "model": "gpt-4o"},
        "decomposer":   {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    },
}

# ─────────────────────────────────────────────
# PROVIDER CLIENTS
# ─────────────────────────────────────────────

def _get_client(provider: str):
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    elif provider == "openai":
        return openai.OpenAI(api_key=config.OPENAI_KEY)
    elif provider == "together":
        return openai.OpenAI(
            api_key=config.TOGETHER_KEY,
            base_url="https://api.together.xyz/v1"
        )
    elif provider == "ollama":
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

    role: one of 'supervisor', 'executor', 'verifier', 'cross_check', 'decomposer'
    system: system prompt string
    user: user message string
    max_tokens: override default if needed
    tier: override config.MODEL_TIER if needed (useful for tests)
    
    Returns: response text as string.
    Raises: RuntimeError with context on API failure.
    """
    active_tier = tier if tier is not None else config.MODEL_TIER
    spec = TIERS[active_tier][role]
    provider = spec["provider"]
    model = spec["model"]

    # Default max_tokens by role
    if max_tokens is None:
        max_tokens = {
            "supervisor":  config.MAX_TOKENS_SUPERVISOR,
            "executor":    config.MAX_TOKENS_EXECUTOR,
            "verifier":    config.MAX_TOKENS_VERIFIER,
            "cross_check": config.MAX_TOKENS_VERIFIER,
            "decomposer":  config.MAX_TOKENS_EXECUTOR,
        }.get(role, 2000)

    try:
        if provider == "anthropic":
            client = _get_client("anthropic")
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text

        else:
            # OpenAI-compatible: Together, Ollama, OpenAI
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

    except Exception as e:
        raise RuntimeError(
            f"router.call failed: role={role}, provider={provider}, "
            f"model={model}, tier={active_tier}\nOriginal error: {e}"
        ) from e


def current_tier_summary() -> str:
    """Print active model assignments. Call at project init for audit trail."""
    tier = config.MODEL_TIER
    lines = [f"Active tier: {tier}"]
    for role, spec in TIERS[tier].items():
        lines.append(f"  {role:12s} → {spec['provider']:10s} / {spec['model']}")
    return "\n".join(lines)
