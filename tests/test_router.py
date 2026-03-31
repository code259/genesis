import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config  # pyre-ignore[21]
from core import router  # pyre-ignore[21]


def test_tier_1_uses_groq_for_primary_roles():
    for role in ["supervisor", "executor", "verifier", "decomposer"]:
        assert router.TIERS[1][role]["provider"] == "groq"


def test_tier_1_cross_check_uses_openai():
    assert router.TIERS[1]["cross_check"]["provider"] == "openai"


def test_groq_client_uses_openai_compatible_base_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "GROQ_KEY", "test-groq-key")
    client = router._get_client("groq")
    assert str(client.base_url) == "https://api.groq.com/openai/v1/"


def test_tier_1_cross_check_falls_back_to_groq_without_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "OPENAI_KEY", None)
    monkeypatch.setattr(config, "GROQ_KEY", "test-groq-key")
    spec = router._resolve_spec(1, "cross_check")
    assert spec["provider"] == "groq"
    assert spec["model"] == "deepseek-r1-distill-llama-70b"
