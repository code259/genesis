import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from core import provider_runtime


def test_groq_targets_skip_blocked_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("GENESIS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(config, "GROQ_KEYS", ["key-a", "key-b"])
    provider_runtime.mark_groq_result("groq_key_1", "llama-3.3-70b-versatile", "rate_limit_day")
    targets = provider_runtime.eligible_groq_targets("executor", "llama-3.3-70b-versatile", max_attempts=3)
    assert all(target["alias"] != "groq_key_1" for target in targets)
    assert any(target["alias"] == "groq_key_2" for target in targets)


def test_classify_provider_error():
    assert provider_runtime.classify_provider_error(RuntimeError("Requests per minute exceeded")) == "rate_limit_minute"
    assert provider_runtime.classify_provider_error(RuntimeError("tokens per day exhausted")) == "rate_limit_day"
    assert provider_runtime.classify_provider_error(RuntimeError("Connection error.")) == "transient"
