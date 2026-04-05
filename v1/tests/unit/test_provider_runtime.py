import json

import pytest

from genesis.agents.runtime import CodingAgentRuntime, ProviderRuntimeError


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        if url.endswith("/api/chat"):
            return _Response(
                {
                    "message": {
                        "content": json_module.dumps(
                            {
                                "summary": "ok",
                                "artifact_plan": [],
                                "experiment_plan": [],
                                "citations": [],
                                "next_action": "continue",
                            }
                        )
                    }
                }
            )
        raise AssertionError(url)


import json as json_module


def test_provider_runtime_parses_ollama_response(tmp_path):
    config_path = tmp_path / "runtime.jsonc"
    config_path.write_text(
        json_module.dumps(
            {
                "categories": {
                    "sisyphus": {
                        "provider": "ollama",
                        "model": "test-model",
                        "fallbacks": [],
                        "temperature": 0.2,
                        "max_tokens": 100,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = CodingAgentRuntime(config_path, session=_Session())
    payload = runtime.generate_task(category="sisyphus", instruction="do work", context={"task": "x"})
    assert payload["summary"] == "ok"


def test_provider_runtime_strips_jsonc_comments_without_breaking_urls(tmp_path):
    config_path = tmp_path / "runtime.jsonc"
    config_path.write_text(
        """
        {
          // runtime categories
          "categories": {
            "sisyphus": {
              "provider": "ollama",
              "model": "test-model",
              "fallbacks": [],
              "temperature": 0.2,
              "max_tokens": 100,
              "endpoint": "http://127.0.0.1:11434"
            }
          }
        }
        """,
        encoding="utf-8",
    )
    runtime = CodingAgentRuntime(config_path, session=_Session())
    assert runtime.categories["sisyphus"].model == "test-model"


def test_provider_runtime_normalizes_missing_response_fields(tmp_path):
    class _MinimalSession:
        def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
            return _Response({"message": {"content": '{"summary":"ok"}'}})

    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json_module.dumps(
            {
                "categories": {
                    "sisyphus": {
                        "provider": "ollama",
                        "model": "test-model",
                        "fallbacks": [],
                        "temperature": 0.2,
                        "max_tokens": 100,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = CodingAgentRuntime(config_path, session=_MinimalSession())
    payload = runtime.generate_task(category="sisyphus", instruction="do work", context={"task": "x"})
    assert payload["artifact_plan"] == []
    assert payload["command_plan"] == []
    assert payload["experiment_plan"] == []
    assert payload["citations"] == []
    assert payload["next_action"] == "continue"


def test_provider_runtime_reports_unknown_category(tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text('{"categories": {}}', encoding="utf-8")
    runtime = CodingAgentRuntime(config_path, session=_Session())
    with pytest.raises(ProviderRuntimeError) as excinfo:
        runtime.generate_task(category="missing", instruction="do work", context={"task": "x"})
    assert excinfo.value.error_class == "unknown_category"


def test_provider_runtime_accepts_legacy_providers_key(tmp_path):
    config_path = tmp_path / "runtime.jsonc"
    config_path.write_text(
        json_module.dumps(
            {
                "providers": {
                    "sisyphus": {
                        "provider": "ollama",
                        "model": "gemma4",
                        "fallbacks": ["llama3.2:3b"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = CodingAgentRuntime(config_path, session=_Session())
    assert runtime.categories["sisyphus"].model == "gemma4"
    assert runtime.categories["sisyphus"].temperature == 0.2
    assert runtime.categories["sisyphus"].max_tokens == 1024


def test_live_runtime_config_uses_gemma4():
    config_path = "/Users/nikhilmaturi/Files/Projects/genesis/v1/.opencode/oh-my-openagent.jsonc"
    runtime = CodingAgentRuntime(config_path, session=_Session())
    assert runtime.categories["sisyphus"].model == "gemma4"
