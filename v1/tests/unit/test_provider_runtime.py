import json

from genesis.agents.runtime import CodingAgentRuntime


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
