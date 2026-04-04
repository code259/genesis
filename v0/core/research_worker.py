from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import config  # type: ignore
from core import cache_manager  # type: ignore
from core import runtime_policy  # type: ignore
from core import router  # type: ignore
from core.artifact_runner import run_python_task
from core.task_parser import extract_stage


PLANNER_PROMPT = Path("prompts/worker_planner_system.md").read_text()
OBSERVATION_PROMPT = Path("prompts/worker_observation_system.md").read_text()
COMPLETION_PROMPT = Path("prompts/worker_completion_system.md").read_text()


@dataclass
class TaskRunResult:
    success: bool
    task_status: str
    output_text: str
    artifact_paths: list[str]
    completion_evidence: list[str]
    blocked_reason: str
    worker_state_path: str


def run_task(task_spec: dict, project_path: Path, revision_context: str = "") -> TaskRunResult:
    project_path = project_path.resolve()
    worker_dir = task_worker_dir(project_path, task_spec["id"])
    worker_dir.mkdir(parents=True, exist_ok=True)
    state = load_worker_state(project_path, task_spec["id"])

    if state["attempt_count"] >= 3 and not state.get("ready_for_verification"):
        state["task_status"] = "ESCALATED"
        state["last_reason"] = "Exceeded maximum task attempts"
        save_worker_state(project_path, task_spec["id"], state)
        return _result_from_state(project_path, task_spec["id"], state)

    state["attempt_count"] += 1
    state["task_status"] = "IN_PROGRESS"
    if revision_context:
        state["revision_context"] = revision_context
    save_worker_state(project_path, task_spec["id"], state)

    for step_index in range(config.WORKER_MAX_STEPS):
        action = _planner_action(task_spec, project_path, state)
        state["last_action"] = action
        observation = execute_tool_action(task_spec, project_path, action)
        summary = _synthesize_observation(task_spec, state, action, observation)
        _append_step(project_path, task_spec["id"], {"action": action, "observation": observation, "summary": summary})

        completion = _completion_check(task_spec, state, summary, observation)
        _record_issues(state, completion, action, observation, step_index)
        state["last_reason"] = completion.get("reason", summary)
        state["artifact_paths"] = sorted(set(state.get("artifact_paths", []) + observation.get("artifact_paths", [])))
        state["completion_evidence"] = sorted(
            set(state.get("completion_evidence", []) + observation.get("evidence", []) + completion.get("completion_evidence", []))
        )

        if action["action"] == "block" or completion.get("status") == "BLOCKED":
            state["task_status"] = "BLOCKED"
            state["blocking_reason"] = completion.get("reason", action.get("reason", "Worker blocked"))
            save_worker_state(project_path, task_spec["id"], state)
            return _result_from_state(project_path, task_spec["id"], state)

        if completion.get("issue_class") == "deferrable_issue" and not completion.get("complete"):
            state["task_status"] = "PARTIAL"
            state["completion_mode"] = "partial"
            state["ready_for_verification"] = False
            state["blocking_reason"] = completion.get("reason", "Deferred non-critical issue")
            state["final_output"] = _final_output(task_spec, project_path, state, summary, partial=True)
            save_worker_state(project_path, task_spec["id"], state)
            return _result_from_state(project_path, task_spec["id"], state)

        if action["action"] == "complete" or completion.get("complete"):
            state["ready_for_verification"] = True
            state["task_status"] = "IN_PROGRESS"
            state["completion_mode"] = "complete"
            state["final_output"] = _final_output(
                task_spec,
                project_path,
                state,
                completion.get("summary_markdown") or action.get("summary_markdown") or summary,
                partial=False,
            )
            save_worker_state(project_path, task_spec["id"], state)
            return _result_from_state(project_path, task_spec["id"], state)

    state["task_status"] = "BLOCKED"
    state["last_reason"] = "Worker exhausted step budget before completion"
    state["blocking_reason"] = state["last_reason"]
    save_worker_state(project_path, task_spec["id"], state)
    return _result_from_state(project_path, task_spec["id"], state)


def load_worker_state(project_path: Path, task_id: str) -> dict:
    project_path = project_path.resolve()
    path = worker_state_path(project_path, task_id)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "task_status": "PENDING",
        "attempt_count": 0,
        "artifact_paths": [],
        "completion_evidence": [],
        "issue_log": [],
        "deferred_issues": [],
        "completion_mode": "pending",
        "blocking_reason": "",
        "last_action": None,
        "last_reason": "",
        "ready_for_verification": False,
    }


def save_worker_state(project_path: Path, task_id: str, state: dict):
    project_path = project_path.resolve()
    path = worker_state_path(project_path, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def mark_task_verified(project_path: Path, task_id: str, verifier_status: str):
    project_path = project_path.resolve()
    state = load_worker_state(project_path, task_id)
    if verifier_status == "ACCEPT":
        state["task_status"] = "VERIFIED"
    elif verifier_status == "ESCALATE":
        state["task_status"] = "ESCALATED"
    save_worker_state(project_path, task_id, state)


def task_worker_dir(project_path: Path, task_id: str) -> Path:
    project_path = project_path.resolve()
    stage = extract_stage(task_id)
    return project_path / "stages" / f"stage_{stage}" / f"{task_id}_worker"


def worker_state_path(project_path: Path, task_id: str) -> Path:
    project_path = project_path.resolve()
    return task_worker_dir(project_path, task_id) / "worker_state.json"


def execute_tool_action(task_spec: dict, project_path: Path, action: dict) -> dict:
    project_path = project_path.resolve()
    kind = action.get("action")
    if kind == "read_local_file":
        policy = runtime_policy.validate_read_path(project_path, action["path"])
        if not policy.allowed:
            return {
                "text": policy.reason,
                "artifact_paths": [],
                "evidence": [],
                "error": True,
                "missing_path": action["path"],
                "issue_class": "deferrable_issue",
                "blocks_task": False,
                "blocks_dependents": False,
            }
        path = policy.normalized_path
        if path is None or not path.exists():
            return {
                "text": f"Local file not found: {action['path']}",
                "artifact_paths": [],
                "evidence": [],
                "error": True,
                "missing_path": action["path"],
                "issue_class": "deferrable_issue",
                "blocks_task": False,
                "blocks_dependents": False,
            }
        return {"text": path.read_text(), "artifact_paths": [], "evidence": [str(path.relative_to(project_path))]}
    if kind in {"fetch_url", "http_request"}:
        return _http_action(action, task_spec, project_path)
    if kind == "run_python":
        result = run_python_task(task_spec, project_path, action)
        missing_artifact_issue = bool(result.missing_artifacts and result.stdout and not result.stderr.strip())
        return {
            "text": result.stdout + ("\n" + result.stderr if result.stderr else ""),
            "artifact_paths": result.artifact_paths,
            "evidence": [result.results_path] + result.artifact_paths,
            "success": result.success,
            "missing_artifacts": result.missing_artifacts,
            "error": not result.success,
            "issue_class": "deferrable_issue" if missing_artifact_issue else None,
            "blocks_task": False if missing_artifact_issue else not result.success,
            "blocks_dependents": False if missing_artifact_issue else not result.success,
        }
    if kind == "complete":
        return {"text": action.get("summary_markdown", ""), "artifact_paths": action.get("artifact_paths", []), "evidence": action.get("completion_evidence", [])}
    if kind == "block":
        return {"text": action.get("reason", "Worker blocked"), "artifact_paths": [], "evidence": []}
    raise ValueError(f"Unknown worker action: {kind}")


def _planner_action(task_spec: dict, project_path: Path, state: dict) -> dict:
    payload = {
        "task_spec": task_spec,
        "project_context": _project_context(project_path),
        "worker_state": state,
        "step_log_tail": _step_log_tail(project_path, task_spec["id"]),
    }
    cached = cache_manager.cache_lookup(
        "worker_planner",
        payload,
        version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], task_spec["id"]],
    )
    if cached is not None:
        return cached["value"]
    user = json.dumps(payload, indent=2)
    raw = router.call("executor", PLANNER_PROMPT, user, max_tokens=1500)
    parsed = _parse_json_block(raw)
    cache_manager.cache_store("worker_planner", payload, {"value": parsed}, version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], task_spec["id"]])
    return parsed


def _synthesize_observation(task_spec: dict, state: dict, action: dict, observation: dict) -> str:
    payload = {"task_spec": task_spec, "worker_state": state, "action": action, "observation": observation}
    cached = cache_manager.cache_lookup(
        "worker_observation",
        payload,
        version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], task_spec["id"]],
    )
    if cached is not None:
        return cached["value"]
    user = json.dumps(payload, indent=2)
    raw = router.call("executor", OBSERVATION_PROMPT, user, max_tokens=500)
    cache_manager.cache_store("worker_observation", payload, {"value": raw.strip()}, version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], task_spec["id"]])
    return raw.strip()


def _completion_check(task_spec: dict, state: dict, summary: str, observation: dict) -> dict:
    payload = {"task_spec": task_spec, "worker_state": state, "summary": summary, "observation": observation}
    cached = cache_manager.cache_lookup(
        "worker_completion",
        payload,
        version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], task_spec["id"]],
    )
    if cached is not None:
        return cached["value"]
    user = json.dumps(payload, indent=2)
    raw = router.call("executor", COMPLETION_PROMPT, user, max_tokens=800)
    parsed = _parse_json_block(raw)
    parsed.setdefault("issue_class", "recoverable_retry")
    parsed.setdefault("blocks_task", parsed.get("status") == "BLOCKED")
    parsed.setdefault("blocks_dependents", parsed.get("status") == "BLOCKED")
    cache_manager.cache_store("worker_completion", payload, {"value": parsed}, version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], task_spec["id"]])
    return parsed


def _append_step(project_path: Path, task_id: str, entry: dict):
    project_path = project_path.resolve()
    path = task_worker_dir(project_path, task_id) / "worker_steps.jsonl"
    with open(path, "a") as handle:
        handle.write(json.dumps(entry) + "\n")


def _step_log_tail(project_path: Path, task_id: str, limit: int = 5) -> list[dict]:
    project_path = project_path.resolve()
    path = task_worker_dir(project_path, task_id) / "worker_steps.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-limit:]
    return [json.loads(line) for line in lines]


def _result_from_state(project_path: Path, task_id: str, state: dict) -> TaskRunResult:
    project_path = project_path.resolve()
    return TaskRunResult(
        success=bool(state.get("ready_for_verification")),
        task_status=state["task_status"],
        output_text=state.get("final_output") or _default_output({"id": task_id, "description": ""}, state),
        artifact_paths=state.get("artifact_paths", []),
        completion_evidence=state.get("completion_evidence", []),
        blocked_reason=state.get("last_reason", ""),
        worker_state_path=str(worker_state_path(project_path, task_id).relative_to(project_path)),
    )


def _default_output(task_spec: dict, state: dict) -> str:
    artifacts = "\n".join(f"- `{path}`" for path in state.get("artifact_paths", [])) or "- None"
    evidence = "\n".join(f"- `{item}`" for item in state.get("completion_evidence", [])) or "- None"
    return f"""## Task ID: {task_spec.get('id')}
### Description
{task_spec.get('description', '')}

### Worker Summary
{state.get('last_reason', 'No summary available.')}

### Completion Evidence
{evidence}

### Produced Artifacts
{artifacts}

CHECKS PERFORMED:
- Ran iterative worker steps and recorded observations.

CHECKS NOT PERFORMED:
- Final scientific verification is deferred to the verifier stage.
"""


def _final_output(task_spec: dict, project_path: Path, state: dict, summary_markdown: str, partial: bool) -> str:
    evidence_lines = "\n".join(f"- `{item}`" for item in state.get("completion_evidence", [])) or "- None"
    artifact_lines = "\n".join(f"- `{path}`" for path in state.get("artifact_paths", [])) or "- None"
    recent_issues = "\n".join(f"- {item['reason']}" for item in state.get("issue_log", [])[-3:]) or "- None"
    snippets = _evidence_snippets(project_path, state.get("artifact_paths", []))
    status = "PARTIAL" if partial else "COMPLETE"
    return f"""## Task ID: {task_spec.get('id')}
### Description
{task_spec.get('description', '')}

### Worker Status
{status}

### Summary
{summary_markdown}

### Completion Evidence
{evidence_lines}

### Produced Artifacts
{artifact_lines}

### Evidence Snapshot
{snippets}

### Recent Issues
{recent_issues}

CHECKS PERFORMED:
- Ran iterative worker steps and recorded observations.
- Referenced concrete project evidence and artifact paths.

CHECKS NOT PERFORMED:
- Final scientific verification is deferred to the verifier stage.
"""


def _project_context(project_path: Path) -> dict:
    project_path = project_path.resolve()
    context = {}
    for filename in ["project_config.json", "global_state.md", "conventions.md", "project_spec.md"]:
        path = project_path / filename
        if path.exists():
            context[filename] = path.read_text()
    return context


def _http_action(action: dict, task_spec: dict, project_path: Path) -> dict:
    payload = {"action": action, "task_id": task_spec["id"]}
    cached = cache_manager.cache_lookup(
        "worker_http",
        payload,
        version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], str(project_path)],
    )
    if cached is not None:
        return cached["value"]
    method = action.get("method", "GET").upper()
    url = action["url"]
    params = action.get("params")
    headers = action.get("headers", {})
    body = action.get("body")
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urlencode(params)}"

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", **headers}

    request = Request(url, method=method, data=data, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8", errors="replace")
            result = {
                "status_code": response.status,
                "text": text[:12000],
                "artifact_paths": [],
                "evidence": [url],
            }
            cache_manager.cache_store("worker_http", payload, {"value": result}, version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], str(project_path)])
            return result
    except HTTPError as exc:
        result = {
            "status_code": exc.code,
            "text": exc.read().decode("utf-8", errors="replace")[:12000],
            "artifact_paths": [],
            "evidence": [url],
            "error": True,
        }
        cache_manager.cache_store("worker_http", payload, {"value": result}, version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], str(project_path)])
        return result
    except URLError as exc:
        result = {
            "status_code": None,
            "text": str(exc),
            "artifact_paths": [],
            "evidence": [url],
            "error": True,
        }
        cache_manager.cache_store("worker_http", payload, {"value": result}, version_parts=[router._resolve_spec(config.MODEL_TIER, "executor")["model"], str(project_path)])
        return result


def _record_issues(state: dict, completion: dict, action: dict, observation: dict, step_index: int):
    issue_class = completion.get("issue_class") or observation.get("issue_class")
    if not issue_class and observation.get("error"):
        issue_class = "recoverable_retry"
    if not issue_class:
        return
    entry = {
        "step": step_index + 1,
        "issue_class": issue_class,
        "reason": completion.get("reason") or observation.get("text", "")[:200],
        "blocks_task": bool(completion.get("blocks_task")),
        "blocks_dependents": bool(completion.get("blocks_dependents")),
        "action": action.get("action"),
    }
    state.setdefault("issue_log", []).append(entry)
    if issue_class == "deferrable_issue":
        state.setdefault("deferred_issues", []).append(entry)


def _evidence_snippets(project_path: Path, artifact_paths: list[str], limit: int = 3) -> str:
    snippets = []
    for path in artifact_paths[:limit]:
        absolute = (project_path / path).resolve()
        if not absolute.exists():
            continue
        if absolute.suffix.lower() in {".md", ".txt", ".json", ".csv"}:
            snippets.append(f"#### {absolute.name}\n```text\n{absolute.read_text()[:300]}\n```")
    return "\n\n".join(snippets) or "No inline snippets available."


def _safe_project_path(project_path: Path, relative_path: str) -> Path:
    policy = runtime_policy.validate_read_path(project_path, relative_path)
    if not policy.allowed or policy.normalized_path is None:
        raise ValueError(policy.reason)
    return policy.normalized_path


def _parse_json_block(raw: str) -> dict:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _extract_json_object(stripped)
        return json.loads(candidate, strict=False)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]
