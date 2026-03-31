from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import config  # type: ignore
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

    for _ in range(config.WORKER_MAX_STEPS):
        action = _planner_action(task_spec, project_path, state)
        observation = execute_tool_action(task_spec, project_path, action)
        summary = _synthesize_observation(task_spec, state, action, observation)
        _append_step(project_path, task_spec["id"], {"action": action, "observation": observation, "summary": summary})

        completion = _completion_check(task_spec, state, summary, observation)
        state["last_reason"] = completion.get("reason", summary)
        state["artifact_paths"] = sorted(set(state.get("artifact_paths", []) + observation.get("artifact_paths", [])))
        state["completion_evidence"] = sorted(
            set(state.get("completion_evidence", []) + observation.get("evidence", []) + completion.get("completion_evidence", []))
        )

        if action["action"] == "block" or completion.get("status") == "BLOCKED":
            state["task_status"] = "BLOCKED"
            save_worker_state(project_path, task_spec["id"], state)
            return _result_from_state(project_path, task_spec["id"], state)

        if action["action"] == "complete" or completion.get("complete"):
            state["ready_for_verification"] = True
            state["task_status"] = "IN_PROGRESS"
            state["final_output"] = completion.get("summary_markdown") or action.get("summary_markdown") or _default_output(task_spec, state)
            save_worker_state(project_path, task_spec["id"], state)
            return _result_from_state(project_path, task_spec["id"], state)

    state["task_status"] = "BLOCKED"
    state["last_reason"] = "Worker exhausted step budget before completion"
    save_worker_state(project_path, task_spec["id"], state)
    return _result_from_state(project_path, task_spec["id"], state)


def load_worker_state(project_path: Path, task_id: str) -> dict:
    path = worker_state_path(project_path, task_id)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "task_status": "PENDING",
        "attempt_count": 0,
        "artifact_paths": [],
        "completion_evidence": [],
        "last_reason": "",
        "ready_for_verification": False,
    }


def save_worker_state(project_path: Path, task_id: str, state: dict):
    path = worker_state_path(project_path, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def mark_task_verified(project_path: Path, task_id: str, verifier_status: str):
    state = load_worker_state(project_path, task_id)
    if verifier_status == "ACCEPT":
        state["task_status"] = "VERIFIED"
    elif verifier_status == "ESCALATE":
        state["task_status"] = "ESCALATED"
    save_worker_state(project_path, task_id, state)


def task_worker_dir(project_path: Path, task_id: str) -> Path:
    stage = extract_stage(task_id)
    return project_path / "stages" / f"stage_{stage}" / f"{task_id}_worker"


def worker_state_path(project_path: Path, task_id: str) -> Path:
    return task_worker_dir(project_path, task_id) / "worker_state.json"


def execute_tool_action(task_spec: dict, project_path: Path, action: dict) -> dict:
    kind = action.get("action")
    if kind == "read_local_file":
        path = _safe_project_path(project_path, action["path"])
        return {"text": path.read_text(), "artifact_paths": [], "evidence": [str(path.relative_to(project_path))]}
    if kind in {"fetch_url", "http_request"}:
        return _http_action(action)
    if kind == "run_python":
        result = run_python_task(task_spec, project_path, action)
        return {
            "text": result.stdout + ("\n" + result.stderr if result.stderr else ""),
            "artifact_paths": result.artifact_paths,
            "evidence": [result.results_path] + result.artifact_paths,
            "success": result.success,
            "missing_artifacts": result.missing_artifacts,
        }
    if kind == "complete":
        return {"text": action.get("summary_markdown", ""), "artifact_paths": action.get("artifact_paths", []), "evidence": action.get("completion_evidence", [])}
    if kind == "block":
        return {"text": action.get("reason", "Worker blocked"), "artifact_paths": [], "evidence": []}
    raise ValueError(f"Unknown worker action: {kind}")


def _planner_action(task_spec: dict, project_path: Path, state: dict) -> dict:
    user = json.dumps(
        {
            "task_spec": task_spec,
            "project_context": _project_context(project_path),
            "worker_state": state,
            "step_log_tail": _step_log_tail(project_path, task_spec["id"]),
        },
        indent=2,
    )
    raw = router.call("executor", PLANNER_PROMPT, user, max_tokens=1500)
    return _parse_json_block(raw)


def _synthesize_observation(task_spec: dict, state: dict, action: dict, observation: dict) -> str:
    user = json.dumps({"task_spec": task_spec, "worker_state": state, "action": action, "observation": observation}, indent=2)
    raw = router.call("executor", OBSERVATION_PROMPT, user, max_tokens=500)
    return raw.strip()


def _completion_check(task_spec: dict, state: dict, summary: str, observation: dict) -> dict:
    user = json.dumps({"task_spec": task_spec, "worker_state": state, "summary": summary, "observation": observation}, indent=2)
    raw = router.call("executor", COMPLETION_PROMPT, user, max_tokens=800)
    return _parse_json_block(raw)


def _append_step(project_path: Path, task_id: str, entry: dict):
    path = task_worker_dir(project_path, task_id) / "worker_steps.jsonl"
    with open(path, "a") as handle:
        handle.write(json.dumps(entry) + "\n")


def _step_log_tail(project_path: Path, task_id: str, limit: int = 5) -> list[dict]:
    path = task_worker_dir(project_path, task_id) / "worker_steps.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-limit:]
    return [json.loads(line) for line in lines]


def _result_from_state(project_path: Path, task_id: str, state: dict) -> TaskRunResult:
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


def _project_context(project_path: Path) -> dict:
    context = {}
    for filename in ["project_config.json", "global_state.md", "conventions.md", "project_spec.md"]:
        path = project_path / filename
        if path.exists():
            context[filename] = path.read_text()
    return context


def _http_action(action: dict) -> dict:
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
            return {
                "status_code": response.status,
                "text": text[:12000],
                "artifact_paths": [],
                "evidence": [url],
            }
    except HTTPError as exc:
        return {
            "status_code": exc.code,
            "text": exc.read().decode("utf-8", errors="replace")[:12000],
            "artifact_paths": [],
            "evidence": [url],
            "error": True,
        }
    except URLError as exc:
        return {
            "status_code": None,
            "text": str(exc),
            "artifact_paths": [],
            "evidence": [url],
            "error": True,
        }


def _safe_project_path(project_path: Path, relative_path: str) -> Path:
    candidate = (project_path / relative_path).resolve()
    project_root = project_path.resolve()
    if project_root not in candidate.parents and candidate != project_root:
        raise ValueError(f"Path escapes project root: {relative_path}")
    return candidate


def _parse_json_block(raw: str) -> dict:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(stripped)
