from __future__ import annotations

import json
from pathlib import Path

from core.cache_manager import cache_summary
from core.provider_runtime import runtime_summary


def build_project_status(project_path: Path) -> dict:
    project_config = _read_json(project_path / "project_config.json", {})
    tasks = _read_json(project_path / "tasks.json", [])
    run_state = _read_json(project_path / "run_state.json", {})

    task_statuses = []
    counts = {}
    current_stage = None
    for task in sorted(tasks, key=lambda item: (item["stage"], item["id"])):
        status = _task_status(project_path, task)
        task_statuses.append(status)
        counts[status["status"]] = counts.get(status["status"], 0) + 1
        if status["status"] not in {"VERIFIED", "PARTIAL"} and current_stage is None:
            current_stage = task["stage"]

    paper_dir = project_path / "paper"
    paper_summary = {
        "exists": paper_dir.exists(),
        "section_count": len(list((paper_dir / "sections").glob("*.tex"))) if (paper_dir / "sections").exists() else 0,
        "figure_count": len(list((paper_dir / "figures").glob("*.png"))) if (paper_dir / "figures").exists() else 0,
        "claim_registry": str((paper_dir / "claim_registry.json").relative_to(project_path)) if (paper_dir / "claim_registry.json").exists() else None,
    }

    blocked_tasks = [task for task in task_statuses if task["status"] in {"BLOCKED", "ESCALATED", "REVISE"}]
    deferred_tasks = [task for task in task_statuses if task["status"] == "PARTIAL"]
    ready_queue = [task["id"] for task in sorted(tasks, key=lambda item: (item["stage"], item["id"])) if _is_ready_task(task, task_statuses)]
    active_workers = [task for task in task_statuses if task["status"] == "IN_PROGRESS"]
    next_human_action = run_state.get("next_human_action")
    if run_state.get("awaiting_human_review"):
        next_human_action = next_human_action or "Review stage summary and paper package"
    recent_actions = []
    for task in task_statuses:
        for action in task.get("recent_actions", [])[-2:]:
            recent_actions.append({"task_id": task["id"], **action})
    recent_actions = recent_actions[-10:]

    return {
        "project": {
            "title": project_config.get("title"),
            "domain": project_config.get("domain"),
        },
        "current_stage": current_stage,
        "task_counts": counts,
        "tasks": task_statuses,
        "blocked_tasks": blocked_tasks,
        "deferred_tasks": deferred_tasks,
        "ready_queue": ready_queue,
        "active_workers": active_workers,
        "recent_actions": recent_actions,
        "provider_runtime": runtime_summary(),
        "cache": cache_summary(),
        "paper": paper_summary,
        "run_state": run_state,
        "last_successful_action": run_state.get("last_successful_action"),
        "next_required_human_action": next_human_action,
    }


def write_project_status(project_path: Path) -> Path:
    status = build_project_status(project_path)
    path = project_path / "project_status.json"
    path.write_text(json.dumps(status, indent=2))
    return path


def write_dashboard(project_path: Path) -> Path:
    status = build_project_status(project_path)
    lines = [
        f"# {status['project'].get('title', 'Project')} Dashboard",
        "",
        f"- Domain: {status['project'].get('domain', 'unknown')}",
        f"- Current stage: {status.get('current_stage')}",
        f"- Last successful action: {status.get('last_successful_action')}",
        f"- Next required human action: {status.get('next_required_human_action')}",
        f"- Ready queue: {', '.join(status.get('ready_queue', [])) or 'None'}",
        "",
        "## Task Counts",
    ]
    for name, count in sorted(status["task_counts"].items()):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Tasks"])
    for task in status["tasks"]:
        lines.append(f"- {task['id']} ({task['status']}): {task['reason']}")

    lines.extend(["", "## Recent Actions"])
    for action in status["recent_actions"]:
        lines.append(f"- {action['task_id']}: {action.get('action')} -> {action.get('summary', '')[:120]}")

    lines.extend(
        [
            "",
            "## Provider Runtime",
            f"- Configured Groq keys: {status['provider_runtime']['configured_groq_keys']}",
            f"- Last provider error: {status['provider_runtime']['last_error']}",
            f"- Request count: {status['provider_runtime']['request_count']}",
            f"- Estimated max tokens: {status['provider_runtime']['estimated_max_tokens']}",
            "",
            "## Cache",
            f"- Hits: {status['cache']['hits']}",
            f"- Misses: {status['cache']['misses']}",
            "",
            "## Paper",
            f"- Sections: {status['paper']['section_count']}",
            f"- Figures: {status['paper']['figure_count']}",
            f"- Claim registry: {status['paper']['claim_registry']}",
        ]
    )

    path = project_path / "dashboard.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def update_run_state(project_path: Path, **updates):
    run_state = _read_json(project_path / "run_state.json", {})
    run_state.update(updates)
    (project_path / "run_state.json").write_text(json.dumps(run_state, indent=2))


def _task_status(project_path: Path, task: dict) -> dict:
    stage_dir = project_path / "stages" / f"stage_{task['stage']}"
    verify_path = stage_dir / f"{task['id']}_verify.json"
    escalation_path = stage_dir / f"{task['id']}_escalation.md"
    worker_state_path = stage_dir / f"{task['id']}_worker" / "worker_state.json"
    worker_steps_path = stage_dir / f"{task['id']}_worker" / "worker_steps.jsonl"

    worker_state = _read_json(worker_state_path, {})
    recent_actions = _read_worker_steps(worker_steps_path)

    if escalation_path.exists():
        verification = _read_json(verify_path, {})
        status = "ESCALATED"
        reason = "Escalation file present"
        if verification.get("status") == "ESCALATE":
            reason = "Verifier status: ESCALATE"
        return {
            "id": task["id"],
            "status": status,
            "reason": reason,
            "verification": verification,
            "recent_actions": recent_actions,
            "deferred_issues": worker_state.get("deferred_issues", []),
        }
    if verify_path.exists():
        verification = _read_json(verify_path, {})
        status = verification.get("status", "UNKNOWN")
        mapped = {"ACCEPT": "VERIFIED", "ESCALATE": "ESCALATED", "REVISE": "REVISE"}.get(status, "IN_PROGRESS")
        return {
            "id": task["id"],
            "status": mapped,
            "reason": f"Verifier status: {status}",
            "verification": verification,
            "recent_actions": recent_actions,
            "deferred_issues": worker_state.get("deferred_issues", []),
        }
    if worker_state_path.exists():
        return {
            "id": task["id"],
            "status": worker_state.get("task_status", "IN_PROGRESS"),
            "reason": worker_state.get("blocking_reason") or worker_state.get("last_reason", "Worker state present"),
            "verification": None,
            "recent_actions": recent_actions,
            "deferred_issues": worker_state.get("deferred_issues", []),
        }
    if (stage_dir / f"{task['id']}.md").exists():
        return {"id": task["id"], "status": "IN_PROGRESS", "reason": "Task output exists without verification", "verification": None, "recent_actions": recent_actions, "deferred_issues": []}
    return {"id": task["id"], "status": "PENDING", "reason": "No output yet", "verification": None, "recent_actions": recent_actions, "deferred_issues": []}


def _read_worker_steps(path: Path, limit: int = 3) -> list[dict]:
    if not path.exists():
        return []
    steps = []
    for line in path.read_text().splitlines()[-limit:]:
        entry = json.loads(line)
        steps.append({"action": entry.get("action", {}).get("action"), "summary": entry.get("summary", "")})
    return steps


def _is_ready_task(task: dict, task_statuses: list[dict]) -> bool:
    status_by_id = {item["id"]: item for item in task_statuses}
    task_status = status_by_id.get(task["id"], {}).get("status")
    if task_status not in {None, "PENDING"}:
        return False
    for dependency in task.get("dependencies", []):
        dep_status = status_by_id.get(dependency, {}).get("status")
        dep_record = status_by_id.get(dependency, {})
        if dep_status == "VERIFIED":
            continue
        if dep_status == "PARTIAL" and not any(issue.get("blocks_dependents") for issue in dep_record.get("deferred_issues", [])):
            continue
        return False
    return True


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())
