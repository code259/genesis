from __future__ import annotations

import json
from pathlib import Path

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
        if status["status"] != "VERIFIED" and current_stage is None:
            current_stage = task["stage"]

    paper_dir = project_path / "paper"
    paper_summary = {
        "exists": paper_dir.exists(),
        "section_count": len(list((paper_dir / "sections").glob("*.tex"))) if (paper_dir / "sections").exists() else 0,
        "figure_count": len(list((paper_dir / "figures").glob("*.png"))) if (paper_dir / "figures").exists() else 0,
        "claim_registry": str((paper_dir / "claim_registry.json").relative_to(project_path)) if (paper_dir / "claim_registry.json").exists() else None,
    }

    blocked_tasks = [task for task in task_statuses if task["status"] in {"BLOCKED", "ESCALATED"}]
    next_human_action = run_state.get("next_human_action")
    if run_state.get("awaiting_human_review"):
        next_human_action = next_human_action or "Review stage summary and paper package"

    return {
        "project": {
            "title": project_config.get("title"),
            "domain": project_config.get("domain"),
        },
        "current_stage": current_stage,
        "task_counts": counts,
        "tasks": task_statuses,
        "blocked_tasks": blocked_tasks,
        "provider_runtime": runtime_summary(),
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
        "",
        "## Task Counts",
    ]
    for name, count in sorted(status["task_counts"].items()):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Tasks"])
    for task in status["tasks"]:
        lines.append(f"- {task['id']} ({task['status']}): {task['reason']}")

    lines.extend(
        [
            "",
            "## Provider Runtime",
            f"- Configured Groq keys: {status['provider_runtime']['configured_groq_keys']}",
            f"- Last provider error: {status['provider_runtime']['last_error']}",
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

    if verify_path.exists():
        verification = _read_json(verify_path, {})
        status = verification.get("status", "UNKNOWN")
        mapped = "VERIFIED" if status == "ACCEPT" else ("ESCALATED" if status == "ESCALATE" else "IN_PROGRESS")
        return {"id": task["id"], "status": mapped, "reason": f"Verifier status: {status}"}
    if escalation_path.exists():
        return {"id": task["id"], "status": "ESCALATED", "reason": "Escalation file present"}
    if worker_state_path.exists():
        state = _read_json(worker_state_path, {})
        return {
            "id": task["id"],
            "status": state.get("task_status", "IN_PROGRESS"),
            "reason": state.get("last_reason", "Worker state present"),
        }
    if (stage_dir / f"{task['id']}.md").exists():
        return {"id": task["id"], "status": "IN_PROGRESS", "reason": "Task output exists without verification"}
    return {"id": task["id"], "status": "PENDING", "reason": "No output yet"}


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())
