from __future__ import annotations

import json
import re
from typing import Any


TASK_ID_RE = re.compile(r"^S(\d+)T(\d+)$")


def extract_stage(task_id: str) -> int:
    match = TASK_ID_RE.match(task_id.strip())
    if not match:
        raise ValueError(f"Invalid task id: {task_id}")
    return int(match.group(1))


def parse_task_tree(content: str) -> list[dict[str, Any]]:
    stripped = content.strip()
    if not stripped:
        return []

    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    json_tasks = _parse_json_tasks(stripped)
    if json_tasks is not None:
        return [_normalize_task(task) for task in json_tasks]

    yaml_tasks = _parse_yaml_task_blocks(content)
    if yaml_tasks:
        return [_normalize_task(task) for task in yaml_tasks]

    markdown_tasks = _parse_markdown_sections(content)
    return [_normalize_task(task) for task in markdown_tasks]


def validate_task_tree(tasks: list[dict[str, Any]]) -> list[str]:
    errors = []
    ids = {task["id"] for task in tasks}

    for task in tasks:
        task_id = task["id"]
        if not TASK_ID_RE.match(task_id):
            errors.append(f"{task_id}: invalid task id format")
        if not task.get("description"):
            errors.append(f"{task_id}: description is empty")

        criteria = task.get("verification_criteria")
        if isinstance(criteria, list):
            if not any(item.strip() for item in criteria):
                errors.append(f"{task_id}: verification criteria is empty")
        elif not criteria:
            errors.append(f"{task_id}: verification criteria is empty")

        for dep in task.get("dependencies", []):
            if dep not in ids:
                errors.append(f"{task_id}: dependency '{dep}' not found in task tree")

    return errors


def build_dependency_graph(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    graph = {task["id"]: [] for task in tasks}
    for task in tasks:
        for dep in task.get("dependencies", []):
            if dep in graph:
                graph[dep].append(task["id"])
    return graph


def tasks_for_stage(tasks: list[dict[str, Any]], stage: int) -> list[dict[str, Any]]:
    return [task for task in tasks if task.get("stage") == stage]


def _parse_json_tasks(content: str) -> list[dict[str, Any]] | None:
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(loaded, dict) and isinstance(loaded.get("tasks"), list):
        return loaded["tasks"]
    if isinstance(loaded, list):
        return loaded
    return None


def _parse_yaml_task_blocks(content: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"```ya?ml\s*(.*?)```", content, re.IGNORECASE | re.DOTALL)
    return [_parse_simple_yaml(block) for block in blocks if block.strip()]


def _parse_markdown_sections(content: str) -> list[dict[str, Any]]:
    sections = re.split(r"\n##\s+(S\d+T\d+)\s*\n", content)
    tasks = []
    idx = 1
    while idx < len(sections):
        task_id = sections[idx].strip()
        body = sections[idx + 1] if idx + 1 < len(sections) else ""
        tasks.append(
            {
                "id": task_id,
                "description": _field(body, "Description") or "",
                "dependencies": _parse_list(_field(body, "Dependencies")),
                "stage": _field(body, "Stage") or extract_stage(task_id),
                "verification_criteria": _parse_bullets_after_field(body, "Verification criteria"),
                "complexity": _field(body, "Complexity") or "STANDARD",
                "foundational": False,
            }
        )
        idx += 2
    return tasks


def _field(body: str, name: str) -> str | None:
    match = re.search(rf"\*\*{re.escape(name)}:\*\*\s*(.+?)(?:\n|$)", body, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _parse_bullets_after_field(body: str, name: str) -> list[str]:
    pattern = rf"\*\*{re.escape(name)}:\*\*\s*(.*?)(?:\n\*\*|\Z)"
    match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    section = match.group(1).strip()
    items = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _parse_simple_yaml(block: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and current_key:
            parsed.setdefault(current_key, [])
            parsed[current_key].append(_coerce_value(stripped[2:].strip()))
            continue

        if ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        current_key = key

        if not raw_value:
            parsed[key] = []
            continue

        parsed[key] = _coerce_value(raw_value)

    return parsed


def _coerce_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_coerce_value(item) for item in value]
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped.isdigit():
        return int(stripped)
    return stripped.strip("'\"")


def _normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("id", "")).strip()
    if not task_id:
        raise ValueError("Task is missing id")

    criteria = task.get("verification_criteria", task.get("verificationCriteria", []))
    if isinstance(criteria, str):
        criteria = [criteria] if criteria else []

    dependencies = task.get("dependencies", [])
    if isinstance(dependencies, str):
        dependencies = _parse_list(dependencies)

    complexity = task.get("complexity", task.get("complexity_tier", "STANDARD"))
    stage = task.get("stage")
    if isinstance(stage, str) and stage.isdigit():
        stage = int(stage)
    if not isinstance(stage, int):
        stage = extract_stage(task_id)

    foundational = task.get("foundational")
    if foundational is None:
        foundational = task.get("complexity_tier", "").upper() == "HIGH" and len(dependencies) == 0

    return {
        "id": task_id,
        "description": str(task.get("description", "")).strip(),
        "dependencies": [str(dep).strip() for dep in dependencies if str(dep).strip()],
        "stage": stage,
        "verification_criteria": [str(item).strip() for item in criteria if str(item).strip()],
        "complexity": str(complexity).replace("complexity_tier", "").upper().strip() or "STANDARD",
        "foundational": bool(foundational),
    }


def _parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    coerced = _coerce_value(raw)
    if isinstance(coerced, list):
        return [str(item).strip() for item in coerced if str(item).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]
