from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from core.task_parser import extract_stage


DENIED_PATH_PARTS = {
    ".git",
    "project_context",
    "site-packages",
    "miniconda",
    "miniconda3",
    "conda",
    "brew",
    ".zshrc",
    ".bashrc",
    ".bash_profile",
    ".profile",
}

FORBIDDEN_CODE_PATTERNS = [
    re.compile(r"\bgit\s+init\b"),
    re.compile(r"\bgit\s+commit\b"),
    re.compile(r"\bconda\b"),
    re.compile(r"\bminiconda\b"),
    re.compile(r"\bbrew\b"),
    re.compile(r"\bsite-packages\b"),
    re.compile(r"\b/project_context/|\bproject_context/"),
    re.compile(r"/Users/"),
    re.compile(r"\brm\s+-rf\b"),
]


@dataclass
class PolicyResult:
    allowed: bool
    reason: str
    normalized_path: Path | None = None


def validate_read_path(project_path: Path, rel_path: str) -> PolicyResult:
    project_root = project_path.resolve()
    stripped = rel_path.strip()
    if not stripped:
        return PolicyResult(False, "Empty read path is not allowed")
    denied = _denied_path_reason(stripped)
    if denied:
        return PolicyResult(False, denied)
    candidate = (project_root / stripped).resolve()
    if project_root not in candidate.parents and candidate != project_root:
        return PolicyResult(False, f"Read path escapes project root: {rel_path}")
    return PolicyResult(True, "allowed", normalized_path=candidate)


def validate_write_path(project_path: Path, task_id: str, rel_path: str) -> PolicyResult:
    project_root = project_path.resolve()
    stripped = rel_path.strip()
    if not stripped:
        return PolicyResult(False, "Empty write path is not allowed")
    denied = _denied_path_reason(stripped)
    if denied:
        return PolicyResult(False, denied)
    artifact_root = _task_artifact_root(project_root, task_id)
    candidate = (artifact_root / stripped).resolve()
    if artifact_root not in candidate.parents and candidate != artifact_root:
        return PolicyResult(False, f"Write path escapes task artifact directory: {rel_path}")
    return PolicyResult(True, "allowed", normalized_path=candidate)


def validate_python_code(task_spec: dict, code: str) -> PolicyResult:
    lowered = code.lower()
    for pattern in FORBIDDEN_CODE_PATTERNS:
        if pattern.search(lowered):
            return PolicyResult(False, f"Python action uses forbidden local-system operation: {pattern.pattern}")

    for item in _extract_string_literals(code):
        denied = _denied_path_reason(item)
        if denied:
            return PolicyResult(False, denied)
        if item.startswith("/") or item.startswith("~"):
            return PolicyResult(False, f"Absolute or home-relative path is denied by worker runtime policy: {item}")

    return PolicyResult(True, "allowed")


def _extract_string_literals(code: str) -> list[str]:
    return re.findall(r"['\"]([^'\"]+)['\"]", code)


def _denied_path_reason(rel_path: str) -> str | None:
    parts = {part for part in Path(rel_path).parts if part not in {"", "."}}
    for denied in DENIED_PATH_PARTS:
        if denied in parts or denied in rel_path:
            return f"Access to '{denied}' is denied by worker runtime policy"
    return None


def _task_artifact_root(project_root: Path, task_id: str) -> Path:
    return project_root / "stages" / f"stage_{extract_stage(task_id)}" / f"{task_id}_artifacts"
