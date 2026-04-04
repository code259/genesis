import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import runtime_policy


def test_validate_read_path_denies_git(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()

    result = runtime_policy.validate_read_path(project_path, ".git/config")

    assert result.allowed is False
    assert ".git" in result.reason


def test_validate_python_code_denies_conda_usage():
    result = runtime_policy.validate_python_code({"id": "S1T1"}, "import subprocess\nsubprocess.check_call(['conda', 'install', 'x'])")

    assert result.allowed is False
    assert "forbidden" in result.reason.lower()


def test_validate_write_path_stays_inside_task_artifacts(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()

    result = runtime_policy.validate_write_path(project_path, "S1T2", "../escape.txt")

    assert result.allowed is False
