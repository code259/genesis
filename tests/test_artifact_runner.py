import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.artifact_runner import run_python_task


def test_run_python_task_produces_expected_artifacts(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    task_spec = {"id": "S1T3"}
    plan = {
        "python_code": """
from pathlib import Path
import json
import os

artifact_dir = Path(os.environ["GENESIS_ARTIFACT_DIR"])
results_path = Path(os.environ["GENESIS_RESULTS_PATH"])
(artifact_dir / "data.csv").write_text("x,y\\n1,2\\n")
(artifact_dir / "figure_1.png").write_bytes(b"png")
results_path.write_text(json.dumps({"oracle_inputs": []}))
""",
        "expected_artifacts": [
            {"path": "data.csv", "kind": "csv"},
            {"path": "figure_1.png", "kind": "png"},
        ],
    }

    result = run_python_task(task_spec, project_path, plan)
    assert result.success is True
    assert any(path.endswith("data.csv") for path in result.artifact_paths)
    assert any(path.endswith("figure_1.png") for path in result.artifact_paths)


def test_run_python_task_fails_when_artifact_missing(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    task_spec = {"id": "S1T4"}
    plan = {
        "python_code": "print('no artifacts')",
        "expected_artifacts": [{"path": "missing.csv", "kind": "csv"}],
    }

    result = run_python_task(task_spec, project_path, plan)
    assert result.success is False
    assert result.missing_artifacts == ["missing.csv"]
