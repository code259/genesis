import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.oracle_manager import run_oracle_checks


def test_non_astro_project_reports_no_oracle_configured(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / "project_config.json").write_text(json.dumps({"oracle_module": None}))
    output_path = project_path / "output.md"
    output_path.write_text("result")

    result = run_oracle_checks({"id": "S1T1"}, output_path, [], project_path)
    assert result.configured is False
    assert result.notes == ["No oracle configured for this project"]


def test_astro_oracle_runs_from_results_json(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / "project_config.json").write_text(json.dumps({"oracle_module": "astro"}))
    stage_dir = project_path / "stages" / "stage_1" / "S1T1_artifacts"
    stage_dir.mkdir(parents=True)
    (stage_dir / "results.json").write_text(
        json.dumps(
            {
                "oracle_inputs": [
                    {"check": "velocity_physical", "kwargs": {"velocity_km_s": 300.0}},
                    {"check": "color_physical", "kwargs": {"filter1": "B", "filter2": "V", "color": 0.5}},
                ]
            }
        )
    )
    output_path = project_path / "stages" / "stage_1" / "S1T1.md"
    output_path.write_text("result")

    result = run_oracle_checks({"id": "S1T1"}, output_path, [], project_path)
    assert result.configured is True
    assert result.applicable is True
    assert result.oracle_pass is True
    assert result.report_path is not None
