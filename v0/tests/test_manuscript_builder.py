import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.manuscript_builder import build_paper_package, build_stage_summary


def test_build_stage_summary_and_paper_package(tmp_path: Path):
    project_path = tmp_path / "project"
    stage_dir = project_path / "stages" / "stage_1"
    artifact_dir = stage_dir / "S1T1_artifacts"
    artifact_dir.mkdir(parents=True)
    (project_path / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "S1T1",
                    "description": "Task",
                    "dependencies": [],
                    "stage": 1,
                    "verification_criteria": ["Output exists"],
                    "complexity": "STANDARD",
                    "foundational": True,
                }
            ]
        )
    )
    (stage_dir / "S1T1.md").write_text("Verified task output")
    (stage_dir / "S1T1_verify.json").write_text(json.dumps({"status": "ACCEPT"}))
    (artifact_dir / "figure_1.png").write_bytes(b"png")

    summary_path = build_stage_summary(project_path, 1)
    paper_dir = build_paper_package(project_path)

    assert summary_path.exists()
    assert (paper_dir / "main.tex").exists()
    assert (paper_dir / "refs.bib").exists()
    assert (paper_dir / "figures" / "figure_1.png").exists()
