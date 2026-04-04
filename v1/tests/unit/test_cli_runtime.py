import json

from click.testing import CliRunner

from genesis.agents.runtime import CodingAgentRuntime
from genesis.cli.main import main


def test_cli_status_results_and_intervention(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: {
            "summary": "Provider executed task successfully.",
            "artifact_plan": [{"path": "notes.md", "content": "generated note"}],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
        },
    )
    spec_path = tmp_path / "spec.json"
    project_root = tmp_path / "projects"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Test cli status",
                "domain": "general",
                "success_criteria": ["Test cli status"],
                "oracle_hints": [],
                "compute_budget": "local_cpu",
                "time_budget_hours": 1,
                "domain_knowledge_model": "none",
                "output_dir": str(project_root),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    run_result = runner.invoke(main, ["run", "--project-id", "cli01", "--spec", str(spec_path)])
    assert run_result.exit_code == 0, run_result.output

    status_result = runner.invoke(main, ["status", "--project-id", "cli01", "--root", str(project_root)])
    assert status_result.exit_code == 0
    assert '"project_id": "cli01"' in status_result.output
    assert '"state"' in status_result.output

    results_result = runner.invoke(main, ["results", "--project-id", "cli01", "--root", str(project_root)])
    assert results_result.exit_code == 0
    assert '"paper_dir"' in results_result.output
    assert '"run_index"' in results_result.output

    intervene_result = runner.invoke(
        main,
        ["intervene", "--project-id", "cli01", "--type", "STOP", "--root", str(project_root)],
    )
    assert intervene_result.exit_code == 0
