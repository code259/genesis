import json

from click.testing import CliRunner

from genesis.cli.main import main


def test_custom_benchmark_suite_runs_minimal_project(tmp_path, monkeypatch):
    spec_path = tmp_path / "benchmark_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Investigate learning rate warmup on a small MLP benchmark",
                "domain": "ml_efficiency",
                "success_criteria": ["Investigate learning rate warmup on a small MLP benchmark"],
                "oracle_hints": ["metric consistency"],
                "compute_budget": "local_gpu",
                "time_budget_hours": 2,
                "domain_knowledge_model": "none",
                "output_dir": str(tmp_path / "projects"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "benchmark01", "--spec", str(spec_path)])
    assert result.exit_code == 0, result.output
    assert "benchmark01" in result.output
