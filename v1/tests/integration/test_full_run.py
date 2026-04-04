import json

from click.testing import CliRunner

from genesis.cli.main import main


def test_full_run(tmp_path, monkeypatch):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Investigate the effect of learning rate on convergence speed in a 2-layer MLP on MNIST",
                "domain": "ml_efficiency",
                "success_criteria": ["Investigate the effect of learning rate on convergence speed in a 2-layer MLP on MNIST"],
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
    result = runner.invoke(main, ["run", "--project-id", "demo1234", "--spec", str(spec_path)])
    assert result.exit_code == 0, result.output
    assert "demo1234" in result.output
