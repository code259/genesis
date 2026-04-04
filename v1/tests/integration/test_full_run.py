import json

from click.testing import CliRunner

from genesis.agents.runtime import CodingAgentRuntime
from genesis.agents.runtime import ProviderRuntimeError
from genesis.cli.main import main


def test_full_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: {
            "summary": "Provider executed task successfully.",
            "artifact_plan": [{"path": "notes.md", "content": "generated note"}],
            "experiment_plan": [
                {
                    "description": "agent proposed experiment",
                    "code_diff": "warmup_ratio=0.3",
                    "expected_metric": 0.62,
                    "expected_trajectory": [0.2, 0.45, 0.62],
                }
            ],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
        },
    )
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
    result = runner.invoke(main, ["run", "--project-id", "demo1234", "--spec", str(spec_path), "--max-runs", "2"])
    assert result.exit_code == 0, result.output
    result_json = json.loads(result.output)
    assert result_json["status"] == "complete"
    paper_dir = tmp_path / "projects" / "demo1234" / "outputs" / "paper"
    assert (paper_dir / "main.tex").exists()
    assert (paper_dir / "synthesis_report.json").exists()


def test_init_and_status_commands(tmp_path, monkeypatch):
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
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Does init create a project?",
                "domain": "general",
                "success_criteria": ["Does init create a project?"],
                "oracle_hints": [],
                "compute_budget": "local_cpu",
                "time_budget_hours": 1,
                "domain_knowledge_model": "none",
                "output_dir": str(tmp_path / "projects"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    init_result = runner.invoke(main, ["init", "--project-id", "initdemo", "--spec", str(spec_path)])
    assert init_result.exit_code == 0, init_result.output
    assert (tmp_path / "projects" / "initdemo" / "spec.json").exists()


def test_provider_failure_writes_halt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: (_ for _ in ()).throw(ProviderRuntimeError("provider down")),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Fail provider runtime",
                "domain": "general",
                "success_criteria": ["Fail provider runtime"],
                "oracle_hints": [],
                "compute_budget": "local_cpu",
                "time_budget_hours": 1,
                "domain_knowledge_model": "none",
                "output_dir": str(tmp_path / "projects"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "halt01", "--spec", str(spec_path)])
    assert result.exit_code == 0, result.output
    assert '"status": "halted"' in result.output
    assert (tmp_path / "projects" / "halt01" / "HALT.json").exists()
