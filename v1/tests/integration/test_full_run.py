import json

from click.testing import CliRunner

from genesis.agents.runtime import CodingAgentRuntime
from genesis.agents.runtime import ProviderRuntimeError
from genesis.cli.main import main
from genesis.models import StoppingDecision


class _PassingReport:
    def __init__(self):
        self.acceptance_ratio = 1.0
        self.stopping_decision = StoppingDecision(True, ["all stopping criteria satisfied"])

    def to_dict(self):
        return {
            "acceptance_ratio": self.acceptance_ratio,
            "claim_flags": [],
            "literature_flags": [],
            "formal_checks": [],
            "grounded_claims": 1,
            "total_claims": 1,
            "stopping_decision": self.stopping_decision.to_dict(),
        }


def test_full_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: {
            "summary": "Provider executed task successfully.",
            "artifact_plan": [{"path": "notes.md", "content": "generated note"}],
            "command_plan": [],
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
    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria: _PassingReport())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None: {"passed": True, "checks": []})
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
            "command_plan": [],
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


def test_non_actionable_plan_is_an_incomplete_run_not_a_halt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: {
            "summary": "Provider wrote prose only.",
            "artifact_plan": [],
            "command_plan": [],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
            "primary_model": "fake-model",
            "attempted_models": ["fake-model"],
            "fallback_used": False,
        },
    )

    class _Report:
        def __init__(self):
            self.acceptance_ratio = 0.0
            self.stopping_decision = StoppingDecision(False, ["continue iteration"])

        def to_dict(self):
            return {
                "acceptance_ratio": self.acceptance_ratio,
                "claim_flags": ["IMPLICIT_ASSUMPTION:test"],
                "literature_flags": [],
                "formal_checks": [],
                "grounded_claims": 0,
                "total_claims": 1,
                "stopping_decision": self.stopping_decision.to_dict(),
            }

    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria: _Report())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None: {"passed": False, "checks": []})

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Non actionable plan",
                "domain": "general",
                "success_criteria": ["Non actionable plan"],
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
    result = runner.invoke(main, ["run", "--project-id", "noop01", "--spec", str(spec_path), "--max-runs", "2"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "incomplete"
    run_result = json.loads((tmp_path / "projects" / "noop01" / "runs" / "1" / "result.json").read_text(encoding="utf-8"))
    assert run_result["classification"] == "non_actionable_plan"
    assert not (tmp_path / "projects" / "noop01" / "HALT.json").exists()


def test_adversarial_stalemate_halts_after_escalation_window(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: {
            "summary": "Provider executed task unsuccessfully.",
            "artifact_plan": [{"path": "notes.md", "content": "generated note"}],
            "command_plan": [],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
        },
    )

    class _Report:
        def __init__(self):
            self.acceptance_ratio = 0.0
            self.stopping_decision = StoppingDecision(False, ["continue iteration"], critical_flags=["needs_work"])

        def to_dict(self):
            return {
                "acceptance_ratio": self.acceptance_ratio,
                "claim_flags": ["IMPLICIT_ASSUMPTION:test"],
                "literature_flags": [],
                "formal_checks": [],
                "grounded_claims": 0,
                "total_claims": 1,
                "stopping_decision": self.stopping_decision.to_dict(),
            }

    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria: _Report())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None: {"passed": False, "checks": []})

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Force stalemate",
                "domain": "general",
                "success_criteria": ["Force stalemate"],
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
    result = runner.invoke(main, ["run", "--project-id", "stalemate01", "--spec", str(spec_path), "--max-runs", "5"])
    assert result.exit_code == 0, result.output
    assert '"status": "halted"' in result.output
    assert (tmp_path / "projects" / "stalemate01" / "HALT.json").exists()
    assert (tmp_path / "projects" / "stalemate01" / "runs" / "3" / "escalation_report.json").exists()


def test_max_runs_exhaustion_returns_incomplete_and_interim_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: {
            "summary": "Provider executed task but work remains.",
            "artifact_plan": [{"path": "notes.md", "content": "generated note"}],
            "command_plan": [],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
        },
    )

    class _Report:
        def __init__(self):
            self.acceptance_ratio = 0.0
            self.stopping_decision = StoppingDecision(False, ["continue iteration"])

        def to_dict(self):
            return {
                "acceptance_ratio": self.acceptance_ratio,
                "claim_flags": ["IMPLICIT_ASSUMPTION:test"],
                "literature_flags": [],
                "formal_checks": [],
                "grounded_claims": 0,
                "total_claims": 1,
                "stopping_decision": self.stopping_decision.to_dict(),
            }

    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria: _Report())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None: {"passed": False, "checks": []})

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Force incomplete project",
                "domain": "general",
                "success_criteria": ["Force incomplete project"],
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
    result = runner.invoke(main, ["run", "--project-id", "incomplete01", "--spec", str(spec_path), "--max-runs", "2"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "incomplete"
    assert "max runs exhausted" in payload["summary"]
    report = json.loads((tmp_path / "projects" / "incomplete01" / "outputs" / "paper" / "synthesis_report.json").read_text(encoding="utf-8"))
    assert report["report_mode"] == "interim"


def test_bad_command_fails_run_gracefully_without_halting_project(tmp_path, monkeypatch):
    calls = {"count": 0}

    def _generate(self, **kwargs):
        calls["count"] += 1
        instruction = str(kwargs.get("instruction", ""))
        if "Produce a bounded research task DAG" in instruction:
            return {
                "summary": "Generated task tree.",
                "artifact_plan": [],
                "command_plan": [],
                "experiment_plan": [],
                "citations": [],
                "next_action": "continue",
                "provider": "test",
                "model": "fake-model",
                "primary_model": "fake-model",
                "attempted_models": ["fake-model"],
                "fallback_used": False,
                "task_tree": [],
            }
        if calls["count"] == 3:
            return {
                "summary": "Tried a bad command.",
                "artifact_plan": [],
                "command_plan": ["BLSSearchCustomMASTData"],
                "experiment_plan": [],
                "citations": [],
                "next_action": "continue",
                "provider": "test",
                "model": "fake-model",
                "primary_model": "fake-model",
                "attempted_models": ["fake-model"],
                "fallback_used": False,
            }
        return {
            "summary": "Wrote a real file.",
            "artifact_plan": [{"path": "notes.md", "content": "generated note"}],
            "command_plan": [],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
            "primary_model": "fake-model",
            "attempted_models": ["fake-model"],
            "fallback_used": False,
        }

    monkeypatch.setattr(CodingAgentRuntime, "generate_task", _generate)

    class _Report:
        def __init__(self):
            self.acceptance_ratio = 0.0
            self.stopping_decision = StoppingDecision(False, ["continue iteration"])

        def to_dict(self):
            return {
                "acceptance_ratio": self.acceptance_ratio,
                "claim_flags": ["IMPLICIT_ASSUMPTION:test"],
                "literature_flags": [],
                "formal_checks": [],
                "grounded_claims": 0,
                "total_claims": 1,
                "stopping_decision": self.stopping_decision.to_dict(),
            }

    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria: _Report())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None: {"passed": False, "checks": []})

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Bad command then recover",
                "domain": "general",
                "success_criteria": ["Bad command then recover"],
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
    result = runner.invoke(main, ["run", "--project-id", "badcmd01", "--spec", str(spec_path), "--max-runs", "2"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "incomplete"
    first = json.loads((tmp_path / "projects" / "badcmd01" / "runs" / "1" / "result.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "projects" / "badcmd01" / "runs" / "2" / "result.json").read_text(encoding="utf-8"))
    assert first["classification"] == "command_failure"
    assert first["failure_type"] == "command_not_found"
    assert second["generated_artifacts"]
    assert not (tmp_path / "projects" / "badcmd01" / "HALT.json").exists()


def test_failed_command_attempt_does_not_pass_verification(tmp_path, monkeypatch):
    calls = {"count": 0}

    def _generate(self, **kwargs):
        calls["count"] += 1
        instruction = str(kwargs.get("instruction", ""))
        if "Produce a bounded research task DAG" in instruction:
            return {
                "summary": "Generated task tree.",
                "artifact_plan": [],
                "command_plan": [],
                "experiment_plan": [],
                "citations": [],
                "next_action": "continue",
                "provider": "test",
                "model": "fake-model",
                "primary_model": "fake-model",
                "attempted_models": ["fake-model"],
                "fallback_used": False,
                "task_tree": [],
            }
        return {
            "summary": "Tried a bad command.",
            "artifact_plan": [],
            "command_plan": ["missing.py"],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
            "primary_model": "fake-model",
            "attempted_models": ["fake-model"],
            "fallback_used": False,
        }

    monkeypatch.setattr(CodingAgentRuntime, "generate_task", _generate)

    class _Report:
        def __init__(self):
            self.acceptance_ratio = 0.0
            self.stopping_decision = StoppingDecision(False, ["continue iteration"])

        def to_dict(self):
            return {
                "acceptance_ratio": self.acceptance_ratio,
                "claim_flags": ["IMPLICIT_ASSUMPTION:test"],
                "literature_flags": [],
                "formal_checks": [],
                "grounded_claims": 0,
                "total_claims": 1,
                "stopping_decision": self.stopping_decision.to_dict(),
            }

    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria: _Report())

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Failed command verification",
                "domain": "general",
                "success_criteria": ["Failed command verification"],
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
    result = runner.invoke(main, ["run", "--project-id", "failedverify01", "--spec", str(spec_path), "--max-runs", "1"])
    assert result.exit_code == 0, result.output
    verification = json.loads((tmp_path / "projects" / "failedverify01" / "runs" / "1" / "verification_report.json").read_text(encoding="utf-8"))
    substantive = next(check for check in verification["checks"] if check["name"] == "substantive_artifacts")
    assert substantive["passed"] is False
    assert verification["passed"] is False
