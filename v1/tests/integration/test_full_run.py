import json
from pathlib import Path

from click.testing import CliRunner

from genesis.agents.runtime import CodingAgentRuntime, ProviderRuntimeError
from genesis.cli.main import main
from genesis.models import StoppingDecision, TaskNode, TaskTree


class _PassingReport:
    def __init__(self):
        self.acceptance_ratio = 1.0
        self.critical_blockers = []
        self.stopping_decision = StoppingDecision(True, ["all stopping criteria satisfied"])

    def to_dict(self):
        return {
            "acceptance_ratio": self.acceptance_ratio,
            "claim_flags": [],
            "literature_flags": [],
            "formal_checks": [],
            "grounded_claims": 1,
            "total_claims": 1,
            "critical_blockers": self.critical_blockers,
            "stopping_decision": self.stopping_decision.to_dict(),
        }


class _FailingReport:
    def __init__(self):
        self.acceptance_ratio = 0.0
        self.critical_blockers = ["IMPLICIT_ASSUMPTION:test"]
        self.stopping_decision = StoppingDecision(False, ["continue iteration"])

    def to_dict(self):
        return {
            "acceptance_ratio": self.acceptance_ratio,
            "claim_flags": ["IMPLICIT_ASSUMPTION:test"],
            "literature_flags": [],
            "formal_checks": [],
            "grounded_claims": 0,
            "total_claims": 1,
            "critical_blockers": self.critical_blockers,
            "stopping_decision": self.stopping_decision.to_dict(),
        }


def _spec_path(tmp_path, question="Test question") -> str:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": question,
                "domain": "general",
                "success_criteria": [question],
                "oracle_hints": [],
                "compute_budget": "local_cpu",
                "time_budget_hours": 1,
                "domain_knowledge_model": "none",
                "output_dir": str(tmp_path / "projects"),
            }
        ),
        encoding="utf-8",
    )
    return str(spec_path)


def _patch_runtime_ok(monkeypatch):
    monkeypatch.setattr(CodingAgentRuntime, "check_health", lambda self, probe_models=False: {"passed": True, "checks": []})


def test_full_run_completes_with_stage_progression(tmp_path, monkeypatch):
    _patch_runtime_ok(monkeypatch)
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
    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria, **kwargs: _PassingReport())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None, **kwargs: {"passed": True, "checks": []})
    monkeypatch.setattr(
        "genesis.modules.oracle.validator.OracleValidator.validate_with_synthetic_data",
        lambda self, oracle_path: {"name": "synthetic_oracle_validation", "passed": True, "result": {"pass_rate": 1.0}},
    )

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "demo1234", "--spec", _spec_path(tmp_path), "--max-runs", "5"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] in {"complete", "incomplete"}
    state = json.loads((tmp_path / "projects" / "demo1234" / "project_state.json").read_text(encoding="utf-8"))
    assert state["current_stage"]
    assert state["task_states"]
    assert (tmp_path / "projects" / "demo1234" / "runs" / "2" / "verification_report.json").exists()
    assert (tmp_path / "projects" / "demo1234" / "outputs" / "paper" / "synthesis_report.json").exists()


def test_repairable_schema_mismatch_does_not_halt_project(tmp_path, monkeypatch):
    _patch_runtime_ok(monkeypatch)
    calls = {"count": 0}

    def _runtime(self, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ProviderRuntimeError(
                "command_plan references workspace file 'validate.py' without creating it in artifact_plan",
                error_class="command_plan_missing_artifact",
                retryable=False,
            )
        return {
            "summary": "Repaired and executed task successfully.",
            "artifact_plan": [{"path": "result.txt", "content": "ok"}],
            "command_plan": [],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
        }

    monkeypatch.setattr(CodingAgentRuntime, "generate_task", _runtime)
    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria, **kwargs: _PassingReport())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None, **kwargs: {"passed": True, "checks": []})
    monkeypatch.setattr(
        "genesis.modules.oracle.validator.OracleValidator.validate_with_synthetic_data",
        lambda self, oracle_path: {"name": "synthetic_oracle_validation", "passed": True, "result": {"pass_rate": 1.0}},
    )

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "repair01", "--spec", _spec_path(tmp_path, "repair schema"), "--max-runs", "4"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] in {"complete", "incomplete"}
    assert not (tmp_path / "projects" / "repair01" / "HALT.json").exists()


def test_runtime_health_failure_halts_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(CodingAgentRuntime, "check_health", lambda self, probe_models=False: {"passed": False, "checks": [{"name": "opencode_binary", "passed": False}]})
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "halt01", "--spec", _spec_path(tmp_path, "health failure")])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "halted"
    assert (tmp_path / "projects" / "halt01" / "HALT.json").exists()


def test_existing_halt_refuses_restart(tmp_path, monkeypatch):
    _patch_runtime_ok(monkeypatch)
    project_dir = tmp_path / "projects" / "halted01"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "HALT.json").write_text(json.dumps({"message": "still halted"}), encoding="utf-8")
    (project_dir / "project_state.json").write_text(json.dumps({"status": "halted", "run_count": 2, "current_stage": "execute", "last_run_status": None}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "halted01", "--spec", _spec_path(tmp_path, "halted restart")])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "halted"


def test_oracle_validation_failure_halts_project(tmp_path, monkeypatch):
    _patch_runtime_ok(monkeypatch)
    monkeypatch.setattr(
        "genesis.harness.decomposer.TaskDecomposer.decompose",
        lambda self, config: TaskTree(
            root_id="oracle",
            tasks=[
                TaskNode(
                    task_id="oracle-task",
                    description="Generate and validate an oracle",
                    acceptance_criteria=["validated oracle"],
                    oracle_checks=[],
                    estimated_compute_budget="local_cpu",
                    task_kind="oracle",
                    expected_artifacts=["oracle.py"],
                    execution_mode="artifact_generation",
                )
            ],
        ),
    )
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
    monkeypatch.setattr(
        "genesis.modules.oracle.validator.OracleValidator.validate_with_synthetic_data",
        lambda self, oracle_path: {"name": "synthetic_oracle_validation", "passed": False, "result": {"pass_rate": 0.0}},
    )
    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria, **kwargs: _PassingReport())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "oraclehalt", "--spec", _spec_path(tmp_path, "oracle failure"), "--max-runs", "3"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "halted"
    assert (tmp_path / "projects" / "oraclehalt" / "HALT.json").exists()


def test_max_runs_exhaustion_returns_incomplete(tmp_path, monkeypatch):
    _patch_runtime_ok(monkeypatch)
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
    monkeypatch.setattr("genesis.harness.loop.MetaHarnessLoop._run_adversarial_check", lambda self, outputs, criteria, **kwargs: _FailingReport())
    monkeypatch.setattr("genesis.modules.verification.pipeline.VerificationPipeline.run", lambda self, outputs_dir, project_id, oracle_path=None, **kwargs: {"passed": False, "checks": []})
    monkeypatch.setattr(
        "genesis.modules.oracle.validator.OracleValidator.validate_with_synthetic_data",
        lambda self, oracle_path: {"name": "synthetic_oracle_validation", "passed": True, "result": {"pass_rate": 1.0}},
    )

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "incomplete01", "--spec", _spec_path(tmp_path, "Force incomplete project"), "--max-runs", "2"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "incomplete"
    assert payload["summary"]


def test_command_failure_does_not_halt_project(tmp_path, monkeypatch):
    _patch_runtime_ok(monkeypatch)
    monkeypatch.setattr(
        CodingAgentRuntime,
        "generate_task",
        lambda self, **kwargs: {
            "summary": "Tried a bad command.",
            "artifact_plan": [],
            "command_plan": ["BLSSearchCustomMASTData"],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
        },
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--project-id", "badcmd01", "--spec", _spec_path(tmp_path, "Bad command"), "--max-runs", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "incomplete"
    run_result = json.loads((tmp_path / "projects" / "badcmd01" / "runs" / "1" / "result.json").read_text(encoding="utf-8"))
    assert run_result["classification"] == "command_failure"
    assert not (tmp_path / "projects" / "badcmd01" / "HALT.json").exists()


def test_doctor_reports_manifold_health(tmp_path, monkeypatch):
    manifold_root = tmp_path / "manifold_index"
    manifold_root.mkdir()
    (manifold_root / "papers.json").write_text("[]", encoding="utf-8")
    (manifold_root / "experiments.json").write_text("[]", encoding="utf-8")
    _patch_runtime_ok(monkeypatch)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--runtime-config", str(Path(__file__).resolve().parents[2] / "configs" / "runtime_omo.jsonc"), "--manifold-root", str(manifold_root)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["manifold_health"]["status"] == "empty"


def test_build_manifold_writes_health_artifact(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    runner = CliRunner()
    root_dir = tmp_path / "manifold_index"
    result = runner.invoke(main, ["build-manifold", "--domain", "general", "--limit", "1", "--root", str(root_dir)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["health_path"]).exists()
