from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

from genesis.agents.runtime import CodingAgentRuntime, ProviderRuntimeError
from genesis.config import ProjectConfig
from genesis.domain_knowledge.registry import DomainKnowledgeRegistry
from genesis.harness.instruction_composer import InstructionComposer
from genesis.harness.decomposer import TaskDecomposer
from genesis.harness.token_budget import TokenBudget
from genesis.models import AdversarialReport, CheckResult, ExperimentProposal, ProjectResult, StoppingDecision
from genesis.modules.adversarial.criteria_generator import AcceptanceCriteriaGenerator
from genesis.modules.adversarial.orchestrator import AdversarialOrchestrator
from genesis.modules.ideation.greedy import GreedyAdjacencySearch
from genesis.modules.ideation.low_density import LowDensityExplorer
from genesis.modules.ideation.orchestrator import IdeationOrchestrator
from genesis.modules.ideation.pollination import PollinationSearch
from genesis.modules.optimizer.oracle_resolver import OracleResolver
from genesis.modules.optimizer.parallel import ParallelExperimentManager
from genesis.modules.optimizer.proposer import ExperimentProposer
from genesis.modules.oracle.generator import DomainOracleGenerator
from genesis.modules.verification.pipeline import VerificationPipeline
from genesis.observability import log_event
from genesis.paper.synthesizer import PaperSynthesizer
from genesis.storage.causal_dag import CausalDAG
from genesis.storage.filesystem import ProjectFilesystem
from genesis.storage.ledger import ExperimentLedger
from genesis.storage.manifold import ManifoldIndex
from genesis.taste.features import ExperimentFeatureExtractor
from genesis.taste.gp_model import TasteGP
from genesis.taste.persistence import TasteModelPersistence

from .history_reader import SelectiveHistoryReader


class MetaHarnessLoop:
    ADVERSARIAL_ITERATION_LIMIT = 3
    ESCALATION_RETRY_LIMIT = 2
    REPEATED_FAILURE_LIMIT = 3
    SCHEMA_REPAIR_LIMIT = 2
    EXECUTION_FOLLOWUP_LIMIT = 1
    STAGE_SEQUENCE = ["survey", "oracle", "execute", "verify", "paper"]

    def __init__(
        self,
        *,
        projects_root: Union[str, Path],
        taste_root: Union[str, Path],
        executor: Optional[Any] = None,
        runtime_config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.filesystem = ProjectFilesystem(projects_root)
        self.token_budget = TokenBudget()
        self.history_reader = SelectiveHistoryReader(self.filesystem, self.token_budget)
        self.composer = InstructionComposer()
        self.agent_runtime = CodingAgentRuntime(
            runtime_config_path or self.repo_root / "configs" / "runtime_omo.jsonc"
        )
        self.decomposer = TaskDecomposer(runtime=self.agent_runtime)
        self.adversarial = AdversarialOrchestrator()
        self.criteria_generator = AcceptanceCriteriaGenerator()
        self.oracle_generator = DomainOracleGenerator(runtime=self.agent_runtime)
        self.taste_persistence = TasteModelPersistence(taste_root)
        self.domain_registry = DomainKnowledgeRegistry()
        self.verification = VerificationPipeline()
        self.feature_extractor = ExperimentFeatureExtractor()
        self.executor = executor or self._default_executor

    def run(self, project_id: str, config: ProjectConfig, max_runs: int = 50) -> ProjectResult:
        project_dir = self.filesystem.init_project(project_id, config.to_dict())
        if (project_dir / "HALT.json").exists():
            halted = self.filesystem.read_json(project_dir / "HALT.json")
            self.filesystem.write_project_state(
                project_id,
                {
                    "status": "halted",
                    "run_count": self.filesystem.read_project_state(project_id).get("run_count", 0),
                    "current_stage": self.filesystem.read_project_state(project_id).get("current_stage", "survey"),
                    "last_run_status": {
                        "run_n": self.filesystem.read_project_state(project_id).get("run_count", 0),
                        "summary": str(halted.get("message", "project halted")),
                    },
                },
            )
            return ProjectResult(
                project_id=project_id,
                status="halted",
                paper_path=None,
                run_count=self.filesystem.read_project_state(project_id).get("run_count", 0),
                summary=str(halted.get("message", "project halted")),
            )
        health = self.agent_runtime.check_health(probe_models=False)
        self.filesystem.write_json(project_dir / "runtime_health.json", health)
        if not health.get("passed", False):
            self.filesystem.write_halt(
                project_id,
                {"type": "RUNTIME_HEALTHCHECK_FAILED", "checks": health.get("checks", [])},
            )
            return ProjectResult(
                project_id=project_id,
                status="halted",
                paper_path=None,
                run_count=0,
                summary="halted due to runtime healthcheck failure",
            )
        try:
            ledger = ExperimentLedger(project_dir / "experiments" / "ledger.sqlite3")
        except Exception as exc:  # noqa: BLE001
            self.filesystem.write_halt(
                project_id,
                {"type": "LEDGER_CORRUPTION", "message": str(exc)},
            )
            raise
        global_manifold = self.repo_root / "manifold_index"
        manifold = ManifoldIndex(global_manifold if global_manifold.exists() else project_dir / "knowledge" / "manifold")
        manifold_health = manifold.assess_health()
        ideation_available = bool(manifold_health.ready_modes)
        self.filesystem.write_json(project_dir / "knowledge" / "manifold_health.json", manifold_health.to_dict())
        global_dag_path = self.taste_persistence.root_dir / "causal_dag_global.json"
        if global_dag_path.exists():
            CausalDAG(project_dir / "causal_dag.json").merge_global_dag(self.filesystem.read_json(global_dag_path))
        optimizer = ParallelExperimentManager(project_dir / "runtime" / "sandboxes")
        ideation = IdeationOrchestrator(
            greedy=GreedyAdjacencySearch(manifold),
            pollination=PollinationSearch(manifold),
            low_density=LowDensityExplorer(manifold),
        )
        log_path = project_dir / "genesis.log"
        criteria = self.criteria_generator.generate(config)["criteria"]
        self.filesystem.write_json(project_dir / "adversarial_criteria.json", {"criteria": criteria})
        decomposition = self.decomposer.decompose(config)
        self.filesystem.write_json(project_dir / "decomposition.json", decomposition.to_dict())
        domain_provider = self.domain_registry.get_provider(config.domain)
        domain_context = domain_provider.initialize(config.to_dict())
        (project_dir / "knowledge" / "domain_context.md").write_text(domain_context, encoding="utf-8")
        oracle_path = project_dir / "knowledge" / "oracle.py"
        taste_model = self.taste_persistence.load_for_project(project_id, project_dir / "knowledge" / "taste_snapshot.json")
        failed_iterations = 0
        redirect_message = ""
        run_n = 0
        stopping_criteria_satisfied = False
        repeated_failure_counts: dict[str, int] = {}
        state = self.filesystem.read_project_state(project_id)
        task_states = self._initialize_task_states(state.get("task_states"), decomposition)
        current_stage = str(state.get("current_stage") or self.STAGE_SEQUENCE[0])
        latest_execute_result: dict[str, Any] | None = None

        result_summary = "unfinished"
        for run_n in range(1, max_runs + 1):
            intervention = self.filesystem.read_human_intervention(project_id)
            if intervention:
                if intervention.get("type") == "STOP":
                    result_summary = "stopped by human intervention"
                    self.filesystem.clear_human_intervention(project_id)
                    break
                if intervention.get("type") == "APPROVE":
                    failed_iterations = 0
                    redirect_message = ""
                if intervention.get("type") == "REDIRECT":
                    redirect_message = str(intervention.get("message", "")).strip()
                if intervention.get("type") == "REJECT":
                    failed_iterations += 1
                self.filesystem.clear_human_intervention(project_id)
            task_states = self._refresh_task_states(project_dir, task_states, decomposition)
            task_node = self._select_next_task(decomposition, task_states)
            if task_node is None:
                result_summary = "all task DAG nodes completed"
                stopping_criteria_satisfied = True
                break
            current_stage = self._stage_for_task(task_node)
            budget_allocations = self.token_budget.allocate(
                128000 if "cloud" in config.compute_budget.lower() or "groq" in config.compute_budget.lower() else 18000
            )
            history = self.history_reader.summarize_experiment_history(project_id)
            if self._ideation_required(failed_iterations=failed_iterations, current_stage=current_stage) and not ideation_available:
                self.filesystem.write_halt(
                    project_id,
                    {
                        "type": "MANIFOLD_HEALTH_REQUIRED",
                        "message": "Ideation was required but the manifold is not healthy enough.",
                        "reasons": manifold_health.reasons,
                        "run_n": run_n,
                    },
                )
                result_summary = "halted due to manifold health requirements"
                break
            requested_modules = self._requested_modules(
                config=config,
                task_node=task_node,
                failed_iterations=failed_iterations,
                current_stage=current_stage,
            )
            stage_domain_context = domain_context
            if hasattr(domain_provider, "get_relevant_context") and current_stage in {"survey", "execute"}:
                try:
                    query = task_node.description if task_node else config.research_question
                    stage_domain_context = domain_provider.get_relevant_context(query)
                except Exception:  # noqa: BLE001
                    stage_domain_context = domain_context
            current_task_context = self._task_context(
                config=config,
                run_n=run_n,
                failed_iterations=failed_iterations,
                redirect_message=redirect_message,
                current_stage=current_stage,
            )
            instruction = self.composer.compose(
                config=config,
                belief_summary=f"tracked_experiments={len(ledger.get_pareto_frontier())}",
                retrieved_history=history,
                domain_context=stage_domain_context,
                current_task_context=current_task_context,
                budget_allocations=budget_allocations,
                requested_modules=requested_modules,
                mode="initial_task_prompt",
                workspace_root=str(self.filesystem.get_run_dir(project_id, run_n) / "workspace"),
                expected_artifacts=list(getattr(task_node, "expected_artifacts", []) or []),
                task_kind=str(getattr(task_node, "task_kind", current_stage)),
            )
            self.filesystem.write_instruction(project_id, run_n, instruction)
            run_dir = self.filesystem.get_run_dir(project_id, run_n)
            self._mark_task_state(
                task_states,
                task_node.task_id,
                status="running",
                run_n=run_n,
                stage=current_stage,
                blockers=[],
            )
            if current_stage == "oracle":
                oracle_source = self.oracle_generator.generate(config)
                oracle_path.write_text(oracle_source, encoding="utf-8")
                oracle_validation = self.verification.oracle_validator.validate_with_synthetic_data(oracle_path)
                result_payload = {
                    "task_id": task_node.task_id if task_node else "oracle",
                    "stage": current_stage,
                    "summary": "Generated and validated project oracle.",
                    "primary_metric": 1.0 if oracle_validation.get("passed") else 0.0,
                    "code_path": str(oracle_path),
                    "artifact_dir": str(oracle_path.parent),
                    "generated_artifacts": [str(oracle_path)],
                    "executed_commands": [],
                    "command_results": [],
                    "classification": "success" if oracle_validation.get("passed") else "oracle_validation_failed",
                    "failure_type": "" if oracle_validation.get("passed") else "oracle_validation_failed",
                    "failure_summary": "" if oracle_validation.get("passed") else "synthetic validation failed",
                    "failed_command": "",
                    "failure_signature": "" if oracle_validation.get("passed") else "oracle_validation_failed",
                    "debug_focus": self._debug_focus(project_dir, run_n + 1),
                    "citations": [],
                    "agent_runtime": {
                        "provider": None,
                        "model": None,
                        "primary_model": None,
                        "attempted_models": [],
                        "fallback_used": False,
                        "next_action": "continue",
                    },
                    "errors": [],
                    "status": "keep" if oracle_validation.get("passed") else "discard",
                }
                report = self._stage_report(current_stage, success=bool(oracle_validation.get("passed")), reason="oracle_ready")
                verification = {"passed": bool(oracle_validation.get("passed")), "checks": [oracle_validation]}
                report = self._run_adversarial_check(
                    result_payload,
                    list(getattr(task_node, "acceptance_criteria", []) or criteria),
                    task_context=self._adversarial_task_context(task_node, current_stage, project_dir),
                    verification=verification,
                    oracle_result=oracle_validation,
                )
                result_payload["adversarial_critical_blockers"] = report.critical_blockers
                self.filesystem.write_json(run_dir / "trace.json", {"stage": current_stage, "oracle_path": str(oracle_path)})
                self.filesystem.write_json(run_dir / "result.json", result_payload)
                self.filesystem.write_json(run_dir / "adversarial_report.json", report.to_dict())
                self.filesystem.write_json(run_dir / "verification_report.json", verification)
                if not oracle_validation.get("passed"):
                    self.filesystem.write_halt(
                        project_id,
                        {"type": "ORACLE_VALIDATION_FAILED", "message": json.dumps(oracle_validation), "run_n": run_n},
                    )
                    result_summary = "halted due to oracle validation failure"
                    break
                if report.critical_blockers:
                    failed_iterations += 1
                    redirect_message = self._format_adversarial_blockers(report.critical_blockers)
                    self._mark_task_state(
                        task_states,
                        task_node.task_id,
                        status="blocked",
                        run_n=run_n,
                        stage=current_stage,
                        blockers=report.critical_blockers,
                    )
                else:
                    failed_iterations = 0
                    redirect_message = ""
                    self._mark_task_state(
                        task_states,
                        task_node.task_id,
                        status="completed",
                        run_n=run_n,
                        stage=current_stage,
                        blockers=[],
                    )
                self.filesystem.write_project_state(
                    project_id,
                    {
                        "status": "running",
                        "run_count": run_n,
                        "current_stage": current_stage,
                        "task_states": task_states,
                        "ideation_available": ideation_available,
                        "manifold_status": manifold_health.status,
                        "ready_ideation_modes": manifold_health.ready_modes,
                        "last_run_status": self._build_run_status(
                            run_n=run_n,
                            summary="oracle ready" if not report.critical_blockers else "oracle blocked by adversarial findings",
                            stopping_decision=report.stopping_decision.to_dict(),
                            verification=verification,
                            result=result_payload,
                        ),
                    },
                )
                continue
            if current_stage == "verify":
                latest_execute_result = latest_execute_result or self._latest_stage_result(project_dir, "execute")
                if latest_execute_result is None:
                    self.filesystem.write_halt(
                        project_id,
                        {"type": "VERIFY_WITHOUT_EXECUTION", "message": "verify stage has no execute-stage artifacts", "run_n": run_n},
                    )
                    result_summary = "halted due to missing execute-stage artifacts"
                    break
                verification = self.verification.run(
                    Path(latest_execute_result["artifact_dir"]),
                    project_id,
                    oracle_path=oracle_path if oracle_path.exists() else None,
                    task_kind=str(getattr(task_node, "task_kind", current_stage)),
                    expected_artifacts=list(getattr(task_node, "expected_artifacts", []) or []),
                )
                report = self._run_adversarial_check(
                    latest_execute_result,
                    list(getattr(task_node, "acceptance_criteria", []) or criteria),
                    task_context=self._adversarial_task_context(task_node, current_stage, project_dir),
                    verification=verification,
                    oracle_result=self.verification.oracle_validator.validate_with_synthetic_data(oracle_path) if oracle_path.exists() else None,
                )
                result_payload = dict(latest_execute_result)
                result_payload["stage"] = current_stage
                result_payload["summary"] = f"Verification review for execute-stage artifacts. {result_payload.get('summary', '')}".strip()
                result_payload["adversarial_critical_blockers"] = report.critical_blockers
                self.filesystem.write_json(run_dir / "trace.json", {"stage": current_stage, "source_run_stage": "execute"})
                self.filesystem.write_json(run_dir / "result.json", result_payload)
                self.filesystem.write_json(run_dir / "adversarial_report.json", report.to_dict())
                self.filesystem.write_json(run_dir / "verification_report.json", verification)
                log_event(
                    log_path,
                    project_id=project_id,
                    run_n=run_n,
                    component="meta_harness",
                    event_type="run_completed",
                    payload={"acceptance_ratio": report.acceptance_ratio, "verification_passed": verification["passed"]},
                )
                if self._check_stopping_criteria(report) and verification["passed"]:
                    result_summary = "stopping criteria satisfied"
                    stopping_criteria_satisfied = True
                    self._mark_task_state(
                        task_states,
                        task_node.task_id,
                        status="completed",
                        run_n=run_n,
                        stage=current_stage,
                        blockers=[],
                    )
                    self.filesystem.write_project_state(
                        project_id,
                        {
                            "status": "complete",
                            "run_count": run_n,
                            "current_stage": current_stage,
                            "task_states": task_states,
                            "last_run_status": self._build_run_status(
                                run_n=run_n,
                                summary=result_summary,
                                stopping_decision=report.stopping_decision.to_dict(),
                                verification=verification,
                                result=result_payload,
                            ),
                        },
                    )
                    break
                failed_iterations += 1
                repeated_failure_counts = self._update_repeated_failures(repeated_failure_counts, result_payload)
                escalation_attempts = max(0, failed_iterations - self.ADVERSARIAL_ITERATION_LIMIT)
                self._mark_task_state(
                    task_states,
                    task_node.task_id,
                    status="blocked",
                    run_n=run_n,
                    stage=current_stage,
                    blockers=report.critical_blockers or ["verification_failed"],
                )
                self.filesystem.write_project_state(
                    project_id,
                    {
                        "status": "running",
                        "run_count": run_n,
                        "current_stage": current_stage,
                        "task_states": task_states,
                        "ideation_available": ideation_available,
                        "manifold_status": manifold_health.status,
                        "ready_ideation_modes": manifold_health.ready_modes,
                        "escalation_attempts": escalation_attempts,
                        "last_run_status": self._build_run_status(
                            run_n=run_n,
                            summary="verification failed; returning to execute stage",
                            stopping_decision=report.stopping_decision.to_dict(),
                            verification=verification,
                            result=result_payload,
                        ),
                    },
                )
                if failed_iterations >= self.ADVERSARIAL_ITERATION_LIMIT:
                    redirect_message = (
                        "Change approach: prior attempts failed verification/adversarial gating. "
                        "Use a materially different tactic and surface the rationale."
                    )
                    stubborn_failure = self._repeated_failure_message(repeated_failure_counts)
                    if stubborn_failure:
                        redirect_message += (
                            f" Avoid repeating failure pattern: {stubborn_failure}. "
                            "First diagnose the root cause from the recorded stderr, then propose a repaired or alternate branch."
                        )
                    self.filesystem.write_json(
                        run_dir / "escalation_report.json",
                        {
                            "failed_iterations": failed_iterations,
                            "escalation_attempts": escalation_attempts,
                            "message": redirect_message,
                        },
                    )
                if failed_iterations >= 5:
                    ideation_result = ideation.run_with_status(config.research_question, failed_iterations, taste_model=taste_model)
                    self.filesystem.write_json(run_dir / "ideation_report.json", ideation_result.to_dict())
                if escalation_attempts >= self.ESCALATION_RETRY_LIMIT:
                    self.filesystem.write_halt(
                        project_id,
                        {
                            "type": "ADVERSARIAL_STALEMATE",
                            "message": "Exceeded adversarial retry threshold after escalation.",
                            "run_n": run_n,
                        },
                    )
                    result_summary = "halted due to adversarial stalemate"
                    break
                continue
            if current_stage == "paper":
                paper = PaperSynthesizer(self.filesystem.base_dir, runtime=self.agent_runtime).synthesize(
                    project_id,
                    final=all(self._task_state_by_id(task_states, task.task_id)["status"] == "completed" for task in decomposition.tasks if task.task_id != task_node.task_id),
                    completion_reason="paper task synthesis",
                    project_status="running",
                )
                result_payload = {
                    "task_id": task_node.task_id,
                    "stage": current_stage,
                    "summary": "Synthesized project paper artifacts.",
                    "primary_metric": 1.0,
                    "code_path": paper["latex_path"],
                    "artifact_dir": str(Path(paper["latex_path"]).parent),
                    "generated_artifacts": [paper["latex_path"], paper["pdf_path"]],
                    "executed_commands": [],
                    "command_results": [],
                    "classification": "success",
                    "failure_type": "",
                    "failure_summary": "",
                    "failed_command": "",
                    "failure_signature": "",
                    "debug_focus": self._debug_focus(project_dir, run_n + 1),
                    "citations": [],
                    "agent_runtime": {
                        "provider": None,
                        "model": None,
                        "primary_model": None,
                        "attempted_models": [],
                        "fallback_used": False,
                        "next_action": "continue",
                    },
                    "errors": [],
                    "status": "keep",
                }
                verification = {"passed": True, "checks": []}
                report = self._run_adversarial_check(
                    result_payload,
                    list(getattr(task_node, "acceptance_criteria", []) or criteria),
                    task_context=self._adversarial_task_context(task_node, current_stage, project_dir),
                    verification=verification,
                    oracle_result=None,
                )
                result_payload["adversarial_critical_blockers"] = report.critical_blockers
                self.filesystem.write_json(run_dir / "trace.json", {"stage": current_stage, "paper_path": paper["pdf_path"]})
                self.filesystem.write_json(run_dir / "result.json", result_payload)
                self.filesystem.write_json(run_dir / "adversarial_report.json", report.to_dict())
                self.filesystem.write_json(run_dir / "verification_report.json", verification)
                if report.critical_blockers:
                    self._mark_task_state(task_states, task_node.task_id, status="blocked", run_n=run_n, stage=current_stage, blockers=report.critical_blockers)
                    failed_iterations += 1
                    redirect_message = self._format_adversarial_blockers(report.critical_blockers)
                else:
                    self._mark_task_state(task_states, task_node.task_id, status="completed", run_n=run_n, stage=current_stage, blockers=[])
                    failed_iterations = 0
                    redirect_message = ""
                self.filesystem.write_project_state(
                    project_id,
                    {
                        "status": "running",
                        "run_count": run_n,
                        "current_stage": current_stage,
                        "task_states": task_states,
                        "ideation_available": ideation_available,
                        "manifold_status": manifold_health.status,
                        "ready_ideation_modes": manifold_health.ready_modes,
                        "last_run_status": self._build_run_status(
                            run_n=run_n,
                            summary="paper synthesized" if not report.critical_blockers else "paper blocked by adversarial findings",
                            stopping_decision=report.stopping_decision.to_dict(),
                            verification=verification,
                            result=result_payload,
                        ),
                    },
                )
                continue
            try:
                execution = self.executor(
                    project_dir=project_dir,
                    run_n=run_n,
                    config=config,
                    task_node=task_node,
                    instruction=instruction,
                    optimizer=optimizer,
                    ledger=ledger,
                    ideation=ideation,
                    oracle_resolver=OracleResolver(),
                    failed_iterations=failed_iterations,
                    taste_model=taste_model,
                )
            except ProviderRuntimeError as exc:
                self.filesystem.write_halt(
                    project_id,
                    {
                        "type": "PROVIDER_FAILURE",
                        "message": str(exc),
                        "run_n": run_n,
                    },
                )
                result_summary = "halted due to provider failure"
                break
            except Exception as exc:  # noqa: BLE001
                self.filesystem.write_halt(
                    project_id,
                    {
                        "type": "UNHANDLED_RUNTIME_FAILURE",
                        "message": str(exc),
                        "run_n": run_n,
                    },
                )
                result_summary = "halted due to unhandled runtime failure"
                break
            execution["trace"]["stage"] = current_stage
            execution["result"]["stage"] = current_stage
            self.filesystem.write_json(run_dir / "trace.json", execution["trace"])
            self.filesystem.write_json(run_dir / "result.json", execution["result"])
            verification = self.verification.run(
                Path(execution["result"]["artifact_dir"]),
                project_id,
                oracle_path=oracle_path if oracle_path.exists() and current_stage == "execute" else None,
                task_kind=str(getattr(task_node, "task_kind", current_stage)),
                expected_artifacts=list(getattr(task_node, "expected_artifacts", []) or []),
            )
            report = self._run_adversarial_check(
                execution["result"],
                list(getattr(task_node, "acceptance_criteria", []) or criteria),
                task_context=self._adversarial_task_context(task_node, current_stage, project_dir),
                verification=verification,
                oracle_result=None,
            )
            self.filesystem.write_json(run_dir / "adversarial_report.json", report.to_dict())
            self.filesystem.write_json(run_dir / "verification_report.json", verification)
            latest_execute_result = execution["result"] if current_stage == "execute" else latest_execute_result
            execution["result"]["adversarial_critical_blockers"] = report.critical_blockers
            execution["result"]["verification_passed"] = bool(verification.get("passed", False))
            execution["result"]["task_kind"] = getattr(task_node, "task_kind", current_stage)
            if self._stage_success(current_stage, execution["result"]) and verification["passed"] and not report.critical_blockers:
                failed_iterations = 0
                redirect_message = ""
                self._mark_task_state(
                    task_states,
                    task_node.task_id,
                    status="completed",
                    run_n=run_n,
                    stage=current_stage,
                    blockers=[],
                    verification_passed=True,
                    adversarial_passed=True,
                    attempt_phase=str(execution["result"].get("attempt_phase", "")),
                    repair_count=int(execution["result"].get("repair_count", 0)),
                    schema_blockers=list(execution["result"].get("schema_blockers", [])),
                    block_reason="",
                )
                result_summary = f"{execution['result'].get('stage', current_stage)} stage complete"
            else:
                failed_iterations += 1
                repeated_failure_counts = self._update_repeated_failures(repeated_failure_counts, execution["result"])
                if report.critical_blockers:
                    redirect_message = self._format_adversarial_blockers(report.critical_blockers)
                elif execution["result"].get("classification") == "plan_materialized":
                    redirect_message = "Execute the existing plan now and produce substantive artifacts for this same task."
                elif execution["result"].get("classification") == "repairable_schema_mismatch":
                    redirect_message = "Repair the schema mismatch and reissue corrected executable JSON for the same task."
                self._mark_task_state(
                    task_states,
                    task_node.task_id,
                    status="blocked",
                    run_n=run_n,
                    stage=current_stage,
                    blockers=report.critical_blockers or [execution["result"].get("classification", "task_failed")],
                    verification_passed=bool(verification.get("passed", False)),
                    adversarial_passed=not bool(report.critical_blockers),
                    attempt_phase=str(execution["result"].get("attempt_phase", "")),
                    repair_count=int(execution["result"].get("repair_count", 0)),
                    schema_blockers=list(execution["result"].get("schema_blockers", [])),
                    block_reason=str(execution["result"].get("failure_type", execution["result"].get("classification", "blocked"))),
                )
                result_summary = "continue iteration"
            self.filesystem.write_project_state(
                project_id,
                {
                    "status": "running",
                    "run_count": run_n,
                    "current_stage": current_stage,
                    "task_states": task_states,
                    "ideation_available": ideation_available,
                    "manifold_status": manifold_health.status,
                    "ready_ideation_modes": manifold_health.ready_modes,
                    "last_run_status": self._build_run_status(
                        run_n=run_n,
                        summary=result_summary,
                        stopping_decision=report.stopping_decision.to_dict(),
                        verification=verification,
                        result=execution["result"],
                    ),
                },
            )
        if (project_dir / "HALT.json").exists():
            self.filesystem.write_project_state(
                project_id,
                {
                    "status": "halted",
                    "run_count": run_n,
                    "current_stage": current_stage,
                    "task_states": task_states,
                    "ideation_available": ideation_available,
                    "manifold_status": manifold_health.status,
                    "last_run_status": {
                        "run_n": run_n,
                        "summary": result_summary,
                    },
                },
            )
            return ProjectResult(
                project_id=project_id,
                status="halted",
                paper_path=None,
                run_count=run_n,
                summary=result_summary,
            )
        if not stopping_criteria_satisfied and result_summary == "unfinished":
            result_summary = "max runs exhausted before stopping criteria were satisfied"
        project_status = "complete" if stopping_criteria_satisfied else "incomplete"
        paper = PaperSynthesizer(self.filesystem.base_dir, runtime=self.agent_runtime).synthesize(
            project_id,
            final=stopping_criteria_satisfied,
            completion_reason=result_summary,
            project_status=project_status,
        )
        verified_experiments = [
            experiment
            for experiment in ledger.get_pareto_frontier()
            if str(experiment.get("status", "")).lower() in {"keep", "success"}
        ]
        self._update_causal_dag(project_dir / "causal_dag.json", project_id, config.domain, verified_experiments)
        self.taste_persistence.save_after_project(project_id, taste_model)
        self.taste_persistence.merge_project_data(project_id, verified_experiments)
        self.filesystem.write_project_state(
            project_id,
            {
                "status": project_status,
                "run_count": run_n,
                "current_stage": current_stage,
                "task_states": task_states,
                "ideation_available": ideation_available,
                "manifold_status": manifold_health.status,
                "ready_ideation_modes": manifold_health.ready_modes,
                "last_run_status": {
                    "run_n": run_n,
                    "summary": result_summary,
                    "stopping_criteria_satisfied": stopping_criteria_satisfied,
                },
                "paper_path": paper["pdf_path"],
            },
        )
        return ProjectResult(
            project_id=project_id,
            status=project_status,
            paper_path=paper["pdf_path"],
            run_count=run_n,
            summary=result_summary,
        )

    def _default_executor(
        self,
        *,
        project_dir: Path,
        run_n: int,
        config: ProjectConfig,
        task_node: Any,
        instruction: str,
        optimizer: ParallelExperimentManager,
        ledger: ExperimentLedger,
        ideation: IdeationOrchestrator,
        oracle_resolver: OracleResolver,
        failed_iterations: int,
        taste_model: Optional[TasteGP] = None,
    ) -> dict[str, Any]:
        active_task = task_node.description if task_node else config.research_question
        run_dir = self.filesystem.get_run_dir(project_dir.name, run_n)
        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        workspace_root = self._workspace_root(artifact_dir)
        workspace_root.mkdir(parents=True, exist_ok=True)
        summary_parts = [f"Run {run_n} investigates {active_task}."]
        base_context = {
            "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
            "task_kind": getattr(task_node, "task_kind", self._stage_for_task(task_node)),
            "execution_mode": getattr(task_node, "execution_mode", "artifact_generation"),
            "expected_artifacts": list(getattr(task_node, "expected_artifacts", []) or []),
            "workspace_root": str(workspace_root),
            "artifacts_root": str(artifact_dir),
            "research_question": config.research_question,
            "domain": config.domain,
            "success_criteria": config.success_criteria,
            "failed_iterations": failed_iterations,
            **self._execution_context(project_dir, run_n),
        }
        attempt_phase = "plan_only"
        repair_count = 0
        schema_blockers: list[str] = []
        agent_result: dict[str, Any] | None = None
        execution_followups = 0
        prompt = instruction

        while True:
            try:
                attempt_phase = "executing"
                agent_result = self.agent_runtime.generate_task(
                    category="sisyphus",
                    instruction=prompt,
                    context=base_context | {"attempt_phase": attempt_phase, "schema_blockers": schema_blockers},
                    budget={"max_runs": 1, "compute_budget": config.compute_budget},
                )
            except ProviderRuntimeError as exc:
                if self._is_repairable_schema_error(exc.error_class) and repair_count < self.SCHEMA_REPAIR_LIMIT:
                    repair_count += 1
                    attempt_phase = "repair_requested"
                    schema_blockers = [self._schema_blocker_from_error(exc)]
                    prompt = self.composer.compose(
                        config=config,
                        belief_summary=f"tracked_experiments={len(ledger.get_pareto_frontier())}",
                        retrieved_history=self.history_reader.summarize_experiment_history(project_dir.name),
                        domain_context="",
                        current_task_context=self._task_context(
                            config=config,
                            run_n=run_n,
                            failed_iterations=failed_iterations,
                            redirect_message="Repair the command/artifact schema mismatch for this same task.",
                            current_stage=self._stage_for_task(task_node),
                        ),
                        budget_allocations={"retrieved_history": 4000},
                        requested_modules=["executor", "verification"],
                        mode="repair_prompt",
                        workspace_root=str(workspace_root),
                        expected_artifacts=list(getattr(task_node, "expected_artifacts", []) or []),
                        schema_blockers=schema_blockers,
                        task_kind=str(getattr(task_node, "task_kind", self._stage_for_task(task_node))),
                    )
                    continue
                if exc.error_class == "non_actionable_plan":
                    return self._executor_result_wrapper(
                        active_task=active_task,
                        failed_iterations=failed_iterations,
                        run_n=run_n,
                        task_node=task_node,
                        summary_parts=summary_parts,
                        artifact_dir=artifact_dir,
                        result={
                            "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
                            "summary": " ".join(summary_parts).strip(),
                            "primary_metric": 0.0,
                            "code_path": "",
                            "errors": [],
                            "artifact_dir": str(artifact_dir),
                            "generated_artifacts": [],
                            "executed_commands": [],
                            "command_results": [],
                            "classification": "non_actionable_plan",
                            "failure_type": exc.error_class,
                            "failure_summary": str(exc),
                            "failed_command": "",
                            "failure_signature": exc.error_class,
                            "debug_focus": self._debug_focus(project_dir, run_n + 1),
                            "citations": [],
                            "agent_runtime": {
                                "provider": None,
                                "model": None,
                                "primary_model": None,
                                "attempted_models": [],
                                "fallback_used": False,
                                "next_action": "continue",
                            },
                            "attempt_phase": attempt_phase,
                            "repair_count": repair_count,
                            "schema_blockers": schema_blockers,
                            "resolved_workspace": str(workspace_root),
                            "status": "discard",
                        },
                    )
                raise

            if str(agent_result.get("validation_mode", "")) == "relaxed_plan_only":
                plan_payload = self._materialize_relaxed_plan_payload(
                    project_dir=project_dir,
                    artifact_dir=artifact_dir,
                    task_node=task_node,
                    run_n=run_n,
                    summary_parts=summary_parts,
                    agent_result=agent_result,
                    attempt_phase="plan_only",
                    repair_count=repair_count,
                    resolved_workspace=str(workspace_root),
                )
                if execution_followups < self.EXECUTION_FOLLOWUP_LIMIT:
                    execution_followups += 1
                    prompt = self.composer.compose(
                        config=config,
                        belief_summary=f"tracked_experiments={len(ledger.get_pareto_frontier())}",
                        retrieved_history=self.history_reader.summarize_experiment_history(project_dir.name),
                        domain_context="",
                        current_task_context=self._task_context(
                            config=config,
                            run_n=run_n,
                            failed_iterations=failed_iterations,
                            redirect_message="The plan exists. Execute it now and materialize real helper files and final artifacts.",
                            current_stage=self._stage_for_task(task_node),
                        ),
                        budget_allocations={"retrieved_history": 4000},
                        requested_modules=["executor", "verification"],
                        mode="execution_followup_prompt",
                        workspace_root=str(workspace_root),
                        expected_artifacts=list(getattr(task_node, "expected_artifacts", []) or []),
                        task_kind=str(getattr(task_node, "task_kind", self._stage_for_task(task_node))),
                    )
                    continue
                return self._executor_result_wrapper(
                    active_task=active_task,
                    failed_iterations=failed_iterations,
                    run_n=run_n,
                    task_node=task_node,
                    summary_parts=summary_parts,
                    artifact_dir=artifact_dir,
                    result=plan_payload,
                )

            generic_result = self._execute_agent_work(
                project_dir=project_dir,
                run_n=run_n,
                task_node=task_node,
                summary_parts=summary_parts,
                agent_result=agent_result,
                artifact_dir=artifact_dir,
                attempt_phase=attempt_phase,
                repair_count=repair_count,
            )
            if generic_result is not None and self._is_repairable_schema_failure(generic_result) and repair_count < self.SCHEMA_REPAIR_LIMIT:
                repair_count += 1
                attempt_phase = "repair_requested"
                schema_blockers = [self._schema_blocker_from_result(generic_result)]
                prompt = self.composer.compose(
                    config=config,
                    belief_summary=f"tracked_experiments={len(ledger.get_pareto_frontier())}",
                    retrieved_history=self.history_reader.summarize_experiment_history(project_dir.name),
                    domain_context="",
                    current_task_context=self._task_context(
                        config=config,
                        run_n=run_n,
                        failed_iterations=failed_iterations,
                        redirect_message="Repair the command/artifact schema mismatch for this same task.",
                        current_stage=self._stage_for_task(task_node),
                    ),
                    budget_allocations={"retrieved_history": 4000},
                    requested_modules=["executor", "verification"],
                    mode="repair_prompt",
                    workspace_root=str(workspace_root),
                    expected_artifacts=list(getattr(task_node, "expected_artifacts", []) or []),
                    schema_blockers=schema_blockers,
                    task_kind=str(getattr(task_node, "task_kind", self._stage_for_task(task_node))),
                )
                continue
            break

        assert agent_result is not None
        if generic_result is not None:
            artifact_payload = generic_result
        elif self._should_use_optimizer(task_node=task_node, config=config, agent_result=agent_result):
            proposals = self._resolve_experiment_proposals(
                task_node=task_node,
                config=config,
                ledger=ledger,
                agent_result=agent_result,
            )
            if taste_model and taste_model.training_targets:
                features = [self.feature_extractor.extract(proposal) for proposal in proposals]
                predicted_means, predicted_variances = taste_model.predict(features)
                for proposal, mean, variance in zip(proposals, predicted_means, predicted_variances):
                    proposal.expected_metric = round(max(mean, proposal.expected_metric), 4)
                    proposal.expected_trajectory = [
                        round(max(mean - variance, 0.0), 4),
                        round(mean, 4),
                        round(mean + variance, 4),
                    ]
            oracle_spec = oracle_resolver.resolve_oracle(task_node, config)
            experiment_results = optimizer.run_batch(
                [
                    {
                        "experiment_id": f"{task_node.task_id}-{run_n}-{index}",
                        "task_id": task_node.task_id,
                        "primary_metric": proposal.expected_metric,
                        "trajectory": proposal.expected_trajectory,
                        "keep_threshold": 0.4,
                        "baseline_metric": proposal.expected_trajectory[0],
                        "step_gain": 0.05,
                        "epochs": len(proposal.expected_trajectory),
                        "command": proposal.command,
                    }
                    for index, proposal in enumerate(proposals, start=1)
                ],
                n_parallel=3,
            )
            if taste_model and taste_model.training_targets:
                predicted_trajectories, predicted_variances = taste_model.predict_trajectory(
                    [self.feature_extractor.extract(proposal) for proposal in proposals]
                )
                for result, predicted, variances in zip(experiment_results, predicted_trajectories, predicted_variances):
                    result.anomaly_score = self._trajectory_anomaly_score(result.trajectory, predicted, variances)
            best = max(experiment_results, key=lambda result: result.primary_metric)
            for proposal, result in zip(proposals, experiment_results):
                ledger.insert_experiment(
                    result,
                    config_diff=proposal.code_diff,
                    timestamp=f"run-{run_n}",
                )
            if taste_model:
                training_examples = [
                    (proposal, result)
                    for proposal, result in zip(proposals, experiment_results)
                    if proposal.command
                    and str(result.status).lower() in {"keep", "success"}
                    and bool(result.trajectory)
                    and not str(result.artifact_path).lower().endswith(("stderr.log", "missing_command.txt", "missing_output.txt"))
                ]
                if training_examples:
                    taste_model.fit(
                        [self.feature_extractor.extract(proposal) for proposal, _ in training_examples],
                        [result.primary_metric for _, result in training_examples],
                        [result.trajectory for _, result in training_examples],
                    )
            summary_parts.append(
                f"Optimizer ran {len(experiment_results)} experiments for metric {oracle_spec.metric_name} and kept {best.experiment_id}."
            )
            artifact_payload = {
                "task_id": task_node.task_id,
                "summary": " ".join(summary_parts + [agent_result.get("summary", "")]).strip(),
                "primary_metric": best.primary_metric,
                "code_path": best.artifact_path,
                "errors": [],
                "artifact_dir": str(artifact_dir),
                "selected_experiment": best.to_dict(),
                "classification": "success",
                "failure_type": "",
                "failure_summary": "",
                "failed_command": "",
                "failure_signature": "",
                "debug_focus": self._debug_focus(project_dir, run_n + 1),
                "citations": agent_result.get("citations", []),
                "agent_runtime": {
                    "provider": agent_result.get("provider"),
                    "model": agent_result.get("model"),
                    "primary_model": agent_result.get("primary_model"),
                    "attempted_models": agent_result.get("attempted_models", []),
                    "fallback_used": agent_result.get("fallback_used", False),
                    "next_action": agent_result.get("next_action"),
                },
                "attempt_phase": "executed_substantively",
                "repair_count": repair_count,
                "schema_blockers": schema_blockers,
                "resolved_workspace": str(workspace_root),
                "artifact_records": [],
                "generated_artifacts": [best.artifact_path],
                "executed_commands": [],
                "command_results": [],
                "status": "keep",
            }
        elif self._has_artifact_attempt(agent_result):
            artifact_payload = {
                "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
                "summary": " ".join(summary_parts + [agent_result.get("summary", "")]).strip(),
                "primary_metric": 0.0,
                "code_path": "",
                "errors": [],
                "artifact_dir": str(artifact_dir),
                "generated_artifacts": [],
                "artifact_records": self._substantive_files(artifact_dir),
                "executed_commands": [],
                "command_results": [],
                "classification": "non_substantive_completion",
                "failure_type": "non_substantive_completion",
                "failure_summary": "artifact plan produced only empty or whitespace-only files",
                "failed_command": "",
                "failure_signature": "non_substantive_completion",
                "debug_focus": self._debug_focus(project_dir, run_n + 1),
                "citations": agent_result.get("citations", []),
                "agent_runtime": {
                    "provider": agent_result.get("provider"),
                    "model": agent_result.get("model"),
                    "primary_model": agent_result.get("primary_model"),
                    "attempted_models": agent_result.get("attempted_models", []),
                    "fallback_used": agent_result.get("fallback_used", False),
                    "next_action": agent_result.get("next_action"),
                },
                "attempt_phase": "blocked",
                "repair_count": repair_count,
                "schema_blockers": schema_blockers,
                "resolved_workspace": str(workspace_root),
                "status": "discard",
            }
        else:
            failure_type = "non_actionable_plan"
            failure_summary = "model returned no files, commands, or usable experiment plan"
            if self._next_action_requires_verified_work(str(agent_result.get("next_action", ""))):
                failure_type = "invalid_next_action"
                failure_summary = "model proposed publication/finalization without substantive verified artifacts"
            artifact_payload = {
                "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
                "summary": " ".join(summary_parts + [agent_result.get("summary", "")]).strip(),
                "primary_metric": 0.0,
                "code_path": "",
                "errors": [],
                "artifact_dir": str(artifact_dir),
                "generated_artifacts": [],
                "executed_commands": [],
                "command_results": [],
                "classification": failure_type,
                "failure_type": failure_type,
                "failure_summary": failure_summary,
                "failed_command": "",
                "failure_signature": failure_type,
                "debug_focus": self._debug_focus(project_dir, run_n + 1),
                "citations": agent_result.get("citations", []),
                "agent_runtime": {
                    "provider": agent_result.get("provider"),
                    "model": agent_result.get("model"),
                    "primary_model": agent_result.get("primary_model"),
                    "attempted_models": agent_result.get("attempted_models", []),
                    "fallback_used": agent_result.get("fallback_used", False),
                    "next_action": agent_result.get("next_action"),
                },
                "attempt_phase": "blocked",
                "repair_count": repair_count,
                "schema_blockers": schema_blockers,
                "resolved_workspace": str(workspace_root),
                "status": "discard",
            }
        return self._executor_result_wrapper(
            active_task=active_task,
            failed_iterations=failed_iterations,
            run_n=run_n,
            task_node=task_node,
            summary_parts=summary_parts,
            artifact_dir=artifact_dir,
            result=artifact_payload,
            provider=agent_result.get("provider"),
            model=agent_result.get("model"),
            attempted_models=agent_result.get("attempted_models", []),
            fallback_used=agent_result.get("fallback_used", False),
        )

    def _materialize_relaxed_plan_payload(
        self,
        *,
        project_dir: Path,
        artifact_dir: Path,
        task_node: Any,
        run_n: int,
        summary_parts: list[str],
        agent_result: dict[str, Any],
        attempt_phase: str = "plan_only",
        repair_count: int = 0,
        resolved_workspace: str = "",
    ) -> dict[str, Any]:
        plan_path = artifact_dir / "execution_plan.md"
        extra = {
            key: value
            for key, value in agent_result.items()
            if key
            not in {
                "summary",
                "artifact_plan",
                "command_plan",
                "experiment_plan",
                "citations",
                "next_action",
                "provider",
                "model",
                "primary_model",
                "attempted_models",
                "fallback_used",
                "raw_response",
                "retryable",
                "error_class",
                "validation_mode",
            }
        }
        sections = [
            "# Genesis Execution Plan",
            "",
            f"Run: {run_n}",
            f"Task: {getattr(task_node, 'task_id', f'run-{run_n}')}",
            "",
            "## Summary",
            str(agent_result.get("summary", "")).strip() or "No summary provided.",
        ]
        if extra:
            sections.extend(["", "## Structured Details", json.dumps(extra, indent=2)])
        raw_response = str(agent_result.get("raw_response", "")).strip()
        if raw_response:
            sections.extend(["", "## Raw Response", "```json", raw_response, "```"])
        plan_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
        return {
            "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
            "summary": " ".join(summary_parts + [str(agent_result.get("summary", "")).strip()]).strip(),
            "primary_metric": 0.5,
            "code_path": str(plan_path),
            "errors": [],
            "artifact_dir": str(artifact_dir),
            "generated_artifacts": [str(plan_path)],
            "artifact_records": [
                {
                    "path": str(plan_path),
                    "role": "final_artifact",
                    "produced_by": "plan_materialization",
                    "size_bytes": plan_path.stat().st_size,
                    "type": "md",
                    "substantive": plan_path.stat().st_size > 0,
                }
            ],
            "executed_commands": [],
            "command_results": [],
            "classification": "plan_materialized",
            "failure_type": "",
            "failure_summary": "",
            "failed_command": "",
            "failure_signature": "",
            "debug_focus": self._debug_focus(project_dir, run_n + 1),
            "citations": agent_result.get("citations", []),
            "agent_runtime": {
                "provider": agent_result.get("provider"),
                "model": agent_result.get("model"),
                "primary_model": agent_result.get("primary_model"),
                "attempted_models": agent_result.get("attempted_models", []),
                "fallback_used": agent_result.get("fallback_used", False),
                "next_action": agent_result.get("next_action"),
                "validation_mode": agent_result.get("validation_mode"),
            },
            "attempt_phase": attempt_phase,
            "repair_count": repair_count,
            "schema_blockers": [],
            "resolved_workspace": resolved_workspace or str(self._workspace_root(artifact_dir)),
            "status": "discard",
        }

    def _executor_result_wrapper(
        self,
        *,
        active_task: str,
        failed_iterations: int,
        run_n: int,
        task_node: Any,
        summary_parts: list[str],
        artifact_dir: Path,
        result: dict[str, Any],
        provider: Any = None,
        model: Any = None,
        attempted_models: list[Any] | None = None,
        fallback_used: bool = False,
    ) -> dict[str, Any]:
        return {
            "trace": {
                "instruction_used": run_n,
                "task": active_task,
                "failed_iterations": failed_iterations,
                "provider": provider,
                "model": model,
                "attempted_models": attempted_models or [],
                "fallback_used": fallback_used,
                "debug_focus": self._debug_focus(artifact_dir.parents[2], run_n),
            },
            "result": result,
        }

    def _is_repairable_schema_error(self, error_class: str) -> bool:
        return error_class in {"command_plan_missing_artifact", "command_plan_requires_shell_wrapper", "command_plan_invalid"}

    def _is_repairable_schema_failure(self, result: dict[str, Any]) -> bool:
        return str(result.get("failure_type", "")) in {"command_plan_missing_artifact", "command_plan_requires_shell_wrapper", "command_plan_invalid"}

    def _schema_blocker_from_error(self, exc: ProviderRuntimeError) -> str:
        return f"{exc.error_class}: {str(exc).split('. First non-empty response', 1)[0][:240]}"

    def _schema_blocker_from_result(self, result: dict[str, Any]) -> str:
        return f"{result.get('failure_type', 'schema_error')}: {result.get('failure_summary', 'repair the schema mismatch')}"

    def _has_artifact_attempt(self, agent_result: dict[str, Any]) -> bool:
        artifact_plan = agent_result.get("artifact_plan", [])
        return isinstance(artifact_plan, list) and any(isinstance(item, dict) and str(item.get("path", "")).strip() for item in artifact_plan)

    def _execute_agent_work(
        self,
        *,
        project_dir: Path,
        run_n: int,
        task_node: Any,
        summary_parts: list[str],
        agent_result: dict[str, Any],
        artifact_dir: Path,
        attempt_phase: str = "executing",
        repair_count: int = 0,
    ) -> dict[str, Any] | None:
        artifact_files = self._materialize_artifact_plan(artifact_dir, agent_result.get("artifact_plan", []))
        command_results = self._run_command_plan(
            project_dir=project_dir,
            artifact_dir=artifact_dir,
            command_plan=agent_result.get("command_plan", []),
        )
        self._promote_workspace_outputs(artifact_dir)
        substantive_file_records = self._substantive_files(artifact_dir)
        substantive_files = [record["path"] for record in substantive_file_records if record["substantive"]]
        if not artifact_files and not command_results and not substantive_files:
            return None
        command_successes = sum(1 for item in command_results if item.get("returncode", 1) == 0)
        command_total = len(command_results)
        command_success_ratio = command_successes / command_total if command_total else 1.0
        last_failure = next((item for item in reversed(command_results) if item.get("returncode", 0) != 0), None)
        classification = "success"
        failure_type = ""
        failure_summary = ""
        next_action = str(agent_result.get("next_action", ""))
        if last_failure is not None:
            classification = "repairable_schema_mismatch" if self._is_repairable_schema_error(str(last_failure.get("failure_type", ""))) else "command_failure"
            failure_type = str(last_failure.get("failure_type", "command_failure"))
            failure_summary = str(last_failure.get("failure_summary", "command failed"))
        elif not substantive_files and not any(item.get("returncode", 1) == 0 and not self._is_setup_command(str(item.get("command", ""))) for item in command_results):
            classification = "non_substantive_completion"
            failure_type = "non_substantive_completion"
            failure_summary = "run produced no non-empty substantive artifacts or successful task-relevant commands"
        elif self._next_action_requires_verified_work(next_action):
            classification = "invalid_next_action"
            failure_type = "invalid_next_action"
            failure_summary = "model proposed publication/finalization before substantive verified work existed"
        primary_metric = round(command_success_ratio if substantive_files else 0.0, 6)
        if command_results:
            (artifact_dir / "command_results.json").write_text(json.dumps(command_results, indent=2), encoding="utf-8")
        manifest_path = artifact_dir.parent / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(substantive_file_records, indent=2), encoding="utf-8")
        result_payload = {
            "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
            "summary": " ".join(summary_parts + [agent_result.get("summary", "")]).strip(),
            "primary_metric": primary_metric,
            "generated_artifacts": substantive_files,
            "artifact_records": substantive_file_records,
            "executed_commands": [item.get("command", "") for item in command_results],
            "command_results": command_results,
            "classification": classification,
            "failure_type": failure_type,
            "failure_summary": failure_summary,
            "failed_command": str(last_failure.get("command", "")) if last_failure else "",
            "failure_signature": str(last_failure.get("failure_signature", "")) if last_failure else "",
            "debug_focus": self._debug_focus(project_dir, run_n + 1),
            "citations": agent_result.get("citations", []),
            "agent_runtime": {
                "provider": agent_result.get("provider"),
                "model": agent_result.get("model"),
                "primary_model": agent_result.get("primary_model"),
                "attempted_models": agent_result.get("attempted_models", []),
                "fallback_used": agent_result.get("fallback_used", False),
                "next_action": agent_result.get("next_action"),
            },
            "artifact_dir": str(artifact_dir),
            "code_path": substantive_files[0] if substantive_files else "",
            "errors": [item.get("stderr_path", "") for item in command_results if item.get("returncode", 0) != 0],
            "attempt_phase": "executed_substantively" if classification == "success" else attempt_phase,
            "repair_count": repair_count,
            "schema_blockers": [failure_summary] if classification == "repairable_schema_mismatch" and failure_summary else [],
            "resolved_workspace": str(self._workspace_root(artifact_dir)),
            "artifact_manifest_path": str(manifest_path),
            "status": "keep" if classification == "success" and (substantive_files or command_success_ratio > 0.0) else "discard",
        }
        return result_payload

    def _substantive_files(self, artifact_dir: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        workspace_root = self._workspace_root(artifact_dir)
        seen: set[str] = set()
        for root in (artifact_dir, workspace_root):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.name in {"result.json", "command_results.json", "artifact_manifest.json"}:
                    continue
                signature = str(path.resolve())
                if signature in seen:
                    continue
                seen.add(signature)
                size_bytes = path.stat().st_size
                substantive = size_bytes > 0 and not path.name.endswith((".stdout.log", ".stderr.log"))
                role = "workspace_helper" if workspace_root == path.parent or workspace_root in path.parents else "final_artifact"
                records.append(
                    {
                        "path": str(path),
                        "size_bytes": size_bytes,
                        "type": path.suffix.lstrip(".") or "file",
                        "substantive": substantive if role == "final_artifact" else False,
                        "role": role,
                        "produced_by": "artifact_plan",
                    }
                )
        return records

    def _materialize_artifact_plan(self, artifact_dir: Path, artifact_plan: list[Any]) -> list[str]:
        artifact_files: list[str] = []
        workspace_root = self._workspace_root(artifact_dir)
        workspace_root.mkdir(parents=True, exist_ok=True)
        for artifact in artifact_plan:
            if not isinstance(artifact, dict):
                continue
            target_path = self._resolve_artifact_target(artifact_dir, workspace_root, artifact)
            if target_path is None:
                continue
            content = str(artifact.get("content", ""))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            if content.strip():
                artifact_files.append(str(target_path))
        return artifact_files

    def _run_command_plan(
        self,
        *,
        project_dir: Path,
        artifact_dir: Path,
        command_plan: list[Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        workspace_root = self._workspace_root(artifact_dir)
        workspace_root.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(command_plan, start=1):
            parsed = self._parse_command_entry(
                entry,
                project_dir=project_dir,
                artifact_dir=artifact_dir,
                workspace_root=workspace_root,
            )
            if parsed is None:
                results.append(
                    {
                        "command": "",
                        "cwd": str(artifact_dir),
                        "returncode": 127,
                        "stdout_path": "",
                        "stderr_path": "",
                        "failure_type": "command_plan_invalid",
                        "failure_summary": "command entry was missing or malformed",
                    }
                )
                break
            if isinstance(parsed, dict):
                results.append(parsed)
                break
            command, cwd, timeout = parsed
            missing_file = self._missing_workspace_reference(command, cwd, artifact_dir, workspace_root)
            if missing_file:
                results.append(
                    {
                        "command": " ".join(command),
                        "cwd": str(cwd),
                        "returncode": 127,
                        "stdout_path": "",
                        "stderr_path": "",
                        "failure_type": "command_plan_missing_artifact",
                        "failure_summary": f"referenced workspace file is missing: {missing_file}",
                        "stderr_excerpt": f"missing workspace file: {missing_file}",
                        "failure_signature": "command_plan_missing_artifact",
                    }
                )
                break
            stdout_path = artifact_dir / f"command_{index}.stdout.log"
            stderr_path = artifact_dir / f"command_{index}.stderr.log"
            try:
                process = subprocess.run(
                    command,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
                stdout_path.write_text(process.stdout, encoding="utf-8")
                stderr_path.write_text(process.stderr, encoding="utf-8")
                results.append(
                    {
                        "command": " ".join(command),
                        "cwd": str(cwd),
                        "returncode": process.returncode,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "failure_type": "command_failure" if process.returncode != 0 else "",
                        "failure_summary": "command returned non-zero exit status" if process.returncode != 0 else "",
                        "stderr_excerpt": process.stderr.strip()[:400],
                        "failure_signature": self._stderr_signature(process.stderr) if process.returncode != 0 else "",
                    }
                )
                if process.returncode != 0:
                    break
            except FileNotFoundError:
                stderr_path.write_text("command not found", encoding="utf-8")
                results.append(
                    {
                        "command": " ".join(command),
                        "cwd": str(cwd),
                        "returncode": 127,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "failure_type": "command_not_found",
                        "failure_summary": "executable or script was not found",
                        "stderr_excerpt": "command not found",
                        "failure_signature": "command_not_found",
                    }
                )
                break
            except subprocess.TimeoutExpired as exc:
                stdout_path.write_text(exc.stdout or "", encoding="utf-8")
                stderr_path.write_text(exc.stderr or "command timed out", encoding="utf-8")
                results.append(
                    {
                        "command": " ".join(command),
                        "cwd": str(cwd),
                        "returncode": 124,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "failure_type": "command_timeout",
                        "failure_summary": "command exceeded timeout",
                        "stderr_excerpt": (exc.stderr or "command timed out")[:400],
                        "failure_signature": "command_timeout",
                    }
                )
                break
        return results

    def _parse_command_entry(
        self,
        entry: Any,
        *,
        project_dir: Path,
        artifact_dir: Path,
        workspace_root: Path,
    ) -> tuple[list[str], Path, int] | dict[str, Any] | None:
        if isinstance(entry, str) and entry.strip():
            return shlex.split(entry), workspace_root, 300
        if not isinstance(entry, dict):
            return None
        command_value = entry.get("command")
        if isinstance(command_value, str) and command_value.strip():
            command = shlex.split(command_value)
        elif isinstance(command_value, list) and all(isinstance(part, str) for part in command_value):
            command = list(command_value)
        else:
            return None
        cwd_value = str(entry.get("cwd", ".")).strip() or "."
        cwd = Path(cwd_value)
        if not cwd.is_absolute():
            if cwd_value.startswith("project:"):
                base = project_dir
                relative = cwd_value.removeprefix("project:")
            elif cwd_value.startswith("workspace:"):
                base = workspace_root
                relative = cwd_value.removeprefix("workspace:")
            else:
                base = workspace_root
                relative = cwd_value
            cwd = (base / relative).resolve()
        timeout = int(entry.get("timeout_seconds", 300))
        if not command:
            return {
                "command": "",
                "cwd": str(cwd),
                "returncode": 127,
                "stdout_path": "",
                "stderr_path": "",
                "failure_type": "command_plan_invalid",
                "failure_summary": "empty command entry",
            }
        return command, cwd, timeout

    def _workspace_root(self, artifact_dir: Path) -> Path:
        return artifact_dir.parent / "workspace"

    def _resolve_artifact_target(self, artifact_dir: Path, workspace_root: Path, artifact: dict[str, Any]) -> Path | None:
        relative_path = str(artifact.get("path", "")).strip()
        if not relative_path:
            return None
        path = Path(relative_path)
        if path.is_absolute():
            try:
                path = path.relative_to(path.anchor)
            except Exception:
                path = Path(path.name)
        normalized = str(path).replace("\\", "/")
        if normalized.startswith("project:"):
            normalized = normalized.removeprefix("project:")
        path = self._normalize_artifact_relative_path(Path(normalized))
        explicit_role = str(artifact.get("role", "")).strip()
        target_root = workspace_root if explicit_role in {"workspace_helper", "helper"} else artifact_dir
        if not explicit_role and path.suffix in {".py", ".sh", ".ipynb", ".R", ".jl"}:
            target_root = workspace_root
        return (target_root / path).resolve()

    def _normalize_artifact_relative_path(self, path: Path) -> Path:
        parts = [part for part in path.parts if part not in {"", "."}]
        while parts and parts[0] in {"workspace", "artifacts"}:
            parts = parts[1:]
        if "outputs" in parts:
            outputs_index = parts.index("outputs")
            parts = parts[outputs_index + 1 :]
            while parts and parts[0] in {"code", "paper", "workspace", "artifacts"}:
                parts = parts[1:]
        if not parts:
            return Path(path.name or "artifact")
        return Path(*parts)

    def _promote_workspace_outputs(self, artifact_dir: Path) -> None:
        workspace_root = self._workspace_root(artifact_dir)
        if not workspace_root.exists():
            return
        helper_suffixes = {".py", ".sh", ".ipynb", ".R", ".jl"}
        for path in sorted(workspace_root.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix in helper_suffixes or path.name.endswith((".stdout.log", ".stderr.log")):
                continue
            relative = path.relative_to(workspace_root)
            target = artifact_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    def _missing_workspace_reference(self, command: list[str], cwd: Path, artifact_dir: Path, workspace_root: Path) -> str:
        referenced = self._workspace_file_reference(command)
        if not referenced:
            return ""
        candidate = Path(referenced)
        candidates = []
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.extend(
                [
                    cwd / candidate,
                    workspace_root / candidate,
                    artifact_dir / candidate,
                ]
            )
        for path in candidates:
            if path.exists():
                return ""
        return referenced

    def _workspace_file_reference(self, command: list[str]) -> str:
        for token in command[1:]:
            cleaned = token.strip()
            if not cleaned or cleaned.startswith("-"):
                continue
            if "/" in cleaned or "." in Path(cleaned).name:
                name = Path(cleaned).name
                if name.endswith((".py", ".sh", ".ipynb", ".R", ".jl", ".md", ".json", ".csv", ".txt")):
                    return cleaned
        return ""

    def _should_use_optimizer(self, *, task_node: Any, config: ProjectConfig, agent_result: dict[str, Any]) -> bool:
        return bool(
            task_node
            and getattr(task_node, "requires_ml_optimizer", False)
            and agent_result.get("experiment_plan")
            and self._plan_has_real_commands(agent_result.get("experiment_plan"))
        )

    def _plan_has_real_commands(self, plan: Any) -> bool:
        if not isinstance(plan, list) or not plan:
            return False
        return all(
            isinstance(item, dict)
            and (
                isinstance(item.get("command"), str)
                and str(item.get("command")).strip()
                or isinstance(item.get("command"), list)
                and bool(item.get("command"))
            )
            for item in plan
        )

    def _execution_context(self, project_dir: Path, run_n: int) -> dict[str, Any]:
        prior_runs: list[dict[str, Any]] = []
        verification_failures: list[str] = []
        for prior_run in range(max(1, run_n - 2), run_n):
            run_dir = project_dir / "runs" / str(prior_run)
            result_path = run_dir / "result.json"
            verification_path = run_dir / "verification_report.json"
            payload: dict[str, Any] = {"run_n": prior_run}
            if result_path.exists():
                result = self.filesystem.read_json(result_path)
                payload["summary"] = result.get("summary", "")
                payload["generated_artifacts"] = result.get("generated_artifacts", [])
                payload["executed_commands"] = result.get("executed_commands", [])
                payload["classification"] = result.get("classification", "")
                payload["failure_type"] = result.get("failure_type", "")
                payload["failure_summary"] = result.get("failure_summary", "")
                payload["failed_command"] = result.get("failed_command", "")
                payload["debug_focus"] = result.get("debug_focus", "")
            if verification_path.exists():
                verification = self.filesystem.read_json(verification_path)
                failed_checks = [
                    str(check.get("name", "unknown"))
                    for check in verification.get("checks", [])
                    if not self.verification._is_check_passing(check)
                ]
                payload["verification_failures"] = failed_checks
                verification_failures.extend(failed_checks)
            prior_runs.append(payload)
        return {
            "prior_runs": prior_runs,
            "verification_failures": verification_failures,
            "debug_focus": self._debug_focus(project_dir, run_n),
        }

    def _build_run_status(
        self,
        *,
        run_n: int,
        summary: str,
        stopping_decision: dict[str, Any],
        verification: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "run_n": run_n,
            "summary": summary,
            "stopping_decision": stopping_decision,
            "verification_passed": bool(verification.get("passed", False)),
            "task_id": result.get("task_id"),
            "stage": result.get("stage", ""),
            "artifact_dir": result.get("artifact_dir"),
            "classification": result.get("classification", ""),
            "failure_type": result.get("failure_type", ""),
            "attempt_phase": result.get("attempt_phase", ""),
            "schema_blockers": result.get("schema_blockers", []),
            "adversarial_passed": not bool(result.get("adversarial_critical_blockers", [])),
            "critical_blockers": result.get("adversarial_critical_blockers", []),
        }

    def _update_repeated_failures(
        self,
        counters: dict[str, int],
        result: dict[str, Any],
    ) -> dict[str, int]:
        classification = str(result.get("classification", "")).strip()
        failure_type = str(result.get("failure_type", "")).strip()
        failed_command = str(result.get("failed_command", "")).strip()
        failure_signature = str(result.get("failure_signature", "")).strip()
        if not classification or classification == "success":
            return {}
        key = "|".join(part for part in (classification, failure_type, failure_signature or failed_command) if part)
        updated = dict(counters)
        updated[key] = updated.get(key, 0) + 1
        return updated

    def _repeated_failure_message(self, counters: dict[str, int]) -> str:
        if not counters:
            return ""
        key, count = max(counters.items(), key=lambda item: item[1])
        if count < self.REPEATED_FAILURE_LIMIT:
            return ""
        return key

    def _next_action_requires_verified_work(self, next_action: str) -> bool:
        lowered = next_action.lower()
        return any(token in lowered for token in ("publish", "publication", "submit", "finalize", "journal"))

    def _stderr_signature(self, stderr: str) -> str:
        lowered = stderr.strip().lower()
        if not lowered:
            return ""
        for line in lowered.splitlines():
            line = line.strip()
            if line:
                return line[:160]
        return lowered[:160]

    def _debug_focus(self, project_dir: Path, run_n: int) -> str:
        signatures: dict[str, int] = {}
        for prior_run in range(1, run_n):
            result_path = project_dir / "runs" / str(prior_run) / "result.json"
            if not result_path.exists():
                continue
            result = self.filesystem.read_json(result_path)
            if str(result.get("classification", "")) != "command_failure":
                continue
            signature = str(result.get("failure_signature", "") or result.get("failure_summary", "")).strip()
            if not signature:
                continue
            signatures[signature] = signatures.get(signature, 0) + 1
        if not signatures:
            return ""
        signature, count = max(signatures.items(), key=lambda item: item[1])
        if count < 2:
            return ""
        return f"Repeated failure signature ({count}x): {signature}"

    def _run_adversarial_check(
        self,
        outputs: dict[str, Any],
        criteria: list[str],
        *,
        task_context: Optional[dict[str, Any]] = None,
        verification: Optional[dict[str, Any]] = None,
        oracle_result: Optional[dict[str, Any]] = None,
    ):
        import asyncio

        return asyncio.run(
            self.adversarial.run(
                outputs,
                criteria,
                task_context=task_context or {},
                verification=verification or {},
                oracle_result=oracle_result or {},
            )
        )

    def _check_stopping_criteria(self, adversarial_report) -> bool:
        return adversarial_report.stopping_decision.should_stop

    def _trajectory_anomaly_score(
        self,
        actual: list[float],
        predicted: list[float],
        variances: list[float],
    ) -> float:
        if not actual or not predicted:
            return 0.0
        score = 0.0
        for actual_value, predicted_value, variance in zip(actual, predicted, variances):
            denominator = max(variance, 1e-6)
            score += ((actual_value - predicted_value) ** 2) / denominator
        return round(score / max(1, len(actual)), 6)

    def _resolve_experiment_proposals(
        self,
        *,
        task_node: Any,
        config: ProjectConfig,
        ledger: ExperimentLedger,
        agent_result: dict[str, Any],
    ) -> list[Any]:
        plan = agent_result.get("experiment_plan")
        if isinstance(plan, list) and plan:
            proposals = []
            for index, item in enumerate(plan, start=1):
                if not isinstance(item, dict):
                    continue
                proposals.append(
                    ExperimentProposal(
                        description=str(item.get("description", f"Experiment variant {index}")),
                        code_diff=str(item.get("code_diff", f"variant {index}")),
                        expected_metric=float(item.get("expected_metric", 0.4 + 0.05 * index)),
                        expected_trajectory=[
                            float(value)
                            for value in item.get("expected_trajectory", [0.2, 0.4, 0.6])
                        ],
                        compute_budget=str(item.get("compute_budget", config.compute_budget)),
                        model_parameter_count=int(item.get("model_parameter_count", 0)),
                        command=item.get("command"),
                    )
                )
            if proposals:
                return proposals
        return ExperimentProposer().propose_next(
            task_node.task_id,
            n=3,
            prior_metric=ledger.get_by_task(task_node.task_id)[0]["primary_metric"] if ledger.get_by_task(task_node.task_id) else 0.0,
            compute_budget=config.compute_budget,
            ledger=ledger,
            causal_dag=CausalDAG(self.taste_persistence.root_dir / "causal_dag_global.json"),
            domain=config.domain,
        )

    def _task_for_stage(self, decomposition: Any, stage: str) -> Any:
        stage_keywords = {
            "survey": ("survey prior work", "literature"),
            "oracle": ("oracle",),
            "execute": ("controlled experiments", "run controlled experiments", "experiment"),
            "paper": ("paper", "synthesize"),
            "verify": ("verify experiment outputs", "verification"),
        }
        tasks = getattr(decomposition, "tasks", []) or []
        for task in tasks:
            description = str(getattr(task, "description", "")).lower()
            if any(token in description for token in stage_keywords.get(stage, ())):
                return task
        if stage == "execute":
            for task in tasks:
                if "paper" not in str(getattr(task, "description", "")).lower():
                    return task
        return tasks[0] if tasks else None

    def _next_stage(self, current_stage: str) -> str:
        try:
            index = self.STAGE_SEQUENCE.index(current_stage)
        except ValueError:
            return self.STAGE_SEQUENCE[0]
        return self.STAGE_SEQUENCE[min(index + 1, len(self.STAGE_SEQUENCE) - 1)]

    def _stage_for_task(self, task_node: Any) -> str:
        task_kind = str(getattr(task_node, "task_kind", "")).strip().lower()
        if task_kind == "oracle":
            return "oracle"
        if task_kind == "verify":
            return "verify"
        if task_kind == "paper":
            return "paper"
        if task_kind == "survey":
            return "survey"
        description = str(getattr(task_node, "description", "")).lower()
        if "oracle" in description:
            return "oracle"
        if "verify" in description:
            return "verify"
        if "paper" in description or "synthesize" in description:
            return "paper"
        if "survey" in description or "literature" in description:
            return "survey"
        return "execute"

    def _initialize_task_states(self, existing: Any, decomposition: Any) -> list[dict[str, Any]]:
        existing_by_id = {
            str(item.get("task_id", "")): item
            for item in (existing or [])
            if isinstance(item, dict) and str(item.get("task_id", "")).strip()
        }
        states: list[dict[str, Any]] = []
        for task in getattr(decomposition, "tasks", []) or []:
            previous = existing_by_id.get(task.task_id, {})
            states.append(
                {
                    "task_id": task.task_id,
                    "stage": self._stage_for_task(task),
                    "status": str(previous.get("status", "pending")),
                    "last_run_n": int(previous.get("last_run_n", 0)),
                    "blockers": list(previous.get("blockers", [])),
                    "verification_passed": bool(previous.get("verification_passed", False)),
                    "adversarial_passed": bool(previous.get("adversarial_passed", False)),
                    "attempt_phase": str(previous.get("attempt_phase", "plan_only")),
                    "repair_count": int(previous.get("repair_count", 0)),
                    "schema_blockers": list(previous.get("schema_blockers", [])),
                    "block_reason": str(previous.get("block_reason", "")),
                    "completion_evidence": list(previous.get("completion_evidence", [])),
                }
            )
        return states

    def _task_state_by_id(self, task_states: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
        for task_state in task_states:
            if str(task_state.get("task_id")) == task_id:
                return task_state
        raise KeyError(task_id)

    def _refresh_task_states(self, project_dir: Path, task_states: list[dict[str, Any]], decomposition: Any) -> list[dict[str, Any]]:
        task_ids = {task.task_id for task in getattr(decomposition, "tasks", []) or []}
        existing = {state["task_id"]: dict(state) for state in task_states if state.get("task_id") in task_ids}
        refreshed = []
        for task in getattr(decomposition, "tasks", []) or []:
            state = existing.get(task.task_id, {"task_id": task.task_id, "stage": self._stage_for_task(task), "status": "pending", "last_run_n": 0, "blockers": [], "verification_passed": False, "adversarial_passed": False, "attempt_phase": "plan_only", "repair_count": 0, "schema_blockers": [], "block_reason": "", "completion_evidence": []})
            snapshot = self._task_completion_snapshot(project_dir, task)
            if snapshot["completed"]:
                state["status"] = "completed"
                state["verification_passed"] = True
                state["adversarial_passed"] = True
                state["blockers"] = []
                state["completion_evidence"] = snapshot["evidence"]
                refreshed.append(state)
                continue
            if state["status"] not in {"completed", "blocked", "running"}:
                deps_complete = all(
                    self._task_state_by_id(task_states, dependency)["status"] == "completed"
                    for dependency in task.dependencies
                ) if task.dependencies else True
                state["status"] = "ready" if deps_complete else "pending"
            refreshed.append(state)
        return refreshed

    def _select_next_task(self, decomposition: Any, task_states: list[dict[str, Any]]) -> Any:
        if not getattr(decomposition, "tasks", None):
            return None
        priority = {"blocked": 0, "ready": 1, "running": 2, "pending": 3, "failed": 4, "completed": 5}
        task_by_id = {task.task_id: task for task in decomposition.tasks}
        candidates = [
            state for state in task_states
            if state.get("status") in {"blocked", "ready", "running"}
        ]
        if not candidates:
            return None
        selected = sorted(
            candidates,
            key=lambda item: (
                priority.get(str(item.get("status")), 99),
                [task.task_id for task in decomposition.tasks].index(item["task_id"]),
            ),
        )[0]
        return task_by_id[selected["task_id"]]

    def _mark_task_state(
        self,
        task_states: list[dict[str, Any]],
        task_id: str,
        *,
        status: str,
        run_n: int,
        stage: str,
        blockers: list[str],
        verification_passed: bool | None = None,
        adversarial_passed: bool | None = None,
        attempt_phase: str | None = None,
        repair_count: int | None = None,
        schema_blockers: list[str] | None = None,
        block_reason: str | None = None,
        completion_evidence: list[str] | None = None,
    ) -> None:
        state = self._task_state_by_id(task_states, task_id)
        state["status"] = status
        state["last_run_n"] = run_n
        state["stage"] = stage
        state["blockers"] = list(blockers)
        if verification_passed is not None:
            state["verification_passed"] = verification_passed
        if adversarial_passed is not None:
            state["adversarial_passed"] = adversarial_passed
        if attempt_phase is not None:
            state["attempt_phase"] = attempt_phase
        if repair_count is not None:
            state["repair_count"] = repair_count
        if schema_blockers is not None:
            state["schema_blockers"] = list(schema_blockers)
        if block_reason is not None:
            state["block_reason"] = block_reason
        if completion_evidence is not None:
            state["completion_evidence"] = list(completion_evidence)

    def _stage_success(self, stage: str, result: dict[str, Any]) -> bool:
        classification = str(result.get("classification", ""))
        generated_artifacts = result.get("generated_artifacts", [])
        task_kind = str(result.get("task_kind", stage)).strip().lower()
        verification_passed = bool(result.get("verification_passed", False))
        if classification != "success":
            return False
        if task_kind == "survey":
            return bool(generated_artifacts)
        if task_kind == "oracle":
            return bool(generated_artifacts)
        if task_kind in {"execute", "acquire_data", "analyze"}:
            return bool(generated_artifacts) or any(
                int(item.get("returncode", 1)) == 0 and not self._is_setup_command(str(item.get("command", "")))
                for item in result.get("command_results", [])
                if isinstance(item, dict)
            )
        if task_kind == "verify":
            return verification_passed
        if task_kind == "paper":
            return bool(generated_artifacts)
        return bool(generated_artifacts)

    def _task_completion_snapshot(self, project_dir: Path, task_node: Any) -> dict[str, Any]:
        expected = set(getattr(task_node, "expected_artifacts", []) or [])
        if not expected:
            return {"completed": False, "evidence": []}
        evidence: list[str] = []
        for run_dir in sorted((project_dir / "runs").glob("*"), reverse=True):
            result_path = run_dir / "result.json"
            verification_path = run_dir / "verification_report.json"
            if not result_path.exists():
                continue
            payload = self.filesystem.read_json(result_path)
            if str(payload.get("task_id", "")) != str(getattr(task_node, "task_id", "")):
                continue
            if str(payload.get("classification", "")) != "success":
                continue
            artifact_records = payload.get("artifact_records", [])
            present = {
                Path(str(item.get("path", ""))).name
                for item in artifact_records
                if isinstance(item, dict) and bool(item.get("substantive")) and str(item.get("role", "")) == "final_artifact"
            }
            verification_passed = False
            if verification_path.exists():
                verification = self.filesystem.read_json(verification_path)
                verification_passed = bool(verification.get("passed", False))
            if expected.issubset(present) and verification_passed:
                evidence = [f"{run_dir.name}:{name}" for name in sorted(expected)]
                return {"completed": True, "evidence": evidence}
        return {"completed": False, "evidence": evidence}

    def _ideation_required(self, *, failed_iterations: int, current_stage: str) -> bool:
        return failed_iterations >= self.ADVERSARIAL_ITERATION_LIMIT and current_stage in {"survey", "execute", "verify"}

    def _adversarial_task_context(self, task_node: Any, stage: str, project_dir: Path) -> dict[str, Any]:
        return {
            "task_id": getattr(task_node, "task_id", ""),
            "task_description": getattr(task_node, "description", ""),
            "stage": stage,
            "task_kind": getattr(task_node, "task_kind", stage),
            "success_metric": getattr(task_node, "success_metric", ""),
            "acceptance_criteria": getattr(task_node, "acceptance_criteria", []),
            "expected_artifacts": getattr(task_node, "expected_artifacts", []),
            "execution_mode": getattr(task_node, "execution_mode", ""),
            "prior_results": self._prior_task_evidence(project_dir, getattr(task_node, "task_id", "")),
        }

    def _format_adversarial_blockers(self, blockers: list[str]) -> str:
        if not blockers:
            return ""
        return "Address adversarial blockers before advancing: " + "; ".join(blockers[:5])

    def _prior_task_evidence(self, project_dir: Path, task_id: str) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for result_path in sorted((project_dir / "runs").glob("*/result.json")):
            payload = self.filesystem.read_json(result_path)
            if str(payload.get("task_id", "")) != str(task_id):
                continue
            evidence.append(
                {
                    "run": result_path.parent.name,
                    "classification": payload.get("classification", ""),
                    "generated_artifacts": payload.get("generated_artifacts", []),
                    "artifact_records": payload.get("artifact_records", []),
                    "summary": payload.get("summary", ""),
                }
            )
        return evidence[-3:]

    def _is_setup_command(self, command: str) -> bool:
        lowered = command.strip().lower()
        return lowered.startswith("pip install") or lowered.startswith("python -m pip install")

    def _latest_stage_result(self, project_dir: Path, stage: str) -> dict[str, Any] | None:
        results: list[dict[str, Any]] = []
        for result_path in sorted((project_dir / "runs").glob("*/result.json")):
            payload = self.filesystem.read_json(result_path)
            if str(payload.get("stage", "")) == stage:
                results.append(payload)
        return results[-1] if results else None

    def _stage_report(self, stage: str, *, success: bool, reason: str) -> AdversarialReport:
        report = AdversarialReport(
            claim_flags=[],
            literature_flags=[],
            formal_checks=[CheckResult(name=f"{stage}_stage", passed=success, evidence=[reason])],
            acceptance_ratio=1.0 if success else 0.0,
            grounded_claims=1 if success else 0,
            total_claims=1,
            stopping_decision=StoppingDecision(
                should_stop=False,
                reasons=[f"{stage}:{'complete' if success else 'incomplete'}"],
                critical_flags=[] if success else [reason or f"{stage}_incomplete"],
            ),
        )
        return report

    def _update_causal_dag(self, dag_path: Path, project_id: str, domain: str, verified_experiments: list[dict[str, Any]]) -> None:
        dag = CausalDAG(dag_path)
        global_dag = CausalDAG(self.taste_persistence.root_dir / "causal_dag_global.json")
        project_dir = self.filesystem.get_project_dir(project_id)
        for experiment in verified_experiments:
            task_id = str(experiment.get("task_id", "")).strip()
            target = f"metric:{task_id}" if task_id else "metric:unknown"
            config_diff = str(experiment.get("config_diff", "")).strip()
            trajectory_summary = experiment.get("trajectory_summary", {}) if isinstance(experiment.get("trajectory_summary", {}), dict) else {}
            effect_size = float(experiment.get("secondary_metrics", {}).get("improvement", 0.0))
            if effect_size == 0.0:
                start = trajectory_summary.get("start")
                end = trajectory_summary.get("end")
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    effect_size = float(end) - float(start)
            confidence = 0.9 if float(experiment.get("anomaly_score", 0.0)) < 0.5 else 0.75
            features = [part.strip() for part in config_diff.split(";") if part.strip()]
            for feature in features:
                try:
                    dag.add_edge(
                        feature,
                        target,
                        effect_size=round(effect_size, 6),
                        confidence=confidence,
                        experiment_ids=[str(experiment.get("experiment_id", ""))],
                        domain=domain,
                    )
                except ValueError:
                    continue
        for result_path in sorted((project_dir / "runs").glob("*/result.json")):
            payload = self.filesystem.read_json(result_path)
            stage = str(payload.get("stage", "unknown"))
            classification = str(payload.get("classification", "unknown"))
            target = "verified_artifact" if classification == "success" else f"failure:{payload.get('failure_type') or classification}"
            effect_size = float(payload.get("primary_metric", 0.0))
            if classification != "success":
                effect_size = -max(0.1, abs(effect_size) or 0.1)
            confidence = 0.85 if classification == "success" else 0.6
            try:
                dag.add_edge(
                    stage,
                    target,
                    effect_size=round(effect_size, 6),
                    confidence=confidence,
                    experiment_ids=[f"{project_id}:{result_path.parent.name}"],
                    domain=domain,
                )
            except ValueError:
                continue
        global_dag.merge_global_dag(self.filesystem.read_json(dag_path))

    def _requested_modules(
        self,
        *,
        config: ProjectConfig,
        task_node: Any,
        failed_iterations: int,
        current_stage: str = "execute",
    ) -> list[str]:
        modules = ["verification", "adversarial"]
        if current_stage in {"survey", "execute"}:
            modules.append("executor")
        if current_stage == "oracle":
            modules.append("oracle")
        if current_stage == "verify":
            modules.extend(["oracle", "verification"])
        if task_node and getattr(task_node, "requires_ml_optimizer", False):
            modules.append("optimizer")
        if failed_iterations >= self.ADVERSARIAL_ITERATION_LIMIT:
            modules.append("ideation")
        if config.domain.lower() == "astrophysics":
            modules.extend(["oracle", "domain_knowledge"])
        return sorted(set(modules))

    def _task_context(
        self,
        *,
        config: ProjectConfig,
        run_n: int,
        failed_iterations: int,
        redirect_message: str,
        current_stage: str,
    ) -> str:
        context = f"Run {run_n} stage={current_stage} for {config.research_question}"
        if redirect_message:
            context += f"\nRedirect: {redirect_message}"
        if failed_iterations >= self.ADVERSARIAL_ITERATION_LIMIT:
            context += (
                "\nEscalation: prior attempts did not satisfy adversarial/verification gates; "
                "a different approach is required."
            )
        return context
