from __future__ import annotations

import json
import shlex
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
    STAGE_SEQUENCE = ["survey", "oracle", "execute", "verify"]

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
            task_node = self._task_for_stage(decomposition, current_stage)
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
            )
            self.filesystem.write_instruction(project_id, run_n, instruction)
            run_dir = self.filesystem.get_run_dir(project_id, run_n)
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
                    criteria,
                    task_context=self._adversarial_task_context(task_node, current_stage),
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
                else:
                    current_stage = self._next_stage(current_stage)
                    failed_iterations = 0
                    redirect_message = ""
                self.filesystem.write_project_state(
                    project_id,
                    {
                        "status": "running",
                        "run_count": run_n,
                        "current_stage": current_stage,
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
                )
                report = self._run_adversarial_check(
                    latest_execute_result,
                    criteria,
                    task_context=self._adversarial_task_context(task_node, current_stage),
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
                    self.filesystem.write_project_state(
                        project_id,
                        {
                            "status": "complete",
                            "run_count": run_n,
                            "current_stage": current_stage,
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
                current_stage = "execute"
                repeated_failure_counts = self._update_repeated_failures(repeated_failure_counts, result_payload)
                escalation_attempts = max(0, failed_iterations - self.ADVERSARIAL_ITERATION_LIMIT)
                self.filesystem.write_project_state(
                    project_id,
                    {
                        "status": "running",
                        "run_count": run_n,
                        "current_stage": current_stage,
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
            try:
                execution = self.executor(
                    project_dir=project_dir,
                    run_n=run_n,
                    config=config,
                    task_node=task_node,
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
            )
            report = self._run_adversarial_check(
                execution["result"],
                criteria,
                task_context=self._adversarial_task_context(task_node, current_stage),
                verification=verification,
                oracle_result=None,
            )
            self.filesystem.write_json(run_dir / "adversarial_report.json", report.to_dict())
            self.filesystem.write_json(run_dir / "verification_report.json", verification)
            latest_execute_result = execution["result"] if current_stage == "execute" else latest_execute_result
            execution["result"]["adversarial_critical_blockers"] = report.critical_blockers
            if self._stage_success(current_stage, execution["result"]) and not report.critical_blockers:
                failed_iterations = 0
                redirect_message = ""
                current_stage = self._next_stage(current_stage)
                result_summary = f"{execution['result'].get('stage', current_stage)} stage complete"
            else:
                failed_iterations += 1
                repeated_failure_counts = self._update_repeated_failures(repeated_failure_counts, execution["result"])
                if report.critical_blockers:
                    redirect_message = self._format_adversarial_blockers(report.critical_blockers)
                result_summary = "continue iteration"
            self.filesystem.write_project_state(
                project_id,
                {
                    "status": "running",
                    "run_count": run_n,
                    "current_stage": current_stage,
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
        self._update_causal_dag(project_dir / "causal_dag.json", project_id)
        self.taste_persistence.save_after_project(project_id, taste_model)
        self.taste_persistence.merge_project_data(project_id, ledger.get_pareto_frontier())
        self.filesystem.write_project_state(
            project_id,
            {
                "status": project_status,
                "run_count": run_n,
                "current_stage": current_stage,
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
        optimizer: ParallelExperimentManager,
        ledger: ExperimentLedger,
        ideation: IdeationOrchestrator,
        oracle_resolver: OracleResolver,
        failed_iterations: int,
        taste_model: Optional[TasteGP] = None,
    ) -> dict[str, Any]:
        active_task = task_node.description if task_node else config.research_question
        artifact_dir = project_dir / "outputs" / "code" / f"run_{run_n}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_parts = [f"Run {run_n} investigates {active_task}."]
        try:
            agent_result = self.agent_runtime.generate_task(
                category="sisyphus",
                instruction=f"Execute research task: {active_task}",
                context={
                    "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
                    "research_question": config.research_question,
                    "domain": config.domain,
                    "success_criteria": config.success_criteria,
                    "failed_iterations": failed_iterations,
                    **self._execution_context(project_dir, run_n),
                },
                budget={"max_runs": 1, "compute_budget": config.compute_budget},
            )
        except ProviderRuntimeError as exc:
            if exc.error_class == "non_actionable_plan":
                return {
                    "trace": {
                        "instruction_used": run_n,
                        "task": active_task,
                        "failed_iterations": failed_iterations,
                        "provider": None,
                        "model": None,
                        "attempted_models": [],
                        "fallback_used": False,
                    },
                    "result": {
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
                    },
                }
            raise
        generic_result = self._execute_agent_work(
            project_dir=project_dir,
            run_n=run_n,
            task_node=task_node,
            summary_parts=summary_parts,
            agent_result=agent_result,
            artifact_dir=artifact_dir,
        )
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
                taste_model.fit(
                    [self.feature_extractor.extract(proposal) for proposal in proposals],
                    [result.primary_metric for result in experiment_results],
                    [result.trajectory for result in experiment_results],
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
                "artifact_dir": str(Path(best.artifact_path).parent),
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
            }
        elif str(agent_result.get("validation_mode", "")) == "relaxed_plan_only":
            artifact_payload = self._materialize_relaxed_plan_payload(
                project_dir=project_dir,
                artifact_dir=artifact_dir,
                task_node=task_node,
                run_n=run_n,
                summary_parts=summary_parts,
                agent_result=agent_result,
            )
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
            }
        if failed_iterations >= 5:
            ideation_result = ideation.run(active_task, failed_iterations)
            artifact_payload["ideation_candidates"] = [idea.to_dict() for idea in ideation_result]
        return {
            "trace": {
                "instruction_used": run_n,
                "task": active_task,
                "failed_iterations": failed_iterations,
                "provider": agent_result.get("provider"),
                "model": agent_result.get("model"),
                "attempted_models": agent_result.get("attempted_models", []),
                "fallback_used": agent_result.get("fallback_used", False),
                "debug_focus": self._debug_focus(project_dir, run_n),
            },
            "result": artifact_payload,
        }

    def _materialize_relaxed_plan_payload(
        self,
        *,
        project_dir: Path,
        artifact_dir: Path,
        task_node: Any,
        run_n: int,
        summary_parts: list[str],
        agent_result: dict[str, Any],
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
            "status": "keep",
        }

    def _execute_agent_work(
        self,
        *,
        project_dir: Path,
        run_n: int,
        task_node: Any,
        summary_parts: list[str],
        agent_result: dict[str, Any],
        artifact_dir: Path,
    ) -> dict[str, Any] | None:
        artifact_files = self._materialize_artifact_plan(artifact_dir, agent_result.get("artifact_plan", []))
        command_results = self._run_command_plan(
            project_dir=project_dir,
            artifact_dir=artifact_dir,
            command_plan=agent_result.get("command_plan", []),
        )
        generated_files = sorted(
            str(path)
            for path in artifact_dir.rglob("*")
            if path.is_file() and path.name not in {"result.json", "command_results.json"}
        )
        substantive_files = [path for path in generated_files if not path.endswith((".stdout.log", ".stderr.log"))]
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
            classification = "command_failure"
            failure_type = str(last_failure.get("failure_type", "command_failure"))
            failure_summary = str(last_failure.get("failure_summary", "command failed"))
        elif self._next_action_requires_verified_work(next_action):
            classification = "invalid_next_action"
            failure_type = "invalid_next_action"
            failure_summary = "model proposed publication/finalization before substantive verified work existed"
        primary_metric = round(command_success_ratio if substantive_files else 0.0, 6)
        if command_results:
            (artifact_dir / "command_results.json").write_text(json.dumps(command_results, indent=2), encoding="utf-8")
        result_payload = {
            "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
            "summary": " ".join(summary_parts + [agent_result.get("summary", "")]).strip(),
            "primary_metric": primary_metric,
            "generated_artifacts": substantive_files,
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
            "status": "keep" if classification == "success" and substantive_files and command_success_ratio > 0.0 else "discard",
        }
        (artifact_dir / "result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        return result_payload

    def _materialize_artifact_plan(self, artifact_dir: Path, artifact_plan: list[Any]) -> list[str]:
        artifact_files: list[str] = []
        for artifact in artifact_plan:
            if not isinstance(artifact, dict):
                continue
            relative_path = str(artifact.get("path", "")).strip()
            if not relative_path:
                continue
            path = artifact_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(artifact.get("content", "")), encoding="utf-8")
            artifact_files.append(str(path))
        return artifact_files

    def _run_command_plan(
        self,
        *,
        project_dir: Path,
        artifact_dir: Path,
        command_plan: list[Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, entry in enumerate(command_plan, start=1):
            parsed = self._parse_command_entry(entry, project_dir=project_dir, artifact_dir=artifact_dir)
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
    ) -> tuple[list[str], Path, int] | dict[str, Any] | None:
        if isinstance(entry, str) and entry.strip():
            return shlex.split(entry), artifact_dir, 300
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
            base = project_dir if cwd_value.startswith("project:") else artifact_dir
            relative = cwd_value.removeprefix("project:")
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

    def _should_use_optimizer(self, *, task_node: Any, config: ProjectConfig, agent_result: dict[str, Any]) -> bool:
        return bool(
            task_node
            and getattr(task_node, "requires_ml_optimizer", False)
            and config.domain == "ml_efficiency"
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
        )

    def _task_for_stage(self, decomposition: Any, stage: str) -> Any:
        stage_keywords = {
            "survey": ("survey prior work", "literature"),
            "oracle": ("oracle",),
            "execute": ("controlled experiments", "run controlled experiments", "experiment"),
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

    def _stage_success(self, stage: str, result: dict[str, Any]) -> bool:
        classification = str(result.get("classification", ""))
        generated_artifacts = result.get("generated_artifacts", [])
        if stage == "survey":
            return classification in {"success", "plan_materialized"} and bool(generated_artifacts)
        if stage == "execute":
            return classification in {"success", "plan_materialized"} and bool(generated_artifacts)
        return classification in {"success", "plan_materialized"}

    def _ideation_required(self, *, failed_iterations: int, current_stage: str) -> bool:
        return failed_iterations >= self.ADVERSARIAL_ITERATION_LIMIT and current_stage in {"survey", "execute", "verify"}

    def _adversarial_task_context(self, task_node: Any, stage: str) -> dict[str, Any]:
        return {
            "task_id": getattr(task_node, "task_id", ""),
            "task_description": getattr(task_node, "description", ""),
            "stage": stage,
            "success_metric": getattr(task_node, "success_metric", ""),
            "acceptance_criteria": getattr(task_node, "acceptance_criteria", []),
        }

    def _format_adversarial_blockers(self, blockers: list[str]) -> str:
        if not blockers:
            return ""
        return "Address adversarial blockers before advancing: " + "; ".join(blockers[:5])

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

    def _update_causal_dag(self, dag_path: Path, project_id: str) -> None:
        dag = CausalDAG(dag_path)
        project_dir = self.filesystem.get_project_dir(project_id)
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
                )
            except ValueError:
                continue

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
