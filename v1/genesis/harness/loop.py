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
from genesis.models import ExperimentProposal, ProjectResult
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

    def __init__(
        self,
        *,
        projects_root: Union[str, Path],
        taste_root: Union[str, Path],
        executor: Optional[Any] = None,
    ) -> None:
        self.filesystem = ProjectFilesystem(projects_root)
        self.token_budget = TokenBudget()
        self.history_reader = SelectiveHistoryReader(self.filesystem, self.token_budget)
        self.composer = InstructionComposer()
        self.agent_runtime = CodingAgentRuntime(
            Path(__file__).resolve().parents[2] / ".opencode" / "oh-my-openagent.jsonc"
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
        try:
            ledger = ExperimentLedger(project_dir / "experiments" / "ledger.sqlite3")
        except Exception as exc:  # noqa: BLE001
            self.filesystem.write_halt(
                project_id,
                {"type": "LEDGER_CORRUPTION", "message": str(exc)},
            )
            raise
        manifold = ManifoldIndex(project_dir / "knowledge" / "manifold")
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
        oracle_source = self.oracle_generator.generate(config)
        oracle_path = project_dir / "knowledge" / "oracle.py"
        oracle_path.write_text(oracle_source, encoding="utf-8")
        taste_model = self.taste_persistence.load_for_project(project_id, project_dir / "knowledge" / "taste_snapshot.json")
        failed_iterations = 0
        redirect_message = ""
        run_n = 0
        stopping_criteria_satisfied = False
        repeated_failure_counts: dict[str, int] = {}

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
            task_node = decomposition.tasks[min(run_n - 1, len(decomposition.tasks) - 1)] if decomposition.tasks else None
            budget_allocations = self.token_budget.allocate(
                128000 if "cloud" in config.compute_budget.lower() or "groq" in config.compute_budget.lower() else 18000
            )
            history = self.history_reader.summarize_experiment_history(project_id)
            requested_modules = self._requested_modules(config=config, task_node=task_node, failed_iterations=failed_iterations)
            current_task_context = self._task_context(
                config=config,
                run_n=run_n,
                failed_iterations=failed_iterations,
                redirect_message=redirect_message,
            )
            instruction = self.composer.compose(
                config=config,
                belief_summary=f"tracked_experiments={len(ledger.get_pareto_frontier())}",
                retrieved_history=history,
                domain_context=domain_context,
                current_task_context=current_task_context,
                budget_allocations=budget_allocations,
                requested_modules=requested_modules,
            )
            self.filesystem.write_instruction(project_id, run_n, instruction)
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
            run_dir = self.filesystem.get_run_dir(project_id, run_n)
            self.filesystem.write_json(run_dir / "trace.json", execution["trace"])
            self.filesystem.write_json(run_dir / "result.json", execution["result"])
            report = self._run_adversarial_check(execution["result"], criteria)
            self.filesystem.write_json(run_dir / "adversarial_report.json", report.to_dict())
            verification = self.verification.run(
                Path(execution["result"]["artifact_dir"]),
                project_id,
                oracle_path=oracle_path,
            )
            self.filesystem.write_json(run_dir / "verification_report.json", verification)
            log_event(
                log_path,
                project_id=project_id,
                run_n=run_n,
                component="meta_harness",
                event_type="run_completed",
                payload={
                    "acceptance_ratio": report.acceptance_ratio,
                    "verification_passed": verification["passed"],
                },
            )
            repeated_failure_counts = self._update_repeated_failures(repeated_failure_counts, execution["result"])
            if self._check_stopping_criteria(report) and verification["passed"]:
                result_summary = "stopping criteria satisfied"
                stopping_criteria_satisfied = True
                failed_iterations = 0
                redirect_message = ""
                self.filesystem.write_project_state(
                    project_id,
                    {
                            "status": "complete",
                            "run_count": run_n,
                            "last_run_status": self._build_run_status(
                                run_n=run_n,
                                summary=result_summary,
                                stopping_decision=report.stopping_decision.to_dict(),
                                verification=verification,
                                result=execution["result"],
                            ),
                    },
                )
                break
            failed_iterations += 1
            escalation_attempts = max(0, failed_iterations - self.ADVERSARIAL_ITERATION_LIMIT)
            self.filesystem.write_project_state(
                project_id,
                {
                    "status": "running",
                    "run_count": run_n,
                    "escalation_attempts": escalation_attempts,
                    "last_run_status": self._build_run_status(
                        run_n=run_n,
                        summary="continue iteration",
                        stopping_decision=report.stopping_decision.to_dict(),
                        verification=verification,
                        result=execution["result"],
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
                ideas = ideation.run(
                    config.research_question,
                    failed_iterations,
                    taste_model=taste_model,
                )
                self.filesystem.write_json(
                    run_dir / "ideation_report.json",
                    {"ideas": [idea.to_dict() for idea in ideas]},
                )
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
        if (project_dir / "HALT.json").exists():
            self.filesystem.write_project_state(
                project_id,
                {
                    "status": "halted",
                    "run_count": run_n,
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
            "artifact_dir": result.get("artifact_dir"),
            "classification": result.get("classification", ""),
            "failure_type": result.get("failure_type", ""),
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

    def _run_adversarial_check(self, outputs: dict[str, Any], criteria: list[str]):
        import asyncio

        return asyncio.run(self.adversarial.run(outputs, criteria))

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

    def _update_causal_dag(self, dag_path: Path, project_id: str) -> None:
        dag = CausalDAG(dag_path)
        dag.add_edge("instruction", "result", effect_size=1.0, confidence=0.9, experiment_ids=[project_id])

    def _requested_modules(self, *, config: ProjectConfig, task_node: Any, failed_iterations: int) -> list[str]:
        modules = ["verification", "adversarial", "paper"]
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
    ) -> str:
        context = f"Run {run_n} for {config.research_question}"
        if redirect_message:
            context += f"\nRedirect: {redirect_message}"
        if failed_iterations >= self.ADVERSARIAL_ITERATION_LIMIT:
            context += (
                "\nEscalation: prior attempts did not satisfy adversarial/verification gates; "
                "a different approach is required."
            )
        return context
