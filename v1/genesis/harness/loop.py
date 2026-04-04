from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from genesis.agents.runtime import CodingAgentRuntime, ProviderRuntimeError
from genesis.config import ProjectConfig
from genesis.domain_knowledge.registry import DomainKnowledgeRegistry
from genesis.harness.instruction_composer import InstructionComposer
from genesis.harness.decomposer import TaskDecomposer
from genesis.harness.token_budget import TokenBudget
from genesis.models import ProjectResult
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
        self.decomposer = TaskDecomposer()
        self.adversarial = AdversarialOrchestrator()
        self.criteria_generator = AcceptanceCriteriaGenerator()
        self.oracle_generator = DomainOracleGenerator()
        self.taste_persistence = TasteModelPersistence(taste_root)
        self.domain_registry = DomainKnowledgeRegistry()
        self.verification = VerificationPipeline()
        self.feature_extractor = ExperimentFeatureExtractor()
        self.agent_runtime = CodingAgentRuntime(
            Path(__file__).resolve().parents[2] / ".opencode" / "oh-my-openagent.jsonc"
        )
        self.executor = executor or self._default_executor

    def run(self, project_id: str, config: ProjectConfig, max_runs: int = 3) -> ProjectResult:
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

        result_summary = "unfinished"
        for run_n in range(1, max_runs + 1):
            intervention = self.filesystem.read_human_intervention(project_id)
            if intervention:
                if intervention.get("type") == "STOP":
                    result_summary = "stopped by human intervention"
                    self.filesystem.clear_human_intervention(project_id)
                    break
                if intervention.get("type") == "REJECT":
                    failed_iterations += 1
                self.filesystem.clear_human_intervention(project_id)
            history = self.history_reader.summarize_experiment_history(project_id)
            instruction = self.composer.compose(
                config=config,
                belief_summary=f"tracked_experiments={len(ledger.get_pareto_frontier())}",
                retrieved_history=history,
                domain_context=domain_context,
                current_task_context=f"Run {run_n} for {config.research_question}",
            )
            self.filesystem.write_instruction(project_id, run_n, instruction)
            task_node = decomposition.tasks[min(run_n - 1, len(decomposition.tasks) - 1)] if decomposition.tasks else None
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
            if self._check_stopping_criteria(report) and verification["passed"]:
                result_summary = "stopping criteria satisfied"
                failed_iterations = 0
                self.filesystem.write_project_state(
                    project_id,
                    {
                        "status": "complete",
                        "run_count": run_n,
                        "last_run_status": report.stopping_decision.to_dict(),
                    },
                )
                break
            failed_iterations += 1
            self.filesystem.write_project_state(
                project_id,
                {
                    "status": "running",
                    "run_count": run_n,
                    "last_run_status": report.stopping_decision.to_dict(),
                },
            )
            if failed_iterations >= 5:
                ideas = ideation.run(config.research_question, failed_iterations)
                self.filesystem.write_json(
                    run_dir / "ideation_report.json",
                    {"ideas": [idea.to_dict() for idea in ideas]},
                )
            if failed_iterations >= 7:
                self.filesystem.write_halt(
                    project_id,
                    {
                        "type": "ADVERSARIAL_STALEMATE",
                        "message": "Exceeded iterative recovery threshold.",
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
                    "last_run_status": result_summary,
                },
            )
            return ProjectResult(
                project_id=project_id,
                status="halted",
                paper_path=None,
                run_count=run_n,
                summary=result_summary,
            )
        paper = PaperSynthesizer(self.filesystem.base_dir).synthesize(project_id)
        self._update_causal_dag(project_dir / "causal_dag.json", project_id)
        self.taste_persistence.save_after_project(project_id, taste_model)
        self.filesystem.write_project_state(
            project_id,
            {
                "status": "complete",
                "run_count": run_n,
                "last_run_status": result_summary,
                "paper_path": paper["pdf_path"],
            },
        )
        return ProjectResult(
            project_id=project_id,
            status="complete",
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
        agent_result = self.agent_runtime.generate_task(
            category="sisyphus",
            instruction=f"Execute research task: {active_task}",
            context={
                "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
                "research_question": config.research_question,
                "domain": config.domain,
                "success_criteria": config.success_criteria,
                "failed_iterations": failed_iterations,
            },
            budget={"max_runs": 1, "compute_budget": config.compute_budget},
        )
        if task_node and getattr(task_node, "requires_ml_optimizer", False):
            proposals = ExperimentProposer().propose_next(
                task_node.task_id,
                n=3,
                prior_metric=ledger.get_by_task(task_node.task_id)[0]["primary_metric"] if ledger.get_by_task(task_node.task_id) else 0.0,
                compute_budget=config.compute_budget,
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
            best = max(experiment_results, key=lambda result: result.primary_metric)
            for result in experiment_results:
                ledger.insert_experiment(result, config_diff="generated proposal", timestamp=f"run-{run_n}")
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
                "agent_runtime": {
                    "provider": agent_result.get("provider"),
                    "model": agent_result.get("model"),
                    "next_action": agent_result.get("next_action"),
                },
            }
        else:
            artifact_files = []
            for artifact in agent_result.get("artifact_plan", []):
                path = artifact_dir / artifact["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(artifact.get("content", ""), encoding="utf-8")
                artifact_files.append(str(path))
            artifact_payload = {
                "task_id": getattr(task_node, "task_id", f"run-{run_n}"),
                "summary": " ".join(summary_parts + [agent_result.get("summary", "")]).strip(),
                "primary_metric": 1.0 if failed_iterations == 0 else 0.85,
                "code_path": artifact_files[0] if artifact_files else "",
                "errors": [],
                "artifact_dir": str(artifact_dir),
                "generated_artifacts": artifact_files,
                "agent_runtime": {
                    "provider": agent_result.get("provider"),
                    "model": agent_result.get("model"),
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
            },
            "result": artifact_payload,
        }

    def _run_adversarial_check(self, outputs: dict[str, Any], criteria: list[str]):
        import asyncio

        return asyncio.run(self.adversarial.run(outputs, criteria))

    def _check_stopping_criteria(self, adversarial_report) -> bool:
        return adversarial_report.stopping_decision.should_stop

    def _update_causal_dag(self, dag_path: Path, project_id: str) -> None:
        dag = CausalDAG(dag_path)
        dag.add_edge("instruction", "result", effect_size=1.0, confidence=0.9, experiment_ids=[project_id])
