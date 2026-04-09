from __future__ import annotations

from hashlib import sha1
from typing import Iterable, Optional

from genesis.agents.runtime import CodingAgentRuntime, ProviderRuntimeError

from genesis.config import ProjectConfig
from genesis.models import TaskNode, TaskTree


class TaskDecomposer:
    MAX_DEPTH = 4
    MAX_BREADTH = 6

    def __init__(self, runtime: Optional[CodingAgentRuntime] = None):
        self.runtime = runtime

    def decompose(self, config: ProjectConfig) -> TaskTree:
        if self.runtime is not None:
            try:
                generated = self.runtime.generate_task(
                    category="genesis-ideation",
                    instruction=f"Produce a bounded research task DAG for: {config.research_question}",
                    context=config.to_dict(),
                    budget={"max_depth": 4, "max_breadth": 6},
                )
                task_tree = self._parse_runtime_task_tree(generated)
                if task_tree.tasks:
                    return task_tree
            except ProviderRuntimeError:
                pass
        tasks: list[TaskNode] = []
        root_id = self._task_id(config.research_question, "root")
        literature_description = f"Survey prior work relevant to: {config.research_question}"
        experiment_description = f"Run controlled experiments for: {config.research_question}"
        if config.domain == "astrophysics":
            literature_description = f"Survey astrophysics literature and datasets relevant to: {config.research_question}"
            experiment_description = f"Download, preprocess, and analyze astrophysics data for: {config.research_question}"
        elif config.domain == "ml_efficiency":
            literature_description = f"Survey prior ML systems and optimization work relevant to: {config.research_question}"
            experiment_description = f"Run controlled ML efficiency experiments for: {config.research_question}"
        elif config.domain == "general":
            literature_description = f"Survey prior work and methods relevant to: {config.research_question}"
            experiment_description = f"Run task-grounded experiments or artifact generation for: {config.research_question}"

        literature = self._task(
            config,
            description=literature_description,
            suffix="literature",
            dependencies=[],
            success_metric="produce grounded literature map",
            requires_ml_optimizer=False,
            task_kind="survey",
            execution_mode="planning",
            expected_artifacts=["literature_review.md", "source_map.json"],
        )
        oracle = self._task(
            config,
            description=f"Generate and validate an oracle for domain {config.domain}",
            suffix="oracle",
            dependencies=[literature.task_id],
            success_metric="validated oracle coverage",
            requires_ml_optimizer=False,
            task_kind="oracle",
            execution_mode="artifact_generation",
            expected_artifacts=["oracle.py", "oracle_validation.json"],
        )
        experiment = self._task(
            config,
            description=experiment_description,
            suffix="experiments",
            dependencies=[literature.task_id],
            success_metric=config.success_criteria[0] if config.success_criteria else "improve primary metric",
            requires_ml_optimizer=config.domain in {"ml_efficiency", "astrophysics"},
            task_kind="acquire_data" if config.domain == "astrophysics" else "analyze",
            execution_mode="command_execution",
            expected_artifacts=(
                ["sample_data.json", "analysis_notes.md", "validation_runner.py"]
                if config.domain == "astrophysics"
                else ["analysis_output.json", "experiment_summary.md"]
            ),
        )
        verification = self._task(
            config,
            description="Verify experiment outputs against oracle and literature",
            suffix="verification",
            dependencies=[oracle.task_id, experiment.task_id],
            success_metric="verification passed",
            requires_ml_optimizer=False,
            task_kind="verify",
            execution_mode="verification",
            expected_artifacts=["verification_report.json"],
        )
        synthesis = self._task(
            config,
            description="Synthesize the final paper and result package",
            suffix="paper",
            dependencies=[verification.task_id, literature.task_id],
            success_metric="paper artifact generated",
            requires_ml_optimizer=False,
            task_kind="paper",
            execution_mode="paper",
            expected_artifacts=["main.tex", "main.pdf", "synthesis_report.json"],
        )
        tasks.extend([literature, oracle, experiment, verification, synthesis])
        return self._validated_tree(TaskTree(root_id=root_id, tasks=tasks))

    def amend(self, tree: TaskTree, rationale: str) -> TaskTree:
        if not tree.tasks:
            return tree
        amended = list(tree.tasks)
        amended.append(
            TaskNode(
                task_id=self._task_id(rationale, f"amend-{len(amended) + 1}"),
                description=f"Amendment task: {rationale}",
                acceptance_criteria=[f"Amendment rationale recorded: {rationale}"],
                oracle_checks=[],
                estimated_compute_budget=amended[-1].estimated_compute_budget,
                dependencies=[amended[-1].task_id],
                success_metric="amendment incorporated",
                requires_ml_optimizer=False,
            )
        )
        return self._validated_tree(TaskTree(root_id=tree.root_id, tasks=amended))

    def _parse_runtime_task_tree(self, payload: dict[str, object]) -> TaskTree:
        nodes = payload.get("task_tree")
        if not isinstance(nodes, list):
            return TaskTree(root_id="root", tasks=[])
        tasks: list[TaskNode] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            task_id = str(node.get("task_id") or self._task_id(str(node.get("description", "")), "runtime"))
            tasks.append(
                TaskNode(
                    task_id=task_id,
                    description=str(node.get("description", "")),
                    acceptance_criteria=[str(item) for item in node.get("acceptance_criteria", [])],
                    oracle_checks=[str(item) for item in node.get("oracle_checks", [])],
                    estimated_compute_budget=str(node.get("estimated_compute_budget", "local_cpu")),
                    dependencies=[str(item) for item in node.get("dependencies", [])],
                    success_metric=str(node.get("success_metric", "")),
                    requires_ml_optimizer=bool(node.get("requires_ml_optimizer", False)),
                    task_kind=str(node.get("task_kind") or self._infer_task_kind(str(node.get("description", "")))),
                    expected_artifacts=[str(item) for item in node.get("expected_artifacts", []) if str(item).strip()],
                    execution_mode=str(node.get("execution_mode") or self._default_execution_mode(str(node.get("task_kind") or ""))),
                )
            )
        root_id = tasks[0].task_id if tasks else "root"
        return self._validated_tree(TaskTree(root_id=root_id, tasks=tasks))

    def _task(
        self,
        config: ProjectConfig,
        *,
        description: str,
        suffix: str,
        dependencies: Iterable[str],
        success_metric: str,
        requires_ml_optimizer: bool,
        task_kind: str,
        execution_mode: str,
        expected_artifacts: list[str],
    ) -> TaskNode:
        return TaskNode(
            task_id=self._task_id(config.research_question, suffix),
            description=description,
            acceptance_criteria=self._acceptance_criteria(
                config,
                description=description,
                success_metric=success_metric,
                task_kind=task_kind,
                expected_artifacts=expected_artifacts,
            ),
            oracle_checks=config.oracle_hints.copy(),
            estimated_compute_budget=config.compute_budget,
            dependencies=list(dependencies),
            success_metric=success_metric,
            requires_ml_optimizer=requires_ml_optimizer,
            task_kind=task_kind,
            expected_artifacts=list(expected_artifacts),
            execution_mode=execution_mode,
        )

    def _task_id(self, question: str, suffix: str) -> str:
        return sha1(f"{question}:{suffix}".encode("utf-8")).hexdigest()[:8]

    def _acceptance_criteria(
        self,
        config: ProjectConfig,
        *,
        description: str,
        success_metric: str,
        task_kind: str,
        expected_artifacts: list[str],
    ) -> list[str]:
        if task_kind == "survey":
            criteria = [
                "Produce a grounded literature review for the active research question.",
                "Produce a source map that ties workflow assumptions to concrete references or official documentation.",
            ]
            if "methodology_note.md" in expected_artifacts:
                criteria.append("Produce a concise methodology note describing the survey-stage workflow.")
            if config.domain == "astrophysics":
                criteria.append("Use domain-relevant astrophysics references and official data-access documentation when available.")
            return criteria
        if task_kind == "oracle":
            return [
                "Produce an oracle implementation artifact.",
                "Demonstrate that the oracle passes synthetic validation.",
            ]
        if task_kind in {"acquire_data", "analyze"}:
            criteria = [
                "Produce substantive execution artifacts for the active analysis task.",
                f"Track success using: {success_metric}",
            ]
            if expected_artifacts:
                criteria.append(f"Expected artifacts: {', '.join(expected_artifacts)}")
            return criteria
        if task_kind == "verify":
            return [
                "Run verification against upstream outputs and the oracle.",
                "Verification must pass for the task to complete.",
            ]
        if task_kind == "paper":
            return [
                "Produce a paper/report artifact package from verified upstream results.",
                "Include substantive narrative sections and supporting artifacts.",
            ]
        return [description, f"Track success using: {success_metric}"]

    def _validated_tree(self, tree: TaskTree) -> TaskTree:
        if len(tree.tasks) > self.MAX_BREADTH * self.MAX_DEPTH:
            raise ValueError("task tree exceeds maximum supported size")
        task_ids = {task.task_id for task in tree.tasks}
        for task in tree.tasks:
            missing = [dependency for dependency in task.dependencies if dependency not in task_ids]
            if missing:
                raise ValueError(f"task {task.task_id} has unknown dependencies: {missing}")
        return tree

    def _infer_task_kind(self, description: str) -> str:
        lowered = description.lower()
        if "oracle" in lowered:
            return "oracle"
        if "verify" in lowered or "validation" in lowered:
            return "verify"
        if "paper" in lowered or "synthesize" in lowered:
            return "paper"
        if "survey" in lowered or "literature" in lowered:
            return "survey"
        if "download" in lowered or "acquire" in lowered or "preprocess" in lowered or "data" in lowered:
            return "acquire_data"
        return "analyze"

    def _default_execution_mode(self, task_kind: str) -> str:
        if task_kind == "survey":
            return "planning"
        if task_kind == "verify":
            return "verification"
        if task_kind == "paper":
            return "paper"
        if task_kind == "oracle":
            return "artifact_generation"
        return "command_execution"
