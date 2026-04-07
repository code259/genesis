from pathlib import Path

from genesis.config import ProjectConfig
from genesis.harness.decomposer import TaskDecomposer
from genesis.harness.instruction_composer import InstructionComposer
from genesis.harness.token_budget import TokenBudget
from genesis.models import TaskNode, TaskTree
import pytest
from genesis.harness.loop import MetaHarnessLoop


def test_token_budget_trims():
    budget = TokenBudget()
    trimmed = budget.trim_to_budget("word " * 2000, 100)
    assert len(trimmed.split()) < 2000


def test_instruction_composer_has_sections():
    composer = InstructionComposer()
    config = ProjectConfig(
        research_question="Test question",
        domain="general",
        compute_budget="local_cpu",
        time_budget_hours=1,
        domain_knowledge_model="none",
        output_dir="projects",
    )
    instruction = composer.compose(
        config=config,
        belief_summary="summary",
        retrieved_history="history",
        budget_allocations={"retrieved_history": 8000},
        requested_modules=["verification", "optimizer"],
    )
    assert "# Objective" in instruction
    assert "# Budget" in instruction
    assert "# Requested Modules" in instruction
    assert "# Validation Expectations" in instruction


def test_decomposer_builds_task_tree():
    decomposer = TaskDecomposer()
    config = ProjectConfig(
        research_question="Test question",
        domain="ml_efficiency",
        compute_budget="local_gpu",
        time_budget_hours=2,
        domain_knowledge_model="none",
        output_dir="projects",
        success_criteria=["baseline", "experiment"],
    )
    tree = decomposer.decompose(config)
    assert len(tree.tasks) == 5
    verification_task = next(task for task in tree.tasks if "Verify experiment outputs" in task.description)
    assert len(verification_task.dependencies) == 2
    experiment_task = next(task for task in tree.tasks if "ML efficiency experiments" in task.description or "controlled ML efficiency experiments" in task.description)
    assert experiment_task.requires_ml_optimizer is True


def test_decomposer_astrophysics_fallback_is_domain_specific():
    decomposer = TaskDecomposer()
    config = ProjectConfig(
        research_question="Estimate redshift",
        domain="astrophysics",
        compute_budget="local_gpu",
        time_budget_hours=2,
        domain_knowledge_model="none",
        output_dir="projects",
    )
    tree = decomposer.decompose(config)
    assert any("astrophysics literature" in task.description.lower() for task in tree.tasks)
    assert any("astrophysics data" in task.description.lower() for task in tree.tasks)


def test_token_budget_cloud_allocation_expands_history():
    budget = TokenBudget()
    allocation = budget.allocate(128000)
    assert allocation["retrieved_history"] >= 20000


def test_decomposer_rejects_unknown_dependencies():
    decomposer = TaskDecomposer()
    with pytest.raises(ValueError):
        decomposer._validated_tree(
            TaskTree(
                root_id="a",
                tasks=[
                    TaskNode(
                        task_id="a",
                        description="broken",
                        acceptance_criteria=[],
                        oracle_checks=[],
                        estimated_compute_budget="local_cpu",
                        dependencies=["b"],
                    )
                ],
            )
        )


def test_harness_requested_modules_expand_on_escalation(tmp_path):
    loop = MetaHarnessLoop(
        projects_root=tmp_path / "projects",
        taste_root=tmp_path / "taste_db",
        executor=lambda **kwargs: {},
    )
    config = ProjectConfig(
        research_question="Q",
        domain="astrophysics",
        compute_budget="local_gpu",
        time_budget_hours=1,
        domain_knowledge_model="none",
        output_dir="projects",
    )
    task = TaskNode(
        task_id="task",
        description="desc",
        acceptance_criteria=[],
        oracle_checks=[],
        estimated_compute_budget="local_gpu",
        requires_ml_optimizer=True,
    )
    modules = loop._requested_modules(config=config, task_node=task, failed_iterations=3)
    assert "optimizer" in modules
    assert "ideation" in modules
    assert "oracle" in modules
    assert "domain_knowledge" in modules


def test_default_executor_runs_generic_command_plan(tmp_path, monkeypatch):
    loop = MetaHarnessLoop(
        projects_root=tmp_path / "projects",
        taste_root=tmp_path / "taste_db",
    )
    project_dir = loop.filesystem.init_project(
        "demo",
        {
            "research_question": "Generate and run a script",
            "domain": "general",
            "compute_budget": "local_cpu",
            "time_budget_hours": 1,
            "domain_knowledge_model": "none",
            "output_dir": str(tmp_path / "projects"),
            "success_criteria": ["Generate and run a script"],
            "oracle_hints": [],
        },
    )
    monkeypatch.setattr(
        loop.agent_runtime,
        "generate_task",
        lambda **kwargs: {
            "summary": "Created and executed a script.",
            "artifact_plan": [
                {
                    "path": "writer.py",
                    "content": "from pathlib import Path\nPath('output.txt').write_text('done', encoding='utf-8')\nprint('ok')\n",
                }
            ],
            "command_plan": ["python3 writer.py"],
            "experiment_plan": [],
            "citations": [],
            "next_action": "continue",
            "provider": "test",
            "model": "fake-model",
        },
    )
    result = loop._default_executor(
        project_dir=project_dir,
        run_n=1,
        config=ProjectConfig(
            research_question="Generate and run a script",
            domain="general",
            compute_budget="local_cpu",
            time_budget_hours=1,
            domain_knowledge_model="none",
            output_dir=str(tmp_path / "projects"),
            success_criteria=["Generate and run a script"],
            oracle_hints=[],
        ),
        task_node=TaskNode(
            task_id="task",
            description="Generate and run a script",
            acceptance_criteria=[],
            oracle_checks=[],
            estimated_compute_budget="local_cpu",
            requires_ml_optimizer=False,
        ),
        optimizer=None,
        ledger=None,
        ideation=None,
        oracle_resolver=None,
        failed_iterations=0,
        taste_model=None,
    )
    payload = result["result"]
    assert payload["executed_commands"] == ["python3 writer.py"]
    assert any(path.endswith("output.txt") for path in payload["generated_artifacts"])
    assert Path(payload["artifact_dir"], "result.json").exists()
