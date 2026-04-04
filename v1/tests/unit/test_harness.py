from genesis.config import ProjectConfig
from genesis.harness.decomposer import TaskDecomposer
from genesis.harness.instruction_composer import InstructionComposer
from genesis.harness.token_budget import TokenBudget
from genesis.models import TaskNode, TaskTree
import pytest


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
