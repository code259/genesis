from genesis.config import ProjectConfig
from genesis.harness.decomposer import TaskDecomposer
from genesis.harness.instruction_composer import InstructionComposer
from genesis.harness.token_budget import TokenBudget


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
    )
    assert "# Objective" in instruction
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
    assert len(tree.tasks) == 2
    assert tree.tasks[1].dependencies
