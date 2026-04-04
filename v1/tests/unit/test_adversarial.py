import asyncio

from genesis.modules.adversarial.criteria_generator import AcceptanceCriteriaGenerator
from genesis.modules.adversarial.orchestrator import AdversarialOrchestrator


def test_criteria_generator_defaults():
    from genesis.config import ProjectConfig

    config = ProjectConfig(
        research_question="Q",
        domain="general",
        compute_budget="local_cpu",
        time_budget_hours=1,
        domain_knowledge_model="none",
        output_dir="projects",
    )
    criteria = AcceptanceCriteriaGenerator().generate(config)
    assert criteria["criteria"]


def test_adversarial_orchestrator_runs():
    orchestrator = AdversarialOrchestrator()
    report = asyncio.run(
        orchestrator.run(
            {
                "summary": "According to evidence, improve convergence by 20 percent.",
                "primary_metric": 0.95,
            },
            ["improve convergence"],
        )
    )
    assert report.acceptance_ratio >= 0.0
