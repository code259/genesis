import asyncio

from genesis.modules.adversarial.criteria_generator import AcceptanceCriteriaGenerator
from genesis.modules.adversarial.socratic import SocraticDebater
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
    assert len(criteria["criteria"]) == len(set(criteria["criteria"]))


def test_adversarial_orchestrator_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_CITATIONS_CACHE", str(tmp_path / "literature_cache.json"))
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


def test_socratic_debater_flags_ungrounded_claims():
    debater = SocraticDebater()
    claims = debater.extract_claims("The model improves convergence speed significantly without citing a source.")
    assert claims
    result = debater.interrogate(claims[0])
    assert not result.grounded
    assert debater.flag_implicit_assumptions([result])
