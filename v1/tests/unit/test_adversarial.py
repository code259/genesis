import asyncio

from genesis.config import ProjectConfig
from genesis.models import CheckResult, ClaimFinding, CriteriaFinding, LiteratureFinding
from genesis.modules.adversarial.criteria_generator import AcceptanceCriteriaGenerator
from genesis.modules.adversarial.orchestrator import AdversarialOrchestrator
from genesis.modules.adversarial.socratic import SocraticDebater


class _Runtime:
    def generate_acceptance_criteria(self, config):
        return [
            f"Answer: {config.research_question}",
            "Provide a verifiable artifact.",
        ]

    def analyze_criteria(self, *, criteria, completed_output, iteration, blockers, task_context):
        return {
            "criteria_findings": [
                {
                    "criterion": criterion,
                    "passed": "artifact" in criterion.lower(),
                    "severity": "high" if "artifact" not in criterion.lower() else "low",
                    "rationale": "stub",
                    "evidence_refs": completed_output.get("generated_artifacts", []),
                }
                for criterion in criteria
            ],
            "critical_blockers": [
                f"criterion_failed:{criterion}"
                for criterion in criteria
                if "artifact" not in criterion.lower()
            ],
            "stop_recommendation": False,
        }

    def analyze_claims(self, *, claims, evidence_context):
        return {
            "claim_findings": [
                {
                    "claim": claim,
                    "classification": "GROUNDED" if "artifact" in claim.lower() else "IMPLICIT_ASSUMPTION",
                    "rationale": "stub",
                    "evidence_refs": evidence_context.get("artifacts", []),
                    "why_chain": ["why1", "why2"],
                }
                for claim in claims
            ]
        }

    def analyze_literature(self, *, claim, search_results):
        return {
            "contradicted": "wrong" in claim.lower(),
            "rationale": "stub literature",
            "evidence_refs": ["paper:1"] if search_results else [],
        }


def test_criteria_generator_uses_runtime():
    config = ProjectConfig(
        research_question="Q",
        domain="general",
        compute_budget="local_cpu",
        time_budget_hours=1,
        domain_knowledge_model="none",
        output_dir="projects",
    )
    criteria = AcceptanceCriteriaGenerator(runtime=_Runtime()).generate(config)
    assert any("Provide a verifiable artifact." == item for item in criteria["criteria"])
    assert len(criteria["criteria"]) == len(set(criteria["criteria"]))


def test_adversarial_orchestrator_runs_with_structured_findings(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_CITATIONS_CACHE", str(tmp_path / "literature_cache.json"))
    orchestrator = AdversarialOrchestrator(runtime=_Runtime())
    report = asyncio.run(
        orchestrator.run(
            {
                "summary": "Artifact proves the task is complete.",
                "primary_metric": 0.95,
                "generated_artifacts": ["notes.md"],
                "code_path": str(tmp_path / "notes.md"),
            },
            ["Provide a verifiable artifact."],
            task_context={"task_id": "task-1", "stage": "execute"},
            verification={"passed": True, "checks": []},
        )
    )
    assert report.criteria_findings
    assert report.claim_findings
    assert report.task_id == "task-1"
    assert report.stage == "execute"


def test_socratic_debater_analyzes_claims_with_runtime():
    debater = SocraticDebater(runtime=_Runtime())
    findings = debater.analyze_claims(
        "The artifact is evidence of convergence. The model improves speed.",
        {"artifacts": ["notes.md"]},
    )
    assert findings
    assert any(finding.classification == "GROUNDED" for finding in findings)
