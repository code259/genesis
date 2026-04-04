import os

from genesis.models import ExperimentProposal
from genesis.modules.adversarial.orchestrator import AdversarialOrchestrator
from genesis.modules.optimizer.proposer import ExperimentProposer
from genesis.taste.features import ExperimentFeatureExtractor
from genesis.taste.gp_model import TasteGP


def test_benchmark_smoke_optimizer_and_taste():
    proposals = ExperimentProposer().propose_next("benchmark-task", n=3)
    extractor = ExperimentFeatureExtractor()
    features = [extractor.extract(proposal) for proposal in proposals]
    model = TasteGP()
    model.fit(features, [proposal.expected_metric for proposal in proposals], [proposal.expected_trajectory for proposal in proposals])
    means, variances = model.predict(features)
    assert len(means) == 3
    assert len(variances) == 3


def test_benchmark_smoke_adversarial():
    os.environ.setdefault("GENESIS_CACHE_ROOT", "/tmp/genesis-smoke-cache")
    report = __import__("asyncio").run(
        AdversarialOrchestrator().run(
            {
                "summary": "According to prior work, this system improves a benchmark by 10 percent.",
                "primary_metric": 0.8,
            },
            ["improves a benchmark"],
        )
    )
    assert report.total_claims >= 0
