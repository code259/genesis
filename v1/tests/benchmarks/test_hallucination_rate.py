import asyncio

from genesis.modules.adversarial.orchestrator import AdversarialOrchestrator


def _dataset():
    hallucinated = [
        {
            "summary": f"This result improves accuracy by {index + 10} percent with no citation or grounding.",
            "primary_metric": 0.7,
        }
        for index in range(25)
    ]
    clean = [
        {
            "summary": f"According to prior work, this benchmark improves by {index + 1} percent with grounded evidence.",
            "primary_metric": 0.7,
        }
        for index in range(25)
    ]
    return hallucinated, clean


def _flagged(report) -> bool:
    return bool(report.claim_flags or report.literature_flags)


def test_hallucination_rate_benchmark():
    orchestrator = AdversarialOrchestrator()
    hallucinated, clean = _dataset()
    hallucinated_results = [
        asyncio.run(orchestrator.run(sample, ["improves accuracy"]))
        for sample in hallucinated
    ]
    clean_results = [
        asyncio.run(orchestrator.run(sample, ["grounded evidence"]))
        for sample in clean
    ]
    true_positive_rate = sum(1 for report in hallucinated_results if _flagged(report)) / len(hallucinated_results)
    false_positive_rate = sum(1 for report in clean_results if _flagged(report)) / len(clean_results)
    assert true_positive_rate >= 0.8
    assert false_positive_rate <= 0.2
