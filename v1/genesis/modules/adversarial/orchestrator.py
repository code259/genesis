from __future__ import annotations

import asyncio

from genesis.models import AdversarialReport, CheckResult, StoppingDecision

from .formal import FormalConsistencyChecker
from .literature import LiteratureCrossExaminer
from .socratic import SocraticDebater


class AdversarialOrchestrator:
    def __init__(
        self,
        *,
        socratic: Optional[SocraticDebater] = None,
        literature: Optional[LiteratureCrossExaminer] = None,
        formal: Optional[FormalConsistencyChecker] = None,
    ):
        self.socratic = socratic or SocraticDebater()
        self.literature = literature or LiteratureCrossExaminer()
        self.formal = formal or FormalConsistencyChecker()

    async def run(self, ca_outputs: dict[str, str], acceptance_criteria: list[str]) -> AdversarialReport:
        text = ca_outputs.get("summary", "")
        claims = self.socratic.extract_claims(text)
        interrogations = await asyncio.gather(*[asyncio.to_thread(self.socratic.interrogate, claim) for claim in claims])
        factual_claims = self.literature.extract_factual_claims(text)
        literature_results = await asyncio.gather(
            *[asyncio.to_thread(self.literature.verify_claim, claim) for claim in factual_claims]
        )

        formal_checks = [
            self.formal.check_metric_plausibility(
                float(ca_outputs.get("primary_metric", 0.0)),
                theoretical_bounds=(0.0, 1.0),
            )
        ]
        if code_path := ca_outputs.get("code_path"):
            formal_checks.append(self.formal.check_parameter_count(code_path, claimed_params=1000))
            formal_checks.append(self.formal.check_implementation_drift(text, code_path))

        claim_flags = self.socratic.flag_implicit_assumptions(interrogations)
        literature_flags = [flag for result in literature_results if not result.verified for flag in result.evidence]
        acceptance_hits = sum(1 for criterion in acceptance_criteria if criterion.lower() in text.lower())
        acceptance_ratio = acceptance_hits / len(acceptance_criteria) if acceptance_criteria else 1.0
        report = AdversarialReport(
            claim_flags=claim_flags,
            literature_flags=literature_flags,
            formal_checks=[check if isinstance(check, CheckResult) else CheckResult(**check) for check in formal_checks],
            acceptance_ratio=acceptance_ratio,
            grounded_claims=sum(1 for result in interrogations if result.grounded),
            total_claims=len(interrogations),
        )
        report.stopping_decision = self._evaluate_stopping_criteria(report, acceptance_criteria)
        return report

    def _evaluate_stopping_criteria(
        self, report: AdversarialReport, criteria: list[str]
    ) -> StoppingDecision:
        critical_flags = [
            flag
            for flag in report.literature_flags
            if flag in {"RESULT_CONTRADICTED_BY_LITERATURE", "ORACLE_FAIL"}
        ]
        should_stop = (
            report.acceptance_ratio >= 0.9
            and not report.claim_flags
            and not critical_flags
            and report.grounded_claims >= max(1, report.total_claims)
        )
        reasons = ["all stopping criteria satisfied"] if should_stop else ["continue iteration"]
        return StoppingDecision(should_stop=should_stop, reasons=reasons, critical_flags=critical_flags)
from typing import Optional
