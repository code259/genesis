from __future__ import annotations

from typing import Any, Optional

from genesis.models import (
    AdversarialReport,
    CheckResult,
    ClaimFinding,
    CriteriaFinding,
    LiteratureFinding,
    StoppingDecision,
)

from .formal import FormalConsistencyChecker
from .literature import LiteratureCrossExaminer
from .runtime import AdversarialRuntime
from .socratic import SocraticDebater


class AdversarialOrchestrator:
    def __init__(
        self,
        *,
        runtime: Optional[AdversarialRuntime] = None,
        socratic: Optional[SocraticDebater] = None,
        literature: Optional[LiteratureCrossExaminer] = None,
        formal: Optional[FormalConsistencyChecker] = None,
    ):
        self.runtime = runtime or AdversarialRuntime()
        self.socratic = socratic or SocraticDebater(runtime=self.runtime)
        self.literature = literature or LiteratureCrossExaminer(runtime=self.runtime)
        self.formal = formal or FormalConsistencyChecker()

    async def run(
        self,
        completed_output: dict[str, Any],
        acceptance_criteria: list[str],
        *,
        task_context: Optional[dict[str, Any]] = None,
        verification: Optional[dict[str, Any]] = None,
        oracle_result: Optional[dict[str, Any]] = None,
        iteration_cap: int = 3,
    ) -> AdversarialReport:
        text = str(completed_output.get("summary", ""))
        generated_criteria = list(acceptance_criteria)
        criteria_findings, criteria_blockers, iteration_count = self._run_criteria_attacker(
            generated_criteria,
            completed_output,
            task_context or {},
            iteration_cap=iteration_cap,
        )
        evidence_context = {
            "task_context": task_context or {},
            "verification": verification or {},
            "oracle_result": oracle_result or {},
            "artifacts": completed_output.get("generated_artifacts", []),
            "code_path": completed_output.get("code_path", ""),
            "summary": text,
        }
        claim_findings = self.socratic.analyze_claims(text, evidence_context)
        major_claims = [finding.claim for finding in claim_findings if finding.classification in {"IMPLICIT_ASSUMPTION", "CONTRADICTED"}]
        literature_findings = [self.literature.analyze_claim(claim) for claim in major_claims]

        formal_checks = [
            self.formal.check_metric_plausibility(
                float(completed_output.get("primary_metric", 0.0)),
                theoretical_bounds=(0.0, 1.0),
            )
        ]
        if code_path := completed_output.get("code_path"):
            formal_checks.append(self.formal.check_parameter_count(code_path, claimed_params=1000))
            formal_checks.append(self.formal.check_implementation_drift(text, code_path))
        if oracle_result and "passed" in oracle_result:
            formal_checks.append(
                CheckResult(
                    name="oracle_gate",
                    passed=bool(oracle_result.get("passed")),
                    evidence=[jsonable(oracle_result)],
                )
            )

        claim_flags = [f"IMPLICIT_ASSUMPTION:{finding.claim}" for finding in claim_findings if finding.classification == "IMPLICIT_ASSUMPTION"]
        literature_flags = [
            f"RESULT_CONTRADICTED_BY_LITERATURE:{finding.claim}"
            for finding in literature_findings
            if finding.contradicted
        ]
        acceptance_ratio = (
            sum(1 for finding in criteria_findings if finding.passed) / len(criteria_findings)
            if criteria_findings
            else 1.0
        )
        grounded_claims = sum(1 for finding in claim_findings if finding.classification == "GROUNDED")
        total_claims = len(claim_findings)
        critical_blockers = sorted(
            set(criteria_blockers)
            | {check.name for check in formal_checks if not check.passed}
            | {flag for flag in literature_flags}
            | {flag for flag in claim_flags}
        )
        report = AdversarialReport(
            generated_criteria=generated_criteria,
            criteria_findings=criteria_findings,
            claim_findings=claim_findings,
            literature_findings=literature_findings,
            claim_flags=claim_flags,
            literature_flags=literature_flags,
            formal_checks=formal_checks,
            acceptance_ratio=acceptance_ratio,
            grounded_claims=grounded_claims,
            total_claims=total_claims,
            critical_blockers=critical_blockers,
            iteration_count=iteration_count,
            task_id=str((task_context or {}).get("task_id", "")),
            stage=str((task_context or {}).get("stage", "")),
        )
        report.stopping_decision = self._evaluate_stopping_criteria(report)
        return report

    def _run_criteria_attacker(
        self,
        criteria: list[str],
        completed_output: dict[str, Any],
        task_context: dict[str, Any],
        *,
        iteration_cap: int,
    ) -> tuple[list[CriteriaFinding], list[str], int]:
        findings: dict[str, CriteriaFinding] = {}
        blockers: set[str] = set()
        seen_blockers: set[str] = set()
        iterations = 0
        for iteration in range(1, iteration_cap + 1):
            iterations = iteration
            try:
                payload = self.runtime.analyze_criteria(
                    criteria=criteria,
                    completed_output=completed_output,
                    iteration=iteration,
                    blockers=sorted(blockers),
                    task_context=task_context,
                )
            except Exception:  # noqa: BLE001
                payload = self._fallback_criteria_payload(criteria, completed_output)
            new_blockers = {
                str(blocker).strip()
                for blocker in payload.get("critical_blockers", [])
                if str(blocker).strip()
            }
            for item in payload.get("criteria_findings", []):
                if not isinstance(item, dict):
                    continue
                criterion = str(item.get("criterion", "")).strip()
                if not criterion:
                    continue
                findings[criterion] = CriteriaFinding(
                    criterion=criterion,
                    passed=bool(item.get("passed", False)),
                    severity=str(item.get("severity", "medium")).strip() or "medium",
                    rationale=str(item.get("rationale", "")).strip(),
                    evidence_refs=[str(ref) for ref in item.get("evidence_refs", []) if str(ref).strip()],
                )
            blockers |= new_blockers
            if not new_blockers or new_blockers <= seen_blockers:
                break
            seen_blockers |= new_blockers
        return list(findings.values()), sorted(blockers), iterations

    def _fallback_criteria_payload(self, criteria: list[str], completed_output: dict[str, Any]) -> dict[str, Any]:
        summary = str(completed_output.get("summary", "")).lower()
        artifacts = completed_output.get("generated_artifacts", [])
        findings = []
        blockers = []
        for criterion in criteria:
            passed = criterion.lower() in summary or ("artifact" in criterion.lower() and bool(artifacts))
            findings.append(
                {
                    "criterion": criterion,
                    "passed": passed,
                    "severity": "high" if not passed else "low",
                    "rationale": "Fallback criteria check used summary/artifact heuristics.",
                    "evidence_refs": [str(path) for path in artifacts[:3]],
                }
            )
            if not passed:
                blockers.append(f"criterion_failed:{criterion}")
        return {"criteria_findings": findings, "critical_blockers": blockers, "stop_recommendation": not blockers}

    def _evaluate_stopping_criteria(self, report: AdversarialReport) -> StoppingDecision:
        failed_formal_checks = [check.name for check in report.formal_checks if not check.passed]
        unresolved_criteria = [
            finding.criterion
            for finding in report.criteria_findings
            if not finding.passed and finding.severity.lower() in {"high", "critical"}
        ]
        unsupported_major_claims = [
            finding.claim
            for finding in report.claim_findings
            if finding.classification in {"IMPLICIT_ASSUMPTION", "CONTRADICTED"}
        ]
        contradicted_claims = [finding.claim for finding in report.literature_findings if finding.contradicted]
        critical_flags = failed_formal_checks + unresolved_criteria + unsupported_major_claims + contradicted_claims
        should_stop = not critical_flags and bool(report.criteria_findings) and report.acceptance_ratio >= 1.0
        reasons = ["all stopping criteria satisfied"] if should_stop else ["continue iteration"]
        reasons.extend(f"formal_check_failed:{name}" for name in failed_formal_checks)
        reasons.extend(f"criterion_unresolved:{criterion}" for criterion in unresolved_criteria)
        reasons.extend(f"unsupported_claim:{claim}" for claim in unsupported_major_claims)
        reasons.extend(f"contradicted_claim:{claim}" for claim in contradicted_claims)
        return StoppingDecision(should_stop=should_stop, reasons=reasons, critical_flags=critical_flags)


def jsonable(payload: dict[str, Any]) -> str:
    return str({key: value for key, value in payload.items() if key != "checks"})
