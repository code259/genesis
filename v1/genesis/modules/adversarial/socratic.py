from __future__ import annotations

import re
from dataclasses import dataclass

from genesis.models import ClaimFinding

from .runtime import AdversarialRuntime

@dataclass
class Claim:
    text: str


@dataclass
class InterrogationResult:
    claim: str
    why_chain: list[str]
    grounded: bool


class SocraticDebater:
    def __init__(self, runtime: AdversarialRuntime | None = None):
        self.runtime = runtime or AdversarialRuntime()

    def extract_claims(self, text: str) -> list[Claim]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        claims: list[Claim] = []
        for sentence in sentences:
            tokens = sentence.split()
            if len(tokens) < 5:
                continue
            if not re.search(r"\b(is|are|improves?|reduces?|achieves?|shows?|demonstrates?)\b", sentence, re.IGNORECASE):
                continue
            claims.append(Claim(text=sentence))
        return claims

    def analyze_claims(self, text: str, evidence_context: dict[str, object]) -> list[ClaimFinding]:
        claims = self.extract_claims(text)
        if not claims:
            return []
        try:
            payload = self.runtime.analyze_claims(
                claims=[claim.text for claim in claims],
                evidence_context=evidence_context,
            )
        except Exception:  # noqa: BLE001
            return [self._fallback_finding(claim) for claim in claims]
        findings: list[ClaimFinding] = []
        for item in payload.get("claim_findings", []):
            if not isinstance(item, dict):
                continue
            findings.append(
                ClaimFinding(
                    claim=str(item.get("claim", "")).strip(),
                    classification=str(item.get("classification", "IMPLICIT_ASSUMPTION")).strip().upper(),
                    rationale=str(item.get("rationale", "")).strip(),
                    evidence_refs=[str(ref) for ref in item.get("evidence_refs", []) if str(ref).strip()],
                    why_chain=[str(step) for step in item.get("why_chain", []) if str(step).strip()],
                )
            )
        return findings or [self._fallback_finding(claim) for claim in claims]

    def interrogate(self, claim: Claim, depth: int = 5) -> InterrogationResult:
        why_chain = [claim.text]
        has_citation = bool(
            re.search(r"\b(doi|arxiv|http|et al\.|according to|citation|source)\b", claim.text, re.IGNORECASE)
        )
        has_measured_result = bool(re.search(r"\b\d+(\.\d+)?(%|x)?\b", claim.text)) and bool(
            re.search(r"\b(metric|accuracy|loss|score|result|artifact|experiment|verification)\b", claim.text, re.IGNORECASE)
        )
        explicitly_unsupported = bool(re.search(r"\b(without|missing|no evidence|unclear)\b", claim.text, re.IGNORECASE))
        grounded = (has_citation or has_measured_result) and not explicitly_unsupported
        for index in range(1, depth):
            if grounded:
                why_chain.append(f"Why {index}: grounded by cited source in the claim.")
                break
            why_chain.append(f"Why {index}: supporting evidence is missing for '{claim.text}'.")
        return InterrogationResult(claim=claim.text, why_chain=why_chain, grounded=grounded)

    def flag_implicit_assumptions(self, results: list[InterrogationResult]) -> list[str]:
        flags = {
            f"IMPLICIT_ASSUMPTION:{result.claim}"
            for result in results
            if not result.grounded and len(result.claim.split()) >= 7
        }
        return sorted(flags)

    def _fallback_finding(self, claim: Claim) -> ClaimFinding:
        interrogation = self.interrogate(claim)
        return ClaimFinding(
            claim=claim.text,
            classification="GROUNDED" if interrogation.grounded else "IMPLICIT_ASSUMPTION",
            rationale=interrogation.why_chain[-1] if interrogation.why_chain else "",
            evidence_refs=[],
            why_chain=interrogation.why_chain,
        )
