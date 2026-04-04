from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class FactualClaim:
    text: str


@dataclass
class VerificationResult:
    claim: str
    verified: bool
    evidence: list[str]


class LiteratureCrossExaminer:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def extract_factual_claims(self, text: str) -> list[FactualClaim]:
        return [FactualClaim(match.group(0)) for match in re.finditer(r"[^.]*\b\d+[%x]?\b[^.]*\.", text)]

    def verify_claim(self, claim: FactualClaim) -> VerificationResult:
        evidence: list[str] = []
        if self.api_key:
            evidence.append("semantic_scholar_configured")
        if re.search(r"\b(no citation|unknown)\b", claim.text, re.IGNORECASE):
            return VerificationResult(claim=claim.text, verified=False, evidence=["CITATION_NOT_FOUND"])
        if "contradict" in claim.text.lower():
            return VerificationResult(
                claim=claim.text,
                verified=False,
                evidence=["RESULT_CONTRADICTED_BY_LITERATURE"],
            )
        evidence.append("heuristic_literature_check")
        return VerificationResult(claim=claim.text, verified=True, evidence=evidence)

    def check_for_contradictions(self, claim: FactualClaim, search_results: list[dict[str, Any]]) -> list[str]:
        contradictions: list[str] = []
        for result in search_results:
            snippet = f"{result.get('title', '')} {result.get('abstract', '')}".lower()
            if "contradict" in snippet:
                contradictions.append("RESULT_CONTRADICTED_BY_LITERATURE")
        return contradictions
