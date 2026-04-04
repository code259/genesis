from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from genesis.scholarly import ScholarlyClient


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
        cache_root = Path(os.getenv("GENESIS_CACHE_ROOT", Path(tempfile.gettempdir()) / "genesis-cache"))
        cache_path = os.getenv("GENESIS_CITATIONS_CACHE", str(cache_root / "literature_cache.json"))
        self.client = ScholarlyClient(
            cache_path=cache_path,
            session=self.session,
            semantic_scholar_api_key=self.api_key,
        )

    def extract_factual_claims(self, text: str) -> list[FactualClaim]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        claims: list[FactualClaim] = []
        seen: set[str] = set()
        for sentence in sentences:
            if not re.search(
                r"\b(?:\d+[%x]?|doi:|arxiv:|according to|et al\.|accuracy|benchmark|improves?|reduces?)\b",
                sentence,
                re.IGNORECASE,
            ):
                continue
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            claims.append(FactualClaim(sentence))
        return claims

    def verify_claim(self, claim: FactualClaim) -> VerificationResult:
        if re.search(r"\b(no citation|unknown)\b", claim.text, re.IGNORECASE):
            return VerificationResult(claim=claim.text, verified=False, evidence=["CITATION_NOT_FOUND"])

        doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", claim.text, re.IGNORECASE)
        candidates: list[dict[str, Any]] = []
        if doi_match:
            paper = self.client.get_paper(f"DOI:{doi_match.group(0)}")
            if paper:
                candidates.append(paper)
        if not candidates:
            cleaned_query = re.sub(r"\s+", " ", claim.text.strip().rstrip("."))
            candidates = self.client.search_semantic_scholar(cleaned_query, limit=5)
        if not candidates:
            title_match = re.search(r'"([^"]+)"', claim.text)
            if title_match:
                candidates = self.client.search_title(title_match.group(1))
        if not candidates:
            return VerificationResult(
                claim=claim.text,
                verified=False,
                evidence=["CITATION_NOT_FOUND"],
            )

        contradictions = self.check_for_contradictions(claim, candidates)
        evidence = [
            f"title:{candidate.get('title', '')}"
            for candidate in candidates[:3]
            if candidate.get("title")
        ]
        if doi_match:
            evidence.append(f"doi:{doi_match.group(0).lower()}")
        evidence.extend(contradictions)
        verified = bool(evidence) and not contradictions
        if not evidence:
            evidence.append("METHODOLOGY_UNSUPPORTED")
            verified = False
        return VerificationResult(claim=claim.text, verified=verified, evidence=evidence)

    def check_for_contradictions(self, claim: FactualClaim, search_results: list[dict[str, Any]]) -> list[str]:
        contradictions: set[str] = set()
        claim_text = claim.text.lower()
        for result in search_results:
            snippet = f"{result.get('title', '')} {result.get('abstract', '')}".lower()
            if "contradict" in snippet:
                contradictions.add("RESULT_CONTRADICTED_BY_LITERATURE")
            if any(term in claim_text for term in ("impossible", "guaranteed", "always")) and "limited" in snippet:
                contradictions.add("METHODOLOGY_UNSUPPORTED")
        return sorted(contradictions)
