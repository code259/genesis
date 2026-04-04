from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Claim:
    text: str


@dataclass
class InterrogationResult:
    claim: str
    why_chain: list[str]
    grounded: bool


class SocraticDebater:
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

    def interrogate(self, claim: Claim, depth: int = 5) -> InterrogationResult:
        why_chain = [claim.text]
        grounded = bool(
            re.search(r"\b(doi|arxiv|http|et al\.|according to|citation|source)\b", claim.text, re.IGNORECASE)
        ) and not bool(re.search(r"\b(without|missing|no)\b", claim.text, re.IGNORECASE))
        for index in range(1, depth):
            if grounded:
                why_chain.append(f"Why {index}: grounded by cited source in the claim.")
                break
            why_chain.append(f"Why {index}: supporting evidence is missing for '{claim.text}'.")
        return InterrogationResult(claim=claim.text, why_chain=why_chain, grounded=grounded)

    def flag_implicit_assumptions(self, results: list[InterrogationResult]) -> list[str]:
        flags = {f"IMPLICIT_ASSUMPTION:{result.claim}" for result in results if not result.grounded}
        return sorted(flags)
