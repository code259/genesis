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
        return [Claim(text=sentence) for sentence in sentences if len(sentence.split()) > 5]

    def interrogate(self, claim: Claim, depth: int = 5) -> InterrogationResult:
        why_chain = [claim.text]
        grounded = bool(re.search(r"\b(doi|arxiv|http|et al\.|according to)\b", claim.text, re.IGNORECASE))
        for index in range(1, depth):
            why_chain.append(f"Why {index}: because {claim.text.lower()}")
        return InterrogationResult(claim=claim.text, why_chain=why_chain, grounded=grounded)

    def flag_implicit_assumptions(self, results: list[InterrogationResult]) -> list[str]:
        return [f"IMPLICIT_ASSUMPTION:{result.claim}" for result in results if not result.grounded]
