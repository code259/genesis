from __future__ import annotations

import hashlib
import math
import re

from genesis.models import ExperimentProposal


class ExperimentFeatureExtractor:
    def __init__(self, embedding_dim: int = 96):
        self.embedding_dim = embedding_dim

    def extract(self, proposal: ExperimentProposal) -> list[float]:
        embedding = self._text_embedding(f"{proposal.description} {proposal.code_diff}")
        trajectory = [float(value) for value in proposal.expected_trajectory]
        trajectory_mean = sum(trajectory) / len(trajectory) if trajectory else 0.0
        trajectory_span = (max(trajectory) - min(trajectory)) if trajectory else 0.0
        compute_budget = proposal.compute_budget.lower()
        structured = [
            math.log10(max(1, len(proposal.description))),
            math.log10(max(1, proposal.model_parameter_count or 1)),
            math.log10(max(1, len(proposal.expected_trajectory) or 1)),
            round(trajectory_mean, 6),
            round(trajectory_span, 6),
            1.0 if "gpu" in compute_budget else 0.0,
            1.0 if "cpu" in compute_budget else 0.0,
            1.0 if any(token in proposal.description.lower() for token in ("warmup", "optimizer", "learning rate")) else 0.0,
        ]
        return embedding + structured

    def _text_embedding(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.embedding_dim)]
        tokens = re.findall(r"[a-z0-9_]{2,}", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, min(len(digest), self.embedding_dim), 2):
                bucket = (digest[offset] + digest[offset + 1]) % self.embedding_dim
                vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [round(value / norm, 6) for value in vector]
