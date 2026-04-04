from __future__ import annotations

import hashlib
import math

from genesis.models import ExperimentProposal


class ExperimentFeatureExtractor:
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

    def extract(self, proposal: ExperimentProposal) -> list[float]:
        digest = hashlib.sha256(proposal.description.encode("utf-8")).digest()
        embedding = [
            digest[index % len(digest)] / 255.0
            for index in range(self.embedding_dim)
        ]
        structured = [
            math.log10(max(1, len(proposal.description))),
            math.log10(max(1, proposal.model_parameter_count or 1)),
            math.log10(max(1, len(proposal.expected_trajectory) or 1)),
        ]
        return embedding + structured
