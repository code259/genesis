import pytest  # pyre-ignore[21]
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import decomposer  # pyre-ignore[21]

GOAL = """
Develop a statistical correction method for batch effects in single-cell RNA-seq 
trajectory inference that preserves biological variation while removing technical 
variation, with formal derivation of the correction factor and validation on 
benchmark datasets.
"""

DOMAIN = """
Single-cell RNA-seq trajectory inference. Relevant methods: Monocle, PAGA, Scanpy.
Key concern: batch effects confound trajectory topology. Current methods apply 
correction before trajectory inference without formal justification.
"""

def test_decompose_produces_tasks(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_call(role, system, user, max_tokens=None, tier=None):
        captured["role"] = role
        captured["system"] = system
        captured["user"] = user
        return "S1T1\nVerification criteria"

    monkeypatch.setattr(decomposer.router, "call", fake_call)
    tree = decomposer.decompose(GOAL, DOMAIN)
    assert "S1T1" in tree
    assert "Verification criteria" in tree
    assert captured["role"] == "decomposer"
    assert "Research goal:" in captured["user"]

def test_adversarial_review_runs(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_call(role, system, user, max_tokens=None, tier=None):
        captured["role"] = role
        captured["system"] = system
        captured["user"] = user
        return '{"review": "ok"}'

    monkeypatch.setattr(decomposer.router, "call", fake_call)
    review = decomposer.adversarial_review(GOAL, "S1T1")
    assert review == '{"review": "ok"}'
    assert captured["role"] == "decomposer"
    assert "Research goal:" in captured["user"]
    assert "Proposed task tree:" in captured["user"]
