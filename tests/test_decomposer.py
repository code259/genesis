import pytest  # pyre-ignore[21]
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.decomposer import decompose, adversarial_review  # pyre-ignore[21]

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

def test_decompose_produces_tasks():
    tree = decompose(GOAL, DOMAIN)
    assert "S1T1" in tree
    assert "Verification criteria" in tree
    print(tree)

def test_adversarial_review_runs():
    tree = decompose(GOAL, DOMAIN)
    review = adversarial_review(tree, GOAL)
    print(review)
