# pyre-ignore-all-errors[21]
from pathlib import Path
from core import router
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = Path("prompts/decomposer_system.md").read_text()
REVIEW_PROMPT = Path("prompts/adversarial_review_system.md").read_text()

def decompose(research_goal: str, domain_context: str) -> str:
    """Generate task tree for a research goal."""
    return router.call(
        role="decomposer",
        system=SYSTEM_PROMPT,
        user=f"Domain context:\n{domain_context}\n\nResearch goal:\n{research_goal}"
    )

def adversarial_review(research_goal: str, task_tree: str) -> str:
    """Review and refine the generated task tree."""
    return router.call(
        role="decomposition_reviewer",
        system=REVIEW_PROMPT,
        user=f"Research goal: {research_goal}\n\nProposed task tree:\n{task_tree}"
    )
