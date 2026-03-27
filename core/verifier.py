# pyre-ignore-all-errors[21]
from pathlib import Path
from core import router  # pyre-ignore[21]
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

VERIFIER_SYSTEM = """You are an adversarial reviewer. You have NO knowledge of how this output was produced.

Your job: evaluate whether this research task output is correct and complete.

For each check produce:
- Check description
- Result: PASS / FAIL / UNABLE TO VERIFY
- If FAIL: specific description of error with location (equation number, line, etc.)
- If UNABLE: what additional information would be needed

End with:
RECOMMENDATION: ACCEPT / REVISE / ESCALATE
If REVISE: specific remediation instructions
If ESCALATE: reason

Be adversarial. Assume errors exist until proven otherwise. Do not give benefit of the doubt."""

def verify(task_spec: dict, output: str, is_foundational: bool = False) -> str:
    """
    Run verification on a task output.
    For foundational results, uses cross_check role (different model/provider at each tier).
    For standard results, uses verifier role.
    """
    user_content = f"""
TASK SPECIFICATION:
{task_spec['description']}

VERIFICATION CRITERIA (what done actually looks like):
{task_spec['verification_criteria']}

OUTPUT TO REVIEW:
{output}
"""
    role = "cross_check" if is_foundational else "verifier"
    return router.call(
        role=role,
        system=VERIFIER_SYSTEM,
        user=user_content
    )
