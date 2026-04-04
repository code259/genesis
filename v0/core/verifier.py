from __future__ import annotations

# pyre-ignore-all-errors[21]
from pathlib import Path
from core import router  # pyre-ignore[21]
import sys
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))

VERIFIER_SYSTEM = Path("prompts/verifier_system.md").read_text()

def verify(task_spec: dict, output: str, oracle_result: dict | None = None, is_foundational: bool = False) -> dict:
    """
    Run verification on a task output.
    For foundational results, uses cross_check role (different model/provider at each tier).
    For standard results, uses verifier role.
    """
    oracle_summary = oracle_result or {}
    user_content = f"""
TASK SPECIFICATION:
{task_spec['description']}

VERIFICATION CRITERIA (what done actually looks like):
{task_spec['verification_criteria']}

ORACLE SUMMARY:
{oracle_summary}

OUTPUT TO REVIEW:
{output}
"""
    role = "cross_check" if is_foundational else "verifier"
    raw = router.call(
        role=role,
        system=VERIFIER_SYSTEM,
        user=user_content
    )
    parsed = parse_verification_report(raw)
    parsed["raw_text"] = raw
    parsed["oracle_summary"] = oracle_summary
    return parsed


def parse_verification_report(report: str) -> dict:
    status_match = re.search(r"RECOMMENDATION:\s*(ACCEPT|REVISE|ESCALATE)", report, re.IGNORECASE)
    status = status_match.group(1).upper() if status_match else "ESCALATE"

    checks = []
    current_check = None
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("### check") or stripped.startswith("- Check description:"):
            if current_check:
                checks.append(current_check)
            current_check = {"description": stripped, "result": "UNABLE TO VERIFY"}
        elif "Result:" in stripped and current_check is not None:
            if "PASS" in stripped:
                current_check["result"] = "PASS"
            elif "FAIL" in stripped:
                current_check["result"] = "FAIL"
            else:
                current_check["result"] = "UNABLE TO VERIFY"
        elif current_check is not None and stripped:
            current_check.setdefault("details", []).append(stripped)
    if current_check:
        checks.append(current_check)

    open_items = []
    collecting = False
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("remediation instructions"):
            collecting = True
            continue
        if collecting and stripped:
            open_items.append(stripped)

    return {
        "status": status,
        "checks": checks,
        "open_items": open_items,
    }
