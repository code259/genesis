import re
from dataclasses import dataclass
from enum import Enum

class Decision(Enum):
    ACCEPT = "accept"
    TRIGGER_VERIFY = "trigger_verify"
    TRIGGER_SOFT_VERIFY = "trigger_soft_verify"  
    HOLD_STAGE_GATE = "hold_stage_gate"
    ESCALATE_HUMAN = "escalate_human"

@dataclass
class SupervisorDecision:
    decision: Decision
    reasons: list[str]
    task_id: str

# Phrases that indicate step-skipping
SKIP_PHRASES = [
    "this becomes", "for consistency", "it follows that",
    "clearly", "one can show", "it is straightforward"
]

# Phrases that indicate fake verification  
FAKE_VERIFY_PHRASES = [
    "verified", "confirmed", "checked", "validated"
]

def evaluate_output(task_id: str, output: str, task_spec: dict) -> SupervisorDecision:
    reasons = []
    decision = Decision.ACCEPT
    
    output_lower = output.lower()
    
    # Check 1: fake verification
    for phrase in FAKE_VERIFY_PHRASES:
        if phrase in output_lower:
            if "checks performed:" not in output_lower:
                reasons.append(f"Contains '{phrase}' but no CHECKS PERFORMED section")
                decision = Decision.TRIGGER_VERIFY
                break
    
    # Check 2: step-skipping phrases
    for phrase in SKIP_PHRASES:
        if phrase in output_lower:
            reasons.append(f"Contains step-skipping phrase: '{phrase}'")
            if decision == Decision.ACCEPT:
                decision = Decision.TRIGGER_SOFT_VERIFY
    
    # Check 3: numerical result without derivation
    has_number = bool(re.search(r'\d+\.\d+', output))
    has_derivation = any(w in output_lower for w in ["derivation", "derivng", "calculate", "integral", "sum"])
    if has_number and not has_derivation and task_spec.get("complexity") == "HIGH":
        reasons.append("Numerical result appears without derivation trace")
        decision = Decision.TRIGGER_VERIFY
    
    # Check 4: INCOMPLETE present — hold gate
    if "INCOMPLETE" in output:
        reasons.append("Task contains INCOMPLETE markers — not ready for stage gate")
        decision = Decision.HOLD_STAGE_GATE
    
    # Check 5: checks not performed section non-empty
    if "checks not performed:" in output_lower:
        idx: int = output_lower.index("checks not performed:")
        stop_idx: int = min(idx + 500, len(output))
        not_performed = "".join(output[i] for i in range(idx, stop_idx))
        if len(not_performed.strip()) > len("CHECKS NOT PERFORMED:") + 10:
            reasons.append("Mandatory checks not performed")
            if task_spec.get("complexity") == "HIGH":
                decision = Decision.HOLD_STAGE_GATE
    
    if not reasons:
        reasons.append("No failure mode signatures detected")
    
    return SupervisorDecision(decision=decision, reasons=reasons, task_id=task_id)

def check_iteration_count(task_id: str, error_history: dict) -> bool:
    """Returns True if human escalation needed (same error 3+ times)."""
    return error_history.get(task_id, 0) >= 3

def generate_escalation_report(task_id: str, reasons: list[str], 
                                 attempts: int, last_output: str) -> str:
    summary_len = min(500, len(last_output))
    summary = "".join(last_output[i] for i in range(summary_len))
    return f"""
ESCALATION REPORT
Task: {task_id}
Attempts: {attempts}
Reasons for escalation: {', '.join(reasons)}

What the model has tried:
[See task output file for full history]

What the human needs to decide:
- Is the verification criteria for this task achievable with current information?
- Is there a simpler decomposition of this task?
- Should this task be marked INCOMPLETE and flagged for post-processing?

Last output summary (first 500 chars):
{summary}
"""
