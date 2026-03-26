# Orchestrate — v1 Implementation Guide

## Philosophy

Build the narrowest thing that proves the core insight. The core insight is: a supervisor layer with domain-specific cross-check triggers + structured file tree + adversarial verification produces dramatically better research outputs than a single long conversation.

v1 proves this on one problem in one domain. Everything else is v2.

Do not add components until the prior component works. Each phase has a concrete test that must pass before moving on.

---

## Stack

```
Python 3.11+
Anthropic SDK (Claude — Haiku for supervisor, Sonnet for execution/verification)
OpenAI SDK (GPT-4o — for cross-provider verification on foundational results)
Plain markdown files (file tree — no database, no vector store)
pytest (oracle test suite)
```

No frameworks. No LangChain. No AutoGen. Direct API calls with explicit prompts. You need to see exactly what's happening at every step.

Cost target: under $50 total for building and testing v1 including all iteration.

---

## Repository Structure

```
orchestrate/
  core/
    decomposer.py
    executor.py
    supervisor.py
    verifier.py
    convention_manager.py
    state_manager.py
    router.py              # model selection logic
  prompts/
    decomposer_system.md
    executor_system.md
    verifier_system.md
    supervisor_system.md
    constraints.md         # anti-hallucination rules, baked into executor
  oracle/
    genomics/
      benchmark_checks.py
      calibration_checks.py
      enrichment_checks.py
  tests/
    test_decomposer.py
    test_supervisor_heuristics.py
    test_verifier.py
    test_oracle.py
  projects/               # runtime — gitignored
    {project_id}/
      master_plan.md
      conventions.md
      global_state.md
      constraints.md
      stages/
  scripts/
    new_project.py        # initialize a project
    run_stage.py          # execute a single stage
    review_stage.py       # human review interface (CLI)
    audit.py              # generate audit trail report
  config.py
  requirements.txt
```

---

## Phase 0 — Environment Setup

**Time estimate:** 2 hours  
**Cost:** $0

```bash
git init orchestrate
cd orchestrate
python -m venv venv
source venv/bin/activate
pip install anthropic openai pytest python-dotenv
```

`.env`:
```
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
```

`config.py`:
```python
import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Model routing
SUPERVISOR_MODEL = "claude-haiku-4-5-20251001"
EXECUTOR_MODEL = "claude-sonnet-4-6"
VERIFIER_MODEL_PRIMARY = "claude-sonnet-4-6"
VERIFIER_MODEL_CROSS = "gpt-4o"  # for foundational results only

# Cost controls
MAX_TOKENS_EXECUTOR = 4000
MAX_TOKENS_VERIFIER = 2000
MAX_TOKENS_SUPERVISOR = 1000
```

**Phase 0 pass criteria:** API calls work, environment loads cleanly.

---

## Phase 1 — Decomposer

**Time estimate:** 1 week  
**Cost:** ~$2–5 in API calls

**Goal:** Given a research goal, produce a structured task tree. Nothing else. No execution.

### 1.1 Decomposer system prompt

`prompts/decomposer_system.md`:
```
You are a research planning agent. Your job is to decompose a research goal into a structured task tree.

For each task produce:
- ID (format: S{stage}T{task} e.g. S1T3)
- Description (what specifically must be done)
- Dependencies (list of task IDs that must complete before this one)
- Stage (integer)
- Verification criteria (how to know this task is actually complete — be specific)
- Complexity tier (STANDARD or HIGH — HIGH means foundational result or novel derivation)

Rules:
- Prefer more tasks over fewer. Under-specification is the main failure mode.
- Every task must have explicit verification criteria.
- Flag any task where the output will be used by 3 or more downstream tasks as FOUNDATIONAL.
- Do not assume steps are obvious. If it requires doing, it requires a task.

Output format: markdown with a YAML frontmatter block per task, then a dependency graph section.
```

### 1.2 Decomposer implementation

`core/decomposer.py`:
```python
import anthropic
from pathlib import Path
import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)

SYSTEM_PROMPT = Path("prompts/decomposer_system.md").read_text()

def decompose(research_goal: str, domain_context: str) -> str:
    """Generate task tree for a research goal."""
    response = client.messages.create(
        model=config.EXECUTOR_MODEL,  # Sonnet for decomposition
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Domain context:\n{domain_context}\n\nResearch goal:\n{research_goal}"
        }]
    )
    return response.content[0].text

def adversarial_review(task_tree: str, research_goal: str) -> str:
    """
    Second model call reviews the decomposition.
    Looks for: missing subtasks, wrong dependencies, under-specified verification criteria.
    """
    response = client.messages.create(
        model=config.EXECUTOR_MODEL,
        max_tokens=2000,
        system="""You are reviewing a research task decomposition for completeness and correctness.
        
Check for:
1. Missing subtasks (steps that will clearly be needed but aren't listed)
2. Dependency errors (tasks that depend on results not yet established)
3. Under-specified verification criteria (vague language like 'results look correct')
4. Tasks marked STANDARD that should be HIGH complexity
5. Foundational results not flagged as such

Output: structured list of issues found. If none, say DECOMPOSITION APPROVED with brief justification.""",
        messages=[{
            "role": "user", 
            "content": f"Research goal: {research_goal}\n\nProposed task tree:\n{task_tree}"
        }]
    )
    return response.content[0].text
```

### 1.3 Test

`tests/test_decomposer.py`:
```python
from core.decomposer import decompose, adversarial_review

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
```

Run: `pytest tests/test_decomposer.py -s`

Read the outputs carefully. Is the task tree actually good? Does the adversarial review catch real gaps? If not, iterate on the system prompt before moving to Phase 2.

**Phase 1 pass criteria:** On the target research problem, the decomposer produces a task tree that a domain expert would agree is complete and well-structured. The adversarial reviewer catches at least one real gap in initial decompositions.

---

## Phase 2 — Executor

**Time estimate:** 1 week  
**Cost:** ~$5–10

**Goal:** Execute a single task from the task tree, write the output to the file tree. No supervisor yet. No verification. Just execution.

### 2.1 Anti-hallucination constraints

`prompts/constraints.md`:
```
NEVER use these phrases to skip steps:
- "this becomes"
- "for consistency"
- "it follows that"  
- "clearly"
- "one can show"
- "it is straightforward to verify"

If you cannot derive something: write INCOMPLETE — [what is missing and why]

NEVER say "verified" without listing every specific check you ran.

NEVER adjust parameters to make results match expectations. If results look wrong, say so explicitly.

NEVER invent terms, citations, or coefficients not established in prior task outputs.

At the end of every output write:
CHECKS PERFORMED: [exhaustive list of every check you ran]
CHECKS NOT PERFORMED: [checks that would strengthen this result but were not done, with reason]
```

### 2.2 Executor system prompt

`prompts/executor_system.md`:
```
You are a research execution agent working on a single, well-defined task.

You will be given:
- The task specification (what must be done and verification criteria)
- Relevant prior outputs from the file tree (look these up, do not rely on memory)
- Domain conventions file
- Anti-hallucination constraints (follow these absolutely)

Your output must:
- Show all reasoning in full detail. No skipping steps.
- Reference specific prior task IDs when using established results.
- End with CHECKS PERFORMED and CHECKS NOT PERFORMED sections.
- Use INCOMPLETE rather than fabricating when you cannot complete something.

Format: structured markdown, equations in LaTeX.
```

### 2.3 Executor implementation

`core/executor.py`:
```python
import anthropic
from pathlib import Path
import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)

SYSTEM_PROMPT = (
    Path("prompts/executor_system.md").read_text() + 
    "\n\n" + 
    Path("prompts/constraints.md").read_text()
)

def load_prior_outputs(project_path: Path, dependency_ids: list[str]) -> str:
    """Load relevant prior task outputs from file tree."""
    outputs = []
    for task_id in dependency_ids:
        stage = task_id[1]  # S1T3 -> stage 1
        task_file = project_path / "stages" / f"stage_{stage}" / f"{task_id}.md"
        if task_file.exists():
            outputs.append(f"### Prior output {task_id}\n{task_file.read_text()}")
    return "\n\n".join(outputs)

def execute_task(
    task_spec: dict,
    project_path: Path,
    use_extended_thinking: bool = False
) -> str:
    prior = load_prior_outputs(project_path, task_spec.get("dependencies", []))
    conventions = (project_path / "conventions.md").read_text()
    
    user_content = f"""
TASK ID: {task_spec['id']}
DESCRIPTION: {task_spec['description']}
VERIFICATION CRITERIA: {task_spec['verification_criteria']}
COMPLEXITY: {task_spec['complexity']}

CONVENTIONS:
{conventions}

PRIOR OUTPUTS:
{prior if prior else 'No prior outputs (this is a foundational task)'}
"""
    
    model = config.EXECUTOR_MODEL
    # Future: add extended thinking for HIGH complexity tasks
    
    response = client.messages.create(
        model=model,
        max_tokens=config.MAX_TOKENS_EXECUTOR,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}]
    )
    
    output = response.content[0].text
    
    # Write to file tree
    stage_dir = project_path / "stages" / f"stage_{task_spec['stage']}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_file = stage_dir / f"{task_spec['id']}.md"
    output_file.write_text(output)
    
    return output

def write_stage_summary(project_path: Path, stage: int) -> str:
    """Synthesize all task outputs for a stage into a summary."""
    stage_dir = project_path / "stages" / f"stage_{stage}"
    task_files = sorted(stage_dir.glob("S*.md"))
    
    combined = "\n\n".join(f.read_text() for f in task_files)
    
    response = client.messages.create(
        model=config.EXECUTOR_MODEL,
        max_tokens=2000,
        system="Synthesize the following task outputs into a coherent stage summary. Be accurate and complete. Do not introduce information not in the task outputs.",
        messages=[{"role": "user", "content": combined}]
    )
    
    summary = response.content[0].text
    (stage_dir / "summary.md").write_text(summary)
    return summary
```

**Phase 2 pass criteria:** Run executor on 3–5 tasks from the decomposed problem. Read every output carefully. Does it show its reasoning? Does it use INCOMPLETE appropriately? Are the CHECKS PERFORMED sections honest? If the model is still faking derivations, the system prompt needs stronger constraints — iterate before moving on.

---

## Phase 3 — Supervisor and Verifier

**Time estimate:** 1.5 weeks  
**Cost:** ~$10–15

**Goal:** Add the supervisor heuristics and verifier. This is the differentiating component.

### 3.1 Supervisor heuristics

`core/supervisor.py`:
```python
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
        idx = output_lower.index("checks not performed:")
        not_performed = output[idx:idx+500]
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
{last_output[:500]}
"""
```

### 3.2 Verifier implementation

`core/verifier.py`:
```python
import anthropic
from pathlib import Path
import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)

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
    Uses cross-provider model for foundational results.
    """
    user_content = f"""
TASK SPECIFICATION:
{task_spec['description']}

VERIFICATION CRITERIA (what done actually looks like):
{task_spec['verification_criteria']}

OUTPUT TO REVIEW:
{output}
"""
    
    if is_foundational:
        # Cross-provider for foundational results
        import openai
        oai_client = openai.OpenAI(api_key=config.OPENAI_KEY)
        response = oai_client.chat.completions.create(
            model=config.VERIFIER_MODEL_CROSS,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": user_content}
            ],
            max_tokens=config.MAX_TOKENS_VERIFIER
        )
        return response.choices[0].message.content
    else:
        # Same provider, cold context (no conversation history passed)
        response = client.messages.create(
            model=config.VERIFIER_MODEL_PRIMARY,
            max_tokens=config.MAX_TOKENS_VERIFIER,
            system=VERIFIER_SYSTEM,
            messages=[{"role": "user", "content": user_content}]
        )
        return response.content[0].text
```

**Phase 3 pass criteria:** Run the supervisor on a batch of executor outputs — including some you know contain errors. Does it catch them? Does it produce false positives? Tune the heuristics. Run verifier on the same outputs. Does the verifier catch errors the supervisor missed? Does cross-provider verification produce meaningfully different results from single-provider?

---

## Phase 4 — State Manager and Stage Gates

**Time estimate:** 1 week  
**Cost:** ~$3–5

### 4.1 Global state manager

`core/state_manager.py`:
```python
from pathlib import Path
from datetime import datetime

def update_global_state(project_path: Path, task_id: str, 
                         result_summary: str, status: str):
    """
    Append to global_state.md after a task completes.
    status: ESTABLISHED | PENDING | INVALIDATED
    """
    state_file = project_path / "global_state.md"
    entry = f"""
## {task_id} — {status} — {datetime.now().strftime('%Y-%m-%d %H:%M')}

{result_summary}

---
"""
    with open(state_file, "a") as f:
        f.write(entry)

def invalidate_dependents(project_path: Path, task_id: str, 
                           task_tree: dict, reason: str):
    """
    When a foundational result is corrected, flag all dependent tasks.
    task_tree: dict mapping task_id -> list of dependent task_ids
    """
    dependents = task_tree.get(task_id, [])
    for dep_id in dependents:
        update_global_state(
            project_path, dep_id, 
            f"FLAGGED: dependency {task_id} was corrected. Reason: {reason}. Re-verification required.",
            "INVALIDATED"
        )
    return dependents

def check_stage_gate(project_path: Path, stage: int, task_specs: list) -> dict:
    """
    Evaluate whether a stage can close.
    Returns dict with: can_close (bool), blocking_items (list)
    """
    stage_dir = project_path / "stages" / f"stage_{stage}"
    blocking = []
    
    for spec in task_specs:
        task_file = stage_dir / f"{spec['id']}.md"
        
        # Check 1: output file exists
        if not task_file.exists():
            blocking.append(f"{spec['id']}: output file missing")
            continue
            
        output = task_file.read_text()
        
        # Check 2: no INCOMPLETE markers
        if "INCOMPLETE" in output:
            blocking.append(f"{spec['id']}: contains INCOMPLETE markers")
        
        # Check 3: verifier sign-off exists
        verify_file = stage_dir / f"{spec['id']}_verify.md"
        if not verify_file.exists():
            blocking.append(f"{spec['id']}: no verifier sign-off")
        elif "RECOMMENDATION: REVISE" in verify_file.read_text():
            blocking.append(f"{spec['id']}: verifier recommends revision")
        elif "RECOMMENDATION: ESCALATE" in verify_file.read_text():
            blocking.append(f"{spec['id']}: verifier escalated to human")
    
    return {
        "can_close": len(blocking) == 0,
        "blocking_items": blocking
    }
```

**Phase 4 pass criteria:** Run a full stage (all tasks, verification, state updates, stage gate check) end-to-end. Does the stage gate correctly block on real issues? Does it pass clean stages?

---

## Phase 5 — Integration and First Full Run

**Time estimate:** 1 week  
**Cost:** ~$10–20

### 5.1 Project initialization script

`scripts/new_project.py`:
```python
import sys
from pathlib import Path
from core.decomposer import decompose, adversarial_review
import uuid

def new_project(research_goal: str, domain_context: str):
    project_id = str(uuid.uuid4())[:8]
    project_path = Path("projects") / project_id
    project_path.mkdir(parents=True)
    
    # Initialize files
    (project_path / "conventions.md").write_text("# Conventions\n\n*To be populated as conventions are established.*\n")
    (project_path / "global_state.md").write_text("# Global State\n\n")
    (project_path / "constraints.md").write_text(Path("prompts/constraints.md").read_text())
    
    # Decompose
    print("Decomposing research goal...")
    task_tree = decompose(research_goal, domain_context)
    
    print("Running adversarial review of decomposition...")
    review = adversarial_review(task_tree, research_goal)
    
    # Save both
    (project_path / "master_plan.md").write_text(task_tree)
    (project_path / "decomposition_review.md").write_text(review)
    
    print(f"\nProject initialized: {project_id}")
    print(f"Review decomposition at: projects/{project_id}/master_plan.md")
    print(f"Adversarial review at: projects/{project_id}/decomposition_review.md")
    print("\n⚠️  Human checkpoint: Review both files before proceeding to execution.")
    
    return project_id

if __name__ == "__main__":
    goal = input("Research goal: ")
    context = input("Domain context: ")
    new_project(goal, context)
```

### 5.2 Stage runner

`scripts/run_stage.py`:
```python
import sys
import json
from pathlib import Path
from core.executor import execute_task
from core.supervisor import evaluate_output, Decision
from core.verifier import verify
from core.state_manager import update_global_state, check_stage_gate

def run_stage(project_id: str, stage: int, task_specs: list):
    project_path = Path("projects") / project_id
    stage_dir = project_path / "stages" / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    error_counts = {}
    
    for spec in task_specs:
        print(f"\n{'='*50}")
        print(f"Executing {spec['id']}: {spec['description'][:60]}...")
        
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            output = execute_task(spec, project_path)
            decision = evaluate_output(spec['id'], output, spec)
            
            print(f"Supervisor decision: {decision.decision.value}")
            for reason in decision.reasons:
                print(f"  - {reason}")
            
            if decision.decision in [Decision.TRIGGER_VERIFY, Decision.TRIGGER_SOFT_VERIFY]:
                print(f"Running verifier...")
                is_foundational = spec.get('complexity') == 'HIGH' and len(spec.get('dependencies', [])) == 0
                verification = verify(spec, output, is_foundational)
                
                verify_file = stage_dir / f"{spec['id']}_verify.md"
                verify_file.write_text(verification)
                
                if "RECOMMENDATION: ACCEPT" in verification:
                    print(f"✓ {spec['id']} verified and accepted")
                    update_global_state(project_path, spec['id'], 
                                       output[:300], "ESTABLISHED")
                    break
                else:
                    print(f"✗ Verifier rejected. Attempt {attempts+1}/{max_attempts}")
                    attempts += 1
                    error_counts[spec['id']] = attempts
                    
                    if attempts >= max_attempts:
                        print(f"⚠️  ESCALATING {spec['id']} to human after {attempts} attempts")
                        # Write escalation report
                        escalation = f"ESCALATION after {attempts} attempts.\nVerifier feedback:\n{verification}"
                        (stage_dir / f"{spec['id']}_escalation.md").write_text(escalation)
                        input("Human: review escalation file and press Enter to continue...")
                        break
            
            elif decision.decision == Decision.ACCEPT:
                print(f"✓ {spec['id']} accepted")
                update_global_state(project_path, spec['id'], output[:300], "ESTABLISHED")
                break
            
            elif decision.decision == Decision.HOLD_STAGE_GATE:
                print(f"⛔ {spec['id']} holding stage gate: {decision.reasons}")
                break
        
    # Stage gate check
    print(f"\n{'='*50}")
    print(f"Stage {stage} gate check...")
    gate = check_stage_gate(project_path, stage, task_specs)
    
    if gate['can_close']:
        print(f"✓ Stage {stage} closed successfully")
    else:
        print(f"⛔ Stage {stage} blocked:")
        for item in gate['blocking_items']:
            print(f"  - {item}")
```

---

## Phase 6 — Domain Oracle (Computational Genomics)

**Time estimate:** 1 week (requires domain knowledge or collaborator)  
**Cost:** $0 (programmatic checks, no API calls)

`oracle/genomics/benchmark_checks.py`:
```python
import numpy as np
from scipy import stats

def check_pvalue_calibration(pvalues: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Under the null, p-values should be uniform.
    A new method's p-values on null data should pass this check.
    """
    stat, p = stats.kstest(pvalues, 'uniform')
    return {
        "check": "p-value calibration under null",
        "statistic": stat,
        "p_value": p,
        "pass": p > alpha,
        "interpretation": "PASS: p-values appear calibrated" if p > alpha 
                         else f"FAIL: p-values not uniform (KS p={p:.4f})"
    }

def check_known_positives_recovered(
    results: dict, 
    known_positive_genes: list,
    fdr_threshold: float = 0.05
) -> dict:
    """
    A valid DE method should recover known differentially expressed genes
    on benchmark datasets.
    """
    sig_genes = {g for g, fdr in results.items() if fdr < fdr_threshold}
    recovered = sig_genes & set(known_positive_genes)
    recall = len(recovered) / len(known_positive_genes)
    
    return {
        "check": "recovery of known positive genes",
        "recall": recall,
        "recovered": list(recovered),
        "missed": list(set(known_positive_genes) - recovered),
        "pass": recall > 0.7,
        "interpretation": f"{'PASS' if recall > 0.7 else 'FAIL'}: {recall:.1%} recall on known positives"
    }

def check_method_degrades_gracefully(
    full_results: dict,
    downsampled_results: dict,
    tolerance: float = 0.3
) -> dict:
    """
    Results on downsampled data should be broadly consistent with full data.
    Large disagreements suggest the method is unstable.
    """
    full_genes = set(full_results.keys())
    down_genes = set(downsampled_results.keys())
    overlap = len(full_genes & down_genes) / len(full_genes) if full_genes else 0
    
    return {
        "check": "graceful degradation under downsampling",
        "overlap": overlap,
        "pass": overlap > (1 - tolerance),
        "interpretation": f"{'PASS' if overlap > (1-tolerance) else 'FAIL'}: {overlap:.1%} overlap between full and downsampled results"
    }
```

---

## Phase 7 — First Paper Run

With all components working, run the full system on the target research problem.

**Suggested target:** A formal statistical correction for batch effects in trajectory inference — novel enough to be publishable, computational enough to be fully verifiable, immediately useful to the field.

**Process:**
1. `python scripts/new_project.py` — decompose and review
2. Human approves task tree
3. `python scripts/run_stage.py` — execute stage by stage with human checkpoints
4. After all stages complete, synthesize outputs into LaTeX draft
5. Domain expert reviews output — this is your pass/fail criteria

**Budget tracking:** Log API call counts and token usage per phase. Target: under $20 for a full run.

---

## Cost Tracking Template

| Phase | Haiku calls | Sonnet calls | GPT-4o calls | Est. cost |
|-------|-------------|--------------|--------------|-----------|
| Decomposition | 0 | 2 | 0 | ~$0.10 |
| Execution (per stage) | 0 | 8–12 | 0 | ~$2–4 |
| Verification (per stage) | 0 | 8–12 | 2–3 | ~$3–5 |
| Supervisor (all stages) | 30–50 | 0 | 0 | ~$0.05 |
| State management | 0 | 2–3 | 0 | ~$0.50 |
| **Total (full run)** | | | | **~$10–20** |

---

## Known Gaps to Investigate

These are the areas flagged in the project spec as needing experimentation. As you build, note findings here.

**Gap 1 — Decomposition quality evaluation**
Current approach: adversarial review by second model call. Hypothesis to test: does independent re-decomposition + diff produce better gap detection than adversarial review of one decomposition?

**Gap 2 — Cross-task consistency checking**
Current approach: global_state.md + model reads it. Known weakness: unreliable. Experiment: lightweight Haiku call after every task that reads the new output + global_state.md and specifically hunts for contradictions. Measure: does it catch real contradictions, what's the false positive rate?

**Gap 3 — Correlated model failure**  
Current approach: cross-provider for foundational results. Hypothesis: if two models agree using identical reasoning paths, agreement is less meaningful than if they use different paths. Test: collect cases where Claude and GPT agree — do they produce similar reasoning? Does divergent reasoning correlate with correctness?

**Gap 4 — Oracle specification automation**
Can a domain expert describe verification constraints in natural language and have the system generate runnable checks? Or does each check need hand-coding? Test on 5 constraints, evaluate quality of generated tests.
