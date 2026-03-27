# Orchestrate — v1 Implementation Guide

## Philosophy

Build the narrowest thing that proves the core insight. The core insight is: a supervisor layer with domain-specific cross-check triggers + structured file tree + adversarial verification produces dramatically better research outputs than a single long conversation.

v1 proves this on one problem in one domain. Everything else is v2.

Do not add components until the prior component works. Each phase has a concrete test that must pass before moving on.

---

## Stack

```
Python 3.11+
Anthropic SDK (Claude — target production stack)
OpenAI SDK (GPT-4o — cross-provider verification, and compatible with open-source endpoints)
Ollama or Together AI SDK (open-source models — early development and iteration)
Plain markdown files (file tree — no database, no vector store)
pytest (oracle test suite)
```

No frameworks. No LangChain. No AutoGen. Direct API calls with explicit prompts. You need to see exactly what's happening at every step.

All model calls are routed through `core/router.py`. No other file imports an SDK client directly. This is what makes the model progression possible — swap the tier in one place and the rest of the system doesn't change.

Cost target: under $50 total for building and testing v1 including all iteration.

---

## Model Progression Strategy

The system is built in tiers. Start cheap. Validate the architecture works. Upgrade models only when the logic is proven.

### Tier 0 — Free / Local (Phases 1–3, development)
Use for: getting the pipeline wiring correct, prompt iteration, supervisor heuristic tuning. Output quality doesn't matter yet — you're testing structure, not science.

| Role | Model | Provider | Cost |
|------|-------|----------|------|
| Supervisor | Llama 3.1 8B | Ollama (local) | $0 |
| Executor | Llama 3.1 8B or 70B | Ollama (local) | $0 |
| Verifier | Llama 3.1 8B | Ollama (local) | $0 |
| Cross-check | Mistral 7B | Ollama (local) | $0 |

Ollama runs models locally with an OpenAI-compatible API endpoint (`http://localhost:11434/v1`). Router points there; no code changes elsewhere.

### Tier 1 — Cheap Cloud (Phase 4–5, integration testing)
Use for: first real research runs, checking whether the pipeline produces coherent multi-stage outputs. Still not production quality — finding where the system breaks.

| Role | Model | Provider | Est. cost/run |
|------|-------|----------|---------------|
| Supervisor | Gemini Flash 2.0 | Google AI | ~$0.01 |
| Executor | Gemini Flash 2.0 or Llama 3.3 70B | Together AI | ~$0.50–1.00 |
| Verifier | Gemini Flash 2.0 | Google AI | ~$0.50 |
| Cross-check | Llama 3.3 70B | Together AI | ~$0.20 |

Together AI and Google AI both expose OpenAI-compatible endpoints. Router swaps the base URL and model string; calling code is unchanged.

### Tier 2 — Production (Phase 6+, real research runs)
Use for: actual paper runs. Models capable enough to produce outputs worth verifying.

| Role | Model | Provider | Est. cost/run |
|------|-------|----------|---------------|
| Supervisor | claude-haiku-4-5 | Anthropic | ~$0.05 |
| Executor | claude-sonnet-4-6 | Anthropic | ~$2–4/stage |
| Verifier (primary) | claude-sonnet-4-6 | Anthropic | ~$2–4/stage |
| Cross-check (foundational only) | gpt-4o | OpenAI | ~$1–2/stage |

### Upgrade criteria
Don't upgrade a tier until:
- The pipeline runs end-to-end without errors on that tier
- The supervisor catches at least one real failure per stage
- Stage gate correctly blocks on bad outputs and passes clean ones
- You have a baseline output to compare against after upgrading

Document what breaks when you upgrade — that's signal about what the system was hiding.

---

## Repository Structure

```
orchestrate/
  core/
    router.py              # ALL model calls go through here — single source of truth
    decomposer.py
    executor.py
    supervisor.py
    verifier.py
    convention_manager.py
    state_manager.py
  prompts/
    decomposer_system.md
    executor_system.md
    verifier_system.md
    supervisor_system.md
    constraints.md         # anti-hallucination rules, baked into executor
  oracle/
    astro/
      physical_checks.py     # constants, dimensional bounds, SB law, distance modulus
      catalog_checks.py      # benchmark star/galaxy recovery
      statistical_checks.py  # uncertainty, chi2, S/N, redshift-distance
      spectral_checks.py     # line identification, redshift consistency, line ratios
      photometry_checks.py   # flux conservation, color bounds, magnitude systems
      run_oracle.py          # entry point: aggregate and report
  tests/
    test_decomposer.py
    test_supervisor_heuristics.py
    test_verifier.py
    test_oracle.py           # astro oracle checks
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
pip install anthropic openai pytest python-dotenv requests
# For Tier 0 local models:
# Install Ollama from https://ollama.com, then: ollama pull llama3.1
```

`.env`:
```
ANTHROPIC_API_KEY=your_key       # Tier 2
OPENAI_API_KEY=your_key          # Tier 2 cross-check
TOGETHER_API_KEY=your_key        # Tier 1
GOOGLE_AI_API_KEY=your_key       # Tier 1
# Tier 0 (Ollama) needs no key — runs locally
```

`config.py` — environment flags only, no model strings (those live in `router.py`):
```python
import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY")

# Active tier: 0 = local/Ollama, 1 = cheap cloud, 2 = production
MODEL_TIER = int(os.getenv("MODEL_TIER", "0"))

# Cost controls (apply at all tiers)
MAX_TOKENS_EXECUTOR = 4000
MAX_TOKENS_VERIFIER = 2000
MAX_TOKENS_SUPERVISOR = 1000
```

---

## `core/router.py` — Single Model Management File

This is the only file that knows which models exist, which tier they belong to, and how to call them. All other modules call `router.call()` — they never import an SDK client directly.

To change tiers: set `MODEL_TIER` in `.env`. To swap a specific model: edit one entry in the `TIERS` dict. Nothing else changes.

```python
import os
import anthropic
import openai
import config

# ─────────────────────────────────────────────
# TIER DEFINITIONS — edit model strings here only
# ─────────────────────────────────────────────

TIERS = {
    0: {
        # Ollama local — OpenAI-compatible endpoint, no key needed
        "supervisor":   {"provider": "ollama", "model": "llama3.1:8b"},
        "executor":     {"provider": "ollama", "model": "llama3.1:8b"},
        "verifier":     {"provider": "ollama", "model": "llama3.1:8b"},
        "cross_check":  {"provider": "ollama", "model": "mistral:7b"},
        "decomposer":   {"provider": "ollama", "model": "llama3.1:8b"},
    },
    1: {
        # Cheap cloud — Together AI (OpenAI-compatible) + Google AI
        "supervisor":   {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        "executor":     {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        "verifier":     {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        "cross_check":  {"provider": "together", "model": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
        "decomposer":   {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    },
    2: {
        # Production — Anthropic + OpenAI
        "supervisor":   {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "executor":     {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "verifier":     {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "cross_check":  {"provider": "openai",    "model": "gpt-4o"},
        "decomposer":   {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    },
}

# ─────────────────────────────────────────────
# PROVIDER CLIENTS
# ─────────────────────────────────────────────

def _get_client(provider: str):
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    elif provider == "openai":
        return openai.OpenAI(api_key=config.OPENAI_KEY)
    elif provider == "together":
        return openai.OpenAI(
            api_key=config.TOGETHER_KEY,
            base_url="https://api.together.xyz/v1"
        )
    elif provider == "ollama":
        return openai.OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ─────────────────────────────────────────────
# UNIFIED CALL INTERFACE
# ─────────────────────────────────────────────

def call(
    role: str,
    system: str,
    user: str,
    max_tokens: int = None,
    tier: int = None
) -> str:
    """
    Single entry point for all model calls.

    role: one of 'supervisor', 'executor', 'verifier', 'cross_check', 'decomposer'
    system: system prompt string
    user: user message string
    max_tokens: override default if needed
    tier: override config.MODEL_TIER if needed (useful for tests)
    
    Returns: response text as string.
    Raises: RuntimeError with context on API failure.
    """
    active_tier = tier if tier is not None else config.MODEL_TIER
    spec = TIERS[active_tier][role]
    provider = spec["provider"]
    model = spec["model"]

    # Default max_tokens by role
    if max_tokens is None:
        max_tokens = {
            "supervisor":  config.MAX_TOKENS_SUPERVISOR,
            "executor":    config.MAX_TOKENS_EXECUTOR,
            "verifier":    config.MAX_TOKENS_VERIFIER,
            "cross_check": config.MAX_TOKENS_VERIFIER,
            "decomposer":  config.MAX_TOKENS_EXECUTOR,
        }.get(role, 2000)

    try:
        if provider == "anthropic":
            client = _get_client("anthropic")
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text

        else:
            # OpenAI-compatible: Together, Ollama, OpenAI
            client = _get_client(provider)
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(
            f"router.call failed: role={role}, provider={provider}, "
            f"model={model}, tier={active_tier}\nOriginal error: {e}"
        ) from e


def current_tier_summary() -> str:
    """Print active model assignments. Call at project init for audit trail."""
    tier = config.MODEL_TIER
    lines = [f"Active tier: {tier}"]
    for role, spec in TIERS[tier].items():
        lines.append(f"  {role:12s} → {spec['provider']:10s} / {spec['model']}")
    return "\n".join(lines)
```

All other modules now use router instead of direct SDK calls. The calling pattern is identical everywhere:

```python
# In decomposer.py, executor.py, verifier.py — all look like this now:
from core import router

output = router.call(
    role="executor",
    system=SYSTEM_PROMPT,
    user=user_content,
    max_tokens=config.MAX_TOKENS_EXECUTOR
)
```

No module ever imports `anthropic` or `openai` directly. No model string appears outside `router.py`.

**Phase 0 pass criteria:** `MODEL_TIER=0 pytest tests/test_router.py` passes — router can call all five roles through Ollama. `python -c "from core.router import current_tier_summary; print(current_tier_summary())"` prints cleanly.

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
from pathlib import Path
from core import router
import config

SYSTEM_PROMPT = Path("prompts/decomposer_system.md").read_text()

def decompose(research_goal: str, domain_context: str) -> str:
    """Generate task tree for a research goal."""
    return router.call(
        role="decomposer",
        system=SYSTEM_PROMPT,
        user=f"Domain context:\n{domain_context}\n\nResearch goal:\n{research_goal}"
    )

def adversarial_review(task_tree: str, research_goal: str) -> str:
    """
    Second model call reviews the decomposition.
    Looks for: missing subtasks, wrong dependencies, under-specified verification criteria.
    """
    system = """You are reviewing a research task decomposition for completeness and correctness.
        
Check for:
1. Missing subtasks (steps that will clearly be needed but aren't listed)
2. Dependency errors (tasks that depend on results not yet established)
3. Under-specified verification criteria (vague language like 'results look correct')
4. Tasks marked STANDARD that should be HIGH complexity
5. Foundational results not flagged as such

Output: structured list of issues found. If none, say DECOMPOSITION APPROVED with brief justification."""

    return router.call(
        role="decomposer",
        system=system,
        user=f"Research goal: {research_goal}\n\nProposed task tree:\n{task_tree}"
    )

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
from pathlib import Path
from core import router
import config

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
    
    output = router.call(
        role="executor",
        system=SYSTEM_PROMPT,
        user=user_content,
        max_tokens=config.MAX_TOKENS_EXECUTOR
    )
    
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

    return router.call(
        role="executor",
        system="Synthesize the following task outputs into a coherent stage summary. Be accurate and complete. Do not introduce information not in the task outputs.",
        user=combined,
        max_tokens=2000
    )
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
from core import router
import config

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
        user=user_content,
        max_tokens=config.MAX_TOKENS_VERIFIER
    )
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

## Phase 6 — Domain Oracle (Astrophysics)

**Time estimate:** 1–1.5 weeks (requires domain knowledge or collaborator)  
**Cost:** $0 (programmatic checks, no API calls)

The oracle layer provides ground-truth verification that does not rely on model judgment. For astrophysics this means: known physical constants and limits, benchmark catalog recovery, dimensional and unit consistency, and statistical sanity checks on derived quantities. These checks run after executor outputs are written and before the verifier model is invoked — catching hard errors cheaply.

Repository structure for this phase:
```
oracle/
  astro/
    physical_checks.py       # dimensional analysis, known constant bounds
    catalog_checks.py        # recovery of benchmark objects with known properties
    statistical_checks.py    # uncertainty propagation, S/N, chi-squared sanity
    spectral_checks.py       # line identification, velocity/redshift consistency
    photometry_checks.py     # flux conservation, magnitude system consistency
    run_oracle.py            # entry point: run all applicable checks for a task output
```

---

### 6.1 Physical constraint checks

`oracle/astro/physical_checks.py`:
```python
import numpy as np

# Fundamental constants (SI)
C_LIGHT = 2.998e8        # m/s
H_PLANCK = 6.626e-34     # J·s
K_BOLTZMANN = 1.381e-23  # J/K
G_NEWTON = 6.674e-11     # m^3 kg^-1 s^-2
M_SUN = 1.989e30         # kg
L_SUN = 3.828e26         # W
PC_TO_M = 3.086e16       # meters per parsec

def check_velocity_physical(velocity_km_s: float, context: str = "") -> dict:
    """
    Velocities must be sub-relativistic for most astrophysical contexts,
    and must not exceed c under any circumstances.
    For stellar/galactic dynamics: flag if |v| > 0.1c (30,000 km/s).
    For cosmological redshifts: apply relativistic formula check separately.
    """
    v_c = abs(velocity_km_s) / (C_LIGHT / 1e3)
    superluminal = v_c >= 1.0
    suspicious = v_c > 0.1 and context not in ("cosmological", "relativistic_jet")

    return {
        "check": "velocity physical bound",
        "velocity_km_s": velocity_km_s,
        "v_over_c": round(v_c, 6),
        "pass": not superluminal,
        "warning": suspicious and not superluminal,
        "interpretation": (
            "FAIL: superluminal velocity" if superluminal else
            f"WARN: v/c = {v_c:.3f}, verify context is relativistic" if suspicious else
            "PASS"
        )
    }

def check_luminosity_physical(luminosity_lsun: float) -> dict:
    """
    Stellar luminosities must be within plausible astrophysical range.
    Below ~1e-5 L_sun: sub-stellar / brown dwarf territory (flag if claimed stellar).
    Above ~1e7 L_sun: hyperluminous, near Eddington for massive stars — flag for justification.
    """
    too_faint = luminosity_lsun < 1e-6
    hyperluminous = luminosity_lsun > 5e6

    return {
        "check": "stellar luminosity plausibility",
        "luminosity_lsun": luminosity_lsun,
        "pass": not too_faint,
        "warning": hyperluminous,
        "interpretation": (
            f"FAIL: luminosity {luminosity_lsun:.2e} L_sun below sub-stellar floor" if too_faint else
            f"WARN: hyperluminous {luminosity_lsun:.2e} L_sun — verify Eddington justification" if hyperluminous else
            "PASS"
        )
    }

def check_eddington_limit(luminosity_lsun: float, mass_msun: float) -> dict:
    """
    For accreting compact objects, luminosity should not exceed Eddington limit
    (unless super-Eddington accretion is explicitly the subject of study).
    L_Edd ≈ 3.2e4 * (M / M_sun) L_sun
    """
    l_edd_lsun = 3.2e4 * mass_msun
    ratio = luminosity_lsun / l_edd_lsun

    return {
        "check": "Eddington luminosity limit",
        "L_over_L_Edd": round(ratio, 4),
        "L_Edd_lsun": l_edd_lsun,
        "pass": ratio <= 10.0,  # allow moderate super-Eddington
        "warning": ratio > 1.0,
        "interpretation": (
            f"FAIL: L/L_Edd = {ratio:.2f}, far exceeds Eddington — requires explicit justification" if ratio > 10.0 else
            f"WARN: super-Eddington L/L_Edd = {ratio:.2f}" if ratio > 1.0 else
            "PASS"
        )
    }

def check_distance_modulus_consistent(
    apparent_mag: float, 
    absolute_mag: float, 
    distance_pc: float,
    extinction_mag: float = 0.0
) -> dict:
    """
    Distance modulus: m - M = 5*log10(d/10 pc) + A
    Check that apparent mag, absolute mag, distance, and extinction are mutually consistent.
    """
    mu_derived = 5 * np.log10(distance_pc / 10.0) + extinction_mag
    m_predicted = absolute_mag + mu_derived
    residual = abs(apparent_mag - m_predicted)

    return {
        "check": "distance modulus self-consistency",
        "mu_derived": round(mu_derived, 3),
        "m_predicted": round(m_predicted, 3),
        "m_reported": apparent_mag,
        "residual_mag": round(residual, 4),
        "pass": residual < 0.05,
        "interpretation": (
            f"FAIL: distance modulus inconsistency of {residual:.3f} mag" if residual >= 0.05 else
            "PASS"
        )
    }

def check_stefan_boltzmann_consistent(
    luminosity_lsun: float, 
    radius_rsun: float, 
    temperature_k: float,
    tolerance: float = 0.05
) -> dict:
    """
    Stefan-Boltzmann: L = 4π R² σ T⁴
    Given any two of {L, R, T}, the third should be consistent.
    """
    SIGMA = 5.670e-8  # W m^-2 K^-4
    R_SUN = 6.957e8   # m

    R_m = radius_rsun * R_SUN
    L_predicted_W = 4 * np.pi * R_m**2 * SIGMA * temperature_k**4
    L_predicted_lsun = L_predicted_W / L_SUN
    
    ratio = luminosity_lsun / L_predicted_lsun
    fractional_error = abs(ratio - 1.0)

    return {
        "check": "Stefan-Boltzmann self-consistency",
        "L_reported_lsun": luminosity_lsun,
        "L_predicted_lsun": round(L_predicted_lsun, 4),
        "ratio": round(ratio, 4),
        "pass": fractional_error < tolerance,
        "interpretation": (
            f"FAIL: L/R/T inconsistency — ratio {ratio:.3f}, expected 1.0 ± {tolerance}" if fractional_error >= tolerance else
            "PASS"
        )
    }
```

---

### 6.2 Catalog recovery checks

`oracle/astro/catalog_checks.py`:
```python
import numpy as np

# Benchmark objects: name -> known properties (literature values)
# Extend this as the target problem requires.
BENCHMARK_STARS = {
    "Vega":        {"T_eff": 9600,  "log_g": 3.95, "M_V": 0.582,  "SpType": "A0V"},
    "Sun":         {"T_eff": 5778,  "log_g": 4.44, "M_V": 4.83,   "SpType": "G2V"},
    "Sirius_A":    {"T_eff": 9940,  "log_g": 4.33, "M_V": 1.43,   "SpType": "A1V"},
    "Betelgeuse":  {"T_eff": 3500,  "log_g": 0.5,  "M_V": -5.85,  "SpType": "M1Ia"},
    "Proxima_Cen": {"T_eff": 3042,  "log_g": 5.20, "M_V": 15.53,  "SpType": "M5.5Ve"},
}

BENCHMARK_GALAXIES = {
    "M87":   {"z": 0.00436, "M_BH_msun": 6.5e9,  "type": "E0"},
    "M31":   {"z": -0.001,  "dist_kpc": 785,      "type": "SA(s)b"},
    "M82":   {"z": 0.000677,"SFR_msun_yr": 13.0,  "type": "starburst"},
    "NGC1052":{"z": 0.00504, "M_BH_msun": 1.5e8,  "type": "E4"},
}

def check_benchmark_star_recovery(
    name: str,
    derived_properties: dict,
    tolerances: dict = None
) -> dict:
    """
    Verify that derived stellar parameters match literature values for benchmark stars.
    tolerances: dict of property -> fractional tolerance (default 5% for T_eff, 0.1 dex for log_g)
    """
    if tolerances is None:
        tolerances = {"T_eff": 0.05, "log_g": 0.1, "M_V": 0.1}

    if name not in BENCHMARK_STARS:
        return {"check": "benchmark star recovery", "pass": None,
                "interpretation": f"SKIP: {name} not in benchmark catalog"}

    known = BENCHMARK_STARS[name]
    failures = []
    warnings = []

    for prop, tol in tolerances.items():
        if prop not in derived_properties or prop not in known:
            continue
        derived = derived_properties[prop]
        reference = known[prop]
        fractional = abs(derived - reference) / abs(reference)
        if fractional > tol:
            failures.append(f"{prop}: derived={derived}, known={reference}, err={fractional:.1%}")

    return {
        "check": f"benchmark star recovery: {name}",
        "failures": failures,
        "pass": len(failures) == 0,
        "interpretation": (
            f"FAIL: {'; '.join(failures)}" if failures else "PASS"
        )
    }

def check_benchmark_galaxy_recovery(
    name: str,
    derived_properties: dict,
    tolerances: dict = None
) -> dict:
    """
    Verify derived galaxy properties against literature values.
    """
    if tolerances is None:
        tolerances = {"z": 0.001, "dist_kpc": 0.05}

    if name not in BENCHMARK_GALAXIES:
        return {"check": "benchmark galaxy recovery", "pass": None,
                "interpretation": f"SKIP: {name} not in benchmark catalog"}

    known = BENCHMARK_GALAXIES[name]
    failures = []

    for prop, tol in tolerances.items():
        if prop not in derived_properties or prop not in known:
            continue
        derived = derived_properties[prop]
        reference = known[prop]
        fractional = abs(derived - reference) / (abs(reference) + 1e-12)
        if fractional > tol:
            failures.append(f"{prop}: derived={derived}, known={reference}, err={fractional:.1%}")

    return {
        "check": f"benchmark galaxy recovery: {name}",
        "failures": failures,
        "pass": len(failures) == 0,
        "interpretation": (
            f"FAIL: {'; '.join(failures)}" if failures else "PASS"
        )
    }
```

---

### 6.3 Statistical checks

`oracle/astro/statistical_checks.py`:
```python
import numpy as np
from scipy import stats

def check_uncertainty_propagation(
    value: float,
    uncertainty: float,
    snr_floor: float = 1.0
) -> dict:
    """
    Uncertainty must be positive and finite.
    Signal-to-noise ratio must exceed floor (default S/N > 1 to be reported).
    Fractional uncertainty > 1 (100%) warrants a warning.
    """
    if uncertainty <= 0 or not np.isfinite(uncertainty):
        return {
            "check": "uncertainty propagation",
            "pass": False,
            "interpretation": f"FAIL: non-physical uncertainty value {uncertainty}"
        }
    snr = abs(value) / uncertainty
    high_frac = (uncertainty / abs(value)) > 1.0 if value != 0 else False

    return {
        "check": "uncertainty propagation",
        "snr": round(snr, 2),
        "fractional_uncertainty": round(uncertainty / abs(value), 4) if value != 0 else None,
        "pass": snr >= snr_floor,
        "warning": high_frac,
        "interpretation": (
            f"FAIL: S/N = {snr:.2f} below floor {snr_floor}" if snr < snr_floor else
            f"WARN: fractional uncertainty > 100%" if high_frac else
            "PASS"
        )
    }

def check_chi_squared_fit(
    chi2: float,
    n_data: int,
    n_params: int,
    tolerance: float = 3.0
) -> dict:
    """
    Reduced chi-squared chi2_r = chi2 / (n_data - n_params).
    chi2_r >> 1: poor fit (underestimated errors or wrong model).
    chi2_r << 1: overfitting or overestimated errors.
    Flag if chi2_r outside [1/tolerance, tolerance].
    """
    dof = n_data - n_params
    if dof <= 0:
        return {"check": "chi-squared fit quality", "pass": False,
                "interpretation": f"FAIL: non-positive DOF ({dof})"}

    chi2_r = chi2 / dof
    p_value = 1.0 - stats.chi2.cdf(chi2, dof)

    good = (1.0 / tolerance) <= chi2_r <= tolerance

    return {
        "check": "chi-squared fit quality",
        "chi2_reduced": round(chi2_r, 4),
        "dof": dof,
        "p_value": round(p_value, 6),
        "pass": good,
        "interpretation": (
            f"FAIL: chi2_r = {chi2_r:.3f}, poor fit (should be near 1.0)" if not good else
            "PASS"
        )
    }

def check_redshift_distance_consistency(
    redshift: float,
    distance_mpc: float,
    H0: float = 70.0,
    tolerance: float = 0.10
) -> dict:
    """
    Hubble law sanity check for low-z sources (z < 0.3): d ≈ cz/H0.
    For z > 0.3, a full cosmological calculation is needed — flag for manual review.
    """
    if redshift > 0.3:
        return {
            "check": "redshift-distance consistency",
            "pass": None,
            "interpretation": "SKIP: z > 0.3 requires full cosmological computation, not Hubble law"
        }

    d_hubble_mpc = (2.998e5 * redshift) / H0
    fractional = abs(distance_mpc - d_hubble_mpc) / d_hubble_mpc

    return {
        "check": "redshift-distance consistency (Hubble law)",
        "z": redshift,
        "d_reported_mpc": distance_mpc,
        "d_hubble_mpc": round(d_hubble_mpc, 3),
        "fractional_discrepancy": round(fractional, 4),
        "pass": fractional < tolerance,
        "interpretation": (
            f"FAIL: d_reported={distance_mpc} Mpc vs d_Hubble={d_hubble_mpc:.1f} Mpc ({fractional:.1%} discrepancy)" if fractional >= tolerance else
            "PASS"
        )
    }

def check_photon_count_statistics(
    counts: float,
    reported_snr: float,
    tolerance: float = 0.05
) -> dict:
    """
    For Poisson-dominated photon counting (CCD/detector data):
    Expected S/N ≈ sqrt(counts). Check reported S/N is consistent.
    """
    expected_snr = np.sqrt(max(counts, 0))
    if expected_snr == 0:
        return {"check": "photon count statistics", "pass": False,
                "interpretation": "FAIL: zero counts"}

    fractional = abs(reported_snr - expected_snr) / expected_snr

    return {
        "check": "photon count Poisson statistics",
        "counts": counts,
        "expected_snr": round(expected_snr, 2),
        "reported_snr": reported_snr,
        "fractional_discrepancy": round(fractional, 4),
        "pass": fractional < tolerance,
        "interpretation": (
            f"WARN: reported S/N={reported_snr:.1f} vs Poisson-expected {expected_snr:.1f} ({fractional:.1%} discrepancy)" if fractional >= tolerance else
            "PASS"
        )
    }
```

---

### 6.4 Spectral checks

`oracle/astro/spectral_checks.py`:
```python
import numpy as np

# Common spectral lines: name -> vacuum rest wavelength (Angstroms)
SPECTRAL_LINES = {
    # Hydrogen Balmer series
    "H_alpha":    6564.61,
    "H_beta":     4862.68,
    "H_gamma":    4341.68,
    "H_delta":    4102.89,
    # Lyman series
    "Ly_alpha":   1215.67,
    "Ly_beta":    1025.72,
    # Metal lines
    "CaII_K":     3933.66,
    "CaII_H":     3968.47,
    "NaI_D1":     5895.92,
    "NaI_D2":     5889.95,
    "MgII_2796":  2796.35,
    "MgII_2803":  2803.53,
    "OII_3727":   3727.09,
    "OIII_4959":  4960.30,
    "OIII_5007":  5008.24,
    "NII_6548":   6549.86,
    "NII_6583":   6585.27,
    "SII_6716":   6718.29,
    "SII_6731":   6732.67,
    # CO bandheads (near-IR)
    "CO_2-0":     22935.0,
    "CO_3-1":     23227.0,
}

def check_redshift_from_lines(
    observed_wavelengths: dict,
    tolerance_km_s: float = 50.0
) -> dict:
    """
    Given observed wavelengths for multiple identified lines, check that all
    implied redshifts are mutually consistent.
    observed_wavelengths: dict of line_name -> observed_wavelength_angstrom
    """
    C_KM_S = 2.998e5

    redshifts = {}
    for line, obs_wl in observed_wavelengths.items():
        if line not in SPECTRAL_LINES:
            continue
        rest_wl = SPECTRAL_LINES[line]
        z = (obs_wl - rest_wl) / rest_wl
        redshifts[line] = z

    if len(redshifts) < 2:
        return {
            "check": "multi-line redshift consistency",
            "pass": None,
            "interpretation": "SKIP: fewer than 2 identified lines"
        }

    z_values = list(redshifts.values())
    z_mean = np.mean(z_values)
    z_std = np.std(z_values)
    max_dv = z_std * C_KM_S  # velocity scatter in km/s

    inconsistent = {k: v for k, v in redshifts.items() 
                    if abs(v - z_mean) * C_KM_S > tolerance_km_s}

    return {
        "check": "multi-line redshift consistency",
        "z_mean": round(z_mean, 6),
        "z_std": round(z_std, 8),
        "velocity_scatter_km_s": round(max_dv, 2),
        "inconsistent_lines": inconsistent,
        "pass": len(inconsistent) == 0,
        "interpretation": (
            f"FAIL: lines {list(inconsistent.keys())} inconsistent with z_mean={z_mean:.5f}" if inconsistent else
            f"PASS: all lines consistent at z={z_mean:.5f} ± {max_dv:.1f} km/s"
        )
    }

def check_line_ratio_physical(
    ratio_name: str,
    observed_ratio: float
) -> dict:
    """
    Diagnostic line ratios must fall within physically allowed ranges.
    Known forbidden ranges indicate calibration errors or misidentification.
    """
    PHYSICAL_RANGES = {
        # BPT diagram bounds
        "NII_Ha":      (1e-3, 10.0),    # [NII]6583 / H_alpha
        "OIII_Hb":     (0.01, 100.0),   # [OIII]5007 / H_beta
        "SII_Ha":      (0.01, 5.0),     # [SII]6716+6731 / H_alpha
        # Balmer decrement (intrinsic H_alpha/H_beta = 2.86 for Case B)
        "Balmer_dec":  (2.0, 20.0),     # reddened values up to ~20
        # [OIII] doublet ratio (density sensitive)
        "OIII_doublet":(0.33, 3.0),     # 4959/5007 ≈ 1/3 (theoretical), allow range
        # [SII] doublet ratio (density sensitive: 0.44 < r < 1.42)
        "SII_doublet": (0.40, 1.50),
    }

    if ratio_name not in PHYSICAL_RANGES:
        return {"check": f"line ratio physical range: {ratio_name}", "pass": None,
                "interpretation": f"SKIP: {ratio_name} not in known ratio catalog"}

    lo, hi = PHYSICAL_RANGES[ratio_name]
    in_range = lo <= observed_ratio <= hi

    return {
        "check": f"line ratio physical range: {ratio_name}",
        "observed": observed_ratio,
        "allowed_range": (lo, hi),
        "pass": in_range,
        "interpretation": (
            f"FAIL: {ratio_name} = {observed_ratio:.3f} outside physical range [{lo}, {hi}]" if not in_range else
            "PASS"
        )
    }
```

---

### 6.5 Photometry checks

`oracle/astro/photometry_checks.py`:
```python
import numpy as np

# AB magnitude zero points (Jy) for common filters
# m_AB = -2.5*log10(f_nu / 3631 Jy)
AB_ZEROPOINT_JY = 3631.0

# Approximate effective wavelengths (Angstroms) for common filter systems
FILTER_WAVELENGTHS = {
    # SDSS
    "u": 3543, "g": 4770, "r": 6231, "i": 7625, "z": 9134,
    # 2MASS
    "J": 12350, "H": 16620, "K": 21590,
    # HST WFC3
    "F275W": 2750, "F336W": 3360, "F435W": 4350,
    "F606W": 6060, "F814W": 8140,
    # Johnson-Cousins
    "U": 3650, "B": 4450, "V": 5510, "R": 6580, "I": 8060,
}

def check_color_physical(
    filter1: str,
    filter2: str,
    color: float
) -> dict:
    """
    Colors (mag1 - mag2) must be within physically plausible stellar ranges.
    Extreme colors indicate: calibration failure, high extinction, or unusual objects
    (which should be noted explicitly if intentional).
    """
    # Bluest to reddest normal stellar colors (approximate)
    COLOR_RANGES = {
        ("B", "V"):   (-0.4, 2.5),
        ("V", "I"):   (-0.5, 4.0),
        ("V", "K"):   (-0.5, 8.0),
        ("g", "r"):   (-0.5, 2.5),
        ("r", "i"):   (-0.4, 1.5),
        ("J", "K"):   (-0.2, 2.5),
    }

    key = (filter1, filter2)
    key_rev = (filter2, filter1)

    if key in COLOR_RANGES:
        lo, hi = COLOR_RANGES[key]
        in_range = lo <= color <= hi
    elif key_rev in COLOR_RANGES:
        lo, hi = COLOR_RANGES[key_rev]
        in_range = -hi <= color <= -lo
    else:
        return {"check": f"color ({filter1}-{filter2}) physical range", "pass": None,
                "interpretation": f"SKIP: no bounds defined for ({filter1}-{filter2})"}

    return {
        "check": f"color ({filter1}-{filter2}) physical range",
        "color": color,
        "allowed_range": (lo, hi),
        "pass": in_range,
        "interpretation": (
            f"FAIL: ({filter1}-{filter2}) = {color:.3f} outside stellar range [{lo}, {hi}]" if not in_range else
            "PASS"
        )
    }

def check_flux_conservation(
    broadband_flux_jy: float,
    integrated_spectrum_flux_jy: float,
    tolerance: float = 0.05
) -> dict:
    """
    Integrated flux from a spectrum convolved with a filter bandpass
    should match the broadband photometry in that filter.
    Discrepancy > tolerance indicates flux calibration error.
    """
    fractional = abs(broadband_flux_jy - integrated_spectrum_flux_jy) / broadband_flux_jy

    return {
        "check": "flux conservation: photometry vs spectrum",
        "broadband_jy": broadband_flux_jy,
        "spectrum_integrated_jy": integrated_spectrum_flux_jy,
        "fractional_discrepancy": round(fractional, 5),
        "pass": fractional < tolerance,
        "interpretation": (
            f"FAIL: {fractional:.1%} flux discrepancy between photometry and spectrum" if fractional >= tolerance else
            "PASS"
        )
    }

def check_magnitude_system_consistent(
    mag_ab: float,
    mag_vega: float,
    filter_name: str,
    tolerance: float = 0.05
) -> dict:
    """
    AB and Vega magnitudes differ by known, filter-dependent offsets.
    If both are reported, verify their difference matches the expected offset.
    Approximate AB-Vega offsets:
    V: +0.02, B: -0.10, R: +0.16, I: +0.40, J: +0.91, H: +1.39, K: +1.85
    g: -0.08, r: +0.16, i: +0.37, z: +0.54
    """
    AB_VEGA_OFFSETS = {
        "V": 0.02, "B": -0.10, "R": 0.16, "I": 0.40,
        "J": 0.91, "H": 1.39, "K": 1.85,
        "g": -0.08, "r": 0.16, "i": 0.37, "z": 0.54,
    }

    if filter_name not in AB_VEGA_OFFSETS:
        return {"check": "magnitude system consistency", "pass": None,
                "interpretation": f"SKIP: no AB-Vega offset defined for filter {filter_name}"}

    expected_offset = AB_VEGA_OFFSETS[filter_name]
    observed_offset = mag_ab - mag_vega
    residual = abs(observed_offset - expected_offset)

    return {
        "check": f"AB/Vega magnitude system consistency: {filter_name}",
        "mag_ab": mag_ab,
        "mag_vega": mag_vega,
        "observed_offset": round(observed_offset, 4),
        "expected_offset": expected_offset,
        "residual": round(residual, 4),
        "pass": residual < tolerance,
        "interpretation": (
            f"FAIL: AB-Vega offset = {observed_offset:.3f}, expected {expected_offset:.3f} for {filter_name}" if residual >= tolerance else
            "PASS"
        )
    }
```

---

### 6.6 Oracle runner

`oracle/astro/run_oracle.py`:
```python
import json
from pathlib import Path
from oracle.astro import physical_checks, statistical_checks

def run_all_checks(checks: list) -> dict:
    """
    Run a list of pre-constructed check calls and aggregate results.
    checks: list of dicts returned by individual check functions
    """
    passed = [c for c in checks if c.get("pass") is True]
    failed = [c for c in checks if c.get("pass") is False]
    warned = [c for c in checks if c.get("warning") is True]
    skipped = [c for c in checks if c.get("pass") is None]

    all_pass = len(failed) == 0

    return {
        "oracle_pass": all_pass,
        "summary": {
            "total": len(checks),
            "passed": len(passed),
            "failed": len(failed),
            "warnings": len(warned),
            "skipped": len(skipped),
        },
        "failures": [{"check": c["check"], "interpretation": c["interpretation"]} for c in failed],
        "warnings": [{"check": c["check"], "interpretation": c["interpretation"]} for c in warned],
    }

def write_oracle_report(project_path: Path, task_id: str, results: dict):
    stage = task_id[1]
    oracle_dir = project_path / "stages" / f"stage_{stage}" / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    report_path = oracle_dir / f"{task_id}_oracle.json"
    report_path.write_text(json.dumps(results, indent=2))
    return report_path
```

**Phase 6 pass criteria:** For each check type, write a unit test with a known-good and a known-bad input. All checks must correctly classify both. Run the oracle on a sample of executor outputs — at least one should produce a real failure that catches an error the verifier model missed.

---

## Phase 7 — First Paper Run (Astrophysics)

With all components working, run the full system on the target research problem.

**Suggested target:** A quantitative analysis of an open problem in stellar physics, galactic structure, or time-domain astronomy that is well-posed (clear inputs and success criteria), amenable to analytical or semi-analytical treatment, and testable against public archival data (SDSS, Gaia, 2MASS, APOGEE, HST, etc.).

Concrete candidate problems suited to v1:
- Derivation and validation of a photometric metallicity calibration for M dwarfs using Gaia + 2MASS colors, with cross-validation against APOGEE spectroscopic metallicities
- Formal treatment of the color-magnitude diagram morphology for a well-studied open cluster, deriving age and distance modulus with explicit uncertainty propagation
- A kinematic analysis of stellar streams in the Gaia DR3 catalog, deriving stream properties and assessing progenitor constraints
- An analytic model for the projected mass profile of a galaxy cluster with X-ray + lensing consistency check

Choose a problem where: (1) the answer is checkable against something you trust, (2) the derivation chain is explicit and traceable, and (3) the oracle checks above are directly applicable.

**Domain context to supply at project initialization:**
```
Include in domain_context:
- Target object(s) or catalog name and data access method (e.g. astroquery, direct download)
- Relevant coordinate system and units conventions for this problem
- Which physical checks are applicable (e.g. stellar vs. extragalactic)
- Known literature values for any benchmark objects in the dataset
- Any instrument-specific calibration issues (e.g. Gaia parallax zero point, SDSS fiber magnitude corrections)
- Preferred magnitude system (AB or Vega) — must be consistent throughout
```

**Conventions file additions for astrophysics** (`projects/{id}/conventions.md`):
```
UNITS: All wavelengths in Angstroms unless noted. Distances in pc/kpc/Mpc (never mix). 
       Magnitudes in AB system unless Vega explicitly stated.
COORDINATES: ICRS J2000 throughout. Always state epoch when using proper motions.
EQUATIONS: All LaTeX. Subscripts: obs = observed, rest = rest-frame, corr = corrected.
CITATIONS: Use ADS bibcodes. Never invent references. Cite the original detection paper, not reviews.
PHYSICAL CONSTANTS: Use values from oracle/astro/physical_checks.py — do not define inline.
SIGNIFICANT FIGURES: Report uncertainties to 2 sig figs; values to matching precision.
INCOMPLETE MARKERS: Use INCOMPLETE — [what is missing] when a derivation cannot be completed.
```

**Process:**
1. `python scripts/new_project.py` — decompose with domain context above
2. Human approves task tree, paying particular attention to: are observational data access tasks explicit? Are unit/system conventions stated per task?
3. `python scripts/run_stage.py` — execute stage by stage
4. After each task, oracle checks run automatically; verifier is triggered on failures and HIGH complexity tasks
5. After all stages complete, synthesize outputs into LaTeX draft
6. Cross-check key numerical results against published values or independent data products

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
