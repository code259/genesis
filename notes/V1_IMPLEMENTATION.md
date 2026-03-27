# Orchestrate — v1 Implementation Guide

## Philosophy

Build the narrowest thing that proves the core insight. The core insight is: a supervisor layer with domain-specific cross-check triggers + structured file tree + adversarial verification produces dramatically better research outputs than a single long conversation.

v1 proves this on one problem in one domain. Everything else is v2.

Do not add components until the prior component works. Each phase has a concrete test that must pass before moving on.

---

## Model Strategy

### The Short Version

**Build the entire system without integrating any model first.** Every component — the supervisor heuristics, the stage gate logic, the verifier structure, the convention drift checker — is pure Python logic that operates on text. None of it depends on which model sits underneath. Get the architecture working with mock responses, then drop in a model.

This is the right way to build this. It keeps iteration fast and cost at zero during architecture development. It also forces clean separation between orchestration logic and model calls, which is what makes the system model-agnostic.

### Model Progression

**Phase 0–5 (architecture and unit tests):** No model at all. Mock responses. Zero cost.

**Phase 6 (first integration test):** Start with a free or very cheap open source model. Recommended options:

- **Groq + Llama 3.3 70B** — fastest, cheapest, good instruction following. Free tier is generous. Best starting point.
- **Ollama + Qwen 2.5 72B** — fully local, zero cost, requires decent hardware (32GB RAM minimum). Good if you want no external API dependency during development.
- **Together AI + Mixtral 8x7B** — cheap, solid, good for parallelism.
- **Google Gemini Flash** — has a free tier, strong reasoning, native search grounding built in.

The 70B class models are good enough to tell you whether your prompts and architecture work. You will see more failures than with Sonnet or Opus — more sycophancy, more step-skipping, more convention drift. That is actually useful: if your supervisor catches these failures and routes them correctly, your architecture is working. Think of weaker models as stress tests for your orchestration layer.

**Required model capabilities — whatever model you choose must have these:**

- **Tool use / function calling** — the executor needs to actually call tools to run code, not just describe code in text
- **Web search** — literature review, citation verification, checking prior results against published work
- **Code execution** — running Python scripts, validating numerical results, generating plots
- **Long context** — minimum 32K tokens, 128K+ preferred for later stages when prior outputs accumulate
- **Structured output / JSON mode** — supervisor and verifier need reliable structured responses

Not all open source models support all of these out of the box. Llama 3.3 70B via Groq has tool use and JSON mode but no built-in search — you would wire search separately using Tavily API or SerpAPI. Gemini Flash has native search grounding which saves integration work. Check capability coverage before committing to a model for integration testing.

**Phase 7 (paper run):** Upgrade the executor and verifier to a frontier model — Claude Sonnet or Opus, GPT-4o, or Gemini 1.5 Pro. Keep the supervisor on a cheap model. **The architecture doesn't change. Only the `MODEL_BACKEND` value in config changes.** This is the whole point of building the abstraction layer first.

### Why This Ordering Works

The supervisor logic, heuristics, stage gates, and file management are all model-independent. By the time you integrate a model, you know the architecture is correct. You are not debugging architecture and model behavior simultaneously — which is the main way these projects spiral in cost and confusion.

When you swap from Llama 70B to Opus for the paper run, the only change is a single environment variable. If something breaks at that point, it's the prompt, not the plumbing.

---

## Stack

```
Python 3.11+
pytest                    # unit and integration tests
pytest-cov                # coverage reporting
python-dotenv             # environment management
pathlib                   # file tree management (stdlib)
numpy
scipy

# Add only when integrating model (Phase 6+):
groq                      # Groq SDK for Llama/Mixtral — start here
tavily-python             # web search tool for models without native search
anthropic                 # Claude — upgrade path
openai                    # GPT-4o — cross-provider verification
```

No LangChain. No AutoGen. No agent frameworks. Direct API calls with explicit prompts. You need to see exactly what is happening at every step — frameworks hide that.

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
    task_parser.py
    model_client.py        # ALL model calls go through here — swap model in one place
  prompts/
    decomposer_system.md
    executor_system.md
    verifier_system.md
    supervisor_system.md
    constraints.md
  oracle/
    astro/
      physical_checks.py     # constants, dimensional bounds, SB law, distance modulus
      catalog_checks.py      # benchmark star/galaxy recovery
      statistical_checks.py  # uncertainty, chi2, S/N, redshift-distance
      spectral_checks.py     # line identification, redshift consistency, line ratios
      photometry_checks.py   # flux conservation, color bounds, magnitude systems
      run_oracle.py          # entry point: aggregate and report
  tests/
    unit/
      test_supervisor_heuristics.py
      test_state_manager.py
      test_convention_manager.py
      test_stage_gate.py
      test_task_parser.py
    integration/
      test_decomposer_integration.py
      test_executor_integration.py
      test_verifier_integration.py
      test_full_pipeline.py
    oracle/
      test_benchmark_checks.py
      test_calibration_checks.py
    fixtures/
      sample_task_tree.md
      sample_good_output.md
      sample_bad_outputs/
        fake_verification.md
        step_skipping.md
        incomplete_checks.md
        convention_drift.md
  projects/               # runtime output — gitignored
    {project_id}/
      master_plan.md
      conventions.md
      global_state.md
      constraints.md
      stages/
  scripts/
    new_project.py
    run_stage.py
    review_stage.py
    audit.py
  config.py
  conftest.py             # pytest fixtures shared across all tests
  requirements.txt
```

---

## Phase 0 — Environment and Mock Infrastructure

**Time estimate:** 2 hours
**Cost:** $0
**Goal:** Full test suite runs, all pass, zero model calls made.

```bash
git init orchestrate
cd orchestrate
python -m venv venv
source venv/bin/activate
pip install pytest pytest-cov python-dotenv numpy scipy
```

### 0.1 Model client — the abstraction layer

This is the most important structural decision in the codebase. Every model call in the entire system goes through `model_client.py`. This is what makes the system model-agnostic. To swap models, you change this file only — nothing else imports anthropic, openai, or groq directly.

`core/model_client.py`:
```python
"""
Model client abstraction layer.

ALL model calls in the system go through this module.
To swap models: set MODEL_BACKEND environment variable.
No other file should import anthropic, openai, or groq directly.

Backends:
  "mock"      — returns fixture responses, zero API calls (default for tests)
  "groq"      — Llama 3.3 70B via Groq (cheap, start here for integration)
  "anthropic" — Claude Sonnet/Opus (upgrade path for paper run)
  "openai"    — GPT-4o (cross-provider verification)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "mock")

MODEL_CONFIG = {
    "mock": {},
    "groq": {
        "executor":  "llama-3.3-70b-versatile",
        "supervisor": "llama-3.3-70b-versatile",
        "verifier":  "llama-3.3-70b-versatile",
        "decomposer": "llama-3.3-70b-versatile",
    },
    "anthropic": {
        "executor":  "claude-sonnet-4-6",
        "supervisor": "claude-haiku-4-5-20251001",
        "verifier":  "claude-sonnet-4-6",
        "decomposer": "claude-sonnet-4-6",
        "verifier_cross": "gpt-4o",  # cross-provider for foundational results
    },
    "openai": {
        "executor":  "gpt-4o",
        "supervisor": "gpt-4o-mini",
        "verifier":  "gpt-4o",
        "decomposer": "gpt-4o",
    },
}

TOKEN_LIMITS = {
    "executor": 4000,
    "supervisor": 1000,
    "verifier": 2000,
    "decomposer": 4000,
}

# Mock responses used when MODEL_BACKEND = "mock"
def _load_fixture(path: str, fallback: str) -> str:
    p = Path(path)
    return p.read_text() if p.exists() else fallback

MOCK_RESPONSES = {
    "decomposer": _load_fixture(
        "tests/fixtures/sample_task_tree.md", "MOCK TASK TREE\n\n## S1T1\n- **Description:** Mock task\n- **Dependencies:** none\n- **Stage:** 1\n- **Verification criteria:** Mock criteria\n- **Complexity:** STANDARD\n"
    ),
    "executor": _load_fixture(
        "tests/fixtures/sample_good_output.md",
        "MOCK EXECUTOR OUTPUT\n\nFull derivation shown.\n\nCHECKS PERFORMED: mock check ran\nCHECKS NOT PERFORMED: none"
    ),
    "supervisor": "SUPERVISOR: no issues detected",
    "verifier": "RECOMMENDATION: ACCEPT\nAll checks passed.",
}


def call(role: str, system_prompt: str, user_content: str) -> str:
    """
    Make a model call for the given role.
    role: "executor" | "supervisor" | "verifier" | "decomposer"
    Returns response text.
    """
    if MODEL_BACKEND == "mock":
        return MOCK_RESPONSES.get(role, f"MOCK RESPONSE for role={role}")
    return _call_backend(role, system_prompt, user_content)


def _call_backend(role: str, system_prompt: str, user_content: str) -> str:
    config = MODEL_CONFIG[MODEL_BACKEND]

    if MODEL_BACKEND == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=config[role],
            max_tokens=TOKEN_LIMITS.get(role, 2000),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ]
        )
        return response.choices[0].message.content

    elif MODEL_BACKEND == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=config[role],
            max_tokens=TOKEN_LIMITS.get(role, 2000),
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )
        return response.content[0].text

    elif MODEL_BACKEND == "openai":
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=config[role],
            max_tokens=TOKEN_LIMITS.get(role, 2000),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ]
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")
```

### 0.2 Test fixtures

Write these carefully. They are the ground truth for what good and bad outputs look like.

`tests/fixtures/sample_good_output.md`:
```markdown
## Task S1T1 — Derivation of batch effect model

Let X_ij denote the expression of gene j in cell i. We model batch effects as additive:

    X_ij = mu_ij + b_k(i) + epsilon_ij

where mu_ij is the true biological signal, b_k(i) is the batch effect for batch k
containing cell i, and epsilon_ij ~ N(0, sigma^2).

Deriving the correction factor from first principles:

Step 1: Take expectation over cells in batch k...
Step 2: Estimate b_hat_k as the batch mean deviation...
Step 3: Correction is therefore C_ij = X_ij - b_hat_k(i)

CHECKS PERFORMED:
- Verified model reduces to standard form when b_k = 0
- Confirmed epsilon term identically distributed across batches under null
- Checked dimensional consistency of all terms
- Verified correction is unbiased: E[C_ij] = mu_ij

CHECKS NOT PERFORMED: none
```

`tests/fixtures/sample_bad_outputs/fake_verification.md`:
```markdown
## Task S1T1

The batch correction formula is X_ij = mu_ij + b_k + epsilon.

For consistency, it follows that the correction factor is C = X - b_hat.

This has been verified and confirmed to be correct.
```

`tests/fixtures/sample_bad_outputs/step_skipping.md`:
```markdown
## Task S1T1

Starting from the general model, it is straightforward to show that
the correction factor becomes C = X - b_hat.

One can show this satisfies the unbiasedness condition. Clearly the
dimensional analysis works out.

CHECKS PERFORMED: verified correctness
CHECKS NOT PERFORMED: none
```

`tests/fixtures/sample_bad_outputs/incomplete_checks.md`:
```markdown
## Task S1T1

Full derivation shown here...

CHECKS PERFORMED: verified model reduces to standard form
CHECKS NOT PERFORMED:
- Dimensional consistency check (ran out of context)
- Verification against known limiting cases
```

`tests/fixtures/sample_bad_outputs/convention_drift.md`:
```markdown
## Task S1T3

Using the standard notation where X represents the count matrix
(note: using X instead of the established M notation from task S1T1),
the normalized expression is computed as follows...
```

`conftest.py`:
```python
import pytest
from pathlib import Path

@pytest.fixture
def good_output():
    return Path("tests/fixtures/sample_good_output.md").read_text()

@pytest.fixture
def bad_fake_verify():
    return Path("tests/fixtures/sample_bad_outputs/fake_verification.md").read_text()

@pytest.fixture
def bad_step_skip():
    return Path("tests/fixtures/sample_bad_outputs/step_skipping.md").read_text()

@pytest.fixture
def bad_incomplete():
    return Path("tests/fixtures/sample_bad_outputs/incomplete_checks.md").read_text()

@pytest.fixture
def bad_convention_drift():
    return Path("tests/fixtures/sample_bad_outputs/convention_drift.md").read_text()

@pytest.fixture
def sample_task_spec():
    return {
        "id": "S1T1",
        "description": "Derive batch effect correction model",
        "dependencies": [],
        "stage": 1,
        "verification_criteria": "Model reduces to standard form under null. Correction is unbiased. Dimensional consistency holds.",
        "complexity": "HIGH",
    }

@pytest.fixture
def tmp_project(tmp_path):
    """Minimal project directory for tests. No model calls needed."""
    (tmp_path / "stages" / "stage_1").mkdir(parents=True)
    (tmp_path / "conventions.md").write_text("# Conventions\n\nM = count matrix\n")
    (tmp_path / "global_state.md").write_text("# Global State\n\n")
    (tmp_path / "constraints.md").write_text("NEVER skip steps.")
    return tmp_path
```

**Phase 0 pass criteria:** `pytest tests/` runs cleanly. Zero model calls. Zero cost.

---

## Phase 1 — Supervisor Heuristics

**Time estimate:** 1 week
**Cost:** $0 — pure logic, no model calls

Write tests first. The tests define exactly what the supervisor must do before you write a single line of implementation.

### 1.1 Tests

`tests/unit/test_supervisor_heuristics.py`:
```python
import pytest
from core.supervisor import evaluate_output, Decision, check_iteration_count


class TestFakeVerificationDetection:
    """Supervisor must catch outputs that claim verification without showing checks."""

    def test_catches_verified_without_checks_section(self, bad_fake_verify, sample_task_spec):
        result = evaluate_output("S1T1", bad_fake_verify, sample_task_spec)
        assert result.decision in [Decision.TRIGGER_VERIFY, Decision.ESCALATE_HUMAN]
        assert any("verified" in r.lower() or "checks performed" in r.lower()
                   for r in result.reasons)

    def test_accepts_verified_with_full_checks_section(self, good_output, sample_task_spec):
        result = evaluate_output("S1T1", good_output, sample_task_spec)
        assert result.decision == Decision.ACCEPT

    def test_catches_confirmed_without_checks(self, sample_task_spec):
        output = "The result is confirmed.\n\nThe formula works."
        result = evaluate_output("S1T1", output, sample_task_spec)
        assert result.decision in [Decision.TRIGGER_VERIFY, Decision.TRIGGER_SOFT_VERIFY]


class TestStepSkippingDetection:
    """Supervisor must flag outputs that skip derivation steps."""

    def test_catches_for_consistency(self, bad_step_skip, sample_task_spec):
        result = evaluate_output("S1T1", bad_step_skip, sample_task_spec)
        assert result.decision in [Decision.TRIGGER_SOFT_VERIFY, Decision.TRIGGER_VERIFY]

    def test_catches_it_follows_that(self, sample_task_spec):
        output = "It follows that the correction is C = X - b.\n\nCHECKS PERFORMED: none needed"
        result = evaluate_output("S1T1", output, sample_task_spec)
        assert result.decision != Decision.ACCEPT

    def test_catches_clearly(self, sample_task_spec):
        output = "Clearly the formula reduces correctly.\n\nCHECKS PERFORMED: visual inspection"
        result = evaluate_output("S1T1", output, sample_task_spec)
        assert result.decision != Decision.ACCEPT

    def test_catches_one_can_show(self, sample_task_spec):
        output = "One can show that this satisfies unbiasedness.\n\nCHECKS PERFORMED: none"
        result = evaluate_output("S1T1", output, sample_task_spec)
        assert result.decision != Decision.ACCEPT

    def test_catches_this_becomes(self, sample_task_spec):
        output = "Substituting, this becomes C = X - b_hat.\n\nCHECKS PERFORMED: substitution"
        result = evaluate_output("S1T1", output, sample_task_spec)
        assert result.decision != Decision.ACCEPT

    def test_good_output_not_flagged_for_step_skipping(self, good_output, sample_task_spec):
        result = evaluate_output("S1T1", good_output, sample_task_spec)
        assert result.decision == Decision.ACCEPT


class TestIncompleteDetection:
    """Supervisor must hold stage gate when INCOMPLETE markers present."""

    def test_catches_incomplete_marker(self, sample_task_spec):
        output = "Derivation...\n\nINCOMPLETE — could not verify limiting case\n\nCHECKS PERFORMED: partial"
        result = evaluate_output("S1T1", output, sample_task_spec)
        assert result.decision == Decision.HOLD_STAGE_GATE

    def test_catches_non_empty_checks_not_performed(self, bad_incomplete, sample_task_spec):
        result = evaluate_output("S1T1", bad_incomplete, sample_task_spec)
        assert result.decision in [Decision.HOLD_STAGE_GATE, Decision.TRIGGER_VERIFY]
        assert any("not performed" in r.lower() for r in result.reasons)

    def test_empty_checks_not_performed_is_ok(self, good_output, sample_task_spec):
        result = evaluate_output("S1T1", good_output, sample_task_spec)
        assert result.decision == Decision.ACCEPT


class TestNumericalResultDetection:
    """HIGH complexity tasks with numerical results must show derivation."""

    def test_number_without_derivation_triggers_verify_for_high_complexity(self, sample_task_spec):
        output = "The correction factor is 3.14159.\n\nCHECKS PERFORMED: computed value"
        result = evaluate_output("S1T1", {**sample_task_spec, "complexity": "HIGH"}, output)
        # Note: evaluate_output(task_id, task_spec, output) — spec before output
        assert result.decision in [Decision.TRIGGER_VERIFY, Decision.ESCALATE_HUMAN]

    def test_number_without_derivation_ok_for_standard_complexity(self, sample_task_spec):
        output = "The mean expression is 3.14159.\n\nCHECKS PERFORMED: computed from data"
        result = evaluate_output("S1T1", {**sample_task_spec, "complexity": "STANDARD"}, output)
        assert result.decision in [Decision.ACCEPT, Decision.TRIGGER_SOFT_VERIFY]


class TestIterationEscalation:
    """After 3 failures on same task, escalate to human."""

    def test_escalates_after_three_failures(self):
        assert check_iteration_count("S1T1", {"S1T1": 3}) is True

    def test_no_escalation_before_three_failures(self):
        assert check_iteration_count("S1T1", {"S1T1": 2}) is False

    def test_no_escalation_for_new_task(self):
        assert check_iteration_count("S1T1", {}) is False


class TestDecisionReasons:
    """Every decision must include human-readable reasons."""

    def test_every_decision_has_reasons(self, good_output, bad_fake_verify, sample_task_spec):
        for output in [good_output, bad_fake_verify]:
            result = evaluate_output("S1T1", sample_task_spec, output)
            assert len(result.reasons) > 0
            assert all(isinstance(r, str) and len(r) > 0 for r in result.reasons)

    def test_accept_decision_explains_why(self, good_output, sample_task_spec):
        result = evaluate_output("S1T1", sample_task_spec, good_output)
        assert result.decision == Decision.ACCEPT
        assert len(result.reasons) > 0
```

### 1.2 Implementation

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
    reasons: list
    task_id: str

SKIP_PHRASES = [
    "this becomes", "for consistency", "it follows that",
    "clearly", "one can show", "it is straightforward",
    "trivially", "obviously",
]

FAKE_VERIFY_PHRASES = ["verified", "confirmed", "checked", "validated"]

def evaluate_output(task_id: str, task_spec: dict, output: str) -> SupervisorDecision:
    reasons = []
    decision = Decision.ACCEPT
    output_lower = output.lower()

    # Check 1: fake verification
    for phrase in FAKE_VERIFY_PHRASES:
        if phrase in output_lower and "checks performed:" not in output_lower:
            reasons.append(
                f"Output contains '{phrase}' but no CHECKS PERFORMED section — claim unsubstantiated"
            )
            decision = Decision.TRIGGER_VERIFY
            break

    # Check 2: step-skipping phrases
    for phrase in SKIP_PHRASES:
        if phrase in output_lower:
            reasons.append(f"Step-skipping phrase detected: '{phrase}'")
            if decision == Decision.ACCEPT:
                decision = Decision.TRIGGER_SOFT_VERIFY

    # Check 3: numerical result in HIGH complexity task without derivation trace
    has_decimal = bool(re.search(r'\b\d+\.\d+\b', output))
    derivation_keywords = [
        "derivation", "deriving", "derive", "calculate", "integral",
        "summation", "proof", "theorem", "lemma", "from first principles",
    ]
    has_derivation = any(w in output_lower for w in derivation_keywords)

    if has_decimal and not has_derivation and task_spec.get("complexity") == "HIGH":
        reasons.append("HIGH complexity task: numerical result present without derivation trace")
        if decision in [Decision.ACCEPT, Decision.TRIGGER_SOFT_VERIFY]:
            decision = Decision.TRIGGER_VERIFY

    # Check 4: INCOMPLETE marker
    if "INCOMPLETE" in output:
        reasons.append("Output contains INCOMPLETE marker — task not finished")
        decision = Decision.HOLD_STAGE_GATE

    # Check 5: non-empty CHECKS NOT PERFORMED section
    if "checks not performed:" in output_lower:
        idx = output_lower.index("checks not performed:")
        after = output[idx + len("checks not performed:"):idx + 500].strip()
        if after and after.lower() not in ["none", "n/a", "-", ""]:
            if len(after) > 4:
                reasons.append(f"Checks not performed: {after[:100]}")
                if task_spec.get("complexity") == "HIGH":
                    if decision not in [Decision.HOLD_STAGE_GATE, Decision.ESCALATE_HUMAN]:
                        decision = Decision.HOLD_STAGE_GATE
                elif decision == Decision.ACCEPT:
                    decision = Decision.TRIGGER_SOFT_VERIFY

    if not reasons:
        reasons.append("No failure mode signatures detected — output appears complete")

    return SupervisorDecision(decision=decision, reasons=reasons, task_id=task_id)


def check_iteration_count(task_id: str, error_history: dict) -> bool:
    """Returns True if human escalation is needed (task failed 3+ times)."""
    return error_history.get(task_id, 0) >= 3


def generate_escalation_report(task_id: str, reasons: list,
                                attempts: int, last_output: str) -> str:
    reason_lines = "\n".join(f"- {r}" for r in reasons)
    return f"""# ESCALATION REPORT

**Task:** {task_id}
**Attempts:** {attempts}
**Reasons:**
{reason_lines}

## What the human needs to decide

- Is the verification criteria achievable with current information?
- Does this task need to be broken into smaller subtasks?
- Should this task be marked INCOMPLETE and deferred?

## Last output (first 500 chars)

{last_output[:500]}
"""
```

Run: `pytest tests/unit/test_supervisor_heuristics.py -v`

All tests must pass before moving to Phase 2.

---

## Phase 2 — State Manager

**Time estimate:** 3 days
**Cost:** $0

### 2.1 Tests

`tests/unit/test_state_manager.py`:
```python
import pytest
from core.state_manager import (
    update_global_state, invalidate_dependents,
    check_stage_gate, get_established_results,
)


class TestUpdateGlobalState:

    def test_creates_entry(self, tmp_project):
        update_global_state(tmp_project, "S1T1", "Batch model derived", "ESTABLISHED")
        content = (tmp_project / "global_state.md").read_text()
        assert "S1T1" in content
        assert "ESTABLISHED" in content
        assert "Batch model derived" in content

    def test_multiple_entries_appended(self, tmp_project):
        update_global_state(tmp_project, "S1T1", "Result A", "ESTABLISHED")
        update_global_state(tmp_project, "S1T2", "Result B", "ESTABLISHED")
        content = (tmp_project / "global_state.md").read_text()
        assert "S1T1" in content
        assert "S1T2" in content

    def test_invalidated_status_recorded(self, tmp_project):
        update_global_state(tmp_project, "S1T2", "Was correct", "ESTABLISHED")
        update_global_state(tmp_project, "S1T2", "Dependency changed", "INVALIDATED")
        content = (tmp_project / "global_state.md").read_text()
        assert "INVALIDATED" in content


class TestInvalidateDependents:

    def test_flags_direct_dependents(self, tmp_project):
        task_tree = {"S1T1": ["S1T2", "S1T3"], "S1T2": ["S2T1"]}
        invalidated = invalidate_dependents(tmp_project, "S1T1", task_tree, "Error found")
        assert "S1T2" in invalidated
        assert "S1T3" in invalidated

    def test_does_not_flag_non_dependents(self, tmp_project):
        task_tree = {"S1T1": ["S1T2"], "S1T3": ["S1T4"]}
        invalidated = invalidate_dependents(tmp_project, "S1T1", task_tree, "Error found")
        assert "S1T3" not in invalidated
        assert "S1T4" not in invalidated

    def test_empty_dependents_returns_empty(self, tmp_project):
        invalidated = invalidate_dependents(tmp_project, "S1T1", {"S1T1": []}, "Error")
        assert invalidated == []

    def test_writes_to_global_state(self, tmp_project):
        invalidate_dependents(tmp_project, "S1T1", {"S1T1": ["S1T2"]}, "Foundational error")
        content = (tmp_project / "global_state.md").read_text()
        assert "S1T2" in content
        assert "INVALIDATED" in content


class TestStageGate:

    def test_passes_when_all_conditions_met(self, tmp_project):
        stage_dir = tmp_project / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        spec = {"id": "S1T1", "complexity": "STANDARD", "description": "test",
                "dependencies": [], "stage": 1, "verification_criteria": "test"}
        (stage_dir / "S1T1.md").write_text(
            "Full derivation.\n\nCHECKS PERFORMED: all checks\nCHECKS NOT PERFORMED: none"
        )
        (stage_dir / "S1T1_verify.md").write_text("RECOMMENDATION: ACCEPT\nAll good.")
        gate = check_stage_gate(tmp_project, 1, [spec])
        assert gate["can_close"] is True
        assert gate["blocking_items"] == []

    def test_blocks_when_output_missing(self, tmp_project):
        spec = {"id": "S1T1", "complexity": "STANDARD", "description": "test",
                "dependencies": [], "stage": 1, "verification_criteria": "test"}
        gate = check_stage_gate(tmp_project, 1, [spec])
        assert gate["can_close"] is False
        assert any("missing" in item for item in gate["blocking_items"])

    def test_blocks_when_verifier_recommends_revise(self, tmp_project):
        stage_dir = tmp_project / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        spec = {"id": "S1T1", "complexity": "STANDARD", "description": "test",
                "dependencies": [], "stage": 1, "verification_criteria": "test"}
        (stage_dir / "S1T1.md").write_text("Output.\n\nCHECKS PERFORMED: done\nCHECKS NOT PERFORMED: none")
        (stage_dir / "S1T1_verify.md").write_text("RECOMMENDATION: REVISE\nError on line 3.")
        gate = check_stage_gate(tmp_project, 1, [spec])
        assert gate["can_close"] is False

    def test_blocks_when_incomplete_marker_present(self, tmp_project):
        stage_dir = tmp_project / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        spec = {"id": "S1T1", "complexity": "HIGH", "description": "test",
                "dependencies": [], "stage": 1, "verification_criteria": "test"}
        (stage_dir / "S1T1.md").write_text("INCOMPLETE — unfinished\n\nCHECKS PERFORMED: none")
        (stage_dir / "S1T1_verify.md").write_text("RECOMMENDATION: ACCEPT")
        gate = check_stage_gate(tmp_project, 1, [spec])
        assert gate["can_close"] is False

    def test_multiple_tasks_all_must_pass(self, tmp_project):
        stage_dir = tmp_project / "stages" / "stage_1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        specs = [
            {"id": "S1T1", "complexity": "STANDARD", "description": "t", "dependencies": [], "stage": 1, "verification_criteria": "t"},
            {"id": "S1T2", "complexity": "STANDARD", "description": "t", "dependencies": [], "stage": 1, "verification_criteria": "t"},
        ]
        # Only S1T1 complete
        (stage_dir / "S1T1.md").write_text("Done.\n\nCHECKS PERFORMED: ok\nCHECKS NOT PERFORMED: none")
        (stage_dir / "S1T1_verify.md").write_text("RECOMMENDATION: ACCEPT")
        gate = check_stage_gate(tmp_project, 1, specs)
        assert gate["can_close"] is False
        assert any("S1T2" in item for item in gate["blocking_items"])
```

### 2.2 Implementation

`core/state_manager.py`:
```python
from pathlib import Path
from datetime import datetime

def update_global_state(project_path: Path, task_id: str, result_summary: str, status: str):
    state_file = project_path / "global_state.md"
    entry = f"\n## {task_id} — {status} — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{result_summary}\n\n---\n"
    with open(state_file, "a") as f:
        f.write(entry)

def invalidate_dependents(project_path: Path, task_id: str, task_tree: dict, reason: str) -> list:
    dependents = task_tree.get(task_id, [])
    for dep_id in dependents:
        update_global_state(
            project_path, dep_id,
            f"FLAGGED: dependency {task_id} corrected. Reason: {reason}. Re-verification required.",
            "INVALIDATED"
        )
    return dependents

def get_established_results(project_path: Path) -> dict:
    state_file = project_path / "global_state.md"
    if not state_file.exists():
        return {}
    results = {}
    for line in state_file.read_text().split("\n"):
        if line.startswith("## S"):
            parts = line.split(" — ")
            if len(parts) >= 2:
                results[parts[0].replace("## ", "").strip()] = parts[1].strip()
    return results

def check_stage_gate(project_path: Path, stage: int, task_specs: list) -> dict:
    stage_dir = project_path / "stages" / f"stage_{stage}"
    blocking = []
    for spec in task_specs:
        task_file = stage_dir / f"{spec['id']}.md"
        if not task_file.exists():
            blocking.append(f"{spec['id']}: output file missing")
            continue
        output = task_file.read_text()
        if "INCOMPLETE" in output:
            blocking.append(f"{spec['id']}: contains INCOMPLETE markers")
        verify_file = stage_dir / f"{spec['id']}_verify.md"
        if not verify_file.exists():
            blocking.append(f"{spec['id']}: no verifier sign-off")
        else:
            verify_content = verify_file.read_text()
            if "RECOMMENDATION: REVISE" in verify_content:
                blocking.append(f"{spec['id']}: verifier recommends revision")
            elif "RECOMMENDATION: ESCALATE" in verify_content:
                blocking.append(f"{spec['id']}: verifier escalated — human review needed")
    return {"can_close": len(blocking) == 0, "blocking_items": blocking}
```

Run: `pytest tests/unit/test_state_manager.py -v`

---

## Phase 3 — Convention Manager

**Time estimate:** 3 days
**Cost:** $0

### 3.1 Tests

`tests/unit/test_convention_manager.py`:
```python
import pytest
from core.convention_manager import check_convention_drift, parse_conventions, add_convention


class TestParsing:

    def test_parses_conventions(self, tmp_project):
        (tmp_project / "conventions.md").write_text("# Conventions\n\nM = count matrix\nX = normalized expression\n")
        c = parse_conventions(tmp_project)
        assert "M" in c
        assert "X" in c

    def test_empty_file_returns_empty_dict(self, tmp_project):
        (tmp_project / "conventions.md").write_text("# Conventions\n\n")
        assert parse_conventions(tmp_project) == {}


class TestDriftDetection:

    def test_flags_alternate_notation(self, tmp_project, bad_convention_drift):
        (tmp_project / "conventions.md").write_text("# Conventions\n\nM = count matrix\n")
        flags = check_convention_drift(bad_convention_drift, tmp_project)
        assert len(flags) > 0

    def test_no_flags_for_consistent_output(self, tmp_project, good_output):
        (tmp_project / "conventions.md").write_text("# Conventions\n\nX = count matrix\nC = correction factor\n")
        flags = check_convention_drift(good_output, tmp_project)
        assert len(flags) == 0

    def test_flags_are_strings_with_context(self, tmp_project, bad_convention_drift):
        (tmp_project / "conventions.md").write_text("# Conventions\n\nM = count matrix\n")
        flags = check_convention_drift(bad_convention_drift, tmp_project)
        assert all(isinstance(f, str) and len(f) > 10 for f in flags)


class TestAddConvention:

    def test_adds_convention(self, tmp_project):
        add_convention(tmp_project, "Z", "latent variable")
        content = (tmp_project / "conventions.md").read_text()
        assert "Z" in content
        assert "latent variable" in content

    def test_no_duplicates(self, tmp_project):
        add_convention(tmp_project, "M", "count matrix")
        add_convention(tmp_project, "M", "count matrix")
        content = (tmp_project / "conventions.md").read_text()
        assert content.count("M = count matrix") == 1
```

### 3.2 Implementation

`core/convention_manager.py`:
```python
import re
from pathlib import Path

def parse_conventions(project_path: Path) -> dict:
    conv_file = project_path / "conventions.md"
    if not conv_file.exists():
        return {}
    conventions = {}
    for line in conv_file.read_text().split("\n"):
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_{}^]*)\s*=\s*(.+)$', line.strip())
        if match:
            conventions[match.group(1)] = match.group(2).strip()
    return conventions

def check_convention_drift(output: str, project_path: Path) -> list:
    conventions = parse_conventions(project_path)
    flags = []
    for line in output.split("\n"):
        for symbol, definition in conventions.items():
            pattern = rf'\b{re.escape(symbol)}\b.*(?:represents|denotes|=|means|is defined as)'
            if re.search(pattern, line, re.IGNORECASE):
                if definition.split()[0].lower() not in line.lower():
                    flags.append(
                        f"Convention drift: '{symbol}' used inconsistently with "
                        f"established '{symbol} = {definition}' — line: {line[:80]}"
                    )
    return flags

def add_convention(project_path: Path, symbol: str, definition: str):
    conv_file = project_path / "conventions.md"
    content = conv_file.read_text() if conv_file.exists() else "# Conventions\n\n"
    entry = f"{symbol} = {definition}"
    if entry not in content:
        with open(conv_file, "a") as f:
            f.write(f"\n{entry}\n")
```

Run: `pytest tests/unit/test_convention_manager.py -v`

---

## Phase 4 — Task Parser

**Time estimate:** 2 days
**Cost:** $0

### 4.1 Tests

`tests/unit/test_task_parser.py`:
```python
import pytest
from core.task_parser import parse_task_tree, validate_task_tree, build_dependency_graph

SAMPLE_TREE = """
## S1T1

- **Description:** Derive batch effect model
- **Dependencies:** none
- **Stage:** 1
- **Verification criteria:** Model reduces to standard form under null
- **Complexity:** HIGH

## S1T2

- **Description:** Implement correction factor
- **Dependencies:** S1T1
- **Stage:** 1
- **Verification criteria:** Output matches analytical derivation within 1e-6
- **Complexity:** STANDARD

## S2T1

- **Description:** Run benchmark validation
- **Dependencies:** S1T1, S1T2
- **Stage:** 2
- **Verification criteria:** Recall > 0.7 on known positive gene sets
- **Complexity:** STANDARD
"""


class TestParsing:

    def test_parses_all_tasks(self):
        assert len(parse_task_tree(SAMPLE_TREE)) == 3

    def test_parses_ids(self):
        ids = [t["id"] for t in parse_task_tree(SAMPLE_TREE)]
        assert "S1T1" in ids and "S2T1" in ids

    def test_parses_single_dependency(self):
        tasks = parse_task_tree(SAMPLE_TREE)
        s1t2 = next(t for t in tasks if t["id"] == "S1T2")
        assert "S1T1" in s1t2["dependencies"]

    def test_parses_no_dependencies(self):
        tasks = parse_task_tree(SAMPLE_TREE)
        s1t1 = next(t for t in tasks if t["id"] == "S1T1")
        assert s1t1["dependencies"] == []

    def test_parses_multiple_dependencies(self):
        tasks = parse_task_tree(SAMPLE_TREE)
        s2t1 = next(t for t in tasks if t["id"] == "S2T1")
        assert "S1T1" in s2t1["dependencies"]
        assert "S1T2" in s2t1["dependencies"]

    def test_parses_complexity(self):
        tasks = parse_task_tree(SAMPLE_TREE)
        s1t1 = next(t for t in tasks if t["id"] == "S1T1")
        assert s1t1["complexity"] == "HIGH"

    def test_parses_stage_number(self):
        tasks = parse_task_tree(SAMPLE_TREE)
        s2t1 = next(t for t in tasks if t["id"] == "S2T1")
        assert s2t1["stage"] == 2


class TestValidation:

    def test_valid_tree_passes(self):
        assert validate_task_tree(parse_task_tree(SAMPLE_TREE)) == []

    def test_catches_empty_verification_criteria(self):
        bad = SAMPLE_TREE.replace(
            "- **Verification criteria:** Model reduces to standard form under null",
            "- **Verification criteria:**"
        )
        errors = validate_task_tree(parse_task_tree(bad))
        assert any("verification" in e.lower() for e in errors)

    def test_catches_dependency_on_nonexistent_task(self):
        bad = SAMPLE_TREE.replace("- **Dependencies:** S1T1\n- **Stage:** 1\n- **Verification criteria:** Output", "- **Dependencies:** S9T9\n- **Stage:** 1\n- **Verification criteria:** Output")
        errors = validate_task_tree(parse_task_tree(bad))
        assert any("S9T9" in e for e in errors)


class TestDependencyGraph:

    def test_correct_reverse_graph(self):
        graph = build_dependency_graph(parse_task_tree(SAMPLE_TREE))
        assert "S1T2" in graph["S1T1"]
        assert "S2T1" in graph["S1T1"]

    def test_leaf_nodes_empty(self):
        graph = build_dependency_graph(parse_task_tree(SAMPLE_TREE))
        assert graph.get("S2T1", []) == []
```

### 4.2 Implementation

`core/task_parser.py`:
```python
import re
from typing import Optional

def parse_task_tree(content: str) -> list:
    tasks = []
    sections = re.split(r'\n## (S\d+T\d+)\n', content)
    i = 1
    while i < len(sections):
        task_id = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        tasks.append({
            "id": task_id,
            "description":          _field(body, "Description") or "",
            "dependencies":         _dependencies(body),
            "stage":                _stage(body, task_id),
            "verification_criteria": _field(body, "Verification criteria") or "",
            "complexity":           _field(body, "Complexity") or "STANDARD",
        })
        i += 2
    return tasks

def _field(body: str, name: str) -> Optional[str]:
    m = re.search(rf'\*\*{name}:\*\*\s*(.+?)(?:\n|$)', body, re.IGNORECASE)
    if m:
        v = m.group(1).strip()
        return None if v.lower() in ["", "none", "n/a"] else v
    return None

def _dependencies(body: str) -> list:
    raw = _field(body, "Dependencies")
    if not raw:
        return []
    return [d.strip() for d in raw.split(",") if re.match(r'S\d+T\d+', d.strip())]

def _stage(body: str, task_id: str) -> int:
    raw = _field(body, "Stage")
    if raw and raw.isdigit():
        return int(raw)
    m = re.match(r'S(\d+)T\d+', task_id)
    return int(m.group(1)) if m else 1

def validate_task_tree(tasks: list) -> list:
    errors = []
    ids = {t["id"] for t in tasks}
    for t in tasks:
        if not t["verification_criteria"]:
            errors.append(f"{t['id']}: verification criteria is empty or missing")
        if not t["description"]:
            errors.append(f"{t['id']}: description is empty")
        for dep in t["dependencies"]:
            if dep not in ids:
                errors.append(f"{t['id']}: dependency '{dep}' not found in task tree")
    return errors

def build_dependency_graph(tasks: list) -> dict:
    graph = {t["id"]: [] for t in tasks}
    for t in tasks:
        for dep in t["dependencies"]:
            if dep in graph:
                graph[dep].append(t["id"])
    return graph
```

Run: `pytest tests/unit/ -v` — all four modules, zero model calls.

---

## Phase 5 — Oracle Tests

**Time estimate:** 3 days
**Cost:** $0

`tests/oracle/test_benchmark_checks.py`:
```python
import numpy as np
import pytest
from oracle.genomics.benchmark_checks import (
    check_pvalue_calibration,
    check_known_positives_recovered,
    check_method_degrades_gracefully,
)


class TestPvalueCalibration:

    def test_uniform_pvalues_pass(self):
        assert check_pvalue_calibration(np.random.uniform(0, 1, 1000))["pass"] is True

    def test_all_small_pvalues_fail(self):
        assert check_pvalue_calibration(np.random.uniform(0, 0.01, 1000))["pass"] is False

    def test_returns_required_fields(self):
        r = check_pvalue_calibration(np.random.uniform(0, 1, 100))
        for field in ["check", "pass", "interpretation"]:
            assert field in r

    def test_interpretation_contains_pass_or_fail(self):
        r = check_pvalue_calibration(np.random.uniform(0, 1, 100))
        assert "PASS" in r["interpretation"] or "FAIL" in r["interpretation"]


class TestKnownPositivesRecovery:

    def test_perfect_recall_passes(self):
        r = check_known_positives_recovered(
            {"GENE1": 0.01, "GENE2": 0.02, "GENE3": 0.03},
            ["GENE1", "GENE2", "GENE3"]
        )
        assert r["pass"] is True and r["recall"] == 1.0

    def test_zero_recall_fails(self):
        r = check_known_positives_recovered(
            {"GENE4": 0.01}, ["GENE1", "GENE2", "GENE3"]
        )
        assert r["pass"] is False and r["recall"] == 0.0

    def test_above_threshold_recall_passes(self):
        r = check_known_positives_recovered(
            {"G1": 0.01, "G2": 0.02, "G3": 0.03, "G4": 0.04},
            ["G1", "G2", "G3", "G4", "G5"]
        )
        assert r["recall"] == 0.8 and r["pass"] is True

    def test_nonsignificant_genes_excluded(self):
        r = check_known_positives_recovered(
            {"GENE1": 0.001, "GENE2": 0.9}, ["GENE1", "GENE2"]
        )
        assert r["recall"] == 0.5


class TestGracefulDegradation:

    def test_identical_results_pass(self):
        d = {"G1": 0.01, "G2": 0.02}
        r = check_method_degrades_gracefully(d, d)
        assert r["pass"] is True and r["overlap"] == 1.0

    def test_completely_different_fails(self):
        r = check_method_degrades_gracefully({"G1": 0.01}, {"G2": 0.01})
        assert r["pass"] is False and r["overlap"] == 0.0

    def test_empty_inputs_handled(self):
        r = check_method_degrades_gracefully({}, {})
        assert "pass" in r
```

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

def check_pvalue_calibration(pvalues: np.ndarray, alpha: float = 0.05) -> dict:
    stat, p = stats.kstest(pvalues, 'uniform')
    passed = p > alpha
    return {
        "check": "p-value calibration under null",
        "statistic": stat, "p_value": p, "pass": passed,
        "interpretation": "PASS: p-values appear calibrated" if passed
                         else f"FAIL: p-values not uniform (KS p={p:.4f})"
    }

def check_known_positives_recovered(
    results: dict, known_positive_genes: list, fdr_threshold: float = 0.05
) -> dict:
    sig = {g for g, fdr in results.items() if fdr < fdr_threshold}
    recovered = sig & set(known_positive_genes)
    recall = len(recovered) / len(known_positive_genes) if known_positive_genes else 0.0
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

def check_method_degrades_gracefully(
    full_results: dict, downsampled_results: dict, tolerance: float = 0.3
) -> dict:
    if not full_results:
        return {"check": "graceful degradation", "overlap": 0.0, "pass": True,
                "interpretation": "PASS: no full results to compare"}
    overlap = len(set(full_results) & set(downsampled_results)) / len(full_results)
    passed = overlap > (1 - tolerance)
    return {
        "check": "graceful degradation under downsampling",
        "overlap": overlap, "pass": passed,
        "interpretation": f"{'PASS' if passed else 'FAIL'}: {overlap:.1%} overlap"
    }
```

Run: `pytest tests/unit/ tests/oracle/ -v --tb=short`

---

## Phase 6 — Model Integration

**Time estimate:** 1 week
**Cost:** ~$0–5 (Groq free tier covers most of this)

Only after all unit and oracle tests pass.

```bash
pip install groq tavily-python
```

`.env`:
```
MODEL_BACKEND=groq
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

`tests/integration/test_decomposer_integration.py`:
```python
import pytest, os
pytest.importorskip("groq")

pytestmark = pytest.mark.skipif(
    os.getenv("MODEL_BACKEND", "mock") == "mock",
    reason="Requires MODEL_BACKEND != mock"
)

from core.decomposer import decompose, adversarial_review

GOAL = "Develop a statistical correction for batch effects in scRNA-seq trajectory inference."
DOMAIN = "Single-cell RNA-seq. Methods: Monocle, PAGA, Scanpy."

def test_produces_structured_output():
    tree = decompose(GOAL, DOMAIN)
    assert len(tree) > 100
    assert any(m in tree for m in ["S1T1", "Stage", "Description", "##"])

def test_adversarial_review_produces_feedback():
    tree = decompose(GOAL, DOMAIN)
    review = adversarial_review(tree, GOAL)
    assert len(review) > 50
    assert any(w in review.upper() for w in ["APPROVED", "MISSING", "ISSUE", "CHECK"])
```

**Run unit tests (free, always):**
```bash
pytest tests/unit/ tests/oracle/ -v
```

**Run integration tests (costs credits, run sparingly):**
```bash
MODEL_BACKEND=groq pytest tests/integration/ -v
```

**What to evaluate during integration testing:**

When first running with a real model, read every output. Specific things to check: Does the decomposer output parse cleanly with `task_parser.py`? Are supervisor heuristics firing on real model failures? Are there false positives? When you push back on a correct result in the verifier prompt, does the model cave? (Expected on Llama 70B — note it, your architecture should catch the changed output.)

---

## Phase 7 — Full Pipeline and Paper Run

Upgrade to frontier model by changing one line:

```
MODEL_BACKEND=anthropic
```

Everything else is unchanged. The architecture you built on Llama 70B runs identically on Claude Opus.

---

## Running Tests

```bash
# Fast, free — run constantly during development
pytest tests/unit/ tests/oracle/ -v

# With coverage
pytest tests/unit/ tests/oracle/ --cov=core --cov=oracle --cov-report=term-missing

# Integration (costs API credits)
MODEL_BACKEND=groq pytest tests/integration/ -v
```

---

## Cost Tracking

| Phase | Model calls | Cost |
|-------|-------------|------|
| 0–5 (architecture + all unit tests) | 0 | $0 |
| 6 (Groq integration) | ~20–30 | ~$0–1 (free tier) |
| 7 (full pipeline, Groq) | ~100–150 | ~$1–5 |
| 7 (paper run, Claude Sonnet) | ~100–150 | ~$15–25 |
| 7 (paper run, Claude Opus) | ~100–150 | ~$40–80 |
| **Total to first paper run** | | **~$20–50** |

---

## Known Gaps to Investigate

**Gap 1 — Decomposition quality:** Does independent re-decomposition + diff catch more gaps than adversarial review of one decomposition? Test with 5 research goals.

**Gap 2 — Cross-task consistency:** Haiku call after every task hunting contradictions with global_state.md. Measure false positive rate on synthetic cases.

**Gap 3 — Correlated model failure:** When Claude and GPT agree, are their reasoning paths identical? Does reasoning path similarity correlate with correctness?

**Gap 4 — Open source sycophancy:** Quantify how often Llama 70B changes a correct answer under pressure. Does the supervisor catch the changed output? This tells you whether architecture compensates for weaker model behavior.
