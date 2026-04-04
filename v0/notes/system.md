# Orchestrate — Project Specification

## System Overview

Orchestrate is a multi-layer research orchestration framework. It is not a single agent, a chatbot wrapper, or an autonomous pipeline. It is a structured system of cooperating components — each with a narrow, well-defined responsibility — coordinated by a supervisor layer that encodes explicit heuristics for when to proceed, when to verify, and when to escalate to a human.

The system is designed around a core principle: **the model that produces a result must never be the model that verifies it.**

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   HUMAN EXPERT                      │
│         (direction, taste, final validation)        │
└───────────────────────┬─────────────────────────────┘
                        │ escalation triggers
┌───────────────────────▼─────────────────────────────┐
│                   SUPERVISOR AGENT                  │
│   - monitors outputs for failure mode signatures    │
│   - triggers cross-checks when heuristics fire      │
│   - enforces stage gating                           │
│   - maintains global state + convention file        │
│   - routes tasks to appropriate model tier          │
└──────┬────────────────┬───────────────────┬─────────┘
       │                │                   │
┌──────▼──────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  DECOMPOSER │  │  EXECUTOR   │  │    VERIFIER     │
│             │  │             │  │                 │
│ Task tree   │  │ Runs tasks  │  │ Adversarial     │
│ generation  │  │ writes to   │  │ cross-check,    │
│ dependency  │  │ file tree   │  │ different model │
│ mapping     │  │             │  │ no prior ctx    │
└─────────────┘  └─────────────┘  └────────┬────────┘
                                            │
                              ┌─────────────▼──────────┐
                              │  DOMAIN ORACLE         │
                              │                        │
                              │ Ground-truth checks:   │
                              │ benchmarks, invariants,│
                              │ theoretical constraints│
                              └────────────────────────┘
```

---

## Component Specifications

### 1. Decomposer

**Responsibility:** Take a research goal and produce a structured task tree with explicit dependencies.

**Inputs:** Research goal (natural language), domain context file, prior literature summary

**Outputs:** Markdown task tree. Each task has: ID, description, dependencies (list of task IDs that must complete first), stage assignment, verification criteria (what does done actually look like), estimated complexity tier

**Model:** Sonnet for generation, then adversarial review of the decomposition itself by a second call before any execution starts.

**Critical design note:** The decomposition is the single most important step (Schwartz confirms this). A bad task tree corrupts everything downstream. The adversarial review of the decomposition — checking for missing subtasks, wrong dependencies, under-specified verification criteria — is not optional.

> ⚠️ **Open problem:** How do you evaluate decomposition quality before execution? The model reviewing its own decomposition has limited value. One approach: have the verifier model independently decompose the same goal, then diff the two trees and flag discrepancies for human review. This needs experimentation — what's the right diff metric, and what level of discrepancy should trigger human escalation vs. automatic resolution?

---

### 2. Executor

**Responsibility:** Work through tasks sequentially within a stage, writing outputs to a structured file tree.

**Inputs:** Single task specification, relevant prior task outputs (looked up from file tree, not held in context), domain constraint file, convention file

**Outputs:** Task result written to dedicated markdown file. Format: derivation/reasoning (full detail, no skipping steps), conclusion, list of checks performed, explicit statement of any assumptions made.

**Model:** Sonnet for standard execution tasks. Stronger model only for tasks flagged as high-complexity by the Decomposer.

**File structure:**
```
project/
  master_plan.md          # full task tree
  constraints.md          # CLAUDE.md equivalent, domain-specific
  conventions.md          # notation, sign conventions, field-specific defaults
  stages/
    stage_1/
      summary.md          # stage-level summary, updated as tasks complete
      task_001.md
      task_002.md
      ...
  global_state.md         # what has been established, what is pending, what is invalidated
```

**Key constraint built into executor prompt:** "NEVER use phrases like 'this becomes', 'for consistency', 'it follows that', or 'clearly' to skip steps. Either show the full derivation or write: INCOMPLETE — [what is missing]."

**Key constraint on completion claims:** Executor must end every task output with: "Checks performed: [exhaustive list]. Checks NOT performed: [list of checks that would strengthen this result but were not done]."

> ⚠️ **Open problem:** The executor operates on single tasks in isolation, which prevents it from noticing cross-task inconsistencies. The global_state.md file partially addresses this, but the model reading it still has to recognize when its current output contradicts prior established results. This is unreliable. A more robust approach might be a lightweight "consistency check" call after every task that diffs the new output against global_state.md specifically looking for contradictions — but this adds cost and latency. The right trigger heuristic for when to run this check needs experimentation.

---

### 3. Supervisor Agent

**Responsibility:** Monitor executor outputs, decide when to trigger verification, enforce stage gating, maintain global state, route escalations to human.

**Model:** Haiku. The supervisor does not need to understand the research content deeply — it needs to pattern-match on outputs and apply routing logic. This is where most cost savings come from.

**Failure mode heuristics (trigger cross-check when any fire):**

| Signal | Trigger |
|--------|---------|
| Output contains "verified" or "confirmed" without listing specific checks | Mandatory verifier call |
| Plot or numerical result appears without derivation trace | Mandatory verifier call |
| Phrases: "for consistency", "this becomes", "it follows", "clearly" | Flag + soft verifier call |
| Task marked complete but "Checks NOT performed" list is non-empty | Hold stage gate, prompt executor to complete checks |
| Same error appears in consecutive task outputs | Escalate to human |
| Model has iterated on same error 3+ times | Escalate to human |
| Stage marked complete — audit all subtask completion claims | Mandatory audit before proceeding |
| Foundational result modified | Flag all dependent tasks for re-verification |
| Convention file reference missing from output | Convention drift warning, soft re-prompt |

**Stage gating logic:** A stage cannot close until every task in it has: (a) a completed output file, (b) a verifier sign-off, (c) no open items in "Checks NOT performed" that the supervisor classifies as mandatory. The supervisor makes this classification using the domain constraint file.

**Escalation to human:** The supervisor generates a structured escalation report: what triggered it, what the model has tried, what specifically the human needs to decide. Not a raw dump of the conversation — a clean summary.

> ⚠️ **Open problem:** The heuristic list above is a starting point derived from Schwartz's documented failure modes. It will miss failure modes that are domain-specific or novel. One direction: after each completed paper, do a retrospective — what errors occurred, were they caught by existing heuristics, if not what heuristic would have caught them — and update the heuristic set. This turns the supervisor into a system that improves with use. The data structure for encoding and versioning heuristics needs design.

---

### 4. Verifier

**Responsibility:** Adversarially review executor outputs. Produce a structured assessment with pass/fail on each check, specific failure descriptions, and recommended remediation.

**Critical design constraint:** The verifier receives the task specification and the executor output. It does NOT receive the executor's conversation history, reasoning process, or any context about what the executor was "trying to do." The goal is cold review — what does this output actually say, is it correct, is it complete.

**Model:** Different from executor. If executor is Sonnet, verifier should be a different provider (GPT-4o or Gemini) for the highest-stakes checks. For routine checks, a second Sonnet call with a cold context is sufficient and much cheaper.

**Verifier output format:**
```
TASK: [task ID]
CHECKS PERFORMED:
  - [check description]: PASS / FAIL / UNABLE TO VERIFY
    [if FAIL]: specific description of error, line/equation reference
    [if UNABLE]: what additional information would be needed
RECOMMENDATION: ACCEPT / REVISE / ESCALATE
  [if REVISE]: specific remediation instructions
  [if ESCALATE]: reason escalation is needed
```

**Multi-model strategy:** For foundational results (anything tagged as a dependency by 3+ downstream tasks), require sign-off from two different model providers before accepting. Cost is justified by the error propagation risk.

> ⚠️ **Open problem:** Model-vs-model verification has a ceiling. Claude and GPT share training data and have correlated failure modes — they can agree on something that's wrong. The domain oracle layer (below) is the real solution, but building it is domain-specific work. For domains where no oracle exists yet, is there a way to detect when two models are agreeing based on shared prior rather than independent verification? One hypothesis: if both models produce identical reasoning chains, that's suspicious — genuine independent verification should produce different paths to the same conclusion. This is worth testing empirically.

---

### 5. Domain Oracle

**Responsibility:** Ground-truth verification that doesn't depend on model judgment. The oracle knows what the output *must* satisfy regardless of whether any model agrees.

**Structure:** A set of domain-specific checks that can be run programmatically or with minimal model involvement.

**For computational genomics (v1 target domain):**
- Does the method recover known results on benchmark datasets (e.g., SEQC, MAQC)?
- Do p-value distributions look calibrated (uniform under null)?
- Do enrichment scores correlate with known pathway databases?
- Does the method degrade gracefully with downsampling?
- Do results agree with published results on shared datasets?

**For theoretical physics (Schwartz's domain):**
- Renormalization group invariance
- Fixed-order limits
- Known limiting cases
- Dimensional analysis

**Implementation:** The oracle is a test suite, not a model call. Where possible it runs automatically after each relevant task. The supervisor checks oracle results as part of stage gating.

> ⚠️ **Open problem:** Building the oracle is the highest-value, highest-effort part of adding a new domain. It requires genuine domain expertise to specify what "must be true" about valid outputs. The open question is how much of this can be automated — can a domain expert describe the constraints in natural language and have the system convert them into runnable checks? Or does each check need to be hand-coded? The answer probably varies by domain. This is worth investing design time in before v1 launch in any new domain.

---

### 6. Convention and Global State Manager

**Responsibility:** Maintain the single source of truth for conventions, established results, and project state across the entire project lifetime.

**Files:**
- `conventions.md` — notation, sign conventions, field-specific terminology, non-standard definitions. Updated only by explicit human approval.
- `global_state.md` — structured record of: what has been established (with task ID reference), what is pending, what has been invalidated (with reason and dependent tasks flagged).

**Convention drift detection:** Every executor output is diff'd against conventions.md before being accepted. Any term, notation, or convention that appears in the output but isn't in conventions.md generates a flag. The supervisor decides whether to: add it to conventions (new convention established), reject the output (convention violation), or escalate to human (ambiguous case).

---

## Model Routing and Cost Structure

| Component | Model | Rationale |
|-----------|-------|-----------|
| Supervisor | Haiku | Pattern matching, routing logic — doesn't need deep reasoning |
| Decomposer | Sonnet | Needs to reason about task structure — Haiku too weak |
| Executor (standard) | Sonnet | Core execution — balance of capability and cost |
| Executor (high-complexity) | Sonnet extended thinking or Opus | Only for tasks flagged as foundational or high-stakes |
| Verifier (routine) | Sonnet, cold context | Different call, no conversation history |
| Verifier (foundational) | GPT-4o or Gemini | Cross-provider for correlated failure mode mitigation |
| Convention drift checker | Haiku | Simple diff task |

**Cost optimization:**
- Prompt caching for: system prompt, constraints file, conventions file, domain context. Pay once per session.
- Batch API for: literature synthesis, non-interactive verification passes, convention drift checks.
- Executor reads from file tree rather than holding prior context. Context window per call stays small.

**Estimated cost per full research pipeline run (narrow bioinformatics problem):** $5–15 with disciplined model routing.

---

## Stage Gate Protocol

Execution proceeds in stages. Each stage has a defined set of tasks. The following must all be true before a stage closes:

1. Every task has a completed output file
2. Every task has verifier sign-off (ACCEPT)
3. No task has mandatory open items in "Checks NOT performed"
4. Domain oracle checks for this stage all pass
5. global_state.md has been updated with this stage's outputs
6. No open convention drift flags

If any condition fails, the stage remains open. The supervisor generates a stage completion report listing what remains and why.

**Human checkpoint:** After every stage closes, a human-readable stage summary is generated. The human can: approve and proceed, request revisions before proceeding, or flag a foundational issue that requires re-opening the stage.

---

## Anti-Hallucination Constraint File (CLAUDE.md equivalent)

Built into every executor system prompt:

```
NEVER use these phrases to skip steps:
- "this becomes"
- "for consistency"  
- "it follows that"
- "clearly"
- "one can show"
- "it is straightforward to verify"

If you cannot derive something, write: INCOMPLETE — [what is missing and why]

NEVER say "verified" unless you list every specific check you ran.

NEVER adjust parameters to make results look better. If a result looks wrong, say so.

NEVER invent terms, coefficients, or citations. If you don't know, say so.

When you find an error, do not stop. Check again. Keep checking until you find nothing new, then list every check you ran.
```

---

## v1 Scope

The v1 system covers:

- Decomposer (with adversarial decomposition review)
- Executor (with structured output format and anti-hallucination constraints)
- Supervisor (with hardcoded heuristic set from Schwartz failure modes)
- Verifier (single-provider cold-context review)
- Convention manager (conventions.md + drift detection)
- Global state manager (global_state.md + dependency flagging)
- Stage gate protocol
- Domain oracle for one target domain (computational genomics, differential expression methods)

Not in v1:
- Multi-provider verifier for foundational results (added in v1.5)
- Dependency-aware error propagation graph (v2)
- Ideation layer / literature synthesis integration (v2)
- Additional domain oracles (added per domain)

---

## Success Criteria for v1

A completed v1 system produces, for a well-specified computational genomics methods problem:

1. A structured task tree with explicit dependencies
2. A complete set of executed task outputs in the file tree
3. Verifier sign-offs for every task
4. Domain oracle pass for all checks
5. A LaTeX-ready paper draft synthesized from task outputs
6. Full audit trail: every result traceable to the task that produced it, every verification step logged

The output is evaluated by a domain expert who did not participate in the run. Pass criteria: the expert agrees the result is correct, novel, and publication-quality.
