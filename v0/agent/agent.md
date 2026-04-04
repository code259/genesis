# AGENT.md — Genesis

## Core Principles

**Build the narrowest thing that works.** Don't add components until the prior component passes its test. v1 proves the core insight on one problem. Everything else is v2.

---

## Code Standards

### Single Responsibility
Each module does one thing. `decomposer.py` decomposes. `executor.py` executes. `verifier.py` verifies. Do not bleed logic across files. If a function is doing two things, split it.

### Functions Should Be Small and Obvious
If you need a comment to explain what a function does, the function is probably too big. Name it so it explains itself. `adversarial_review()` is better than `run_second_pass()`.

### Explicit Over Implicit
No magic. No hidden defaults that change behavior silently. Model routing lives in `config.py`, not scattered inline. Prompts live in `prompts/`, not buried in function bodies.

### Fail Loudly
Don't silently swallow errors or return empty strings on failure. Raise exceptions with context. The supervisor/verifier pipeline depends on detecting failures — a silent failure is worse than a crash.

### No Premature Abstraction
Don't build a general framework for something that has one use case. Build the specific thing. Generalize only when you have two concrete cases that share a pattern.

---

## Project-Specific Rules

**Prompts are files, not strings.** System prompts live in `prompts/` as `.md` files and are read at runtime. Never hardcode a multi-line prompt inline.

**State is explicit.** All cross-task context flows through `global_state.md`. Don't pass context implicitly through function arguments across module boundaries.

**Human checkpoints are not optional.** The stage gate check, decomposition review, and escalation paths exist for a reason. Do not auto-proceed past them.

**Token costs are real.** Respect the model routing: Haiku for supervisor, Sonnet for execution/verification, GPT-4o only for foundational cross-checks. Don't use a heavier model where a lighter one is specified.

**Tests gate phases.** Each phase has a pass criteria. Don't move to the next phase until it's met. See `V1_IMPLEMENTATION.md` for the criteria per phase.

---

## Testing

- Write tests in `tests/` before considering a component done
- Oracle checks in `oracle/` are programmatic and must run without API calls
- Use `pytest` — no custom test runners
- A component without a test is not done

---

## What Not To Do

- Don't add a database or vector store — plain markdown files only
- Don't add retry logic beyond the 3-attempt max already specified
- Don't generalize `run_stage.py` into a general task runner until v1 is complete
- Don't change model assignments without updating `config.py` and noting the cost delta