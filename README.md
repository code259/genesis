# Genesis

> *"This may be the most important paper I've ever written—not for the physics, but for the method. There is no going back."*
> — Matthew Schwartz, Harvard Professor of Physics

---

## Overview

Orchestrate is a research orchestration system designed to close the gap between what frontier AI models *can* do and what they *reliably produce* in real scientific work. It is not an autonomous AI scientist. It is a structured scaffold that lets a domain expert direct AI assistants through rigorous, verifiable, multi-stage research — producing outputs that are genuinely novel, not just plausible-looking.

The core insight: current LLMs fail at long research tasks not because they lack capability, but because they lack architecture. They lose context, drift from conventions, fake verification, stop checking too early, and collapse under pressure. Orchestrate treats these as engineering problems and solves them systematically.

---

## Inspiration

In December 2025, Harvard physicist Matthew Schwartz guided Claude Opus 4.5 through a complete theoretical physics calculation — producing a genuine contribution to quantum field theory in two weeks rather than the typical one to two years. He used no autonomous AI pipeline. Instead, he built, largely by intuition, a structured workflow: hierarchical task trees, cross-model adversarial verification, explicit anti-hallucination constraints, and stage-gated progression.

The paper was real. The result was novel. The method was reproducible.

Schwartz's conclusion: *"AI is not doing end-to-end science yet. But this project proves that I could create a set of prompts that can get Claude to do frontier science."*

Orchestrate systematizes what Schwartz did manually. The supervisor logic, cross-check triggers, verification heuristics, and escalation paths that lived in his head become explicit, reusable, domain-adaptable components.

Previous attempts at AI research systems — Sakana's AI Scientist, Google's research agent, FutureHouse's pipeline — failed not primarily because models were too weak, but because they optimized for the wrong thing: end-to-end autonomy with no graceful degradation, no domain-grounded verification, no mechanism to catch foundational errors before they propagated. Their outputs looked like papers. They were not.

---

## Goals

**v1 — Make the Schwartz workflow reproducible**
- Structured task decomposition with dependency tracking
- Supervisor agent that monitors for known failure modes and triggers cross-checks
- Multi-model adversarial verification as a first-class workflow primitive
- Convention tracking across long projects
- Stage gating: no downstream work until upstream outputs pass verification
- Domain-specific verification oracles (starting with computational genomics)
- Cost-efficient model routing: Haiku for orchestration, Sonnet for execution, targeted use of stronger models only when needed

**v2 — Extend to ideation**
- Integration with literature synthesis tools to identify open problems
- Krenn-style idea generation as an upstream layer feeding the execution pipeline
- Human-in-the-loop taste layer: system proposes directions, expert selects and refines
- Moving from G2 problems (well-defined, known answer) toward G3 problems (open-ended, requires judgment about which direction matters)

---

## Future Directions

**Auto-Research Integration**
Karpathy's auto-research framing — giving models tools to search, read, and synthesize literature autonomously — sits naturally upstream of this system. The planned integration: auto-research identifies open problems and relevant prior work, Orchestrate executes the chosen problem rigorously, human expert provides direction and validates outputs. Neither layer works well alone. Together they cover the full research cycle.

**Dependency-Aware Error Propagation**
When a foundational result is corrected, all downstream work that depends on it should be automatically flagged for re-verification. This requires maintaining an explicit dependency graph across the task tree — not just a flat list of completed tasks. This is one of the most impactful unsolved problems and a priority for v2.

**Domain-Specific Verification Oracles**
Moving beyond model-vs-model cross-checking toward ground-truth verification: mathematical constraints the output must satisfy, benchmark datasets with known results, theoretical properties that are checkable without human review. Each new domain requires building this oracle layer. The long-term vision is a library of domain verification modules that can be plugged into the core orchestration framework.

**Collaborative Multi-Agent Research**
Multiple specialized agents working in parallel on different stages — a derivation agent, a numerical validation agent, a literature agent, a writing agent — coordinated by the supervisor. Each agent is narrow and verifiable. The supervisor maintains global coherence.

**Taste Layer**
The hardest long-term problem: encoding research taste. Which problems are worth solving? Which approximations matter? When is a result interesting enough to publish? Currently this lives entirely with the human expert. Eventually, with enough examples of good vs. bad research choices, this could be learned. This is the difference between a very capable G2 system and a genuine G3 system.

---

## The Target Output

A system that enables a domain expert to go from a well-specified research question to a publication-quality result — with genuine novelty, verified correctness, and full auditability of every step — in days rather than months.
