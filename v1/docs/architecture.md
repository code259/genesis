# Genesis v1 Architecture Notes

This note reconciles the v1 spec, implementation guide, and the architecture
issue for the overnight implementation branch.

## Cross-cutting concerns

- **Persistent state**: project filesystem, experiment ledger, manifold index,
  taste-model persistence, and causal DAG are explicit first-class surfaces.
- **Token budgeting**: all proposer-side history selection and instruction
  generation must stay budget-aware.
- **Domain knowledge injection**: provider registry controls optional
  domain-specific context rather than hard-coding astrophysics behavior into the
  harness.
- **Observability**: run-level events must be written as structured JSON logs so
  the system can support postmortems and future harness learning.
- **Failure handling**: intervention files, adversarial stalemates, and HALT
  files are part of the runtime contract and not ad hoc side channels.

## Parallelization update

Runtime experiment parallelization does **not** use `git worktree`.

The implementation guide originally described experiment variants running in Git
worktrees. For v1 runtime execution we replace that with **isolated execution
sandboxes** under the project runtime tree. This keeps experiment batches
parallel without coupling runtime experiment management to repository branch
state.

Current branch/worktree usage is for **developer workflow only**, not for the
system's internal experiment runner.

## Graph-VAE manifold update

The ideation manifold preserves a **continuous latent space** with meaningful
interpolation. Greedy adjacency, pollination, and low-density exploration all
share this latent representation.

## Adversarial stalemate policy

Current implementation keeps the spec defaults as the initial runtime contract:

- 3 adversarial iterations on the same CA output before raising
  `ADVERSARIAL_STALEMATE`
- 2 additional meta-harness escalation attempts before writing a
  human-required HALT condition

This remains explicitly reviewable in the dedicated stalemate-policy issue and
should be treated as configurable rather than hard-coded doctrine.
