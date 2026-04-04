# Genesis v1 proposer skill contract

- The meta-harness may create and update files under `projects/<id>/`, `taste_db/`,
  and `manifold_index/`.
- The meta-harness must never modify an existing `projects/<id>/spec.json`.
- The meta-harness must never delete `projects/<id>/runs/`.
- `instruction.md` must contain: objective, selected context, budget, requested
  modules, explicit next action, and validation expectations.
- Stopping criteria are governed by the adversarial checker contract and the
  project intervention / HALT rules.
