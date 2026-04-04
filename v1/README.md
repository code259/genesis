# Genesis v1

Genesis v1 is a prototype autonomous research system that combines a meta-harness,
specialist research modules, persistent project state, and paper synthesis.

This directory contains the v1 implementation scaffold, CLI, storage/runtime
contracts, and tests for the end-to-end orchestration path described in the
specification and implementation guide.

## Layout

- `genesis/` — application package
- `.opencode/` — agent/runtime configuration
- `scripts/` — operational scripts
- `tests/` — unit, integration, and benchmark tests

## Quick start

```bash
cd v1
python -m pip install -e ".[dev]"
pytest
python -m genesis.cli.main --help
```

## Runtime note

The implementation work happened in a dedicated Git worktree branch for
isolation, but runtime experiment parallelization inside the system uses
isolated execution sandboxes rather than `git worktree`.
