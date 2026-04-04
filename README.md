# Genesis

Genesis is an autonomous AI research system that takes a project spec, decomposes
the work, executes research iterations, verifies outputs, and synthesizes a
paper-ready artifact set.

The repository currently contains two major lines of work:

- `v0/` — earlier experimentation and prototype research infrastructure
- `v1/` — the current Genesis v1 system, including the harness, specialist
  modules, storage layer, CLI, benchmarks, and paper synthesis pipeline

## Current State

The active system is `v1/`.

Genesis v1 is designed around:

- a meta-harness that manages project runs
- specialist modules for adversarial checking, optimization, ideation, oracle
  generation, citations, plotting, verification, and paper synthesis
- persistent project state under per-project directories
- a local-first execution model that uses isolated runtime sandboxes rather than
  git worktrees for experiment execution

This repository has gone through several implementation passes, and the most
complete delivery branch may be ahead of the local `main` checkout depending on
your workspace state. If you are evaluating or extending v1, verify the branch
you are on before starting work.

## Repository Layout

- `v0/` — legacy/prototype code and experiments
- `v1/` — Genesis v1 package, configs, scripts, docs, and tests
- `v1/genesis/` — core Python package
- `v1/scripts/` — operational scripts such as taste initialization and manifold building
- `v1/tests/` — unit, integration, and benchmark coverage

## Working With v1

Start in `v1/`:

```bash
cd v1
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev,data,verification,paper]"
python -m pytest -q
python -m genesis.cli.main --help
```

Additional optional dependency groups are available for heavier local runs:

- `ml` — embeddings, graph/manifold, and model-oriented dependencies
- `domain` — astrophysics-oriented domain tooling

Example:

```bash
cd v1
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev,data,verification,paper,ml,domain]"
```

If you are installing the `ml` extra in a fresh environment, Genesis currently expects a `NumPy 1.x` stack and a modern `torch` build. If you previously installed an incompatible mix, repair it with:

```bash
python -m pip install --upgrade --force-reinstall "numpy<2" "torch>=2.4" "sentence-transformers>=3.0"
python -m pip install -e ".[dev,data,verification,paper,ml,domain]"
```

## CLI Overview

Genesis v1 currently exposes these operator-facing commands:

- `genesis init`
- `genesis run`
- `genesis status`
- `genesis intervene`
- `genesis results`
- `genesis build-manifold`
- `genesis init-taste`

These commands operate on project specs and project runtime directories under
the configured output root.

## Typical Local Flow

For a local-first run:

1. Install the required extras for the path you want to exercise.
2. Initialize the taste store with `python3 scripts/init_taste.py`.
3. Build or seed the manifold with `python3 scripts/build_manifold.py --domain ...`.
4. Prepare a project spec JSON.
5. Run the system with `python3 -m genesis.cli.main run --project-id ... --spec ...`.

Outputs are written per project, including:

- instructions and traces
- adversarial and verification reports
- generated code/output artifacts
- paper assets such as LaTeX, figures, and synthesis reports

## Notes

- Runtime experiment execution is sandbox-based, not git-worktree-based.
- Local development may use git worktrees extensively for implementation lanes.
- Some heavier backends have optional support paths; install the relevant extras
  if you want to exercise them locally.

## See Also

- [v1/README.md](/Users/nikhilmaturi/Files/Projects/genesis/v1/README.md)
- `v1/genesis_v1_spec.docx`
- `v1/genesis_v1_implementation_guide.docx`
