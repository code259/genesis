# Genesis v1

Genesis v1 is an autonomous research-system prototype built around a Python meta-harness, a real OpenCode/Oh My OpenAgent executor path, persistent project state, adversarial/verification modules, ideation/manifold tooling, and paper synthesis.

This `v1/` directory contains the current working implementation, CLI, runtime configuration, scripts, and tests.

## What This Repo Assumes

Genesis is split into two layers:

- `Genesis` is the coordinator written in Python.
- `Oh My OpenAgent (OMO)` is the executor backend that Genesis drives through `opencode run`.

Genesis does **not** call a Python OMO library and does **not** treat OMO as a REST server. The active runtime path shells out to OpenCode/OMO.

## Repository Layout

- `genesis/` — application package
- `.opencode/` — OpenCode and OMO project configuration
- `configs/` — Genesis runtime routing config
- `scripts/` — operational scripts such as manifold build
- `tests/` — unit and integration coverage

## Prerequisites

You need:

- Python 3.11 recommended
- `pip`
- `opencode` installed and on `PATH`
- OMO installed as an OpenCode plugin
- at least one working cloud provider
  - preferred: Ollama Cloud
  - optional fallback: Groq

Optional but helpful:

- `pdflatex` for real PDF compilation
- a local Ollama daemon if you want local fallback models

## 1. Create a Python Environment

From the repository root:

```bash
cd /Users/nikhilmaturi/Files/Projects/genesis/v1

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev,data,verification,paper]"
```

Quick verification:

```bash
python -m genesis.cli.main --help
python -m pytest -q tests/unit tests/integration
```

## 2. Install OpenCode

If `opencode` is not already installed:

```bash
npm install -g opencode-ai
```

Verify:

```bash
which opencode
opencode --version
```

## 3. Install Oh My OpenAgent

OMO is installed into OpenCode, not into Python:

```bash
npx --yes oh-my-openagent install --no-tui \
  --claude=no \
  --openai=no \
  --gemini=no \
  --copilot=no \
  --opencode-zen=no \
  --zai-coding-plan=no \
  --kimi-for-coding=no \
  --opencode-go=no \
  --skip-auth
```

What this should create:

- `~/.config/opencode/opencode.json`
- `~/.config/opencode/oh-my-openagent.json`

Project-local config that Genesis uses is already checked in:

- [`.opencode/opencode.json`](/Users/nikhilmaturi/Files/Projects/genesis/v1/.opencode/opencode.json)
- [`.opencode/oh-my-openagent.jsonc`](/Users/nikhilmaturi/Files/Projects/genesis/v1/.opencode/oh-my-openagent.jsonc)
- [`.opencode/oh-my-opencode.jsonc`](/Users/nikhilmaturi/Files/Projects/genesis/v1/.opencode/oh-my-opencode.jsonc)
- [`configs/runtime_omo.jsonc`](/Users/nikhilmaturi/Files/Projects/genesis/v1/configs/runtime_omo.jsonc)

## 4. Provider Setup

### Ollama Cloud

Genesis prefers Ollama Cloud as the primary execution lane.

Export your key:

```bash
export OLLAMA_API_KEY='...'
```

Sign in once with the Ollama CLI:

```bash
ollama signin
ollama pull glm-5:cloud
```

Smoke test:

```bash
env OLLAMA_API_KEY="$OLLAMA_API_KEY" \
opencode run --model ollama-cloud/glm-5:cloud --format json 'Reply with OK only'
```

Expected result:

- OpenCode returns an event stream ending with `OK`

### Groq

Genesis supports Groq as a fallback route.

Export one or more keys:

```bash
export GROQ_API_KEY='...'
export GROQ_API_KEY_2='...'
export GROQ_API_KEY_3='...'
export GROQ_API_KEY_4='...'
export GROQ_API_KEY_5='...'
export GROQ_API_KEY_6='...'
export GROQ_API_KEY_7='...'
```

Direct API smoke test:

```bash
python3 - <<'PY'
import json, requests, os

key = os.environ["GROQ_API_KEY"]
r = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json={
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Reply with OK"}],
        "max_tokens": 8,
    },
    timeout=30,
)
print(r.status_code)
print(r.text[:300])
PY
```

Notes:

- valid Groq keys do **not** guarantee the full OMO/OpenCode executor path will fit within your TPM limits
- large OpenCode/OMO prompts may still hit Groq quota/context constraints even when the API key is valid

### Local Ollama Fallback

Genesis keeps a local Ollama fallback configured for degraded mode. If you want that path available:

```bash
ollama serve
ollama pull gemma4:e4b
ollama pull llama3.2:3b
```

## 5. Sanity Check Genesis Runtime

Run the built-in doctor:

```bash
python3 -m genesis.cli.main doctor --runtime-config configs/runtime_omo.jsonc
```

What you want to see:

- `opencode_binary: passed`
- runtime config files present
- `ollama_cloud_auth: passed`
- `groq_auth: passed` if you exported Groq keys

If `doctor` fails:

- check the env vars are actually present in the current shell
- check `opencode` is on `PATH`
- check `ollama pull glm-5:cloud` succeeded

## 6. Build a Manifold

For general experimentation:

```bash
python3 -m genesis.cli.main build-manifold --domain general
```

For astrophysics:

```bash
python3 -m genesis.cli.main build-manifold --domain astrophysics
```

The build writes into `manifold_index/` by default.

## 7. Create a Spec

Example minimal smoke spec:

```bash
cat >/tmp/genesis_smoke_spec.json <<'EOF'
{
  "research_question": "Create one short markdown artifact proving Genesis can control Oh My OpenAgent.",
  "domain": "general",
  "success_criteria": [
    "Produce at least one substantive artifact relevant to the task."
  ],
  "oracle_hints": [],
  "compute_budget": "cloud_small",
  "time_budget_hours": 1,
  "domain_knowledge_model": "none",
  "output_dir": "/tmp/genesis-smoke-projects"
}
EOF
```

## 8. Run an End-to-End Smoke Test

```bash
python3 -m genesis.cli.main run \
  --project-id smoke01 \
  --spec /tmp/genesis_smoke_spec.json \
  --runtime-config configs/runtime_omo.jsonc \
  --max-runs 3
```

Inspect results:

```bash
python3 -m genesis.cli.main status --project-id smoke01 --root /tmp/genesis-smoke-projects
find /tmp/genesis-smoke-projects/smoke01 -maxdepth 4 -type f | sort
cat /tmp/genesis-smoke-projects/smoke01/project_state.json
```

Minimum healthy signs:

- `decomposition.json` exists
- `runs/1/instruction.md` exists
- `runs/1/result.json` exists
- at least one artifact exists under `outputs/code/`
- `project_state.json` has task state and current stage information

## 9. Useful Commands

Initialize from a spec:

```bash
python3 -m genesis.cli.main init --project-id demo --spec /path/to/spec.json
```

Run:

```bash
python3 -m genesis.cli.main run --project-id demo --spec /path/to/spec.json
```

Status:

```bash
python3 -m genesis.cli.main status --project-id demo --root projects
```

Results:

```bash
python3 -m genesis.cli.main results --project-id demo --root projects
```

Manual intervention:

```bash
python3 -m genesis.cli.main intervene --project-id demo --type REDIRECT
python3 -m genesis.cli.main intervene --project-id demo --type APPROVE
python3 -m genesis.cli.main intervene --project-id demo --type REJECT
python3 -m genesis.cli.main intervene --project-id demo --type STOP
```

Initialize taste storage:

```bash
python3 -m genesis.cli.main init-taste
```

## 10. Running the Test Suite

Fast full regression pass:

```bash
python3 -m pytest -q tests/unit tests/integration
```

Targeted tests:

```bash
python3 -m pytest -q tests/unit/test_provider_runtime.py
python3 -m pytest -q tests/unit/test_harness.py
python3 -m pytest -q tests/unit/test_optimizer_and_taste.py
python3 -m pytest -q tests/unit/test_adversarial.py
python3 -m pytest -q tests/integration/test_full_run.py
```

## 11. Common Failure Modes

### `opencode: command not found`

Install OpenCode:

```bash
npm install -g opencode-ai
```

### `Unauthorized: unauthorized` from `ollama-cloud`

Usually means one of:

- `OLLAMA_API_KEY` is not exported in the current shell
- `ollama signin` was not completed
- `ollama pull glm-5:cloud` has not been run yet

Check:

```bash
python3 - <<'PY'
import os
value = os.getenv("OLLAMA_API_KEY", "")
print("present:", bool(value))
print("length:", len(value))
PY
```

### Groq fallback fails even though the key is valid

This often means the request is too large for the current Groq tier, not that the key is invalid. Validate the key directly, but prefer Ollama Cloud for the full executor path.

### `doctor` crashes in manifold/Chroma code

This usually means the local environment has a stale or partially written manifold store. Rebuild it:

```bash
rm -rf manifold_index
python3 -m genesis.cli.main build-manifold --domain general
```

### Project refuses to restart because of `HALT.json`

That is intentional. Inspect the halt reason first:

```bash
cat projects/<project_id>/HALT.json
cat projects/<project_id>/project_state.json
```

## 12. Notes

- Genesis is designed to be truthful. If a subsystem is not ready, it should gate or disable honestly rather than fake output.
- The current primary executor lane is Ollama Cloud through OpenCode/OMO.
- Groq is best treated as a fallback provider, not the default full-context executor lane.
- Some deeper live/manual validation is still recommended for:
  - multi-task DAG runs
  - real optimizer tasks with runnable commands
  - cross-project causal carryover
  - scientific paper quality on realistic domain-heavy prompts
