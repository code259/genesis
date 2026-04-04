#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPEC_PATH="${1:-notes/PROJECT_SPEC.md}"
DOMAIN="${DOMAIN:-astrophysics}"
SKIP_CHECKPOINTS="${GENESIS_SKIP_HUMAN_CHECKPOINTS:-1}"

if [[ ! -f "$SPEC_PATH" ]]; then
  echo "Spec file not found: $SPEC_PATH" >&2
  exit 1
fi

export GENESIS_SKIP_HUMAN_CHECKPOINTS="$SKIP_CHECKPOINTS"

echo "Initializing project from $SPEC_PATH"
PROJECT_ID="$(
  venv/bin/python - <<'PY' "$SPEC_PATH" "$DOMAIN" | tail -n 1
from scripts.new_project import new_project
import sys

project_id = new_project(spec_path=sys.argv[1], domain=sys.argv[2])
print(project_id)
PY
)"

echo "Project ID: $PROJECT_ID"

MAX_STAGE="$(
  venv/bin/python - <<'PY' "$PROJECT_ID"
from pathlib import Path
import json
import sys

project_id = sys.argv[1]
tasks = json.loads((Path("projects") / project_id / "tasks.json").read_text())
print(max(task["stage"] for task in tasks) if tasks else 0)
PY
)"

echo "Detected $MAX_STAGE stage(s)"

for STAGE in $(seq 1 "$MAX_STAGE"); do
  echo
  echo "Running stage $STAGE"
  venv/bin/python - <<'PY' "$PROJECT_ID" "$STAGE"
from scripts.run_stage import run_stage
import sys

run_stage(sys.argv[1], int(sys.argv[2]))
PY

  venv/bin/python scripts/project_status.py "$PROJECT_ID"

  PHASE="$(
    venv/bin/python - <<'PY' "$PROJECT_ID"
from pathlib import Path
import json
import sys

status = json.loads((Path("projects") / sys.argv[1] / "project_status.json").read_text())
print(status.get("run_state", {}).get("phase", "unknown"))
PY
  )"

  if [[ "$PHASE" == "blocked" ]]; then
    echo "Project blocked during stage $STAGE. Inspect projects/$PROJECT_ID/dashboard.md"
    exit 2
  fi
done

echo
echo "Project run complete: $PROJECT_ID"
echo "Dashboard: projects/$PROJECT_ID/dashboard.md"
echo "Paper package: projects/$PROJECT_ID/paper"
