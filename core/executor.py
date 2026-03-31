from __future__ import annotations

# pyre-ignore-all-errors[21]
import json
from pathlib import Path
from core import router  # pyre-ignore[21]
from core.artifact_runner import run_python_task, task_artifact_dir
from core.task_parser import extract_stage
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = (
    Path("prompts/executor_system.md").read_text() + 
    "\n\n" + 
    Path("prompts/constraints.md").read_text()
)
ARTIFACT_PLAN_PROMPT = Path("prompts/artifact_plan_system.md").read_text()

def load_prior_outputs(project_path: Path, dependency_ids: list[str]) -> str:
    """Load relevant prior task outputs from file tree."""
    outputs = []
    for task_id in dependency_ids:
        stage = extract_stage(task_id)
        task_file = project_path / "stages" / f"stage_{stage}" / f"{task_id}.md"
        if task_file.exists():
            outputs.append(f"### Prior output {task_id}\n{task_file.read_text()}")
    return "\n\n".join(outputs)

def execute_task(
    task_spec: dict,
    project_path: Path,
    revision_context: str = "",
) -> str:
    prior = load_prior_outputs(project_path, task_spec.get("dependencies", []))
    conventions = (project_path / "conventions.md").read_text()
    verification = _verification_text(task_spec["verification_criteria"])
    
    user_content = f"""
TASK ID: {task_spec['id']}
DESCRIPTION: {task_spec['description']}
VERIFICATION CRITERIA:
{verification}
COMPLEXITY: {task_spec['complexity']}

CONVENTIONS:
{conventions}

PRIOR OUTPUTS:
{prior if prior else 'No prior outputs (this is a foundational task)'}
"""
    if revision_context:
        user_content += f"\n\nREVISION CONTEXT:\n{revision_context}"

    stage_dir = project_path / "stages" / f"stage_{task_spec['stage']}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_file = stage_dir / f"{task_spec['id']}.md"

    execution_plan = build_execution_plan(task_spec, project_path, user_content)
    if execution_plan is not None:
        result = run_python_task(task_spec, project_path, execution_plan)
        output = render_artifact_task_output(task_spec, project_path, execution_plan, result)
        output_file.write_text(output)
        return output
    
    output = router.call(
        role="executor",
        system=SYSTEM_PROMPT,
        user=user_content
    )
    
    output_file.write_text(output)
    
    return output

def write_stage_summary(project_path: Path, stage: int) -> str:
    """Synthesize all task outputs for a stage into a summary."""
    stage_dir = project_path / "stages" / f"stage_{stage}"
    task_files = sorted(stage_dir.glob("S*.md"))
    combined = "\n\n".join(f.read_text() for f in task_files)

    return router.call(
        role="executor",
        system="Synthesize the following task outputs into a coherent stage summary. Be accurate and complete. Do not introduce information not in the task outputs.",
        user=combined,
        max_tokens=2000
    )


def artifact_paths_for_task(project_path: Path, task_id: str) -> list[str]:
    artifact_dir = task_artifact_dir(project_path, task_id)
    if not artifact_dir.exists():
        return []
    return sorted(
        str(path.relative_to(project_path))
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "run.py"
    )


def build_execution_plan(task_spec: dict, project_path: Path, user_content: str) -> dict | None:
    if not requires_artifacts(task_spec):
        return None

    raw_plan = router.call(
        role="executor",
        system=ARTIFACT_PLAN_PROMPT,
        user=user_content,
        max_tokens=2000,
    )

    plan = parse_execution_plan(raw_plan)
    artifact_dir = task_artifact_dir(project_path, task_spec["id"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "execution_plan.json").write_text(json.dumps(plan, indent=2))
    return plan


def parse_execution_plan(raw_plan: str) -> dict:
    stripped = raw_plan.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    plan = json.loads(stripped)
    if "python_code" not in plan:
        raise ValueError("Execution plan missing python_code")
    plan.setdefault("expected_artifacts", [])
    plan.setdefault("notes", [])
    return plan


def render_artifact_task_output(task_spec: dict, project_path: Path, execution_plan: dict, result) -> str:
    verification = _verification_text(task_spec["verification_criteria"])
    artifacts = "\n".join(f"- `{path}`" for path in result.artifact_paths) or "- None"
    checks_not_performed = []
    if result.missing_artifacts:
        checks_not_performed.append(
            "Missing required artifacts: " + ", ".join(result.missing_artifacts)
        )
    if not result.success and result.stderr:
        checks_not_performed.append("Execution failed before all planned validations completed")
    if not checks_not_performed:
        checks_not_performed.append("None")

    status_line = "INCOMPLETE — artifact execution failed" if not result.success else "Execution completed."
    return f"""## Task ID: {task_spec['id']}
### Description
{task_spec['description']}

### Verification Criteria
{verification}

### Artifact Execution
{status_line}

Script: `{result.script_path}`
Results summary: `{result.results_path}`

### Planned Notes
{_bullet_list(execution_plan.get('notes', []))}

### Produced Artifacts
{artifacts}

### Stdout
```text
{result.stdout.strip()}
```

### Stderr
```text
{result.stderr.strip()}
```

CHECKS PERFORMED:
- Executed the generated Python script.
- Collected artifact paths from the task artifact directory.
- Verified required artifacts exist.

CHECKS NOT PERFORMED:
{_bullet_list(checks_not_performed)}
"""


def requires_artifacts(task_spec: dict) -> bool:
    text = " ".join(task_spec.get("verification_criteria", []) + [task_spec.get("description", "")]).lower()
    keywords = ["dataset", "data points", "csv", "figure", "plot", "png", "regression", "analysis", "artifact"]
    return any(keyword in text for keyword in keywords)


def _verification_text(criteria: list[str]) -> str:
    return "\n".join(f"- {item}" for item in criteria)


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None"
