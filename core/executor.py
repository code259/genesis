# pyre-ignore-all-errors[21]
from pathlib import Path
from core import router  # pyre-ignore[21]
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = (
    Path("prompts/executor_system.md").read_text() + 
    "\n\n" + 
    Path("prompts/constraints.md").read_text()
)

def load_prior_outputs(project_path: Path, dependency_ids: list[str]) -> str:
    """Load relevant prior task outputs from file tree."""
    outputs = []
    for task_id in dependency_ids:
        stage = task_id[1]  # S1T3 -> stage 1
        task_file = project_path / "stages" / f"stage_{stage}" / f"{task_id}.md"
        if task_file.exists():
            outputs.append(f"### Prior output {task_id}\n{task_file.read_text()}")
    return "\n\n".join(outputs)

def execute_task(
    task_spec: dict,
    project_path: Path,
) -> str:
    prior = load_prior_outputs(project_path, task_spec.get("dependencies", []))
    conventions = (project_path / "conventions.md").read_text()
    
    user_content = f"""
TASK ID: {task_spec['id']}
DESCRIPTION: {task_spec['description']}
VERIFICATION CRITERIA: {task_spec['verification_criteria']}
COMPLEXITY: {task_spec['complexity']}

CONVENTIONS:
{conventions}

PRIOR OUTPUTS:
{prior if prior else 'No prior outputs (this is a foundational task)'}
"""
    
    output = router.call(
        role="executor",
        system=SYSTEM_PROMPT,
        user=user_content
    )
    
    # Write to file tree
    stage_dir = project_path / "stages" / f"stage_{task_spec['stage']}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_file = stage_dir / f"{task_spec['id']}.md"
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
