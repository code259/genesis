from __future__ import annotations

import sys
import json
import os
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.decomposer import decompose, adversarial_review  # pyre-ignore[21]
from core.project_spec import legacy_project_spec_markdown, normalize_project_spec, parse_project_spec, project_spec_to_context, validate_project_spec  # pyre-ignore[21]
from core.status_manager import update_run_state, write_dashboard, write_project_status  # pyre-ignore[21]
from core.task_parser import build_dependency_graph, parse_task_tree, validate_task_tree  # pyre-ignore[21]
import uuid


def _write_temp_spec(project_path: Path, content: str) -> Path:
    temp_path = project_path / "_legacy_project_spec.md"
    temp_path.write_text(content)
    return temp_path


def _skip_human_checkpoints() -> bool:
    return os.getenv("GENESIS_SKIP_HUMAN_CHECKPOINTS", "").lower() in {"1", "true", "yes", "on"}

def new_project(
    research_goal: str | None = None,
    domain_context: str | None = None,
    domain: str = "general",
    oracle_module: str | None = None,
    spec_path: str | None = None,
):
    full_uuid = str(uuid.uuid4())
    project_id = "".join(full_uuid[i] for i in range(8))
    project_path = Path("projects") / project_id

    normalized_spec = None
    parsed_spec = None
    project_spec_markdown = None

    if spec_path is not None:
        parsed_spec = parse_project_spec(spec_path)
        spec_errors = validate_project_spec(parsed_spec)
        if spec_errors:
            raise ValueError("Project spec is invalid:\n" + "\n".join(spec_errors))
        normalized_spec = normalize_project_spec(parsed_spec)
        project_spec_markdown = Path(spec_path).read_text()
        research_goal = normalized_spec["research_goal"]
        domain_context = project_spec_to_context(parsed_spec)
    else:
        if research_goal is None or domain_context is None:
            raise ValueError("Either spec_path or both research_goal and domain_context are required")
        project_spec_markdown = legacy_project_spec_markdown(research_goal, domain_context, domain=domain)

    project_created = False
    try:
        project_path.mkdir(parents=True, exist_ok=False)
        project_created = True
        os.environ["GENESIS_RUNTIME_DIR"] = str(project_path / "runtime")

        # Initialize files
        (project_path / "conventions.md").write_text("# Conventions\n\n*To be populated as conventions are established.*\n")
        (project_path / "global_state.md").write_text("# Global State\n\n")
        (project_path / "constraints.md").write_text(Path("prompts/constraints.md").read_text())

        if spec_path is None:
            parsed_spec = parse_project_spec(_write_temp_spec(project_path, project_spec_markdown))
            normalized_spec = normalize_project_spec(parsed_spec)

        # Decompose
        print("Decomposing research goal...")
        task_tree = decompose(research_goal, domain_context)

        print("Running adversarial review of decomposition...")
        review = adversarial_review(research_goal, task_tree)

        tasks = parse_task_tree(task_tree)
        errors = validate_task_tree(tasks)
        if errors:
            raise ValueError("Task tree is not parseable:\n" + "\n".join(errors))

        if oracle_module is None and domain.lower() in {"astrophysics", "astro", "physics"}:
            oracle_module = "astro"

        # Save both
        (project_path / "project_spec.md").write_text(project_spec_markdown)
        (project_path / "master_plan.md").write_text(task_tree)
        (project_path / "decomposition_review.md").write_text(review)
        (project_path / "tasks.json").write_text(json.dumps(tasks, indent=2))
        (project_path / "dependency_graph.json").write_text(json.dumps(build_dependency_graph(tasks), indent=2))
        (project_path / "project_config.json").write_text(
            json.dumps(
                {
                    "title": normalized_spec["title"],
                    "domain": domain,
                    "oracle_module": oracle_module,
                    "paper": {"manual_refs_only": True},
                    "project_spec": normalized_spec,
                },
                indent=2,
            )
        )
        update_run_state(
            project_path,
            phase="initialized",
            awaiting_human_review=not _skip_human_checkpoints(),
            next_human_action=(
                None
                if _skip_human_checkpoints()
                else "Review decomposition and decomposition review before stage execution"
            ),
            last_successful_action="Project initialized",
        )
        write_project_status(project_path)
        write_dashboard(project_path)

        print(f"\nProject initialized: {project_id}")
        print(f"Review decomposition at: projects/{project_id}/master_plan.md")
        print(f"Adversarial review at: projects/{project_id}/decomposition_review.md")
        if _skip_human_checkpoints():
            print("\nTest mode: human checkpoint bypassed for initialization.")
        else:
            print("\n⚠️  Human checkpoint: Review both files before proceeding to execution.")

        return project_id
    except Exception:
        if project_created and project_path.exists():
            shutil.rmtree(project_path)
        raise

if __name__ == "__main__":
    goal = input("Research goal: ")
    context = input("Domain context: ")
    new_project(goal, context)
