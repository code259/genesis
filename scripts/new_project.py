from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.decomposer import decompose, adversarial_review  # pyre-ignore[21]
from core.task_parser import build_dependency_graph, parse_task_tree, validate_task_tree  # pyre-ignore[21]
import uuid

def new_project(
    research_goal: str,
    domain_context: str,
    domain: str = "general",
    oracle_module: str | None = None,
):
    full_uuid = str(uuid.uuid4())
    project_id = "".join(full_uuid[i] for i in range(8))
    project_path = Path("projects") / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize files
    (project_path / "conventions.md").write_text("# Conventions\n\n*To be populated as conventions are established.*\n")
    (project_path / "global_state.md").write_text("# Global State\n\n")
    (project_path / "constraints.md").write_text(Path("prompts/constraints.md").read_text())
    
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
    (project_path / "master_plan.md").write_text(task_tree)
    (project_path / "decomposition_review.md").write_text(review)
    (project_path / "tasks.json").write_text(json.dumps(tasks, indent=2))
    (project_path / "dependency_graph.json").write_text(json.dumps(build_dependency_graph(tasks), indent=2))
    (project_path / "project_config.json").write_text(
        json.dumps(
            {
                "title": research_goal,
                "domain": domain,
                "oracle_module": oracle_module,
                "paper": {"manual_refs_only": True},
            },
            indent=2,
        )
    )
    
    print(f"\nProject initialized: {project_id}")
    print(f"Review decomposition at: projects/{project_id}/master_plan.md")
    print(f"Adversarial review at: projects/{project_id}/decomposition_review.md")
    print("\n⚠️  Human checkpoint: Review both files before proceeding to execution.")
    
    return project_id

if __name__ == "__main__":
    goal = input("Research goal: ")
    context = input("Domain context: ")
    new_project(goal, context)
