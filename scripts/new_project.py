import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.decomposer import decompose, adversarial_review  # pyre-ignore[21]
import uuid

def new_project(research_goal: str, domain_context: str):
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
    
    # Save both
    (project_path / "master_plan.md").write_text(task_tree)
    (project_path / "decomposition_review.md").write_text(review)
    
    print(f"\nProject initialized: {project_id}")
    print(f"Review decomposition at: projects/{project_id}/master_plan.md")
    print(f"Adversarial review at: projects/{project_id}/decomposition_review.md")
    print("\n⚠️  Human checkpoint: Review both files before proceeding to execution.")
    
    return project_id

if __name__ == "__main__":
    goal = input("Research goal: ")
    context = input("Domain context: ")
    new_project(goal, context)
