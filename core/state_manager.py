from pathlib import Path
from datetime import datetime

def update_global_state(project_path: Path, task_id: str, 
                         result_summary: str, status: str):
    """
    Append to global_state.md after a task completes.
    status: ESTABLISHED | PENDING | INVALIDATED
    """
    state_file = project_path / "global_state.md"
    entry = f"""
## {task_id} — {status} — {datetime.now().strftime('%Y-%m-%d %H:%M')}

{result_summary}

---
"""
    with open(state_file, "a") as f:
        f.write(entry)

def invalidate_dependents(project_path: Path, task_id: str, 
                           task_tree: dict, reason: str):
    """
    When a foundational result is corrected, flag all dependent tasks.
    task_tree: dict mapping task_id -> list of dependent task_ids
    """
    dependents = task_tree.get(task_id, [])
    for dep_id in dependents:
        update_global_state(
            project_path, dep_id, 
            f"FLAGGED: dependency {task_id} was corrected. Reason: {reason}. Re-verification required.",
            "INVALIDATED"
        )
    return dependents

def check_stage_gate(project_path: Path, stage: int, task_specs: list) -> dict:
    """
    Evaluate whether a stage can close.
    Returns dict with: can_close (bool), blocking_items (list)
    """
    stage_dir = project_path / "stages" / f"stage_{stage}"
    blocking = []
    
    for spec in task_specs:
        task_file = stage_dir / f"{spec['id']}.md"
        
        # Check 1: output file exists
        if not task_file.exists():
            blocking.append(f"{spec['id']}: output file missing")
            continue
            
        output = task_file.read_text()
        
        # Check 2: no INCOMPLETE markers
        if "INCOMPLETE" in output:
            blocking.append(f"{spec['id']}: contains INCOMPLETE markers")
        
        # Check 3: verifier sign-off exists
        verify_file = stage_dir / f"{spec['id']}_verify.md"
        if not verify_file.exists():
            blocking.append(f"{spec['id']}: no verifier sign-off")
        elif "RECOMMENDATION: REVISE" in verify_file.read_text():
            blocking.append(f"{spec['id']}: verifier recommends revision")
        elif "RECOMMENDATION: ESCALATE" in verify_file.read_text():
            blocking.append(f"{spec['id']}: verifier escalated to human")
    
    return {
        "can_close": len(blocking) == 0,
        "blocking_items": blocking
    }
