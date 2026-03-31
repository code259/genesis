from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.executor import artifact_paths_for_task, execute_task  # pyre-ignore[21]
from core.manuscript_builder import build_paper_package, build_stage_summary  # pyre-ignore[21]
from core.oracle_manager import oracle_result_to_dict, run_oracle_checks  # pyre-ignore[21]
from core.supervisor import evaluate_output, Decision  # pyre-ignore[21]
from core.verifier import verify  # pyre-ignore[21]
from core.state_manager import update_global_state, check_stage_gate  # pyre-ignore[21]
from core.task_parser import tasks_for_stage  # pyre-ignore[21]

def run_stage(project_id: str, stage: int, task_specs: list | None = None):
    project_path = Path("projects") / project_id
    stage_dir = project_path / "stages" / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if task_specs is None:
        task_specs = _load_stage_tasks(project_path, stage)
    
    error_counts = {}
    
    for spec in task_specs:
        print(f"\n{'='*50}")
        print(f"Executing {spec['id']}: {spec['description'][:60]}...")
        
        max_attempts: int = 3
        revision_context = ""
        
        for attempt in range(max_attempts):
            output = execute_task(spec, project_path, revision_context=revision_context)
            decision = evaluate_output(spec['id'], output, spec)
            output_path = stage_dir / f"{spec['id']}.md"
            oracle_result = oracle_result_to_dict(
                run_oracle_checks(spec, output_path, artifact_paths_for_task(project_path, spec["id"]), project_path)
            )
            is_foundational = bool(spec.get("foundational")) or (
                spec.get('complexity') == 'HIGH' and len(spec.get('dependencies', [])) == 0
            )
            verification = verify(spec, output, oracle_result=oracle_result, is_foundational=is_foundational)
            _write_verification_artifacts(stage_dir, spec["id"], verification)
            
            print(f"Supervisor decision: {decision.decision.value}")
            for reason in decision.reasons:
                print(f"  - {reason}")
            print(f"Verifier status: {verification['status']}")
            
            if decision.decision == Decision.HOLD_STAGE_GATE:
                print(f"⛔ {spec['id']} holding stage gate: {decision.reasons}")
                revision_context = verification["raw_text"]
                if attempt + 1 >= max_attempts:
                    _write_escalation(stage_dir, spec["id"], verification["raw_text"], attempt + 1)
                continue

            if verification["status"] == "ACCEPT" and oracle_result["oracle_pass"] is not False:
                print(f"✓ {spec['id']} accepted")
                update_global_state(project_path, spec['id'], output[:300], "ESTABLISHED")
                break
            else:
                print(f"✗ {spec['id']} requires revision. Attempt {attempt+1}/{max_attempts}")
                error_counts[spec['id']] = attempt + 1
                revision_context = verification["raw_text"]
                if attempt + 1 >= max_attempts:
                    _write_escalation(stage_dir, spec["id"], verification["raw_text"], attempt + 1)
                    input("Human: review escalation file and press Enter to continue...")
                    break
        
    # Stage gate check
    print(f"\n{'='*50}")
    print(f"Stage {stage} gate check...")
    gate = check_stage_gate(project_path, stage, task_specs)
    
    if gate['can_close']:
        summary_path = build_stage_summary(project_path, stage)
        paper_path = build_paper_package(project_path)
        print(f"✓ Stage {stage} closed successfully")
        print(f"Stage summary written to: {summary_path}")
        print(f"Paper package updated at: {paper_path}")
        input("Human checkpoint: review the stage summary and paper package, then press Enter to continue...")
    else:
        print(f"⛔ Stage {stage} blocked:")
        for item in gate['blocking_items']:
            print(f"  - {item}")


def _load_stage_tasks(project_path: Path, stage: int) -> list:
    tasks_path = project_path / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks.json in {project_path}")
    tasks = json.loads(tasks_path.read_text())
    return tasks_for_stage(tasks, stage)


def _write_verification_artifacts(stage_dir: Path, task_id: str, verification: dict):
    (stage_dir / f"{task_id}_verify.md").write_text(verification["raw_text"])
    structured = {k: v for k, v in verification.items() if k != "raw_text"}
    (stage_dir / f"{task_id}_verify.json").write_text(json.dumps(structured, indent=2))


def _write_escalation(stage_dir: Path, task_id: str, verification_text: str, attempts: int):
    escalation = f"ESCALATION after {attempts} attempts.\nVerifier feedback:\n{verification_text}"
    (stage_dir / f"{task_id}_escalation.md").write_text(escalation)
