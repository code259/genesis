from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import json
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # pyre-ignore[21]
from core.executor import artifact_paths_for_task, execute_task  # pyre-ignore[21]
from core.manuscript_builder import build_paper_package, build_stage_summary  # pyre-ignore[21]
from core.oracle_manager import oracle_result_to_dict, run_oracle_checks  # pyre-ignore[21]
from core.research_worker import load_worker_state, mark_task_verified  # pyre-ignore[21]
from core.status_manager import build_project_status, update_run_state, write_dashboard, write_project_status  # pyre-ignore[21]
from core.supervisor import evaluate_output, Decision  # pyre-ignore[21]
from core.verifier import verify  # pyre-ignore[21]
from core.state_manager import update_global_state, check_stage_gate  # pyre-ignore[21]
from core.task_parser import build_dependency_graph, tasks_for_stage  # pyre-ignore[21]


def _skip_human_checkpoints() -> bool:
    return os.getenv("GENESIS_SKIP_HUMAN_CHECKPOINTS", "").lower() in {"1", "true", "yes", "on"}


def run_stage(project_id: str, stage: int, task_specs: list | None = None):
    project_path = Path("projects") / project_id
    stage_dir = project_path / "stages" / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GENESIS_RUNTIME_DIR"] = str(project_path / "runtime")
    if task_specs is None:
        task_specs = _load_stage_tasks(project_path, stage)
    update_run_state(
        project_path,
        current_stage=stage,
        phase="running_stage",
        awaiting_human_review=False,
        next_human_action=None,
        last_successful_action=f"Started stage {stage}",
    )
    write_project_status(project_path)
    write_dashboard(project_path)
    
    remaining = {spec["id"]: spec for spec in task_specs}
    dependency_graph = build_dependency_graph(task_specs)

    while remaining:
        ready = _ready_tasks(project_path, list(remaining.values()))
        if not ready:
            break
        for group in _parallel_groups(ready):
            outputs: dict[str, str] = {}
            if len(group) == 1:
                spec = group[0]
                outputs[spec["id"]] = execute_task(spec, project_path, revision_context="")
            else:
                with ThreadPoolExecutor(max_workers=len(group)) as executor:
                    futures = {executor.submit(execute_task, spec, project_path, ""): spec for spec in group}
                    for future, spec in futures.items():
                        outputs[spec["id"]] = future.result()

            for spec in group:
                print(f"\n{'='*50}")
                print(f"Executing {spec['id']}: {spec['description'][:60]}...")
                _process_task_result(project_path, stage_dir, spec, outputs[spec["id"]])
                remaining.pop(spec["id"], None)
                write_project_status(project_path)
                write_dashboard(project_path)

    # Stage gate check
    print(f"\n{'='*50}")
    print(f"Stage {stage} gate check...")
    gate = check_stage_gate(project_path, stage, task_specs)
    
    if gate['can_close']:
        summary_path = build_stage_summary(project_path, stage)
        paper_path = build_paper_package(project_path)
        update_run_state(
            project_path,
            phase="stage_closed" if _skip_human_checkpoints() else "awaiting_human_review",
            awaiting_human_review=not _skip_human_checkpoints(),
            next_human_action=(
                None if _skip_human_checkpoints() else f"Review stage {stage} summary and paper package"
            ),
            last_successful_action=f"Closed stage {stage}",
        )
        write_project_status(project_path)
        write_dashboard(project_path)
        print(f"✓ Stage {stage} closed successfully")
        print(f"Stage summary written to: {summary_path}")
        print(f"Paper package updated at: {paper_path}")
        if _skip_human_checkpoints():
            print("Test mode: stage-close checkpoint bypassed.")
        else:
            input("Human checkpoint: review the stage summary and paper package, then press Enter to continue...")
    else:
        update_run_state(
            project_path,
            phase="blocked",
            awaiting_human_review=False,
            next_human_action=f"Resolve blocking items for stage {stage}",
            last_successful_action=f"Stage {stage} blocked",
        )
        write_project_status(project_path)
        write_dashboard(project_path)
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


def _process_task_result(project_path: Path, stage_dir: Path, spec: dict, output: str):
    max_attempts = 3
    attempt = 1
    revision_context = ""
    current_output = output
    while attempt <= max_attempts:
        worker_state = load_worker_state(project_path, spec["id"])
        if worker_state.get("task_status") == "PARTIAL" and not worker_state.get("ready_for_verification"):
            print(f"~ {spec['id']} deferred as partial: {worker_state.get('last_reason')}")
            update_run_state(project_path, last_successful_action=f"Deferred {spec['id']}")
            return

        decision = evaluate_output(spec['id'], current_output, spec)
        output_path = stage_dir / f"{spec['id']}.md"
        oracle_result = oracle_result_to_dict(
            run_oracle_checks(spec, output_path, artifact_paths_for_task(project_path, spec["id"]), project_path)
        )
        is_foundational = bool(spec.get("foundational")) or (
            spec.get('complexity') == 'HIGH' and len(spec.get('dependencies', [])) == 0
        )
        verification = verify(spec, current_output, oracle_result=oracle_result, is_foundational=is_foundational)
        _write_verification_artifacts(stage_dir, spec["id"], verification)

        print(f"Supervisor decision: {decision.decision.value}")
        for reason in decision.reasons:
            print(f"  - {reason}")
        print(f"Verifier status: {verification['status']}")

        if decision.decision == Decision.HOLD_STAGE_GATE:
            print(f"⛔ {spec['id']} holding stage gate: {decision.reasons}")
            revision_context = verification["raw_text"]
            if attempt >= max_attempts:
                _write_escalation(stage_dir, spec["id"], verification["raw_text"], attempt)
            else:
                attempt += 1
                current_output = execute_task(spec, project_path, revision_context=revision_context)
            continue

        if verification["status"] == "ACCEPT" and oracle_result["oracle_pass"] is not False:
            print(f"✓ {spec['id']} accepted")
            mark_task_verified(project_path, spec["id"], "ACCEPT")
            update_global_state(project_path, spec['id'], current_output[:300], "ESTABLISHED")
            update_run_state(project_path, last_successful_action=f"Accepted {spec['id']}")
            return

        print(f"✗ {spec['id']} requires revision. Attempt {attempt}/{max_attempts}")
        revision_context = verification["raw_text"]
        if attempt >= max_attempts:
            mark_task_verified(project_path, spec["id"], "ESCALATE")
            _write_escalation(stage_dir, spec["id"], verification["raw_text"], attempt)
            update_run_state(
                project_path,
                phase="blocked",
                next_human_action=f"Review escalation for {spec['id']}",
                last_successful_action=f"Escalated {spec['id']}",
            )
            if _skip_human_checkpoints():
                print("Test mode: escalation checkpoint bypassed.")
            else:
                input("Human: review escalation file and press Enter to continue...")
            return
        attempt += 1
        current_output = execute_task(spec, project_path, revision_context=revision_context)


def _ready_tasks(project_path: Path, task_specs: list[dict]) -> list[dict]:
    status = build_project_status(project_path)
    task_status_by_id = {task["id"]: task for task in status["tasks"]}
    ready = []
    for spec in task_specs:
        if spec["id"] not in status["ready_queue"]:
            continue
        if all(_dependency_satisfied(task_status_by_id.get(dep, {})) for dep in spec.get("dependencies", [])):
            ready.append(spec)
    return ready


def _dependency_satisfied(status_record: dict) -> bool:
    status = status_record.get("status")
    if status == "VERIFIED":
        return True
    if status == "PARTIAL":
        return not any(issue.get("blocks_dependents") for issue in status_record.get("deferred_issues", []))
    return False


def _parallel_groups(ready: list[dict]) -> list[list[dict]]:
    max_parallel = max(1, min(config.MAX_PARALLEL_TASKS, config.MAX_PARALLEL_GROQ_CALLS, len(ready)))
    groups = []
    for index in range(0, len(ready), max_parallel):
        groups.append(ready[index : index + max_parallel])
    return groups
