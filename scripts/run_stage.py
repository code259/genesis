import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.executor import execute_task  # pyre-ignore[21]
from core.supervisor import evaluate_output, Decision  # pyre-ignore[21]
from core.verifier import verify  # pyre-ignore[21]
from core.state_manager import update_global_state, check_stage_gate  # pyre-ignore[21]

def run_stage(project_id: str, stage: int, task_specs: list):
    project_path = Path("projects") / project_id
    stage_dir = project_path / "stages" / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    error_counts = {}
    
    for spec in task_specs:
        print(f"\n{'='*50}")
        print(f"Executing {spec['id']}: {spec['description'][:60]}...")
        
        max_attempts: int = 3
        
        for attempt in range(max_attempts):
            output = execute_task(spec, project_path)
            decision = evaluate_output(spec['id'], output, spec)
            
            print(f"Supervisor decision: {decision.decision.value}")
            for reason in decision.reasons:
                print(f"  - {reason}")
            
            if decision.decision in [Decision.TRIGGER_VERIFY, Decision.TRIGGER_SOFT_VERIFY]:
                print(f"Running verifier...")
                is_foundational = spec.get('complexity') == 'HIGH' and len(spec.get('dependencies', [])) == 0
                verification = verify(spec, output, is_foundational)
                
                verify_file = stage_dir / f"{spec['id']}_verify.md"
                verify_file.write_text(verification)
                
                if "RECOMMENDATION: ACCEPT" in verification:
                    print(f"✓ {spec['id']} verified and accepted")
                    update_global_state(project_path, spec['id'], 
                                       output[:300], "ESTABLISHED")
                    break
                else:
                    print(f"✗ Verifier rejected. Attempt {attempt+1}/{max_attempts}")
                    error_counts[spec['id']] = attempt + 1
                    
                    if attempt + 1 >= max_attempts:
                        print(f"⚠️  ESCALATING {spec['id']} to human after {attempt+1} attempts")
                        # Write escalation report
                        escalation = f"ESCALATION after {attempt+1} attempts.\nVerifier feedback:\n{verification}"
                        (stage_dir / f"{spec['id']}_escalation.md").write_text(escalation)
                        input("Human: review escalation file and press Enter to continue...")
                        break
            
            elif decision.decision == Decision.ACCEPT:
                print(f"✓ {spec['id']} accepted")
                update_global_state(project_path, spec['id'], output[:300], "ESTABLISHED")
                break
            
            elif decision.decision == Decision.HOLD_STAGE_GATE:
                print(f"⛔ {spec['id']} holding stage gate: {decision.reasons}")
                break
        
    # Stage gate check
    print(f"\n{'='*50}")
    print(f"Stage {stage} gate check...")
    gate = check_stage_gate(project_path, stage, task_specs)
    
    if gate['can_close']:
        print(f"✓ Stage {stage} closed successfully")
    else:
        print(f"⛔ Stage {stage} blocked:")
        for item in gate['blocking_items']:
            print(f"  - {item}")
