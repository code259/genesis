# pyre-ignore-all-errors
import json
from pathlib import Path
from oracle.astro import physical_checks, statistical_checks  # pyre-ignore[21]
from core.task_parser import extract_stage

def run_all_checks(checks: list) -> dict:
    """
    Run a list of pre-constructed check calls and aggregate results.
    checks: list of dicts returned by individual check functions
    """
    passed = [c for c in checks if c.get("pass") is True]
    failed = [c for c in checks if c.get("pass") is False]
    warned = [c for c in checks if c.get("warning") is True]
    skipped = [c for c in checks if c.get("pass") is None]

    all_pass = len(failed) == 0

    return {
        "oracle_pass": all_pass,
        "summary": {
            "total": len(checks),
            "passed": len(passed),
            "failed": len(failed),
            "warnings": len(warned),
            "skipped": len(skipped),
        },
        "failures": [{"check": c["check"], "interpretation": c["interpretation"]} for c in failed],
        "warnings": [{"check": c["check"], "interpretation": c["interpretation"]} for c in warned],
    }

def write_oracle_report(project_path: Path, task_id: str, results: dict):
    stage = extract_stage(task_id)
    oracle_dir = project_path / "stages" / f"stage_{stage}" / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    report_path = oracle_dir / f"{task_id}_oracle.json"
    report_path.write_text(json.dumps(results, indent=2))
    return report_path
