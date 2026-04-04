import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.status_manager import write_dashboard, write_project_status  # pyre-ignore[21]


def main(project_id: str):
    project_path = Path("projects") / project_id
    status_path = write_project_status(project_path)
    dashboard_path = write_dashboard(project_path)
    status = json.loads(status_path.read_text())
    print(f"Project: {status['project'].get('title')}")
    print(f"Current stage: {status.get('current_stage')}")
    print(f"Task counts: {status.get('task_counts')}")
    print(f"Runtime: {status.get('runtime', {}).get('container_state')}")
    print(f"Next required human action: {status.get('next_required_human_action')}")
    print(f"Status written to: {status_path}")
    print(f"Dashboard written to: {dashboard_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: project_status.py <project_id>")
    main(sys.argv[1])
