from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .models import ensure_parent


def log_event(
    log_path: Union[str, Path],
    *,
    project_id: str,
    run_n: Optional[int],
    component: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    destination = Path(log_path)
    ensure_parent(destination)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "run_n": run_n,
        "component": component,
        "event_type": event_type,
        "payload": payload,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
