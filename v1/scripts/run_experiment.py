from __future__ import annotations

import argparse
import json
import math
import tracemalloc
from pathlib import Path


def _run_warmup_schedule(steps: int, lr: float, warmup_ratio: float) -> list[float]:
    parameter = 5.0
    trajectory = []
    warmup_steps = max(1, int(steps * warmup_ratio))
    for step in range(steps):
        effective_lr = lr * min(1.0, float(step + 1) / warmup_steps)
        gradient = 2.0 * parameter
        parameter -= effective_lr * gradient
        loss = parameter * parameter
        trajectory.append(round(1.0 / (1.0 + loss), 6))
    return trajectory


def _run_linear_probe(steps: int, scale: float) -> list[float]:
    value = 0.0
    trajectory = []
    for step in range(steps):
        value += scale / max(1.0, step + 1)
        trajectory.append(round(math.tanh(value), 6))
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    tracemalloc.start()
    steps = int(plan.get("steps", 8))
    task_type = plan.get("task_type", "ml_efficiency")
    if task_type == "ml_efficiency":
        trajectory = _run_warmup_schedule(
            steps=steps,
            lr=float(plan.get("learning_rate", 0.12)),
            warmup_ratio=float(plan.get("warmup_ratio", 0.35)),
        )
    else:
        trajectory = _run_linear_probe(steps=steps, scale=float(plan.get("scale", 0.8)))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = {
        "primary_metric": trajectory[-1],
        "secondary_metrics": {
            "start_metric": trajectory[0],
            "end_metric": trajectory[-1],
            "improvement": round(trajectory[-1] - trajectory[0], 6),
        },
        "trajectory": trajectory,
        "peak_memory": peak / (1024 * 1024),
        "status": "keep" if trajectory[-1] >= float(plan.get("keep_threshold", 0.4)) else "discard",
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
