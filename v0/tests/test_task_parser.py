import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.task_parser import build_dependency_graph, extract_stage, parse_task_tree, tasks_for_stage, validate_task_tree


MARKDOWN_TREE = """
### Task 1
```yml
id: S1T1
description: Define the problem.
dependencies: []
stage: 1
verification_criteria:
  - The problem statement is written down.
complexity_tier: STANDARD
foundational: true
```

### Task 2
```yml
id: S10T1
description: Validate the result.
dependencies: [S1T1]
stage: 10
verification_criteria:
  - Validation report exists.
complexity_tier: HIGH
foundational: false
```
"""


def test_parse_markdown_yaml_blocks():
    tasks = parse_task_tree(MARKDOWN_TREE)
    assert len(tasks) == 2
    assert tasks[0]["id"] == "S1T1"
    assert tasks[1]["stage"] == 10
    assert tasks[1]["dependencies"] == ["S1T1"]


def test_parse_json_review_tasks():
    payload = {
        "tasks": [
            {
                "id": "S2T1",
                "description": "Write output",
                "dependencies": [],
                "stage": 2,
                "verification_criteria": ["Output exists"],
                "complexity_tier": "STANDARD",
                "foundational": False,
            }
        ]
    }
    tasks = parse_task_tree(json.dumps(payload))
    assert tasks[0]["verification_criteria"] == ["Output exists"]


def test_validate_task_tree_reports_missing_dependency():
    tasks = parse_task_tree(MARKDOWN_TREE)
    tasks[1]["dependencies"] = ["S9T9"]
    errors = validate_task_tree(tasks)
    assert "dependency 'S9T9' not found" in errors[0]


def test_dependency_graph_and_tasks_for_stage():
    tasks = parse_task_tree(MARKDOWN_TREE)
    graph = build_dependency_graph(tasks)
    assert graph["S1T1"] == ["S10T1"]
    assert [task["id"] for task in tasks_for_stage(tasks, 10)] == ["S10T1"]


def test_extract_stage_supports_multi_digit():
    assert extract_stage("S10T1") == 10
