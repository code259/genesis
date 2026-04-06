import json

from genesis.harness.history_reader import SelectiveHistoryReader
from genesis.harness.token_budget import TokenBudget
from genesis.storage.filesystem import ProjectFilesystem


def test_history_reader_summarizes_progress_and_failures_separately(tmp_path):
    filesystem = ProjectFilesystem(tmp_path / "projects")
    project_dir = filesystem.init_project("demo", {"research_question": "q"})
    run1 = filesystem.get_run_dir("demo", 1)
    run2 = filesystem.get_run_dir("demo", 2)
    (run1 / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "summary": "Executed a useful command.",
                "primary_metric": 0.6,
                "generated_artifacts": ["artifact.txt"],
                "executed_commands": ["python3 main.py"],
                "classification": "success",
            }
        ),
        encoding="utf-8",
    )
    (run2 / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-2",
                "summary": "Did nothing useful.",
                "primary_metric": 0.0,
                "generated_artifacts": [],
                "executed_commands": [],
                "classification": "non_actionable_plan",
                "failure_summary": "model returned no files, commands, or usable experiment plan",
            }
        ),
        encoding="utf-8",
    )
    reader = SelectiveHistoryReader(filesystem, TokenBudget())
    summary = reader.summarize_experiment_history("demo")
    assert "- task-1: metric=0.6 | Executed a useful command." in summary
    assert "- failed task-2: non_actionable_plan | model returned no files, commands, or usable experiment plan" in summary

