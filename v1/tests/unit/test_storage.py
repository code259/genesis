from genesis.storage.causal_dag import CausalDAG
from genesis.storage.filesystem import ProjectFilesystem
from genesis.storage.ledger import ExperimentLedger
from genesis.models import ExperimentResult


def test_filesystem_init_and_results(tmp_path):
    fs = ProjectFilesystem(tmp_path / "projects")
    fs.init_project("demo", {"research_question": "q"})
    run_dir = fs.get_run_dir("demo", 1)
    fs.write_json(run_dir / "result.json", {"primary_metric": 0.8})
    assert fs.list_all_results("demo")[0]["primary_metric"] == 0.8


def test_ledger_insert_and_query(tmp_path):
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3")
    ledger.insert_experiment(
        ExperimentResult(
            experiment_id="exp-1",
            task_id="task-1",
            primary_metric=0.9,
            secondary_metrics={},
            trajectory=[0.1, 0.9],
            peak_memory=0.0,
            runtime_seconds=0.1,
            status="keep",
            code_hash="abc",
            artifact_path="artifact.json",
        ),
        timestamp="2026-04-04T00:00:00Z",
    )
    assert ledger.get_by_task("task-1")[0]["primary_metric"] == 0.9


def test_causal_dag_cycle_detection(tmp_path):
    dag = CausalDAG(tmp_path / "dag.json")
    dag.add_edge("a", "b", effect_size=1.0, confidence=0.9, experiment_ids=["1"])
    try:
        dag.add_edge("b", "a", effect_size=1.0, confidence=0.9, experiment_ids=["1"])
    except ValueError:
        return
    raise AssertionError("expected cycle detection")
