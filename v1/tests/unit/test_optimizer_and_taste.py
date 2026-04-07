from pathlib import Path

from genesis.models import ExperimentProposal, ExperimentResult
from genesis.modules.ideation.greedy import GreedyAdjacencySearch
from genesis.modules.ideation.low_density import LowDensityExplorer
from genesis.modules.ideation.orchestrator import IdeationOrchestrator
from genesis.modules.ideation.pollination import PollinationSearch
from genesis.modules.optimizer.oracle_resolver import OracleResolver
from genesis.modules.optimizer.parallel import ParallelExperimentManager
from genesis.modules.optimizer.proposer import ExperimentProposer
from genesis.modules.optimizer.runner import ExperimentRunner
from genesis.storage.ledger import ExperimentLedger
from genesis.storage.causal_dag import CausalDAG
from genesis.storage.manifold import ManifoldIndex
from genesis.taste.features import ExperimentFeatureExtractor
from genesis.taste.gp_model import TasteGP
from genesis.taste.persistence import TasteModelPersistence
from genesis.config import ProjectConfig
from genesis.models import TaskNode


def _proposal(description: str, metric: float, steps: int = 3) -> ExperimentProposal:
    return ExperimentProposal(
        description=description,
        code_diff="learning_rate=0.1; warmup_ratio=0.3",
        expected_metric=metric,
        expected_trajectory=[metric * 0.5, metric * 0.8, metric],
        compute_budget="local_gpu",
        model_parameter_count=1000,
    )


def test_parallel_experiment_manager_runs_and_preserves_order(tmp_path):
    manager = ParallelExperimentManager(tmp_path / "sandboxes")
    results = manager.run_batch(
        [
            {
                "experiment_id": "exp1",
                "task_id": "task",
                "command": ["python3", "-c", "import json; open('result.json','w').write(json.dumps({'primary_metric':0.6,'trajectory':[0.2,0.4,0.6]}))"],
                "expected_trajectory": [0.2, 0.4, 0.6],
            },
            {
                "experiment_id": "exp2",
                "task_id": "task",
                "command": ["python3", "-c", "import json; open('result.json','w').write(json.dumps({'primary_metric':0.3,'trajectory':[0.1,0.2,0.3]}))"],
                "expected_trajectory": [0.1, 0.2, 0.3],
            },
        ],
        n_parallel=2,
    )
    assert [result.experiment_id for result in results] == ["exp1", "exp2"]
    assert all(result.trajectory for result in results)
    assert all(Path(result.artifact_path).exists() for result in results)


def test_experiment_runner_requires_real_command(tmp_path):
    runner = ExperimentRunner(tmp_path / "sandboxes")
    result = runner.run("task", {"experiment_id": "exp1", "expected_trajectory": [0.1, 0.2]})
    assert result.status == "crash"
    assert "did not provide a runnable command" in Path(result.artifact_path).read_text(encoding="utf-8")


def test_experiment_proposer_uses_ledger_history(tmp_path):
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3")
    ledger.insert_experiment(
        ExperimentResult(
            experiment_id="exp-1",
            task_id="task-1",
            primary_metric=0.72,
            secondary_metrics={"improvement": 0.3},
            trajectory=[0.2, 0.5, 0.72],
            peak_memory=1.0,
            runtime_seconds=0.1,
            status="keep",
            code_hash="hash1",
            artifact_path="artifact.json",
            trajectory_path="trajectory.npz",
            anomaly_score=0.75,
        ),
        timestamp="2026-04-04T00:00:00Z",
    )
    dag = CausalDAG(tmp_path / "dag.json")
    dag.add_edge("learning_rate", "metric", effect_size=0.2, confidence=0.9, experiment_ids=["exp-1"], domain="ml_efficiency")
    proposals = ExperimentProposer().propose_next("task-1", n=2, ledger=ledger, causal_dag=dag, domain="ml_efficiency")
    assert len(proposals) == 2
    assert proposals[0].expected_metric >= 0.72
    assert "causal" in proposals[0].description.lower() or "stabilization" in proposals[0].description.lower()


def test_feature_extractor_gp_and_persistence_shape(tmp_path):
    extractor = ExperimentFeatureExtractor()
    proposals = [_proposal(f"experiment {index}", 0.3 + index * 0.05) for index in range(4)]
    features = [extractor.extract(proposal) for proposal in proposals]
    assert len(features[0]) > 90
    assert features[0] == extractor.extract(proposals[0])

    model = TasteGP()
    model.fit(features[:2], [proposal.expected_metric for proposal in proposals[:2]], [proposal.expected_trajectory for proposal in proposals[:2]])
    low_data_variances = model.predict(features[2:])[1]
    model.fit(features, [proposal.expected_metric for proposal in proposals], [proposal.expected_trajectory for proposal in proposals])
    high_data_variances = model.predict(features[2:])[1]
    assert sum(high_data_variances) <= sum(low_data_variances)

    save_path = tmp_path / "taste_model.json"
    model.save(save_path)
    restored = TasteGP.load(save_path)
    means, _ = restored.predict(features[2:])
    assert len(means) == 2
    assert model.backend in {"scipy", "gpytorch"}


def test_taste_gp_uses_nearest_neighbor_before_enough_points():
    model = TasteGP()
    x = [[0.0, 0.0], [10.0, 10.0]]
    y = [0.2, 0.9]
    model.fit(x, y, [[0.2], [0.9]])
    means, variances = model.predict([[9.5, 9.5]])
    assert means[0] == 0.9
    assert variances[0] > 0.0


def test_taste_persistence_merges_only_verified_real_outcomes(tmp_path):
    persistence = TasteModelPersistence(tmp_path / "taste_db")
    persistence.merge_project_data(
        "proj",
        [
            {
                "experiment_id": "good",
                "status": "keep",
                "artifact_path": str(tmp_path / "result.json"),
                "trajectory": [0.1, 0.2],
            },
            {
                "experiment_id": "bad-missing-command",
                "status": "keep",
                "artifact_path": str(tmp_path / "missing_command.txt"),
                "trajectory": [0.1, 0.2],
            },
            {
                "experiment_id": "bad-crash",
                "status": "crash",
                "artifact_path": str(tmp_path / "stderr.log"),
                "trajectory": [0.1],
            },
        ],
    )
    merged = persistence.dataset_path.read_text(encoding="utf-8")
    assert "good" in merged
    assert "bad-missing-command" not in merged
    assert "bad-crash" not in merged


def test_oracle_resolver_and_ideation_orchestrator(tmp_path):
    manifold = ManifoldIndex(tmp_path / "manifold")
    manifold.upsert_collection(
        [
            {
                "paper_id": "paper-1",
                "title": "Warmup schedules improve convergence",
                "abstract": "Warmup stabilizes early optimization.",
                "latent_z": [1.0, 0.0, 0.0],
                "density_score": 0.2,
                "citations": [{"paper_id": "paper-2"}],
            },
            {
                "paper_id": "paper-2",
                "title": "Sparse optimizer regimes",
                "abstract": "Explores distant sparse regions of optimizer space.",
                "latent_z": [0.0, 1.0, 0.0],
                "density_score": 0.9,
                "citations": [{"paper_id": "paper-1"}],
            },
            {
                "paper_id": "paper-3",
                "title": "Gradient smoothing in compact models",
                "abstract": "Analyzes stable optimization in smaller networks.",
                "latent_z": [0.0, 0.0, 1.0],
                "density_score": 0.7,
                "citations": [],
            },
        ],
        collection="papers",
    )
    manifold.add_experiment({"experiment_id": "exp-used", "paper_id": "paper-1", "density_score": 0.1})

    orchestrator = IdeationOrchestrator(
        greedy=GreedyAdjacencySearch(manifold),
        pollination=PollinationSearch(manifold),
        low_density=LowDensityExplorer(manifold),
    )
    ideation_result = orchestrator.run_with_status("optimizer warmup stability", n_failed_iterations=5)
    ideas = ideation_result.ideas
    assert ideation_result.status == "enabled_with_candidates"
    assert "greedy" in ideation_result.health.ready_modes
    assert ideas
    assert ideas[0].score.composite_score >= ideas[-1].score.composite_score
    assert any(idea.idea.source == "pollination" for idea in ideas)
    assert all(idea.idea.metadata.get("source_paper_id") != "paper-1" for idea in ideas if idea.idea.source == "greedy")

    task = TaskNode(
        task_id="task-1",
        description="Optimize training",
        acceptance_criteria=[],
        oracle_checks=[],
        estimated_compute_budget="local_gpu",
        success_metric="validation accuracy > 0.8",
        requires_ml_optimizer=True,
    )
    config = ProjectConfig(
        research_question="Optimize training",
        domain="ml_efficiency",
        compute_budget="local_gpu",
        time_budget_hours=2,
        domain_knowledge_model="none",
        output_dir=str(tmp_path),
        success_criteria=["loss < 0.2"],
    )
    spec = OracleResolver().resolve_oracle(task, config)
    assert spec.metric_name == "loss"
    assert spec.direction == "minimize"


def test_manifold_chromadb_round_trips_structured_metadata(tmp_path):
    manifold = ManifoldIndex(tmp_path / "manifold")
    manifold.upsert_collection(
        [
            {
                "paper_id": "paper-1",
                "title": "Warmup schedules improve convergence",
                "abstract": "Warmup stabilizes early optimization.",
                "latent_z": [1.0, 0.0, 0.0],
                "embedding": [1.0, 0.0, 0.0],
                "density_score": 0.2,
                "authors": [{"name": "A. Researcher"}],
                "citations": ["paper-2"],
                "domain": "ml_efficiency",
            }
        ],
        collection="papers",
    )
    papers = manifold.all_papers()
    assert papers[0]["authors"] == [{"name": "A. Researcher"}]
    assert papers[0]["citations"] == ["paper-2"]


def test_manifold_health_reports_missing_prereqs(tmp_path):
    manifold = ManifoldIndex(tmp_path / "manifold")
    health = manifold.assess_health()
    assert health.status == "empty"
    assert "papers_missing" in health.reasons


def test_manifold_health_reports_ready_modes(tmp_path):
    manifold = ManifoldIndex(tmp_path / "manifold")
    manifold.upsert_collection(
        [
            {
                "paper_id": "paper-1",
                "title": "Paper 1",
                "abstract": "A",
                "embedding": [1.0, 0.0],
                "latent_z": [1.0, 0.0],
                "density_score": 0.4,
                "citations": [{"paper_id": "paper-2"}],
            },
            {
                "paper_id": "paper-2",
                "title": "Paper 2",
                "abstract": "B",
                "embedding": [0.0, 1.0],
                "latent_z": [0.0, 1.0],
                "density_score": 0.6,
                "citations": [],
            },
        ],
        collection="papers",
    )
    health = manifold.assess_health()
    assert health.status in {"ready", "degraded"}
    assert "greedy" in health.ready_modes
