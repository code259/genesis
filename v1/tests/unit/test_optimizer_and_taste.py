from genesis.models import ExperimentProposal
from genesis.modules.optimizer.parallel import ParallelExperimentManager
from genesis.modules.optimizer.proposer import ExperimentProposer
from genesis.taste.features import ExperimentFeatureExtractor
from genesis.taste.gp_model import TasteGP


def test_parallel_experiment_manager_runs(tmp_path):
    manager = ParallelExperimentManager(tmp_path / "sandboxes")
    results = manager.run_batch(
        [
            {"experiment_id": "exp1", "task_id": "task", "primary_metric": 0.5},
            {"experiment_id": "exp2", "task_id": "task", "primary_metric": 0.6},
        ],
        n_parallel=2,
    )
    assert len(results) == 2


def test_feature_extractor_and_gp():
    proposal = ExperimentProposal(
        description="test experiment",
        code_diff="diff",
        expected_metric=0.5,
        expected_trajectory=[0.1, 0.2],
        compute_budget="local_gpu",
        model_parameter_count=1000,
    )
    features = ExperimentFeatureExtractor().extract(proposal)
    model = TasteGP()
    model.fit([features], [0.5], [[0.1, 0.2]])
    means, variances = model.predict([features])
    assert means and variances


def test_experiment_proposer():
    proposals = ExperimentProposer().propose_next("task", n=2)
    assert len(proposals) == 2
