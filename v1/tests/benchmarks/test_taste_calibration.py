from genesis.models import ExperimentProposal
from genesis.taste.features import ExperimentFeatureExtractor
from genesis.taste.gp_model import TasteGP


def test_taste_model_calibration_properties():
    proposals = [
        ExperimentProposal(
            description=f"experiment {index}",
            code_diff=f"diff {index}",
            expected_metric=0.3 + 0.02 * index,
            expected_trajectory=[0.1 + 0.01 * index, 0.2 + 0.01 * index, 0.3 + 0.02 * index],
            compute_budget="local_gpu",
            model_parameter_count=1000 * (index + 1),
        )
        for index in range(20)
    ]
    extractor = ExperimentFeatureExtractor()
    features = [extractor.extract(proposal) for proposal in proposals]
    targets = [proposal.expected_metric for proposal in proposals]
    trajectories = [proposal.expected_trajectory for proposal in proposals]

    model = TasteGP()
    model.fit(features[:5], targets[:5], trajectories[:5])
    _, low_data_variances = model.predict(features[5:10])
    model.fit(features, targets, trajectories)
    _, high_data_variances = model.predict(features[5:10])

    assert sum(high_data_variances) <= sum(low_data_variances)
