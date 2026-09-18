"""Selection must not change when held-out test scores change."""

import pandas as pd

from src.config import Config
from src.pipeline import ExperimentPipeline


def test_diagnostic_model_uses_validation_mean_and_ignores_test_winner():
    experiment = ExperimentPipeline(Config())
    experiment.results["validation"] = pd.DataFrame(
        [
            {"model": "validation_winner", "auc": 0.8},
            {"model": "validation_winner", "auc": 0.7},
            {"model": "test_winner", "auc": 0.9},
            {"model": "test_winner", "auc": 0.4},
        ]
    )
    experiment.results["summary"] = pd.DataFrame(
        [
            {"model": "validation_winner", "auc": 0.51},
            {"model": "test_winner", "auc": 0.99},
        ]
    )
    assert experiment._select_diagnostic_model() == "validation_winner"
    experiment.results["summary"]["auc"] = [0.99, 0.51]
    assert experiment._select_diagnostic_model() == "validation_winner"


def test_elastic_net_repeated_fit_uses_configured_seed():
    import numpy as np

    from src.models.regularized_model import ElasticNetModel

    rng = np.random.default_rng(8)
    features = rng.normal(size=(80, 5))
    target = (features[:, 0] + rng.normal(size=80) > 0).astype(int)
    config = Config()
    params = {"params": {"l1_ratio": 0.5, "max_iter": 1000}}
    first = ElasticNetModel(params, "point_and_interval", "Elastic Net", config)
    second = ElasticNetModel(params, "point_and_interval", "Elastic Net", config)
    first.fit(features, target)
    second.fit(features, target)
    assert first.model.random_state == config.project.seed
    np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))
