"""Selection must not change when held-out test scores change."""

import joblib
import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.synthetic_generator import generate_synthetic_data
from src.features.interval_features import build_interval_features
from src.features.pipeline import FeaturePipeline
from src.models.logistic_baseline import LogisticBaselineModel
from src.pipeline import ExperimentPipeline, build_prepared_frame_pipeline


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


def test_saved_point_model_pipeline_matches_in_memory_prediction(tmp_path):
    config = load_config("config/full_benchmark.yaml").model_copy(deep=True)
    config.data.synthetic.n_companies = 30
    raw = generate_synthetic_data(config)
    prepared = build_interval_features(raw, config)
    train = prepared[prepared.report_date < "2021-01-01"]
    test = prepared[prepared.report_date >= "2021-01-01"]

    features = FeaturePipeline(config)
    X_train, y_train, _ = features.fit_transform(train, engineer_features=False)
    names = features.get_feature_names()
    point_names = config.features.point_features
    point_indices = [names.index(name) for name in point_names]
    model = LogisticBaselineModel({}, "point_only", "point model", config)
    model.feature_names = point_names
    model.fit(X_train[:, point_indices], y_train)

    path = tmp_path / "prepared_frame_pipeline.joblib"
    joblib.dump(build_prepared_frame_pipeline(model, features), path)
    restored = joblib.load(path)
    X_test, _, _ = features.transform(test, engineer_features=False)
    np.testing.assert_allclose(
        restored.predict_proba(test)[:, 1],
        model.predict_proba(X_test[:, point_indices]),
    )
