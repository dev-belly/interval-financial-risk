"""Unit tests for feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.data.synthetic_generator import generate_synthetic_data
from src.features.interval_features import build_interval_features
from src.features.pipeline import FeaturePipeline


@pytest.fixture
def config():
    project_root = Path(__file__).resolve().parent.parent
    return load_config(project_root / "config" / "config.yaml")


@pytest.fixture
def synthetic_df(config):
    return generate_synthetic_data(config)


def test_synthetic_data_has_required_columns(synthetic_df):
    required = {"company_id", "report_date", "industry", "risk_label"}
    assert required.issubset(synthetic_df.columns)


def test_interval_features_created(config, synthetic_df):
    df = build_interval_features(synthetic_df, config)
    for feature in config.features.interval_features:
        assert f"{feature}_mean" in df.columns
        assert f"{feature}_std" in df.columns


def test_feature_pipeline_fit_transform(config, synthetic_df):
    pipeline = FeaturePipeline(config)
    X, y, meta = pipeline.fit_transform(synthetic_df)
    assert X.shape[0] == len(synthetic_df)
    assert X.shape[1] == len(pipeline.get_feature_names())
    assert len(y) == len(synthetic_df)
    assert not pd.isna(X).any()
