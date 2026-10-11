"""Regression tests for time-aware validation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import Config, load_config
from src.evaluation.conformal import _conformal_radius
from src.evaluation.rolling_validator import RollingWindowValidator


@pytest.mark.parametrize("field", ["initial_train_quarters", "test_quarters", "step_quarters"])
@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_nonpositive_rolling_windows(field, value):
    with pytest.raises(ValidationError, match=field):
        Config.model_validate({"validation": {field: value}})


def test_rolling_validator_rejects_mutated_zero_step_before_yielding():
    config = Config()
    config.validation.step_quarters = 0
    frame = pd.DataFrame({"report_date": pd.date_range("2020-03-31", periods=12, freq="QE")})
    with pytest.raises(ValueError, match="positive"):
        next(RollingWindowValidator(config, frame["report_date"]).split(frame))


@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_nonpositive_interval_window(value):
    with pytest.raises(ValidationError, match="interval_window"):
        Config.model_validate({"features": {"interval_window": value}})


def test_rolling_validator_returns_iloc_positions_for_custom_index():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "full_benchmark.yaml")
    dates = pd.date_range("2020-03-31", periods=12, freq="QE")
    frame = pd.DataFrame(
        {"report_date": np.repeat(dates, 10)},
        index=np.arange(1_000, 1_000 + len(dates) * 10),
    )
    train_idx, val_idx, test_idx = next(
        RollingWindowValidator(config, frame["report_date"]).split(frame)
    )
    assert train_idx.min() == 0
    assert train_idx.max() < len(frame)
    assert frame.iloc[test_idx]["report_date"].min() > frame.iloc[val_idx]["report_date"].max()


def test_conformal_radius_uses_upper_order_statistic():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.6, 0.5])
    # alpha=.5 gives corrected quantile level ceil(5*.5)/4=.75;
    # k=3 selects the third score (zero-based index 2), namely 0.4.
    assert _conformal_radius(y, p, alpha=0.5, n_cal=4) == 0.4
    assert _conformal_radius(y, p, alpha=0.01, n_cal=4) == 1.0
