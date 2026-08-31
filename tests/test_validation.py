"""Regression tests for time-aware validation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.conformal import _conformal_radius
from src.evaluation.rolling_validator import RollingWindowValidator


def test_rolling_validator_returns_iloc_positions_for_custom_index():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "full_benchmark.yaml")
    dates = pd.date_range("2020-03-31", periods=12, freq="QE")
    frame = pd.DataFrame(
        {"report_date": np.repeat(dates, 10)},
        index=np.arange(1_000, 1_000 + len(dates) * 10),
    )
    train_idx, val_idx, test_idx = next(RollingWindowValidator(config, frame["report_date"]).split(frame))
    assert train_idx.min() == 0
    assert train_idx.max() < len(frame)
    assert frame.iloc[test_idx]["report_date"].min() > frame.iloc[val_idx]["report_date"].max()


def test_conformal_radius_uses_upper_order_statistic():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.6, 0.5])
    # alpha=.5 gives corrected quantile level ceil(5*.5)/4=.75;
    # the conformal order statistic is 0.5, not an interpolated 0.425.
    assert _conformal_radius(y, p, alpha=0.5, n_cal=4) == 0.5
