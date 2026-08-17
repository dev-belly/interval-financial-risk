"""Time-based rolling-window cross-validation."""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class RollingWindowValidator:
    """Generate train/validation/test splits by report_date.

    Ensures no forward-looking leakage: training data always precedes test data.
    """

    def __init__(self, config: Config, dates: pd.Series):
        self.config = config
        self.dates = pd.to_datetime(dates)
        self.unique_dates = sorted(self.dates.unique())

    def _date_index(self, date: pd.Timestamp) -> int:
        return self.unique_dates.index(date)

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx, test_idx) for each rolling window.

        Validation set is the quarter immediately preceding the test window.
        """

        cfg = self.config.validation
        initial = cfg.initial_train_quarters
        test_window = cfg.test_quarters
        step = cfg.step_quarters
        min_samples = cfg.min_train_samples

        n_dates = len(self.unique_dates)
        start = initial

        while start + test_window <= n_dates:
            train_end = start
            val_start = start
            val_end = min(start + test_window, n_dates)
            test_start = val_end
            test_end = min(test_start + test_window, n_dates)

            if test_end > n_dates:
                break

            train_dates = self.unique_dates[:train_end]
            val_dates = self.unique_dates[val_start:val_end]
            test_dates = self.unique_dates[test_start:test_end]

            train_idx = df.index[df["report_date"].isin(train_dates)].to_numpy()
            val_idx = df.index[df["report_date"].isin(val_dates)].to_numpy()
            test_idx = df.index[df["report_date"].isin(test_dates)].to_numpy()

            if len(train_idx) < min_samples or len(test_idx) < 10:
                start += step
                continue

            logger.info(
                "Rolling split: train=%s, val=%s, test=%s (n_train=%d, n_test=%d)",
                train_dates[-1].date(),
                val_dates[-1].date(),
                test_dates[-1].date(),
                len(train_idx),
                len(test_idx),
            )

            yield train_idx, val_idx, test_idx
            start += step
