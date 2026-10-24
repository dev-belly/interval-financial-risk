"""Interval and distribution feature engineering.

Transforms point-in-time quarterly financial metrics into rolling-window
distribution features (mean, std, quantiles, skew, kurtosis, interval width).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer

from src.config import Config

logger = logging.getLogger(__name__)


class _QuarterWindow(BaseIndexer):
    """Select observed rows in the current and preceding calendar quarters."""

    def __init__(self, dates: pd.Series, window: int):
        super().__init__(window_size=window)
        self.quarters = dates.dt.to_period("Q").astype("int64").to_numpy()

    def get_window_bounds(
        self, num_values=0, min_periods=None, center=None, closed=None, step=None
    ):
        starts = np.searchsorted(self.quarters, self.quarters - self.window_size + 1)
        ends = np.arange(1, len(self.quarters) + 1, dtype=np.int64)
        return starts, ends


def _rolling_stats(
    group: pd.DataFrame, feature: str, window: int, stats: list[str]
) -> pd.DataFrame:
    """Compute rolling distribution statistics for a single feature."""

    s = group[feature]
    calendar = _QuarterWindow(group["report_date"], window)
    result = pd.DataFrame(index=group.index)

    if "mean" in stats:
        result[f"{feature}_mean"] = s.rolling(calendar, min_periods=1).mean()
    if "std" in stats:
        result[f"{feature}_std"] = s.rolling(calendar, min_periods=min(2, window)).std().fillna(0)
    if "min" in stats:
        result[f"{feature}_min"] = s.rolling(calendar, min_periods=1).min()
    if "max" in stats:
        result[f"{feature}_max"] = s.rolling(calendar, min_periods=1).max()
    if "q25" in stats:
        result[f"{feature}_q25"] = s.rolling(calendar, min_periods=1).quantile(0.25)
    if "q50" in stats:
        result[f"{feature}_q50"] = s.rolling(calendar, min_periods=1).quantile(0.50)
    if "q75" in stats:
        result[f"{feature}_q75"] = s.rolling(calendar, min_periods=1).quantile(0.75)
    if "skew" in stats:
        result[f"{feature}_skew"] = s.rolling(calendar, min_periods=min(3, window)).skew().fillna(0)
    if "kurt" in stats:
        result[f"{feature}_kurt"] = s.rolling(calendar, min_periods=min(4, window)).kurt().fillna(0)

    return result


def build_interval_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build point and interval/distribution features from raw panel data.

    For each interval feature defined in config, compute rolling statistics
    over the current and preceding ``interval_window - 1`` calendar quarters per
    company. Gaps do not extend the window or synthesize observations. There must
    be at most one row per company/quarter. Original point features are retained.
    Interval width (max - min) is added when configured.

    Parameters
    ----------
    df:
        Raw panel data with company_id, report_date, and financial features.
    config:
        Project configuration.

    Returns
    -------
    pd.DataFrame
        Augmented dataframe with point features and interval features.
    """

    window = config.features.interval_window
    if type(window) is not int or window <= 0:
        raise ValueError("interval_window must be a positive integer number of quarters")
    if df["company_id"].isna().any():
        raise ValueError("company_id must be present for every quarterly observation")
    df = df.copy()
    try:
        if pd.api.types.is_numeric_dtype(df["report_date"]):
            raise ValueError("Numeric timestamps are not quarterly calendar dates")
        dates = pd.to_datetime(df["report_date"], errors="raise")
        if dates.isna().any() or dates.dt.tz is not None:
            raise ValueError("Missing or timezone-aware report date")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("report_date must contain valid timezone-naive calendar dates") from exc
    df["report_date"] = dates
    keys = pd.MultiIndex.from_arrays([df["company_id"], dates.dt.to_period("Q")])
    if keys.has_duplicates:
        raise ValueError("Expected at most one observation per company and calendar quarter")
    df = df.sort_values(["company_id", "report_date"])
    stats = config.features.interval_stats

    feature_frames = []
    for feature in config.features.interval_features:
        if feature not in df.columns:
            logger.warning("Interval feature %s not found in data; skipping", feature)
            continue

        rolled = df.groupby("company_id", group_keys=False)[[feature, "report_date"]].apply(
            lambda g: _rolling_stats(g, feature, window, stats)
        )
        feature_frames.append(rolled)

    if feature_frames:
        interval_df = pd.concat(feature_frames, axis=1)
        df = pd.concat([df.reset_index(drop=True), interval_df.reset_index(drop=True)], axis=1)

    # Add interval width
    if config.features.use_interval_width:
        for feature in config.features.interval_features:
            min_col = f"{feature}_min"
            max_col = f"{feature}_max"
            if min_col in df.columns and max_col in df.columns:
                df[f"{feature}_width"] = df[max_col] - df[min_col]
            elif feature in df.columns:
                bounds = df.groupby("company_id", group_keys=False)[[feature, "report_date"]].apply(
                    lambda g: _rolling_stats(g, feature, window, ["min", "max"])
                )
                df[f"{feature}_width"] = (bounds[max_col] - bounds[min_col]).to_numpy()

    # Add a few engineered ratio features
    for feature in config.features.interval_features:
        mean_col = f"{feature}_mean"
        std_col = f"{feature}_std"
        if mean_col in df.columns and std_col in df.columns:
            df[f"{feature}_cv"] = df[std_col] / (df[mean_col].abs() + 1e-6)

    logger.info(
        "Built interval features: window=%d, stats=%s, total_columns=%d",
        window,
        stats,
        len(df.columns),
    )
    return df
