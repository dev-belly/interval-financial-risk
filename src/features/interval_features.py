"""Interval and distribution feature engineering.

Transforms point-in-time quarterly financial metrics into rolling-window
distribution features (mean, std, quantiles, skew, kurtosis, interval width).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


def _rolling_stats(group: pd.DataFrame, feature: str, window: int, stats: list[str]) -> pd.DataFrame:
    """Compute rolling distribution statistics for a single feature."""

    s = group[feature]
    result = pd.DataFrame(index=group.index)

    if "mean" in stats:
        result[f"{feature}_mean"] = s.rolling(window=window, min_periods=1).mean()
    if "std" in stats:
        result[f"{feature}_std"] = s.rolling(window=window, min_periods=2).std().fillna(0)
    if "min" in stats:
        result[f"{feature}_min"] = s.rolling(window=window, min_periods=1).min()
    if "max" in stats:
        result[f"{feature}_max"] = s.rolling(window=window, min_periods=1).max()
    if "q25" in stats:
        result[f"{feature}_q25"] = s.rolling(window=window, min_periods=1).quantile(0.25)
    if "q50" in stats:
        result[f"{feature}_q50"] = s.rolling(window=window, min_periods=1).quantile(0.50)
    if "q75" in stats:
        result[f"{feature}_q75"] = s.rolling(window=window, min_periods=1).quantile(0.75)
    if "skew" in stats:
        result[f"{feature}_skew"] = s.rolling(window=window, min_periods=3).skew().fillna(0)
    if "kurt" in stats:
        result[f"{feature}_kurt"] = s.rolling(window=window, min_periods=4).kurt().fillna(0)

    return result


def build_interval_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build point and interval/distribution features from raw panel data.

    For each interval feature defined in config, compute rolling statistics
    over the past ``interval_window`` quarters per company. Original point
    features are retained. Interval width (max - min) is added when configured.

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

    df = df.sort_values(["company_id", "report_date"]).copy()
    window = config.features.interval_window
    stats = config.features.interval_stats

    feature_frames = []
    for feature in config.features.interval_features:
        if feature not in df.columns:
            logger.warning("Interval feature %s not found in data; skipping", feature)
            continue

        rolled = df.groupby("company_id", group_keys=False).apply(
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
