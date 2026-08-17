"""Ablation studies to quantify incremental value of feature groups."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.metrics import compute_metrics
from src.models.base import RiskModel

logger = logging.getLogger(__name__)


def ablation_study(
    model_builder: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: Config,
) -> pd.DataFrame:
    """Run ablation experiments by removing groups of features.

    Parameters
    ----------
    model_builder:
        Callable that returns a fresh, unfitted RiskModel instance.
    X, y:
        Full training data.
    feature_names:
        Ordered list of feature names corresponding to X columns.
    config:
        Project configuration with ablation_groups defined.

    Returns
    -------
    pd.DataFrame
        Ablation results with metric differences vs. full model.
    """

    # Full model baseline
    full_model = model_builder()
    full_model.fit(X, y)
    full_proba = full_model.predict_proba(X)
    full_metrics = compute_metrics(y, full_proba)

    rows = [{"ablation": "full", "description": "All features", **full_metrics}]

    for group in config.evaluation.ablation_groups:
        remove_patterns = group.remove_patterns
        keep_cols = [
            i
            for i, name in enumerate(feature_names)
            if not any(pattern in name for pattern in remove_patterns)
        ]

        if not keep_cols:
            logger.warning("Ablation group %s removed all features; skipping", group.name)
            continue

        X_abl = X[:, keep_cols]
        model = model_builder()
        model.fit(X_abl, y)
        proba = model.predict_proba(X_abl)
        metrics = compute_metrics(y, proba)

        row = {"ablation": group.name, "description": group.description}
        for k, v in metrics.items():
            row[k] = v
            row[f"{k}_delta"] = v - full_metrics[k]
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Ablation study complete: %d configurations", len(df))
    return df
