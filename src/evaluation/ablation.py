"""Ablation studies to quantify incremental value of feature groups."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def ablation_study(
    model_builder: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    config: Config,
) -> pd.DataFrame:
    """Run ablation experiments by removing groups of features.

    Parameters
    ----------
    model_builder:
        Callable that returns a fresh, unfitted RiskModel instance.
    X_train, y_train:
        Training data used to fit each candidate model.
    X_test, y_test:
        Held-out data used for every metric comparison.
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
    full_model.fit(X_train, y_train)
    full_proba = full_model.predict_proba(X_test)
    full_metrics = compute_metrics(y_test, full_proba)

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

        X_train_abl = X_train[:, keep_cols]
        X_test_abl = X_test[:, keep_cols]
        model = model_builder()
        model.fit(X_train_abl, y_train)
        proba = model.predict_proba(X_test_abl)
        metrics = compute_metrics(y_test, proba)

        row = {"ablation": group.name, "description": group.description}
        for k, v in metrics.items():
            row[k] = v
            row[f"{k}_delta"] = v - full_metrics[k]
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Ablation study complete: %d configurations", len(df))
    return df
