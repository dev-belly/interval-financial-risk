"""Evaluation metrics for risk classification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import average_precision_score as pr_auc_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray | None = None) -> dict[str, float]:
    """Compute a battery of classification and calibration metrics."""

    if y_pred is None:
        y_pred = (y_proba >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(pr_auc_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # Expected Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    n = len(y_true)
    bin_counts, bin_edges = np.histogram(y_proba, bins=np.linspace(0, 1, 11))
    # Only use bins that calibration_curve returned (non-empty bins)
    nonempty_mask = bin_counts > 0
    ece = float(
        np.sum(bin_counts[nonempty_mask] * np.abs(prob_true - prob_pred)) / max(n, 1)
    )
    metrics["calibration_error"] = ece

    return metrics


def compute_calibration_curve(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Return calibration curve data."""

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return {"prob_true": prob_true, "prob_pred": prob_pred}


def compute_grouped_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: np.ndarray,
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics stratified by group (e.g., industry or time period)."""

    if metrics is None:
        metrics = ["auc", "pr_auc", "brier"]

    result: dict[str, dict[str, float]] = {}
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 2:
            continue
        y_g = y_true[mask]
        p_g = y_proba[mask]
        try:
            result[str(group)] = {m: compute_metrics(y_g, p_g)[m] for m in metrics}
        except Exception as exc:
            logger.warning("Could not compute metrics for group %s: %s", group, exc)

    return result
