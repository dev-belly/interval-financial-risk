"""Evaluation metrics for risk classification."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score as pr_auc_score
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray | None = None) -> dict[str, float]:
    """Compute a battery of classification and calibration metrics."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0 or y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must be non-empty and have matching lengths")
    if not np.isfinite(y_proba).all() or ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("y_proba must contain finite probabilities in [0, 1]")

    if y_pred is None:
        y_pred = (y_proba >= 0.5).astype(int)

    auc_value = float("nan")
    pr_auc_value = float("nan")
    if np.unique(y_true).size >= 2:
        auc_value = float(roc_auc_score(y_true, y_proba))
        pr_auc_value = float(pr_auc_score(y_true, y_proba))

    metrics: dict[str, float] = {
        "auc": auc_value,
        "pr_auc": pr_auc_value,
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # Expected Calibration Error (ECE).  Computing the occupied bins directly
    # keeps the counts aligned with their observed/forecast means, including
    # single-class samples and probabilities exactly equal to one.
    n_bins = 10
    bin_ids = np.minimum((y_proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if not np.any(in_bin):
            continue
        ece += float(np.sum(in_bin)) * abs(
            float(np.mean(y_true[in_bin])) - float(np.mean(y_proba[in_bin]))
        )
    ece /= float(y_true.size)
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
