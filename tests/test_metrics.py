"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import compute_grouped_metrics, compute_metrics


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_proba)
    assert metrics["auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_grouped_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    result = compute_grouped_metrics(y_true, y_proba, groups, metrics=["auc"])
    assert "A" in result
    assert "B" in result
    assert 0 <= result["A"]["auc"] <= 1


def test_single_class_auc_is_nan_instead_of_crashing():
    metrics = compute_metrics(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.isnan(metrics["auc"])
    assert np.isnan(metrics["pr_auc"])


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError, match="probabilities"):
        compute_metrics(np.array([0, 1]), np.array([0.2, 1.2]))
