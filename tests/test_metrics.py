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
