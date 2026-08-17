"""Permutation importance test for feature significance."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.models.base import RiskModel

logger = logging.getLogger(__name__)


def permutation_importance_test(
    model: RiskModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 20,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimate feature importance via permutation on the validation set.

    For each feature, shuffle its values, compute the drop in AUC, and report
    mean drop, standard deviation, and an approximate p-value.
    """

    rng = np.random.default_rng(random_state)
    baseline_proba = model.predict_proba(X)
    baseline_score = roc_auc_score(y, baseline_proba)

    results: dict[str, dict[str, float]] = {}

    for col_idx, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, col_idx])
            perm_proba = model.predict_proba(X_perm)
            perm_score = roc_auc_score(y, perm_proba)
            scores.append(baseline_score - perm_score)

        scores_arr = np.array(scores)
        results[name] = {
            "mean_drop": float(scores_arr.mean()),
            "std_drop": float(scores_arr.std()),
            "p_value": float(np.mean(scores_arr <= 0)),
            "baseline_auc": float(baseline_score),
        }

    logger.info("Permutation test complete for %d features", len(feature_names))
    return results
