"""Logistic regression baseline using only point-in-time features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.models.base import ModelRegistry, RiskModel

logger = logging.getLogger(__name__)


class LogisticBaselineModel(RiskModel):
    """Baseline logistic regression restricted to point features only."""

    def __init__(self, config_section: dict[str, Any], feature_set: str, name: str, project_config: Config):
        super().__init__(config_section, feature_set, name)
        self.project_config = project_config
        self.point_features = project_config.features.point_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        params = self.config_section.get("params", {}).copy()
        params.setdefault("max_iter", 1000)
        params.setdefault("class_weight", "balanced")
        self.model = LogisticRegression(**params)
        self.model.fit(X, y)
        logger.info("Trained %s on %d samples, %d features", self.name, X.shape[0], X.shape[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float] | None:
        if self.model is None:
            return None
        return {name: float(coef) for name, coef in zip(self.point_features, self.model.coef_[0])}


ModelRegistry.register("logistic_regression", LogisticBaselineModel)
