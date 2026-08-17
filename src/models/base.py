"""Abstract base classes and model registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RiskModel(ABC):
    """Abstract interface for all risk-prediction models."""

    def __init__(self, config_section: dict[str, Any], feature_set: str, name: str):
        self.config_section = config_section
        self.feature_set = feature_set
        self.name = name
        self.model: Any = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importances if available."""
        raise NotImplementedError


class ModelRegistry:
    """Registry mapping model_type strings to concrete RiskModel classes."""

    _registry: dict[str, type[RiskModel]] = {}

    @classmethod
    def register(cls, model_type: str, model_class: type[RiskModel]) -> None:
        cls._registry[model_type] = model_class
        logger.debug("Registered model type %s", model_type)

    @classmethod
    def get(cls, model_type: str) -> type[RiskModel]:
        if model_type not in cls._registry:
            raise KeyError(f"Unknown model_type '{model_type}'. Available: {list(cls._registry.keys())}")
        return cls._registry[model_type]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())
