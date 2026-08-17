"""Elastic-Net regularized logistic regression with interval features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.models.base import ModelRegistry, RiskModel

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ElasticNetModel(RiskModel):
    """Logistic regression with elastic-net penalty over point + interval features."""

    def __init__(self, config_section: dict[str, Any], feature_set: str, name: str, project_config: Config):
        super().__init__(config_section, feature_set, name)
        self.project_config = project_config
        self.optimize = config_section.get("optimize", False)
        self.optuna_trials = config_section.get("optuna_trials", 50)
        self.feature_names: list[str] = []

    def _compute_class_weight(self, y: np.ndarray) -> dict[int, float]:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        return {0: 1.0, 1: n_neg / max(n_pos, 1)}

    def _build_estimator(self, params: dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        params = self.config_section.get("params", {}).copy()
        params.setdefault("max_iter", 5000)
        params.setdefault("solver", "saga")
        params.setdefault("class_weight", "balanced")
        # sklearn >= 1.9 deprecates explicit `penalty`; use `l1_ratio` to select penalty type.
        params.pop("penalty", None)

        if self.optimize:
            logger.info("Optimizing %s hyperparameters with Optuna (%d trials)", self.name, self.optuna_trials)
            params = self._optimize_params(X, y, params)

        self.model = self._build_estimator(params)
        self.model.fit(X, y)
        logger.info("Trained %s on %d samples, %d features", self.name, X.shape[0], X.shape[1])

    def _optimize_params(self, X: np.ndarray, y: np.ndarray, base_params: dict[str, Any]) -> dict[str, Any]:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.project_config.project.seed)

        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            trial_params = {**base_params, "C": C, "l1_ratio": l1_ratio}
            estimator = self._build_estimator(trial_params)

            scores = []
            for tr_idx, val_idx in cv.split(X, y):
                estimator.fit(X[tr_idx], y[tr_idx])
                val_proba = estimator.predict_proba(X[val_idx])[:, 1]
                scores.append(roc_auc_score(y[val_idx], val_proba))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.project_config.project.seed))
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

        best = {**base_params, **study.best_params}
        logger.info("Best params for %s: %s", self.name, best)
        return best

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float] | None:
        if self.model is None or not self.feature_names:
            return None
        return {name: float(coef) for name, coef in zip(self.feature_names, self.model.coef_[0])}


ModelRegistry.register("elastic_net", ElasticNetModel)
