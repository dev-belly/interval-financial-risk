"""Gradient-boosted tree models: XGBoost, LightGBM, CatBoost."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

from src.config import Config
from src.models.base import ModelRegistry, RiskModel

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostModel(RiskModel):
    """XGBoost classifier with optional Optuna tuning."""

    def __init__(self, config_section: dict[str, Any], feature_set: str, name: str, project_config: Config):
        super().__init__(config_section, feature_set, name)
        self.project_config = project_config
        self.optimize = config_section.get("optimize", False)
        self.optuna_trials = config_section.get("optuna_trials", 50)
        self.feature_names: list[str] = []

    def _scale_pos_weight(self, y: np.ndarray) -> float:
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return float(n_neg / max(n_pos, 1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb

        params = self.config_section.get("params", {}).copy()
        params.setdefault("n_estimators", 300)
        params.setdefault("max_depth", 5)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.8)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.pop("use_label_encoder", None)
        params.setdefault("random_state", self.project_config.project.seed)
        params.setdefault("n_jobs", self.project_config.project.n_jobs)
        params["scale_pos_weight"] = self._scale_pos_weight(y)

        if self.optimize:
            params = self._optimize_params(X, y, params, xgb.XGBClassifier)

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y, verbose=False)
        logger.info("Trained %s on %d samples, %d features", self.name, X.shape[0], X.shape[1])

    def _optimize_params(self, X: np.ndarray, y: np.ndarray, base_params: dict[str, Any], estimator_cls: Any) -> dict[str, Any]:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.project_config.project.seed)
        base_copy = {k: v for k, v in base_params.items() if k not in {"max_depth", "learning_rate", "subsample", "colsample_bytree", "n_estimators"}}

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                **base_copy,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            }
            estimator = estimator_cls(**trial_params)
            scores = []
            for tr_idx, val_idx in cv.split(X, y):
                estimator.fit(X[tr_idx], y[tr_idx])
                val_proba = estimator.predict_proba(X[val_idx])[:, 1]
                scores.append(roc_auc_score(y[val_idx], val_proba))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.project_config.project.seed))
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

        best = {**base_copy, **study.best_params}
        logger.info("Best params for %s: %s", self.name, best)
        return best

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float] | None:
        if self.model is None or not self.feature_names:
            return None
        return {name: float(imp) for name, imp in zip(self.feature_names, self.model.feature_importances_)}


class LightGBMModel(RiskModel):
    """LightGBM classifier with optional Optuna tuning."""

    def __init__(self, config_section: dict[str, Any], feature_set: str, name: str, project_config: Config):
        super().__init__(config_section, feature_set, name)
        self.project_config = project_config
        self.optimize = config_section.get("optimize", False)
        self.optuna_trials = config_section.get("optuna_trials", 50)
        self.feature_names: list[str] = []

    def _scale_pos_weight(self, y: np.ndarray) -> float:
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return float(n_neg / max(n_pos, 1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import lightgbm as lgb

        params = self.config_section.get("params", {}).copy()
        params.setdefault("n_estimators", 300)
        params.setdefault("max_depth", 5)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.8)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("objective", "binary")
        params.setdefault("random_state", self.project_config.project.seed)
        params.setdefault("n_jobs", self.project_config.project.n_jobs)
        params.setdefault("verbose", -1)
        params["class_weight"] = "balanced"

        if self.optimize:
            params = self._optimize_params(X, y, params, lgb.LGBMClassifier)

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y)
        logger.info("Trained %s on %d samples, %d features", self.name, X.shape[0], X.shape[1])

    def _optimize_params(self, X: np.ndarray, y: np.ndarray, base_params: dict[str, Any], estimator_cls: Any) -> dict[str, Any]:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.project_config.project.seed)
        base_copy = {k: v for k, v in base_params.items() if k not in {"max_depth", "learning_rate", "subsample", "colsample_bytree", "n_estimators", "num_leaves"}}

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                **base_copy,
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            }
            estimator = estimator_cls(**trial_params)
            scores = []
            for tr_idx, val_idx in cv.split(X, y):
                estimator.fit(X[tr_idx], y[tr_idx])
                val_proba = estimator.predict_proba(X[val_idx])[:, 1]
                scores.append(roc_auc_score(y[val_idx], val_proba))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.project_config.project.seed))
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

        best = {**base_copy, **study.best_params}
        logger.info("Best params for %s: %s", self.name, best)
        return best

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float] | None:
        if self.model is None or not self.feature_names:
            return None
        return {name: float(imp) for name, imp in zip(self.feature_names, self.model.feature_importances_)}


ModelRegistry.register("xgboost", XGBoostModel)
ModelRegistry.register("lightgbm", LightGBMModel)
