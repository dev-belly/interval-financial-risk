"""Conformal prediction for statistically valid risk-probability intervals.

Standard ML models output a single point probability p in [0, 1]. In financial
risk work that number alone is dangerous: a 0.76 default probability and a 0.74
probability should not be treated as confidently different. Split conformal
prediction turns the point estimate into a *valid* interval [p - eps, p + eps]
whose marginal coverage P(y in interval) >= 1 - alpha is guaranteed by
finite-sample theory (Vovk et al., 2005; Angelopoulos & Bates, 2021), with no
distributional assumptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    alpha: float
    eps: float
    test_coverage: float
    test_avg_width: float
    calibration_fraction: float
    coverage_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def _conformal_radius(y_cal: np.ndarray, p_cal: np.ndarray, alpha: float, n_cal: int) -> float:
    """Conformal radius for marginal coverage >= 1 - alpha.

    Nonconformity score s = |y - p| lives in [0, 1]. Finite-sample quantile
    level uses the standard (n+1) correction for split conformal.
    """
    scores = np.abs(y_cal - p_cal)
    level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    level = float(min(max(level, 0.0), 1.0))
    return float(np.quantile(scores, level))


def conformal_interval(p_test: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.clip(p_test - eps, 0.0, 1.0)
    upper = np.clip(p_test + eps, 0.0, 1.0)
    return lower, upper


def empirical_coverage(y_test: np.ndarray, p_test: np.ndarray, eps: float) -> float:
    lower, upper = conformal_interval(p_test, eps)
    covered = (y_test >= lower - 1e-9) & (y_test <= upper + 1e-9)
    return float(np.mean(covered))


def coverage_curve(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    alphas: np.ndarray,
    n_cal: int,
) -> pd.DataFrame:
    """Empirical vs nominal coverage across a grid of alpha levels."""
    rows = []
    for a in alphas:
        eps = _conformal_radius(y_cal, p_cal, a, n_cal)
        cov = empirical_coverage(y_test, p_test, eps)
        lower, upper = conformal_interval(p_test, eps)
        width = float(np.mean(upper - lower))
        rows.append(
            {
                "alpha": float(a),
                "nominal_coverage": 1.0 - a,
                "empirical_coverage": cov,
                "avg_interval_width": width,
            }
        )
    return pd.DataFrame(rows)


def run_conformal_experiment(
    model_class,
    model_cfg,
    feature_pipe,
    full_train: pd.DataFrame,
    config,
    alpha: float = 0.1,
    cal_fraction: float = 0.2,
) -> ConformalResult:
    """Run a properly held-out split-conformal experiment.

    full_train is split into train / calibration / test by time order so that
    the conformal model never sees calibration or test data (valid coverage).
    """
    from src.features.pipeline import FeaturePipeline
    from src.models.base import RiskModel

    n = len(full_train)
    i1 = int(n * (1.0 - 2 * cal_fraction))
    i2 = int(n * (1.0 - cal_fraction))
    train_part = full_train.iloc[:i1].copy()
    cal_part = full_train.iloc[i1:i2].copy()
    test_part = full_train.iloc[i2:].copy()

    fp = FeaturePipeline(config)
    X_tr, y_tr, _ = fp.fit_transform(train_part)
    if model_cfg.feature_set == "point_only":
        point = config.features.point_features
        idx = [i for i, nm in enumerate(fp.get_feature_names()) if nm in point]
        X_tr = X_tr[:, idx]

    conf_model: RiskModel = model_class(
        model_cfg.model_dump(), model_cfg.feature_set, model_cfg.name, config
    )
    conf_model.fit(X_tr, y_tr)

    def _proba(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xp, yp, _ = fp.transform(part)
        if model_cfg.feature_set == "point_only":
            Xp = Xp[:, idx]
        return conf_model.predict_proba(Xp), yp

    p_cal, y_cal = _proba(cal_part)
    p_test, y_test = _proba(test_part)

    n_cal = len(y_cal)
    eps = _conformal_radius(y_cal, p_cal, alpha, n_cal)
    test_cov = empirical_coverage(y_test, p_test, eps)
    lower, upper = conformal_interval(p_test, eps)
    test_width = float(np.mean(upper - lower))

    alphas = np.linspace(0.05, 0.4, 8)
    curve = coverage_curve(y_cal, p_cal, y_test, p_test, alphas, n_cal)

    logger.info(
        "Conformal (alpha=%.2f): eps=%.3f, test coverage=%.3f, avg width=%.3f",
        alpha,
        eps,
        test_cov,
        test_width,
    )
    return ConformalResult(
        alpha=alpha,
        eps=eps,
        test_coverage=test_cov,
        test_avg_width=test_width,
        calibration_fraction=cal_fraction,
        coverage_curve=curve,
    )
