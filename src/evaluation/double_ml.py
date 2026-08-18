"""Double / Debiased Machine Learning for orthogonalized effect estimation.

A plain logistic regression of risk on interval features is confounded:
industry, scale and business cycle drive both the financial ratios and the
risk label, so naive coefficients mix the interval-feature signal with
confounder bias. Double ML (Chernozhukov et al., 2018) removes this bias via
orthogonalization with cross-fitting:

    y = theta * X + g(W) + u
    X = m(W) + v

We regress the *residuals* r_y = y - g(W) on r_x = X - m(W). Because g and m
are estimated by flexible ML on held-out folds, theta is root-n consistent and
the confounder bias vanishes at rate 1/sqrt(n) instead of staying O(1). The
result is the *clean* marginal contribution of each interval feature after
controlling for industry / scale / cycle -- the number the research question
actually wants.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def _encode_confounds(W: pd.DataFrame) -> tuple[Any, list[str]]:
    cat_cols = W.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = W.select_dtypes(include=[np.number]).columns.tolist()
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    ct = ColumnTransformer(transformers)
    W_enc = ct.fit_transform(W)
    if hasattr(W_enc, "toarray"):
        W_enc = W_enc.toarray()
    return W_enc, cat_cols + num_cols


def double_ml_partial_linear(
    X: np.ndarray,
    W: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate orthogonalized effect theta of each X column on y, controlling W.

    Returns a DataFrame with the debiased coefficient, standard error, 95% CI,
    and a naive (confounded) logistic coefficient for comparison.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    W_enc, _ = _encode_confounds(W)
    n, d = X.shape

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    r_y = np.zeros(n)
    r_x = np.zeros_like(X)

    for train_idx, test_idx in kf.split(X):
        g = RandomForestClassifier(
            n_estimators=120, max_depth=6, n_jobs=1, random_state=random_state
        )
        g.fit(W_enc[train_idx], y[train_idx])
        r_y[test_idx] = y[test_idx] - g.predict_proba(W_enc[test_idx])[:, 1]

        for j in range(d):
            m = RandomForestRegressor(
                n_estimators=120, max_depth=6, n_jobs=1, random_state=random_state
            )
            m.fit(W_enc[train_idx], X[train_idx, j])
            r_x[test_idx, j] = X[test_idx, j] - m.predict(W_enc[test_idx])

    # Final orthogonalized OLS: r_y ~ theta * r_x (+ intercept)
    Xd = np.column_stack([np.ones(n), r_x])
    beta, _, _, _ = np.linalg.lstsq(Xd, r_y, rcond=None)
    resid = r_y - Xd @ beta
    dof = n - Xd.shape[1]
    sigma2 = float(resid @ resid) / max(dof, 1)
    # Pseudo-inverse: interval features are highly collinear, the normal
    # equations matrix can be near-singular. pinv gives a stable (min-norm)
    # solution and finite standard errors instead of crashing.
    xtx_inv = np.linalg.pinv(Xd.T @ Xd)
    var_beta = np.clip(sigma2 * np.diag(xtx_inv), 0.0, None)
    se = np.sqrt(var_beta)
    theta = beta[1:]
    z = 1.959963984540054  # 95%
    ci_low = theta - z * se[1:]
    ci_high = theta + z * se[1:]

    # Naive benchmark: logistic regression of y on X and W together
    naive = LogisticRegression(max_iter=3000, class_weight="balanced")
    naive.fit(np.column_stack([X, W_enc]), y)
    naive_coef = naive.coef_[0][:d]

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "theta_orthogonalized": theta,
            "se": se[1:],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "naive_coef": naive_coef,
        }
    )
    out["significant"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    out["bias_reduction"] = out["naive_coef"] - out["theta_orthogonalized"]
    logger.info(
        "Double ML: %d features, %d significant after orthogonalization",
        d,
        int(out["significant"].sum()),
    )
    return out
