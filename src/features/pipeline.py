"""End-to-end feature pipeline: engineering + preprocessing."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.features.interval_features import build_interval_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Encapsulates feature engineering and sklearn preprocessing.

    The pipeline is fit on training data and can transform validation/test data
    without leakage.
    """

    def __init__(self, config: Config):
        self.config = config
        self.feature_names: list[str] = []
        self.numeric_features: list[str] = []
        self._preprocessor: ColumnTransformer | None = None

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select columns that should enter the model."""

        exclude = {"company_id", "report_date", "industry", "risk_label"}
        feature_cols = [c for c in df.columns if c not in exclude]
        return df[feature_cols]

    def fit_transform(
        self, df: pd.DataFrame, *, engineer_features: bool = True
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Build features, fit preprocessor, and transform data.

        Returns
        -------
        X, y, metadata
        """

        if engineer_features:
            df = build_interval_features(df, self.config)
        feature_df = self._select_features(df)
        self.numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = self.numeric_features.copy()

        X = feature_df[self.numeric_features]
        y = df["risk_label"].values if "risk_label" in df.columns else np.array([])

        self._preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy=self.config.features.impute_strategy)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_features,
                )
            ],
            remainder="drop",
        )

        X_processed = self._preprocessor.fit_transform(X)

        metadata = {
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "industry": df.get("industry"),
            "report_date": df.get("report_date"),
            "company_id": df.get("company_id"),
        }

        logger.info(
            "FeaturePipeline fit_transform complete: n_samples=%d, n_features=%d",
            X_processed.shape[0],
            X_processed.shape[1],
        )
        return X_processed, y, metadata

    def transform(
        self, df: pd.DataFrame, *, engineer_features: bool = True
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Transform new data using the already-fit preprocessor."""

        if self._preprocessor is None:
            raise RuntimeError("Pipeline has not been fit yet. Call fit_transform first.")

        if engineer_features:
            df = build_interval_features(df, self.config)
        feature_df = self._select_features(df)

        # Align columns to training features
        aligned = pd.DataFrame(index=feature_df.index)
        for col in self.numeric_features:
            aligned[col] = feature_df[col] if col in feature_df.columns else np.nan

        X = aligned[self.numeric_features]
        y = df["risk_label"].values if "risk_label" in df.columns else np.array([])

        X_processed = self._preprocessor.transform(X)

        metadata = {
            "feature_names": self.feature_names,
            "industry": df.get("industry"),
            "report_date": df.get("report_date"),
            "company_id": df.get("company_id"),
        }

        return X_processed, y, metadata

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of numeric feature names."""
        return self.feature_names
