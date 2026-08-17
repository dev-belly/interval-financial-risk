"""End-to-end experiment pipeline orchestration."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.data.loader import load_data
from src.evaluation.ablation import ablation_study
from src.evaluation.metrics import compute_grouped_metrics, compute_metrics
from src.evaluation.permutation_test import permutation_importance_test
from src.evaluation.rolling_validator import RollingWindowValidator
from src.features.pipeline import FeaturePipeline
from src.models.base import ModelRegistry, RiskModel
from src.visualization import plots

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Run the complete interval-feature risk identification experiment."""

    def __init__(self, config: Config):
        self.config = config
        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline and return aggregated results."""

        logger.info("Starting experiment: %s", self.config.project.name)

        # 1. Load data
        df = load_data(self.config)
        logger.info("Loaded data: %d rows, %d columns", len(df), len(df.columns))

        # 2. Rolling validation
        validator = RollingWindowValidator(self.config, df["report_date"])

        all_results: dict[str, list[dict[str, Any]]] = {}
        rolling_rows = []
        fold_idx = 0

        for train_idx, val_idx, test_idx in validator.split(df):
            fold_idx += 1
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # Feature pipeline fit on training data
            feature_pipe = FeaturePipeline(self.config)
            X_train, y_train, train_meta = feature_pipe.fit_transform(train_df)
            X_test, y_test, test_meta = feature_pipe.transform(test_df)
            feature_names = feature_pipe.get_feature_names()

            for model_key, model_cfg in self.config.models.items():
                model_class = ModelRegistry.get(model_cfg.model_type)
                risk_model = model_class(
                    model_cfg.model_dump(),
                    model_cfg.feature_set,
                    model_cfg.name,
                    self.config,
                )

                # Select feature set
                if model_cfg.feature_set == "point_only":
                    X_tr = self._select_point_features(X_train, feature_names)
                    X_te = self._select_point_features(X_test, feature_names)
                    fn = self._point_feature_names(feature_names)
                else:
                    X_tr, X_te, fn = X_train, X_test, feature_names

                risk_model.feature_names = fn
                risk_model.fit(X_tr, y_train)
                y_proba = risk_model.predict_proba(X_te)
                y_pred = risk_model.predict(X_te)

                metrics = compute_metrics(y_test, y_proba, y_pred)

                fold_result = {
                    "fold": fold_idx,
                    "model": model_cfg.name,
                    "model_key": model_key,
                    "y_true": y_test,
                    "y_proba": y_proba,
                    "y_pred": y_pred,
                    "metrics": metrics,
                    "feature_names": fn,
                    "test_end_date": test_df["report_date"].max(),
                }
                all_results.setdefault(model_cfg.name, []).append(fold_result)

                rolling_rows.append(
                    {
                        "fold": fold_idx,
                        "model": model_cfg.name,
                        "test_end_date": test_df["report_date"].max(),
                        **metrics,
                    }
                )

        rolling_df = pd.DataFrame(rolling_rows)
        self.results["rolling"] = rolling_df

        # Aggregate metrics across folds
        summary_rows = []
        for model_name, folds in all_results.items():
            metrics_list = [f["metrics"] for f in folds]
            avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            std_metrics = {f"{k}_std": float(np.std([m[k] for m in metrics_list])) for k in metrics_list[0]}
            summary_rows.append({"model": model_name, **avg_metrics, **std_metrics})
        summary_df = pd.DataFrame(summary_rows)
        self.results["summary"] = summary_df

        logger.info("\n%s", summary_df.to_string(index=False))

        # 3. Visualizations
        figures_dir = Path(self.config.output.figures_dir)
        plots.plot_roc_curves(all_results, figures_dir / "roc_curves.png")
        plots.plot_calibration_curves(
            all_results, figures_dir / "calibration_curves.png", n_bins=self.config.evaluation.calibration_bins
        )
        plots.plot_rolling_metrics(rolling_df, figures_dir / "rolling_metrics.png")

        # 4. Best model: permutation + ablation on the last fold
        best_model_name = self._select_best_model(summary_df)
        logger.info("Best model by AUC: %s", best_model_name)

        last_fold = all_results[best_model_name][-1]
        best_model_key = last_fold["model_key"]
        best_model_cfg = self.config.models[best_model_key]

        # Refit best model on all historical data for diagnostics
        feature_pipe_full = FeaturePipeline(self.config)
        full_train = df[df["report_date"] <= last_fold["test_end_date"]].copy()
        X_full, y_full, _ = feature_pipe_full.fit_transform(full_train)
        if best_model_cfg.feature_set == "point_only":
            X_full = self._select_point_features(X_full, feature_pipe_full.get_feature_names())
            fn_full = self._point_feature_names(feature_pipe_full.get_feature_names())
        else:
            fn_full = feature_pipe_full.get_feature_names()

        best_model_class = ModelRegistry.get(best_model_cfg.model_type)
        best_model = best_model_class(
            best_model_cfg.model_dump(),
            best_model_cfg.feature_set,
            best_model_cfg.name,
            self.config,
        )
        best_model.feature_names = fn_full
        best_model.fit(X_full, y_full)

        # Feature importance
        importances = best_model.get_feature_importance()
        if importances:
            plots.plot_feature_importance(importances, figures_dir / "feature_importance.png")
            self.results["feature_importance"] = importances

        # Permutation test
        perm_results = permutation_importance_test(
            best_model,
            X_full,
            y_full,
            fn_full,
            n_repeats=self.config.evaluation.permutation_n_repeats,
            random_state=self.config.project.seed,
        )
        plots.plot_permutation_importance(perm_results, figures_dir / "permutation_importance.png")
        self.results["permutation"] = perm_results

        # Ablation study
        def model_builder() -> RiskModel:
            m = best_model_class(
                best_model_cfg.model_dump(),
                best_model_cfg.feature_set,
                best_model_cfg.name,
                self.config,
            )
            m.feature_names = fn_full
            return m

        ablation_df = ablation_study(model_builder, X_full, y_full, fn_full, self.config)
        plots.plot_ablation_study(ablation_df, figures_dir / "ablation_study.png")
        self.results["ablation"] = ablation_df

        # Grouped analysis
        if self.config.groups.by_industry and "industry" in full_train.columns:
            industry_groups = full_train["industry"].values
            self.results["grouped_industry"] = compute_grouped_metrics(
                y_full, best_model.predict_proba(X_full), industry_groups
            )

        # 5. Save artifacts
        self._save_artifacts(summary_df, rolling_df, ablation_df, best_model, feature_pipe_full)

        return self.results

    def _select_point_features(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        point_features = self.config.features.point_features
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        indices = [name_to_idx[name] for name in point_features if name in name_to_idx]
        return X[:, indices]

    def _point_feature_names(self, feature_names: list[str]) -> list[str]:
        point_features = self.config.features.point_features
        return [name for name in point_features if name in feature_names]

    def _select_best_model(self, summary_df: pd.DataFrame) -> str:
        best = summary_df.loc[summary_df["auc"].idxmax()]
        return str(best["model"])

    def _save_artifacts(
        self,
        summary_df: pd.DataFrame,
        rolling_df: pd.DataFrame,
        ablation_df: pd.DataFrame,
        best_model: RiskModel,
        feature_pipe: FeaturePipeline,
    ) -> None:
        reports_dir = Path(self.config.output.reports_dir)
        models_dir = Path(self.config.output.models_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(reports_dir / "model_summary.csv", index=False)
        rolling_df.to_csv(reports_dir / "rolling_metrics.csv", index=False)
        ablation_df.to_csv(reports_dir / "ablation_study.csv", index=False)

        with open(reports_dir / "permutation_importance.json", "w", encoding="utf-8") as fh:
            json.dump(self.results["permutation"], fh, indent=2, ensure_ascii=False)

        joblib.dump(best_model.model, models_dir / "best_model.joblib")
        joblib.dump(feature_pipe._preprocessor, models_dir / "preprocessor.joblib")
        with open(models_dir / "feature_names.pkl", "wb") as fh:
            pickle.dump(feature_pipe.get_feature_names(), fh)

        logger.info("Saved artifacts to %s and %s", reports_dir, models_dir)
