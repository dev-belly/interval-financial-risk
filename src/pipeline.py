"""End-to-end experiment pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import load_data
from src.evaluation.ablation import ablation_study
from src.evaluation.conformal import run_conformal_experiment
from src.evaluation.double_ml import double_ml_partial_linear
from src.evaluation.metrics import compute_grouped_metrics, compute_metrics
from src.evaluation.permutation_test import permutation_importance_test
from src.evaluation.rolling_validator import RollingWindowValidator
from src.features.interval_features import build_interval_features
from src.features.pipeline import FeaturePipeline
from src.models.base import ModelRegistry, RiskModel
from src.visualization import html_report as html_report_mod
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

        # Build trailing-window features once on the chronological panel.  A
        # test fold can then use observations from the immediately preceding
        # training/validation quarters without fitting its imputer/scaler on
        # any future row.
        prepared_df = build_interval_features(df, self.config)

        # 2. Rolling validation
        validator = RollingWindowValidator(self.config, prepared_df["report_date"])

        all_results: dict[str, list[dict[str, Any]]] = {}
        rolling_rows = []
        validation_rows = []
        split_rows = []
        fold_idx = 0

        for train_idx, val_idx, test_idx in validator.split(prepared_df):
            fold_idx += 1
            train_df = prepared_df.iloc[train_idx].copy()
            test_df = prepared_df.iloc[test_idx].copy()
            val_df = prepared_df.iloc[val_idx].copy()
            split_rows.append(
                {
                    "fold": fold_idx,
                    **{
                        f"{name}_{field}": value
                        for name, frame in [
                            ("train", train_df),
                            ("validation", val_df),
                            ("test", test_df),
                        ]
                        for field, value in [
                            ("start", str(frame.report_date.min().date())),
                            ("end", str(frame.report_date.max().date())),
                            ("n", len(frame)),
                            ("positives", int(frame.risk_label.sum())),
                        ]
                    },
                }
            )

            # Feature pipeline fit on training data
            feature_pipe = FeaturePipeline(self.config)
            X_train, y_train, train_meta = feature_pipe.fit_transform(
                train_df, engineer_features=False
            )
            X_test, y_test, test_meta = feature_pipe.transform(test_df, engineer_features=False)
            feature_names = feature_pipe.get_feature_names()
            X_val, y_val, _ = feature_pipe.transform(val_df, engineer_features=False)

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
                X_va = (
                    self._select_point_features(X_val, feature_names)
                    if model_cfg.feature_set == "point_only"
                    else X_val
                )
                validation_rows.append(
                    {
                        "fold": fold_idx,
                        "model": model_cfg.name,
                        **compute_metrics(y_val, risk_model.predict_proba(X_va)),
                    }
                )
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
                    "company_id": test_df["company_id"].to_numpy(),
                    "report_date": test_df["report_date"].to_numpy(),
                    "metrics": metrics,
                    "feature_names": fn,
                    "train_end_date": train_df["report_date"].max(),
                    "test_start_date": test_df["report_date"].min(),
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
        if rolling_df.empty:
            raise ValueError(
                "Rolling validation produced no usable folds; increase the date range "
                "or reduce validation.min_train_samples"
            )
        self.results["rolling"] = rolling_df
        self.results["splits"] = pd.DataFrame(split_rows)
        self.results["validation"] = pd.DataFrame(validation_rows)
        self.results["data_summary"] = {
            "source": "synthetic" if self.config.data.use_synthetic else "local data",
            "rows": len(df),
            "companies": int(df.company_id.nunique()),
            "quarters": int(df.report_date.nunique()),
            "positive_rate": float(df.risk_label.mean()),
            "date_start": str(df.report_date.min().date()),
            "date_end": str(df.report_date.max().date()),
            "data_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(df, index=True).values.tobytes()
            ).hexdigest(),
        }

        # Aggregate metrics across folds
        summary_rows = []
        for model_name, folds in all_results.items():
            metrics_list = [f["metrics"] for f in folds]
            avg_metrics = {
                k: float(np.nanmean([m[k] for m in metrics_list])) for k in metrics_list[0]
            }
            std_metrics = {
                f"{k}_std": float(np.nanstd([m[k] for m in metrics_list])) for k in metrics_list[0]
            }
            summary_rows.append({"model": model_name, **avg_metrics, **std_metrics})
        summary_df = pd.DataFrame(summary_rows)
        self.results["summary"] = summary_df

        logger.info("\n%s", summary_df.to_string(index=False))

        # 3. Visualizations
        figures_dir = Path(self.config.output.figures_dir)
        plots.plot_roc_curves(all_results, figures_dir / "roc_curves.png")
        plots.plot_calibration_curves(
            all_results,
            figures_dir / "calibration_curves.png",
            n_bins=self.config.evaluation.calibration_bins,
        )
        plots.plot_rolling_metrics(rolling_df, figures_dir / "rolling_metrics.png")

        # 4. Best model: permutation + ablation on the last fold
        best_model_name = self._select_diagnostic_model()
        self.results["diagnostic_model"] = best_model_name
        logger.info("Diagnostic model selected by validation AUC: %s", best_model_name)

        last_fold = all_results[best_model_name][-1]
        best_model_key = last_fold["model_key"]
        best_model_cfg = self.config.models[best_model_key]

        # Refit using only observations available before the final test fold.
        # Permutation, ablation and grouped metrics are then evaluated on that
        # held-out fold rather than optimistically on the fitting sample.
        diagnostic_train = prepared_df[
            prepared_df["report_date"] < last_fold["test_start_date"]
        ].copy()
        diagnostic_test = prepared_df[
            (prepared_df["report_date"] >= last_fold["test_start_date"])
            & (prepared_df["report_date"] <= last_fold["test_end_date"])
        ].copy()
        feature_pipe_full = FeaturePipeline(self.config)
        X_train_diag, y_train_diag, _ = feature_pipe_full.fit_transform(
            diagnostic_train, engineer_features=False
        )
        X_test_diag, y_test_diag, _ = feature_pipe_full.transform(
            diagnostic_test, engineer_features=False
        )
        if best_model_cfg.feature_set == "point_only":
            X_train_diag = self._select_point_features(
                X_train_diag, feature_pipe_full.get_feature_names()
            )
            X_test_diag = self._select_point_features(
                X_test_diag, feature_pipe_full.get_feature_names()
            )
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
        best_model.fit(X_train_diag, y_train_diag)

        # Feature importance
        importances = best_model.get_feature_importance()
        if importances:
            plots.plot_feature_importance(importances, figures_dir / "feature_importance.png")
            self.results["feature_importance"] = importances

        # Permutation test
        perm_results = permutation_importance_test(
            best_model,
            X_test_diag,
            y_test_diag,
            fn_full,
            n_repeats=self.config.evaluation.permutation_n_repeats,
            random_state=self.config.project.seed,
        )
        plots.plot_permutation_importance(perm_results, figures_dir / "permutation_importance.png")
        self.results["permutation"] = perm_results

        # Fix an interval-capable model for the ablation; the selected winner
        # may be point-only, in which case removing interval columns is a no-op.
        ablation_cfg = self.config.models.get("logistic_interval", best_model_cfg)
        ablation_class = ModelRegistry.get(ablation_cfg.model_type)
        self.results["ablation_model"] = ablation_cfg.name
        X_ab_train, _, _ = feature_pipe_full.transform(diagnostic_train, engineer_features=False)
        X_ab_test, _, _ = feature_pipe_full.transform(diagnostic_test, engineer_features=False)
        ablation_names = feature_pipe_full.get_feature_names()
        if ablation_cfg.feature_set == "point_only":
            X_ab_train = self._select_point_features(X_ab_train, ablation_names)
            X_ab_test = self._select_point_features(X_ab_test, ablation_names)
            ablation_names = self._point_feature_names(ablation_names)

        def model_builder() -> RiskModel:
            m = ablation_class(
                ablation_cfg.model_dump(),
                ablation_cfg.feature_set,
                ablation_cfg.name,
                self.config,
            )
            m.feature_names = ablation_names
            return m

        ablation_df = ablation_study(
            model_builder,
            X_ab_train,
            y_train_diag,
            X_ab_test,
            y_test_diag,
            ablation_names,
            self.config,
        )
        plots.plot_ablation_study(ablation_df, figures_dir / "ablation_study.png")
        self.results["ablation"] = ablation_df

        # 5. Conformal prediction -- valid uncertainty intervals (HIGHLIGHT)
        try:
            # Fix the conformal model before observing any validation/test scores.
            conformal_cfg = self.config.models.get(
                "logistic_interval", next(iter(self.config.models.values()))
            )
            self.results["conformal_model"] = conformal_cfg.name
            conformal_res = run_conformal_experiment(
                ModelRegistry.get(conformal_cfg.model_type),
                conformal_cfg,
                feature_pipe_full,
                prepared_df[prepared_df["report_date"] <= last_fold["test_end_date"]],
                self.config,
                alpha=0.1,
                features_precomputed=True,
            )
            self.results["conformal"] = conformal_res
            plots.plot_conformal_coverage(
                conformal_res.coverage_curve, figures_dir / "conformal_coverage.png"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Conformal prediction skipped: %s", exc)

        # 6. Double ML -- orthogonalized interval-feature effects (HIGHLIGHT)
        try:
            confounders = diagnostic_train[["industry"]].copy()
            dm_df = double_ml_partial_linear(
                X_train_diag,
                confounders,
                y_train_diag,
                fn_full,
                n_folds=5,
                random_state=self.config.project.seed,
            )
            self.results["double_ml"] = dm_df
            plots.plot_double_ml(dm_df, figures_dir / "double_ml_effects.png")
        except Exception as exc:  # pragma: no cover
            logger.warning("Double ML skipped: %s", exc)

        # keep per-fold predictions for the HTML ROC
        self.results["all_results"] = all_results

        # Grouped analysis
        if self.config.groups.by_industry and "industry" in diagnostic_test.columns:
            industry_groups = diagnostic_test["industry"].values
            self.results["grouped_industry"] = compute_grouped_metrics(
                y_test_diag, best_model.predict_proba(X_test_diag), industry_groups
            )

        self.results["config_snapshot"] = json.loads(self.config.model_dump_json())
        for section, keys in {
            "data": ["raw_path", "processed_path"],
            "output": ["figures_dir", "reports_dir", "models_dir"],
        }.items():
            for key in keys:
                path = Path(self.results["config_snapshot"][section][key])
                self.results["config_snapshot"][section][key] = (
                    str(path.relative_to(self.config.project_root))
                    if path.is_relative_to(self.config.project_root)
                    else str(path)
                )
        self.results["config_snapshot"].pop("project_root", None)
        # 7. Save extra artifacts + interactive HTML report
        reports_dir = Path(self.config.output.reports_dir)
        if "conformal" in self.results:
            self.results["conformal"].coverage_curve.to_csv(
                reports_dir / "conformal_coverage.csv", index=False
            )
        if "double_ml" in self.results:
            self.results["double_ml"].to_csv(reports_dir / "double_ml_effects.csv", index=False)
        try:
            html_report_mod.build_html_report(self.results, self.config, figures_dir, reports_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning("HTML report skipped: %s", exc)

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

    def _select_diagnostic_model(self) -> str:
        """Choose using validation predictions only, never the test summary."""
        selection = self.results["validation"].groupby("model", as_index=False)["auc"].mean()
        return self._select_best_model(selection)

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
        self.results["validation"].to_csv(reports_dir / "validation_metrics.csv", index=False)
        self.results["splits"].to_csv(reports_dir / "fold_splits.csv", index=False)
        predictions = [
            pd.DataFrame(
                {
                    "fold": f["fold"],
                    "model": model,
                    "company_id": f["company_id"],
                    "report_date": f["report_date"],
                    "y_true": f["y_true"],
                    "y_proba": f["y_proba"],
                }
            )
            for model, folds in self.results["all_results"].items()
            for f in folds
        ]
        pd.concat(predictions, ignore_index=True).to_csv(
            reports_dir / "predictions.csv", index=False
        )
        manifest = {
            "data": self.results["data_summary"],
            "diagnostic_model": self.results["diagnostic_model"],
            "ablation_model": self.results["ablation_model"],
            "conformal_model": self.results.get("conformal_model"),
            "selection": "mean validation AUC, final test excluded",
            "config": self.results["config_snapshot"],
            "python": platform.python_version(),
            "dependencies": {
                pkg: version(pkg)
                for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "plotly"]
            },
        }
        sources = sorted(self.config.project_root.glob("src/**/*.py"))
        manifest["source_sha256"] = hashlib.sha256(
            b"".join(
                str(p.relative_to(self.config.project_root)).encode() + p.read_bytes()
                for p in sources
            )
        ).hexdigest()
        (reports_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rolling_df.to_csv(reports_dir / "rolling_metrics.csv", index=False)
        ablation_df.to_csv(reports_dir / "ablation_study.csv", index=False)

        with open(reports_dir / "permutation_importance.json", "w", encoding="utf-8") as fh:
            json.dump(self.results["permutation"], fh, indent=2, ensure_ascii=False)

        joblib.dump(best_model.model, models_dir / "best_model.joblib")
        joblib.dump(feature_pipe._preprocessor, models_dir / "preprocessor.joblib")
        with open(models_dir / "feature_names.pkl", "wb") as fh:
            pickle.dump(feature_pipe.get_feature_names(), fh)

        logger.info("Saved artifacts to %s and %s", reports_dir, models_dir)
