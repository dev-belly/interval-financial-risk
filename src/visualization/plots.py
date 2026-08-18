"""Visualization utilities for experiment results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)

# Use a dark-friendly palette
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
# Render Chinese titles correctly on macOS / CI
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False

PALETTE = sns.color_palette("tab10")


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_roc_curves(
    results_by_model: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Overlay ROC curves for each model across rolling windows."""

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (model_name, folds) in enumerate(results_by_model.items()):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for fold in folds:
            fpr, tpr, _ = roc_curve(fold["y_true"], fold["y_proba"])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={mean_auc:.3f})", color=PALETTE[idx])

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mean ROC Curves")
    ax.legend(loc="lower right")
    _savefig(fig, output_path)


def plot_calibration_curves(
    results_by_model: dict[str, list[dict[str, Any]]],
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot calibration curves for each model."""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

    for idx, (model_name, folds) in enumerate(results_by_model.items()):
        all_y = np.concatenate([fold["y_true"] for fold in folds])
        all_p = np.concatenate([fold["y_proba"] for fold in folds])
        prob_true, prob_pred = _calibration_data(all_y, all_p, n_bins)
        ax.plot(prob_pred, prob_true, marker="o", label=model_name, color=PALETTE[idx])

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    _savefig(fig, output_path)


def _calibration_data(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_proba)
    bin_sums = np.bincount(binids, weights=y_proba, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_count = np.bincount(binids, minlength=n_bins)
    nonzero = bin_count != 0
    prob_true = bin_true[nonzero] / bin_count[nonzero]
    prob_pred = bin_sums[nonzero] / bin_count[nonzero]
    return prob_true, prob_pred


def plot_rolling_metrics(
    rolling_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Line plot of AUC and PR-AUC across rolling test windows."""

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    metrics = [("auc", "AUC"), ("pr_auc", "PR-AUC")]

    for ax, (metric, title) in zip(axes, metrics):
        for model_name in rolling_results["model"].unique():
            sub = rolling_results[rolling_results["model"] == model_name]
            ax.plot(sub["test_end_date"], sub[metric], marker="o", label=model_name)
        ax.set_ylabel(title)
        ax.set_title(f"Rolling {title} Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Test Window End Date")
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    _savefig(fig, output_path)


def plot_feature_importance(
    importances: dict[str, float],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Horizontal bar plot of feature importances."""

    df = pd.Series(importances).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    df.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    _savefig(fig, output_path)


def plot_permutation_importance(
    perm_results: dict[str, dict[str, float]],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot mean AUC drop from permutation test."""

    df = pd.DataFrame.from_dict(perm_results, orient="index")
    df = df.sort_values("mean_drop", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ax.barh(df.index, df["mean_drop"], xerr=df["std_drop"], color="coral")
    ax.set_xlabel("Mean AUC Drop")
    ax.set_title(f"Permutation Importance (Top {top_n})")
    _savefig(fig, output_path)


def plot_ablation_study(
    ablation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar plot of AUC delta from ablation study."""

    df = ablation_df.copy()
    if "auc_delta" not in df.columns:
        logger.warning("No auc_delta column in ablation results; skipping plot")
        return

    df = df.set_index("ablation")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if v >= 0 else "red" for v in df["auc_delta"]]
    ax.bar(df.index, df["auc_delta"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("AUC Delta vs. Full Model")
    ax.set_title("Ablation Study: AUC Impact of Feature Groups")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    _savefig(fig, output_path)


def plot_conformal_coverage(
    coverage_curve: pd.DataFrame,
    output_path: Path,
) -> None:
    """Nominal vs empirical coverage for split conformal prediction."""

    if coverage_curve is None or len(coverage_curve) == 0:
        logger.warning("Empty conformal coverage curve; skipping plot")
        return

    df = coverage_curve.copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(df["nominal_coverage"], df["empirical_coverage"], "o-", color="darkorange",
            label="经验覆盖率")
    ax.plot([0, 1], [0, 1], "k--", label="理想对角线")
    ax.set_xlabel("名义覆盖率 (1 - alpha)")
    ax.set_ylabel("经验覆盖率")
    ax.set_title("保形预测: 覆盖率校准 (有效性检验)")
    ax.legend()
    _savefig(fig, output_path)


def plot_double_ml(
    dm_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Orthogonalized interval-feature effects with 95% CI error bars."""

    if dm_df is None or len(dm_df) == 0:
        logger.warning("Empty double-ml results; skipping plot")
        return

    df = dm_df.sort_values("theta_orthogonalized", key=lambda s: s.abs(), ascending=False).head(15).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    theta = df["theta_orthogonalized"].values
    err_lo = theta - df["ci_low"].values
    err_hi = df["ci_high"].values - theta
    ax.barh(y, theta, xerr=[err_lo, err_hi], color="steelblue", alpha=0.85,
            error_kw={"ecolor": "black", "capsize": 3})
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("theta (正交化边际效应, 控制行业后)")
    ax.set_title("双机器学习: 区间特征的纯净边际效应")
    _savefig(fig, output_path)
