"""Interactive HTML report builder (Plotly).

Replaces the static matplotlib figures with a single self-contained, browser
openable dashboard: model comparison, ROC/PR, calibration, rolling stability,
feature importance, permutation & ablation diagnostics, plus the two new
highlight modules -- conformal prediction coverage and double-ML orthogonalized
effects. Plotly.js is loaded from CDN so the file is portable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_OK = True
except Exception:  # pragma: no cover
    _PLOTLY_OK = False


def _roc_traces(all_results: dict[str, list[dict[str, Any]]]):
    from sklearn.metrics import auc, roc_curve

    traces = []
    for name, folds in all_results.items():
        y_true = np.concatenate([f["y_true"] for f in folds])
        y_proba = np.concatenate([f["y_proba"] for f in folds])
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        a = auc(fpr, tpr)
        traces.append(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={a:.3f})")
        )
    traces.append(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    return traces


def build_html_report(
    results: dict[str, Any],
    config,
    figures_dir: Path,
    reports_dir: Path,
) -> Path | None:
    if not _PLOTLY_OK:
        logger.warning("plotly not available; skip interactive HTML report")
        return None

    parts: list[str] = []
    first = True

    def _add(fig: go.Figure, title: str):
        nonlocal first
        div = fig.to_html(
            full_html=False,
            include_plotlyjs=("cdn" if first else False),
            config={"displayModeBar": False},
        )
        first = False
        parts.append(f"<div class='card'><h2>{title}</h2>{div}</div>")

    # ---- Title ----
    parts.append(
        "<div class='card'><h1>区间型财务数据与企业风险识别 — 交互报告</h1>"
        f"<p>项目: {config.project.name} | 随机种子: {config.project.seed}</p></div>"
    )

    # ---- Model comparison table ----
    if "summary" in results:
        df = results["summary"].copy()
        styled = df.round(3).to_html(index=False, classes="tbl")
        parts.append(f"<div class='card'><h2>模型对比 (滚动验证均值)</h2>{styled}</div>")

    # ---- ROC ----
    if "all_results" in results:
        fig = go.Figure(_roc_traces(results["all_results"]))
        fig.update_layout(
            title="ROC 曲线 (跨所有滚动折)", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate", template="plotly_dark", height=420,
        )
        _add(fig, "ROC 曲线")

    # ---- Rolling metrics ----
    if "rolling" in results:
        rdf = results["rolling"]
        fig = make_subplots(rows=1, cols=1)
        for name in rdf["model"].unique():
            sub = rdf[rdf["model"] == name]
            fig.add_trace(
                go.Scatter(x=sub["test_end_date"].astype(str), y=sub["auc"],
                           mode="lines+markers", name=name)
            )
        fig.update_layout(title="滚动 AUC 稳定性", template="plotly_dark", height=380)
        _add(fig, "滚动验证稳定性")

    # ---- Feature importance ----
    if results.get("feature_importance"):
        fi = results["feature_importance"]
        items = sorted(fi.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
        names = [k for k, _ in items]
        vals = [float(v) for _, v in items]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h"))
        fig.update_layout(title="特征重要性 (最佳模型)", template="plotly_dark", height=420)
        _add(fig, "特征重要性")

    # ---- Permutation ----
    if results.get("permutation"):
        perm = results["permutation"]
        # permutation_importance_test returns one metrics dictionary per
        # feature.  Convert that mapping to plotting vectors explicitly.
        items = sorted(
            perm.items(), key=lambda item: item[1].get("mean_drop", 0.0), reverse=True
        )[:20]
        feats = [name for name, _ in items]
        drops = [values.get("mean_drop", 0.0) for _, values in items]
        fig = go.Figure(go.Bar(x=drops, y=feats, orientation="h"))
        fig.update_layout(title="置换检验 (AUC 下降)", template="plotly_dark", height=380)
        _add(fig, "置换重要性检验")

    # ---- Ablation ----
    if "ablation" in results:
        ab = results["ablation"]
        if "auc_delta" in ab.columns:
            fig = go.Figure(go.Bar(x=ab["ablation"], y=ab["auc_delta"]))
            fig.update_layout(title="消融检验 (AUC Δ vs 全模型)", template="plotly_dark", height=360)
            _add(fig, "消融检验")

    # ---- Conformal coverage ----
    conf = results.get("conformal")
    if conf is not None and hasattr(conf, "coverage_curve"):
        cv = conf.coverage_curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cv["nominal_coverage"], y=cv["empirical_coverage"],
                                 mode="lines+markers", name="经验覆盖率"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="理想对角线",
                                 line=dict(dash="dash")))
        fig.update_layout(title="保形预测: 经验 vs 名义覆盖率",
                          xaxis_title="名义覆盖率 (1-α)", yaxis_title="经验覆盖率",
                          template="plotly_dark", height=380)
        _add(fig, "保形预测校准 (覆盖率有效性)")

    # ---- Double ML ----
    dm = results.get("double_ml")
    if dm is not None and len(dm):
        dm = dm.sort_values("theta_orthogonalized", key=lambda s: s.abs(), ascending=False).head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dm["theta_orthogonalized"], y=dm["feature"], orientation="h",
            error_x=dict(type="data", symmetric=False,
                         array=dm["ci_high"] - dm["theta_orthogonalized"],
                         arrayminus=dm["theta_orthogonalized"] - dm["ci_low"]),
        ))
        fig.update_layout(
            title="双机器学习: 正交化后的区间特征边际效应 (含 95% CI)",
            xaxis_title="theta (控制行业/规模后)", template="plotly_dark", height=460,
        )
        _add(fig, "双机器学习正交化效应")

    html = (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
        "<title>区间型财务风险识别报告</title>"
        "<style>body{background:#0e1116;color:#e6e6e6;font-family:-apple-system,"
        "Segoe UI,Roboto,sans-serif;margin:0;padding:24px;}"
        ".card{background:#161b22;border:1px solid #2d333b;border-radius:12px;"
        "padding:18px;margin-bottom:20px;}h1{margin:0 0 8px;}h2{margin-top:0;}"
        "table.tbl{border-collapse:collapse;width:100%;font-size:13px;}"
        "table.tbl th,table.tbl td{border:1px solid #2d333b;padding:6px 10px;}"
        "table.tbl th{background:#21262d;}</style></head><body>"
        + "".join(parts)
        + "</body></html>"
    )
    out = reports_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Saved interactive HTML report: %s", out)
    return out
