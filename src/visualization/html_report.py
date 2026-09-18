"""Interactive HTML report builder (Plotly).

Replaces the static matplotlib figures with a single self-contained, browser
openable dashboard: model comparison, ROC/PR, calibration, rolling stability,
feature importance, permutation & ablation diagnostics, plus the two new
highlight modules -- conformal prediction coverage and double-ML orthogonalized
effects. Plotly.js is embedded once so the report also opens offline.
"""

from __future__ import annotations

import html
import json
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
        traces.append(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={a:.3f})"))
    traces.append(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
    )
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
            include_plotlyjs=(True if first else False),
            config={"displayModeBar": False},
        )
        first = False
        parts.append(f"<div class='card'><h2>{title}</h2>{div}</div>")

    # ---- Title ----
    parts.append(
        "<div class='card'><h1>区间型财务数据与企业风险识别 — 交互报告</h1>"
        f"<p>项目: {html.escape(config.project.name)} | 随机种子: {config.project.seed}</p>"
        "<p class=notice>合成数据演示 · 结果用于检验研究流程，不代表真实企业风险识别能力。</p>"
        "<p>比较点特征与过去四个季度的分布特征；同一公司的未来季度进入测试集。"
        "报告期是合成季度标记，尚未证明真实财报发布日期可得性。</p></div>"
    )

    data = results.get("data_summary", {})
    parts.append(
        "<div class=card><h2>实验来源与评价口径</h2><p>"
        + html.escape(json.dumps(data, ensure_ascii=False))
        + "</p><p>表格为各测试折指标的等权均值 ± 折间标准差；ROC/PR 图将测试预测合并，口径不同。标准差不是置信区间。分类阈值固定 0.5；PR-AUC 使用 average precision。</p><p>诊断模型由验证集平均 AUC 选择："
        + html.escape(results.get("diagnostic_model", ""))
        + "。消融与置换重要性在最后测试折评估，属于探索性诊断。</p></div>"
    )
    if "splits" in results:
        parts.append(
            "<div class=card><h2>时间验证边界</h2><p>按 report_date 扩展训练窗口；预处理仅拟合训练集。验证窗口用于模型选择，测试窗口不参与该选择。</p><div class=scroll>"
            + results["splits"].to_html(index=False, classes="tbl")
            + "</div></div>"
        )

    # ---- Model comparison table ----
    if "summary" in results:
        df = results["summary"].copy()
        styled = df.round(3).to_html(index=False, classes="tbl")
        parts.append(
            f"<div class='card'><h2>模型对比 (滚动验证均值)</h2><div class=scroll>{styled}</div></div>"
        )

    # ---- ROC ----
    if "all_results" in results:
        fig = go.Figure(_roc_traces(results["all_results"]))
        fig.update_layout(
            title="ROC 曲线 (跨所有滚动折)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            height=420,
        )
        _add(fig, "ROC 曲线")

    # ---- Calibration and precision-recall on held-out predictions ----
    if "all_results" in results:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import average_precision_score, precision_recall_curve

        calibration = go.Figure()
        pr = go.Figure()
        for name, folds in results["all_results"].items():
            truth = np.concatenate([f["y_true"] for f in folds])
            probability = np.concatenate([f["y_proba"] for f in folds])
            observed, predicted = calibration_curve(
                truth, probability, n_bins=config.evaluation.calibration_bins
            )
            calibration.add_trace(
                go.Scatter(x=predicted, y=observed, mode="lines+markers", name=name)
            )
            precision, recall, _ = precision_recall_curve(truth, probability)
            pr.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"{name} (AP={average_precision_score(truth, probability):.3f})",
                )
            )
        calibration.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(dash="dash"))
        )
        calibration.update_layout(
            template="plotly_dark",
            height=430,
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed positive fraction",
        )
        pr.add_hline(y=float(truth.mean()), line_dash="dash", annotation_text="Test prevalence")
        pr.update_layout(
            template="plotly_dark", height=430, xaxis_title="Recall", yaxis_title="Precision"
        )
        _add(calibration, "概率校准：预测风险与实际标签频率")
        _add(pr, "Precision–Recall：与测试正例率对照")

    # ---- Rolling metrics ----
    if "rolling" in results:
        rdf = results["rolling"]
        fig = make_subplots(rows=1, cols=1)
        for name in rdf["model"].unique():
            sub = rdf[rdf["model"] == name]
            fig.add_trace(
                go.Scatter(
                    x=sub["test_end_date"].astype(str),
                    y=sub["auc"],
                    mode="lines+markers",
                    name=name,
                )
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
        items = sorted(perm.items(), key=lambda item: item[1].get("mean_drop", 0.0), reverse=True)[
            :20
        ]
        feats = [name for name, _ in items]
        drops = [values.get("mean_drop", 0.0) for _, values in items]
        fig = go.Figure(go.Bar(x=drops, y=feats, orientation="h"))
        fig.update_layout(title="置换检验 (AUC 下降)", template="plotly_dark", height=380)
        _add(fig, "置换重要性检验")

    # ---- Ablation ----
    if "ablation" in results:
        parts.append(
            "<div class=card><h2>消融模型</h2><p>预先固定为 "
            + html.escape(results.get("ablation_model", ""))
            + "，各消融变体重新拟合，使用同一最后测试折；这样即使点模型获胜，移除区间特征也不会变成空操作。</p></div>"
        )
        ab = results["ablation"]
        if "auc_delta" in ab.columns:
            fig = go.Figure(go.Bar(x=ab["ablation"], y=ab["auc_delta"]))
            fig.update_layout(
                title="消融检验 (AUC Δ vs 全模型)", template="plotly_dark", height=360
            )
            _add(fig, "消融检验")

    # ---- Conformal coverage ----
    conf = results.get("conformal")
    if conf is not None and hasattr(conf, "coverage_curve"):
        parts.append(
            f"<div class=card><h2>标签预测区间的覆盖与宽度</h2><p>名义覆盖 {1 - conf.alpha:.0%} · 经验覆盖 {conf.test_coverage:.1%} · 平均宽度 {conf.test_avg_width:.3f}</p><p>覆盖对象是二元标签 y，不是未知真实风险概率。时间漂移与公司内相关性违反经典可交换性假设；只报告本次经验覆盖。区间很宽也可能得到高覆盖，应同时阅读宽度。</p></div>"
        )
        parts.append(
            "<div class=card><p>预先固定模型："
            + html.escape(results.get("conformal_model", ""))
            + "；不按测试或校准得分挑选模型。</p><p>"
            + html.escape(json.dumps(conf.split_summary, ensure_ascii=False))
            + "</p></div>"
        )
        cv = conf.coverage_curve
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cv["nominal_coverage"],
                y=cv["empirical_coverage"],
                mode="lines+markers",
                name="经验覆盖率",
            )
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="理想对角线", line=dict(dash="dash"))
        )
        fig.update_layout(
            title="保形预测: 经验 vs 名义覆盖率",
            xaxis_title="名义覆盖率 (1-α)",
            yaxis_title="经验覆盖率",
            template="plotly_dark",
            height=380,
        )
        _add(fig, "保形预测校准 (覆盖率有效性)")

    # ---- Double ML ----
    dm = results.get("double_ml")
    if dm is not None and len(dm):
        dm = dm.sort_values("theta_orthogonalized", key=lambda s: s.abs(), ascending=False).head(15)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=dm["theta_orthogonalized"],
                y=dm["feature"],
                orientation="h",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=dm["ci_high"] - dm["theta_orthogonalized"],
                    arrayminus=dm["theta_orthogonalized"] - dm["ci_low"],
                ),
            )
        )
        fig.update_layout(
            title="正交化线性关联（探索性；常规 OLS 区间）",
            xaxis_title="theta (仅控制行业)",
            template="plotly_dark",
            height=460,
        )
        _add(fig, "正交化关联诊断")
        parts.append(
            "<div class=card><p>仅控制行业；随机交叉拟合未按公司分组，OLS 标准误未做公司聚类修正。系数不构成因果效应或稳健统计显著性证据。</p></div>"
        )

    snapshot = html.escape(
        json.dumps(results.get("config_snapshot", {}), ensure_ascii=False, indent=2)
    )
    parts.append(
        "<div class=card><h2>复现配置</h2><p>生成命令：<code>python scripts/run_experiment.py --config config/config.yaml</code>。随附 run_manifest.json 记录依赖、数据摘要与源码哈希，predictions.csv 支持独立复算。</p><details><summary>展开完整配置</summary><pre>"
        + snapshot
        + "</pre></details></div>"
    )
    document = (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
        "<title>区间型财务风险识别报告</title><meta name=viewport content='width=device-width, initial-scale=1'>"
        "<style>body{background:#0e1116;color:#e6e6e6;font-family:-apple-system,"
        "Segoe UI,Roboto,sans-serif;margin:0;padding:24px;}"
        ".card{background:#161b22;border:1px solid #2d333b;border-radius:12px;"
        "padding:18px;margin-bottom:20px;}h1{margin:0 0 8px;}h2{margin-top:0;}"
        "table.tbl{border-collapse:collapse;width:100%;font-size:13px;}"
        "table.tbl th,table.tbl td{border:1px solid #2d333b;padding:6px 10px;}"
        "table.tbl th{background:#21262d;}.scroll{overflow:auto}.notice{color:#8ed8e8;font-weight:600}body{max-width:1280px;margin:auto}p{line-height:1.7;overflow-wrap:anywhere}pre{white-space:pre-wrap}h1{font-size:28px}details{padding:12px 0}</style></head><body>"
        + "".join(parts)
        + "</body></html>"
    )
    out = reports_dir / "report.html"
    out.write_text(document, encoding="utf-8")
    logger.info("Saved interactive HTML report: %s", out)
    return out
