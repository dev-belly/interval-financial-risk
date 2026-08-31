# 区间型财务数据与企业风险识别研究

> Interval Financial Data for Enterprise Risk Identification

基于公开季度财务报表与行情数据，将营收增速、利润率、现金流和波动率由单点指标扩展为季度区间与分布型特征，系统比较点估计模型与区间/分布特征模型对企业风险标签的增量识别能力。

---

## 核心贡献

1. **区间型特征工程**：把传统单点财务指标扩展为均值、区间宽度、分位数、偏度、峰度等分布型特征。
2. **多模型对比**：逻辑回归基线（仅均值点数据）→ Elastic Net 正则化模型 → XGBoost/LightGBM 树模型。
3. **时间滚动验证**：严格按财报发布日期划分训练/验证/测试窗口，避免前视偏差。
4. **稳健性检验**：置换检验（Permutation Test）+ 消融检验（Ablation Study）量化复杂特征的真实信息增量。
5. **可解释评估**：AUC、PR-AUC、Brier 校准误差、行业分组性能、时期稳定性。
6. **结论限定为预测关联**，不将相关性表述为因果关系。

---

## 项目亮点

本项目不止是"跑几个模型比 AUC"，以下三个模块具备学术与实务上的差异化锋芒：

### 1. 保形预测（Conformal Prediction）—— 给概率装上"误差条"

普通模型只吐一个点概率（如"风险 0.76"），在金融场景里这是危险的：0.76 和 0.74 常被当作确信不同的两个数。
本平台用按报告期切分的 **Split Conformal** 把点预测转成标签预测区间
`[p-ε, p+ε]`，并报告样本外经验覆盖率。经典有限样本覆盖结论依赖样本可交换性；
金融时间序列存在漂移和截面相关，因此这里把覆盖率当作诊断指标，而不是无条件保证或“概率置信区间”。

### 2. 双机器学习（Double / Debiased ML）—— 剥离混淆后的"纯净"效应

朴素逻辑回归会把行业、规模等混淆变量的影响混进区间特征系数。本平台用 **Chernozhukov et al. (2018) 的双机器学习**
做正交化：用交叉拟合估计 `g(W)=E[Y|W]` 与 `m(X|W)=E[X|W]`，再对残差 `r_y ~ θ·r_x` 做回归。
结果 `θ` 是控制配置中混杂变量后的正交化线性关联，并附 95% 置信区间。
它用于稳健性诊断；没有外生处理或完整识别假设时，不解释为因果效应。

### 3. 可交互 HTML 报告（Plotly）

所有结果自动汇总为一个浏览器直接打开的仪表盘 `outputs/reports/report.html`：模型对比表、ROC、滚动稳定性、
特征重要性、置换/消融检验，以及上述保形覆盖曲线与双机器学习效应条形图（均带交互与误差棒）。

---

## 项目结构

```text
interval-financial-risk/
├── config/                  # 实验配置
├── data/                    # 原始/处理/合成数据
├── notebooks/               # Jupyter 实验报告
├── src/                     # 核心源码
│   ├── data/                # 数据加载与合成数据生成
│   ├── features/            # 区间特征工程与 Pipeline
│   ├── models/              # 基线、正则化、树模型
│   ├── evaluation/          # 评估指标、滚动验证、置换/消融检验
│   └── visualization/       # 可视化
├── scripts/                 # 一键运行脚本
├── tests/                   # 单元测试
└── outputs/                 # 结果输出
```

---

## 快速开始

### 1. 环境准备

```bash
cd interval-financial-risk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 运行完整实验

```bash
python scripts/run_experiment.py --config config/config.yaml
```

> 默认配置对算力要求较低，可在普通笔记本/CI 上稳定运行。如需更大样本量 + Optuna 深度超参优化，请使用 `config/full_benchmark.yaml`（需要更多内存和运行时间）。

### 3. 查看报告

```bash
open outputs/reports/report.html       # macOS
# 或直接用浏览器打开 outputs/reports/report.html
```

---

## 数据说明

当前仓库内置**合成数据生成器**（`src/data/synthetic_generator.py`），可生成具有真实统计特性的季度财报与行情数据，用于在没有付费数据接口的情况下验证完整 Pipeline。

若接入真实数据，只需实现 `src/data/loader.py` 中的 `RealDataLoader`，保持输出列名与合成数据一致即可无缝替换。支持的数据源示例：

- A 股财报：Tushare / AkShare / CSMAR
- 行情数据：Tushare / Yahoo Finance / 东方财富
- 风险标签：ST 公告、违约事件、评级下调等

---

## 技术栈

- **Python 3.11+**
- **数据处理**：Polars、Pandas、NumPy
- **机器学习**：scikit-learn、XGBoost、LightGBM、CatBoost
- **超参优化**：Optuna
- **数据验证**：Great Expectations
- **配置管理**：Pydantic、Hydra
- **可视化**：Plotly、Matplotlib、Seaborn
- **可复现**：Git LFS、DVC（可选）
- **测试**：pytest

---

## 主要结果

实验结果将输出到 `outputs/` 目录：

- `outputs/figures/`：校准曲线、ROC/PR 曲线、滚动性能、特征重要性、置换/消融检验图
- `outputs/reports/`：指标表格、LaTeX/Markdown 实验报告
- `outputs/models/`：序列化模型与预处理 Pipeline

---

## 作者与协议

独立研究项目，2026.08 — 至今。  
代码采用 [MIT License](LICENSE)。
