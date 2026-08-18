#!/usr/bin/env bash
# 一键运行区间型财务风险识别实验
# 用法：在终端里进入项目目录后执行  bash run.sh
set -e

# 无论在哪里执行，都切换到脚本所在目录
cd "$(dirname "$0")"

# 如果虚拟环境不存在，就自动建好并装依赖（只第一次需要）
if [ ! -d .venv ]; then
  echo "首次运行：创建虚拟环境并安装依赖..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

# 运行实验（用小规模配置，速度快）
.venv/bin/python scripts/run_experiment.py --config config/full_benchmark.yaml

echo ""
echo "实验完成！结果在 outputs/ 目录："
echo "  图表：outputs/figures/"
echo "  表格：outputs/reports/"
