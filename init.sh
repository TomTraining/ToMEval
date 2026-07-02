#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# BenchEval 是 Python 评测/数据治理框架，没有 npm。
# 解释器固定用用户的虚拟环境，避免触发 source 权限确认。
PY="/Users/yangmeili/Downloads/Code/.venv/bin/python"

echo "==> 当前目录: $PWD"

if [ ! -x "$PY" ]; then
  echo "!! 找不到解释器 $PY，请先建立虚拟环境（见 README 安装段）。"
  exit 1
fi

echo "==> 解释器: $PY"
"$PY" --version

echo "==> 同步依赖"
"$PY" -m pip install -q -r requirements.txt

echo "==> 基础 smoke test（核心模块可导入）"
"$PY" - <<'PYEOF'
import importlib
mods = [
    "src.runner",
    "src.dataloader",
    "src.evaluation",
    "src.evaluation.pipeline",
    "src.evaluation.open_judge",
    "src.evaluation.task_metrics",
    "src.llm",
    "src.visualization",
]
for m in mods:
    importlib.import_module(m)
    print(f"  ok  {m}")
print("smoke test passed")
PYEOF

cat <<'EOF'

==> 三条流水线入口（按需运行）
    评测:     python run_eval.py            (配置 experiment_config.yaml)
    数据质量: python run_filter.py          (配置 src/filter/config.yaml)
    数据合成: python run_feedback.py        (配置 src/feedback/config.yaml)
    报告:     python src/report/generate_dataset_tables.py && python src/report/generate_summary.py
EOF
