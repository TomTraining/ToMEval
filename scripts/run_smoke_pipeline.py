"""ToMBench 端到端 smoke 流程编排：eval → feedback → filter。

用途
----
把三段流程（评测 / 数据合成 / 过滤）用一份隔离配置串起来跑通一遍，作为回归 smoke。
被测模型 qwen3-8b、teacher 强模型 deepseek-v4-flash 都走 tokenkey.dev，
API key 从仓库根目录 .env 注入（占位符，用时替换成真实 key）。

⚠ 当前只测 ToMBench。后续要换/加数据集时，改这三份 smoke 配置里的数据集列表即可：
    scripts/smoke/experiment_config.smoke.yaml  (datasets)
    scripts/smoke/feedback.smoke.yaml           (synthesis_datasets，models.name 要与 eval 的 model_name 一致)
    scripts/smoke/filter.smoke.yaml             (datasets)
本脚本的 DATASETS 常量也要同步（只用于产物校验与 parquet 搬运）。

数据流
------
    eval      run_eval.py    → results/<DS>/qwen3-8b/exp_*/{prediction.jsonl,metrics.json}
    feedback  run_feedback   → feedback_output/datasets/<DS>.parquet
    [搬运]    feedback_output/datasets/<DS>.parquet → train_datasets/<DS>/synthetic.parquet
    filter    run_filter     → filter_output_smoke/datasets/<DS>_filtered.parquet

用法
----
    # 先在 .env 里填好真实 key，然后：
    python scripts/run_smoke_pipeline.py
    python scripts/run_smoke_pipeline.py --skip eval        # 跳过某段（复用已有产物）
    python scripts/run_smoke_pipeline.py --only feedback    # 只跑某段

配置里的 ${VAR} 占位符由本脚本读 .env 后展开，写成 .resolved 临时配置再交给各段
（各段的 yaml.safe_load 不会自己展开环境变量）。.resolved 文件仅本地临时用，退出前清理。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = REPO_ROOT / "scripts" / "smoke"
ENV_FILE = REPO_ROOT / ".env"

# 当前 smoke 覆盖的数据集（换/加数据集时与三份 smoke 配置同步修改）。
DATASETS: List[str] = ["ToMBench"]

# eval 落盘用的 model 目录名，必须与 experiment_config.smoke.yaml 的 llm.model_name 一致。
EVAL_MODEL_NAME = "qwen3-8b"

STAGES = ("eval", "feedback", "filter")


# ── .env 加载 & 配置展开 ────────────────────────────────────────────────────────

def load_dotenv(path: Path) -> None:
    """极简 .env 解析（KEY=VALUE，# 注释，去引号），注入 os.environ。不覆盖已存在的环境变量。"""
    if not path.exists():
        print(f"[env] .env 不存在: {path}（将只依赖已导出的环境变量）")
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_config(src: Path) -> Path:
    """把配置里的 ${VAR} 用环境变量展开，写成同目录 .resolved 文件，返回其路径。

    缺失的变量会明确报错，避免把字面量 ${VAR} 传给下游。
    """
    text = src.read_text(encoding="utf-8")
    try:
        resolved = Template(text).substitute(os.environ)
    except KeyError as e:
        raise SystemExit(f"[config] {src.name} 引用了未设置的环境变量: {e}. 请在 .env 中补全。")
    out = src.with_suffix(src.suffix + ".resolved")
    out.write_text(resolved, encoding="utf-8")
    return out


def check_placeholder_keys() -> None:
    """校验关键 key 已从占位符替换成真实值，否则明确提示。"""
    missing = []
    for key in ("TOKENKEY_QWEN_KEY", "TOKENKEY_DEEPSEEK_KEY", "TOKENKEY_BASE_URL"):
        val = os.environ.get(key, "")
        if not val or val.startswith("<") or "REPLACE" in val.upper() or "占位" in val:
            missing.append(key)
    if missing:
        raise SystemExit(
            f"[env] 以下变量仍是占位符/未设置: {missing}\n"
            f"      请在 {ENV_FILE} 中填入真实 key 后重试。"
        )


# ── 子进程执行 ──────────────────────────────────────────────────────────────────

def run_cmd(cmd: List[str], env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return proc.returncode


# ── 各段 ───────────────────────────────────────────────────────────────────────

def stage_eval() -> bool:
    cfg = resolve_config(SMOKE_DIR / "experiment_config.smoke.yaml")
    rc = run_cmd([sys.executable, "run_eval.py", "--experiment-config", str(cfg)])
    if rc != 0:
        print(f"[eval] run_eval.py 退出码 {rc}")
        return False
    # 校验产物：每个数据集都要有 prediction.jsonl
    for ds in DATASETS:
        base = REPO_ROOT / "results" / ds / EVAL_MODEL_NAME
        exps = sorted(base.glob("exp_*")) if base.exists() else []
        if not exps or not (exps[-1] / "prediction.jsonl").exists():
            print(f"[eval] ✗ 缺产物: {base}/exp_*/prediction.jsonl")
            return False
        print(f"[eval] ✓ {ds}: {exps[-1]}/prediction.jsonl")
    return True


def stage_feedback() -> bool:
    cfg = resolve_config(SMOKE_DIR / "feedback.smoke.yaml")
    rc = run_cmd([sys.executable, "run_feedback.py", "--config", str(cfg)])
    if rc != 0:
        print(f"[feedback] run_feedback.py 退出码 {rc}")
        return False
    for ds in DATASETS:
        out = REPO_ROOT / "feedback_output" / "datasets" / f"{ds}.parquet"
        if not out.exists():
            print(f"[feedback] ✗ 缺产物: {out}")
            return False
        print(f"[feedback] ✓ {ds}: {out}")
    return True


def stage_transfer() -> bool:
    """把 feedback 产物搬到 filter 的 input_root（train_datasets/<DS>/synthetic.parquet）。"""
    import shutil

    for ds in DATASETS:
        src = REPO_ROOT / "feedback_output" / "datasets" / f"{ds}.parquet"
        if not src.exists():
            print(f"[transfer] ✗ 源文件不存在: {src}")
            return False
        dst_dir = REPO_ROOT / "train_datasets" / ds
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "synthetic.parquet"
        shutil.copyfile(src, dst)
        print(f"[transfer] ✓ {src} → {dst}")
    return True


def stage_filter() -> bool:
    cfg = resolve_config(SMOKE_DIR / "filter.smoke.yaml")
    # filter 侧通过 FILTER_CONFIG 环境变量指定隔离配置（base._CONFIG_PATH 会读取）。
    rc = run_cmd([sys.executable, "run_filter.py"], env_extra={"FILTER_CONFIG": str(cfg)})
    if rc != 0:
        print(f"[filter] run_filter.py 退出码 {rc}")
        return False
    # 产物目录以配置里的 output_root 为准（smoke 用隔离目录 filter_output_smoke，避免覆盖生产 filter_output）。
    import yaml
    output_root = yaml.safe_load(cfg.read_text(encoding="utf-8")).get("paths", {}).get("output_root", "filter_output")
    for ds in DATASETS:
        out = REPO_ROOT / output_root / "datasets" / f"{ds}_filtered.parquet"
        if not out.exists():
            print(f"[filter] ✗ 缺产物: {out}")
            return False
        print(f"[filter] ✓ {ds}: {out}")
    return True


# ── 主流程 ───────────────────────────────────────────────────────────────────────

def cleanup_resolved() -> None:
    for f in SMOKE_DIR.glob("*.resolved"):
        f.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="ToMBench 端到端 smoke（eval→feedback→filter）")
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGES,
                        help="跳过指定段（复用已有产物）")
    parser.add_argument("--only", nargs="*", default=[], choices=STAGES,
                        help="只跑指定段（其余全跳过）")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)

    if args.only:
        to_run = set(args.only)
    else:
        to_run = set(STAGES) - set(args.skip)

    # 只有真正要调模型的段才校验 key
    if to_run & {"eval", "feedback", "filter"}:
        check_placeholder_keys()

    steps = [
        ("eval", stage_eval, "eval" in to_run),
        ("feedback", stage_feedback, "feedback" in to_run),
        ("transfer", stage_transfer, "filter" in to_run),  # 搬运随 filter 段
        ("filter", stage_filter, "filter" in to_run),
    ]

    results = {}
    try:
        for name, fn, enabled in steps:
            if not enabled:
                print(f"\n=== [{name}] 跳过 ===")
                continue
            print(f"\n{'='*60}\n=== [{name}] 开始 ===\n{'='*60}")
            ok = fn()
            results[name] = ok
            if not ok:
                print(f"\n✗ [{name}] 失败，流程中止。")
                return 1
    finally:
        cleanup_resolved()

    print(f"\n{'='*60}\n=== smoke 结果 ===\n{'='*60}")
    for name in ("eval", "feedback", "transfer", "filter"):
        if name in results:
            print(f"  {name:10s}: {'PASS' if results[name] else 'FAIL'}")
    print("全流程通过 ✅" if all(results.values()) else "存在失败 ❌")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
