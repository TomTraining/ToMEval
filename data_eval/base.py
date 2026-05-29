"""data_eval 公共工具：parquet 加载、报告写出、结果 dataclass。"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.llm.content_client import ContentClient


DATASETS = ["BigToM", "EmoBench", "FanToM", "HiToM", "SocialIQA", "ToMBench"]

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class EvalResult:
    dataset: str
    eval_type: str
    total_rows: int
    pass_: bool
    records: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


_REQUIRED_ROLES = ("strong", "simple")


def _build_client(entry: Dict[str, Any], default_max_tokens: int = 512) -> ContentClient:
    return ContentClient(
        model_name=entry["model_name"],
        api_key=entry["api_key"],
        api_url=entry["api_url"],
        temperature=entry.get("temperature", 0.3),
        max_workers=entry.get("max_workers", 8),
        max_tokens=entry.get("max_tokens", default_max_tokens),
        enable_thinking=False,
    )


def load_answer_models() -> Dict[str, ContentClient]:
    """F039：解析 config.yaml eval_model dict，返回 {'strong': client, 'simple': client}。

    schema 显式区分角色，禁止 list 形式（D039-01）。缺角色 raise ValueError。
    """
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    em = cfg.get("eval_model")
    if not isinstance(em, dict) or any(r not in em for r in _REQUIRED_ROLES):
        raise ValueError(
            f"config.yaml eval_model 必须为 dict 且含键 {_REQUIRED_ROLES}，得到: {type(em).__name__} keys={list(em) if isinstance(em, dict) else None}"
        )
    return {role: _build_client(em[role]) for role in _REQUIRED_ROLES}


def load_sample_rows(dataset: str) -> int:
    """F041：从 config.yaml 读 LLM 评估采样行数。

    schema:
      sample_rows:
        default: <int>           # 全局默认，必填
        per_dataset:             # 可选；为指定数据集覆盖 default
          BigToM: <int>
          ...

    缺 default 或类型非法 → raise ValueError；per_dataset 缺失即回落 default。
    """
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    sr = cfg.get("sample_rows")
    if not isinstance(sr, dict) or "default" not in sr:
        raise ValueError(
            "config.yaml 必须含 sample_rows.default（int），当前: "
            f"{type(sr).__name__} keys={list(sr) if isinstance(sr, dict) else None}"
        )
    per = sr.get("per_dataset") or {}
    if not isinstance(per, dict):
        raise ValueError(f"config.yaml sample_rows.per_dataset 必须为 dict 或缺省，得到 {type(per).__name__}")
    val = per.get(dataset, sr["default"])
    if not isinstance(val, int) or val <= 0:
        raise ValueError(
            f"config.yaml sample_rows[{dataset!r}] 必须为正整数，得到 {val!r}"
        )
    return val


def load_run_config() -> Dict[str, Any]:
    """F042：解析 config.yaml 的 paths/run/datasets 三段，返回规范化运行配置。

    返回 dict：
      {
        "datasets": List[str],        # 顶层 datasets，长度≥1
        "paths": {train_root, output_root, audit_root, report_path},
        "run": {datasets: List[str], report_only: bool, no_report: bool},
      }

    校验：
      - 顶层 datasets 非空 list
      - paths 四键必须存在；缺任一 → raise ValueError 含键名
      - run.datasets 非空时必须 ⊆ 顶层 datasets
      - run.report_only 与 run.no_report 同时为 True → raise
    """
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    top_datasets = cfg.get("datasets")
    if not isinstance(top_datasets, list) or not top_datasets:
        raise ValueError(f"config.yaml datasets 必须是非空 list，得到 {top_datasets!r}")

    paths = cfg.get("paths") or {}
    required_paths = ("train_root", "output_root", "audit_root", "report_path")
    missing = [k for k in required_paths if k not in paths]
    if missing:
        raise ValueError(f"config.yaml 缺 paths.{', paths.'.join(missing)}")

    run = cfg.get("run") or {}
    run_datasets = run.get("datasets")
    if run_datasets in (None, []):
        run_datasets = list(top_datasets)
    elif isinstance(run_datasets, list):
        unknown = [d for d in run_datasets if d not in top_datasets]
        if unknown:
            raise ValueError(
                f"config.yaml run.datasets 含未注册数据集 {unknown}（顶层 datasets={top_datasets}）"
            )
    else:
        raise ValueError(f"config.yaml run.datasets 必须为 list 或 null，得到 {type(run_datasets).__name__}")

    report_only = bool(run.get("report_only", False))
    no_report = bool(run.get("no_report", False))
    if report_only and no_report:
        raise ValueError("config.yaml run.report_only 与 run.no_report 不能同时为 true")

    return {
        "datasets": list(top_datasets),
        "paths": {k: str(paths[k]) for k in required_paths},
        "run": {
            "datasets": list(run_datasets),
            "report_only": report_only,
            "no_report": no_report,
        },
    }


def load_synth_parquet(
    dataset: str,
    iter_n: int = 1,
    model: str = "*",
    root: str = "feedback_data/synth_clean",
) -> pd.DataFrame:
    """加载 stage4 产出的 clean parquet。model 支持 glob 通配。"""
    root_path = Path(root) / dataset
    if not root_path.exists():
        raise FileNotFoundError(f"目录不存在: {root_path}")

    pattern = f"synthetic_iter{iter_n}_{model}.parquet"
    files = sorted(root_path.glob(pattern))
    # 排除 _hard.parquet
    files = [f for f in files if not f.name.endswith("_hard.parquet")]
    if not files:
        raise FileNotFoundError(f"在 {root_path} 未找到匹配 {pattern} 的文件")

    return pd.read_parquet(files[0])


def difficulty_artifact_path(dataset: str, file_stem: str, output_root: str = "data_eval_output") -> Path:
    """F038 D038-01：F035 难度评估中间产物路径单一来源。"""
    return Path(output_root) / "difficulty" / f"{dataset}_{file_stem}.json"


def write_report(report: Dict[str, Any], path: str | Path) -> None:
    """将报告写到指定路径，自动创建父目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
