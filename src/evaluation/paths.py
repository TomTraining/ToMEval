from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_experiment_dir(
    dataset_name: str,
    model_name: str,
    results_path: str,
    exp_dir: Optional[str] = None,
    create: bool = False,
) -> Path:
    base_dir = Path(results_path) / dataset_name / model_name
    if exp_dir:
        if Path(exp_dir).name.startswith("exp_"):
            target_dir = base_dir / Path(exp_dir).name
        else:
            target_dir = base_dir / f"exp_{exp_dir}"
    else:
        timestamp = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = base_dir / f"exp_{timestamp}"

    if create:
        target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def find_latest_experiment_dir(dataset_name: str, model_name: str, results_path: str) -> Path:
    base_dir = Path(results_path) / dataset_name / model_name
    candidates = sorted(base_dir.glob("exp_*"))
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found under {base_dir}")
    return candidates[-1]


def save_config(
    output_dir: Path,
    dataset_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
) -> None:
    config_data = {
        "dataset": dataset_config["dataset"],
        "dataset_config": dict(dataset_config),
        "experiment_config": dict(experiment_config),
    }

    # experiment_config 的 llm_config 脱敏。
    if "llm_config" in config_data["experiment_config"]:
        cleaned = dict(config_data["experiment_config"]["llm_config"])
        cleaned.pop("api_key", None)
        cleaned.pop("api_url", None)
        config_data["experiment_config"]["llm_config"] = cleaned

    # judge 参数现在落在 dataset_config(config.yaml)，同样剥掉密钥/地址。
    for section in ("judge1", "judge2"):
        if isinstance(config_data["dataset_config"].get(section), dict):
            cleaned = dict(config_data["dataset_config"][section])
            cleaned.pop("api_key", None)
            cleaned.pop("api_url", None)
            config_data["dataset_config"][section] = cleaned

    (output_dir / "config.json").write_text(
        json.dumps(config_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def average_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all_metrics:
        return {}

    # 只对数值和嵌套 dict 做跨 repeat 平均，其它字段不进入 avg_metrics。
    avg: Dict[str, Any] = {}
    keys = set()
    for metric in all_metrics:
        keys.update(metric.keys())

    for key in sorted(keys):
        values = [metric.get(key) for metric in all_metrics if key in metric]
        if not values:
            continue
        first = values[0]
        if isinstance(first, (int, float)):
            numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
            avg[key] = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
        elif isinstance(first, dict):
            dict_values = [value for value in values if isinstance(value, dict)]
            avg[key] = average_metrics(dict_values)
    return avg
