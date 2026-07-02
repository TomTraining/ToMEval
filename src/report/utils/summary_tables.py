"""总览表格（SUMMARY）共享逻辑。

generate_summary.py（纯 Markdown）与 generate_summary_html.py（HTML 合并单元格）
此前各自复制了同一套「读 tables/ 目录基础指标 → 解析 → 归并 model/judge」的逻辑，
这里集中一份，两个脚本只保留各自的渲染部分（generate_summary_table）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import read_json


def config_display_name(config_path: Path, dataset_dir: Path) -> str:
    config_data = read_json(config_path)
    table_display = str(config_data.get("table_display") or "").strip()
    if table_display:
        return table_display

    rel_parent = config_path.parent.relative_to(dataset_dir)
    parts = rel_parent.parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else config_path.parent.name


def collect_config_display_names(dataset_dir: Path) -> List[str]:
    labels: List[str] = []

    for config_path in sorted(dataset_dir.glob("*/*/config.json")):
        label = config_display_name(config_path, dataset_dir)
        if label not in labels:
            labels.append(label)

    for config_path in sorted(dataset_dir.glob("*/config.json")):
        label = config_display_name(config_path, dataset_dir)
        if label not in labels:
            labels.append(label)

    return labels


def resolve_metric_label(raw_label: str, config_labels: List[str]) -> Optional[str]:
    if raw_label in config_labels:
        return raw_label

    matches = [label for label in config_labels if label == raw_label or label.startswith(f"{raw_label}/")]
    if len(matches) == 1:
        return matches[0]

    return raw_label


def parse_basic_metrics_table(table_dir: Path) -> Dict[str, Dict[str, Any]]:
    basic_metrics_file = table_dir / "基础指标.md"
    if not basic_metrics_file.exists():
        return {}

    metrics: Dict[str, Dict[str, Any]] = {}
    config_labels = collect_config_display_names(table_dir)
    content = basic_metrics_file.read_text(encoding="utf-8")

    lines = content.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("|") and "---" not in line]

    if len(data_lines) < 2:
        return {}

    header = [cell.strip() for cell in data_lines[0].split("|")[1:-1]]
    models = [resolve_metric_label(model, config_labels) for model in header[1:]]

    for line in data_lines[1:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != len(models) + 1:
            continue

        metric_name = cells[0]
        for i, model in enumerate(models):
            if not model:
                continue

            value: Any = cells[i + 1]
            if value == "-":
                continue

            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass

            metrics.setdefault(model, {})[metric_name] = value

    return metrics


def collect_metrics_from_tables(tables_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    tables_path = Path(tables_dir)
    metrics_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dataset_dir in tables_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue

        dataset_metrics = parse_basic_metrics_table(dataset_dir)
        if dataset_metrics:
            metrics_data[dataset_dir.name] = dataset_metrics

    return metrics_data


def split_model_judge(label: str) -> Tuple[str, str]:
    if "/" not in label:
        return label.strip(), "unknown"

    model_label, judge_label = label.split("/", 1)
    judge_label = judge_label.strip()
    lowered = judge_label.lower()

    for prefix in ("judge-", "judge:", "judge_"):
        if lowered.startswith(prefix):
            judge_label = judge_label[len(prefix):].strip()
            break

    return model_label.strip(), judge_label or "unknown"
