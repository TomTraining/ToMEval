from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, hierarchical_metrics


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            # difficulty/task_type/order 在标准化数据的 meta 里并不存在（恒为 unknown），
            # 改用真实字段：dimension（考察维度）/ task（bAbI 任务型 fb/tb/sofb）。
            ("dimension", by_meta("dimension")),
            ("task", by_meta("task")),
        ],
    )
