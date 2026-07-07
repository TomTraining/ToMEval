from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import hierarchical_metrics


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 注意：PUB 标准化后 meta 很稀薄，只剩 dimension(恒为单值 pragmatics)+n_options，
    # source/difficulty/ethics_category/task_type 等原始切面在转换时已丢失（恒为 unknown），
    # 故这里仅保留有区分度的 option_count（=候选个数），其余交给固定的 type 维度。
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            ("option_count", lambda record: [str(len(record.get("options") or {}))]),
        ],
    )
