from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, hierarchical_metrics, meta, value_list


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            # task_type/source_split/original_category 在标准化数据的 meta 里并不存在
            # （恒为 unknown），改用真实字段 category（官方 <层级>-<能力>-<子任务> 类别码）。
            ("category", by_meta("category")),
            ("dimension", lambda record: value_list(meta(record).get("dimension"))),
            ("lang", by_meta("lang")),
            ("num_choices", lambda record: [str(len(record.get("options") or {}))]),
        ],
    )
