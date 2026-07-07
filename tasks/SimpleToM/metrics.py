from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, hierarchical_metrics


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            # source(dataset_source)/difficulty 在标准化数据的 meta 里并不存在（恒为 unknown），
            # 改用真实字段：dimension / qa_type / scenario_name。
            ("dimension", by_meta("dimension")),
            ("qa_type", by_meta("qa_type")),
            ("scenario_name", by_meta("scenario_name")),
        ],
    )
