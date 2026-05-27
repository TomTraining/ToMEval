from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import first_value, generic_group_metrics, meta, value_list


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_task_type", lambda record: [first_value(meta(record).get("task_type"))]),
            ("by_source_split", lambda record: [first_value(meta(record).get("source_split"))]),
            ("by_lang", lambda record: [first_value(meta(record).get("lang"))]),
            ("by_original_category", lambda record: [first_value(meta(record).get("original_category"))]),
            ("by_dimension", lambda record: value_list(meta(record).get("dimension"))),
            ("by_num_choices", lambda record: [str(len(record.get("options") or {}))]),
        ],
    )
