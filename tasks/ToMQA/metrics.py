from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import first_value, generic_group_metrics, meta


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_dimension", lambda record: [first_value(meta(record).get("dimension"))]),
            ("by_difficulty", lambda record: [first_value(meta(record).get("difficulty"))]),
            ("by_task_type", lambda record: [first_value(meta(record).get("task_type"))]),
            ("by_order", lambda record: [first_value(meta(record).get("order"))]),
        ],
    )
