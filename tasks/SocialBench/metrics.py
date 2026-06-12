from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, generic_group_metrics, meta, value_list


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_task_type", by_meta("task_type")),
            ("by_source_split", by_meta("source_split")),
            ("by_lang", by_meta("lang")),
            ("by_original_category", by_meta("original_category")),
            ("by_dimension", lambda record: value_list(meta(record).get("dimension"))),
            ("by_num_choices", lambda record: [str(len(record.get("options") or {}))]),
        ],
    )
