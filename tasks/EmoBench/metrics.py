from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import first_value, generic_group_metrics, meta, value_list


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_subset", lambda record: [first_value(meta(record).get("subset"))]),
            ("by_language", lambda record: [first_value(meta(record).get("language"))]),
            ("by_question_subtype", lambda record: [first_value(meta(record).get("question_subtype"))]),
            ("by_coarse_category", lambda record: [first_value(meta(record).get("coarse_category"))]),
            ("by_finegrained_category", lambda record: [first_value(meta(record).get("finegrained_category"))]),
            ("by_dimension", lambda record: value_list(meta(record).get("dimension"))),
            ("by_num_choices", lambda record: [str(len(record.get("options") or {}))]),
        ],
    )
