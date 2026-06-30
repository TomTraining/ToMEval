from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, hierarchical_metrics, meta, value_list


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return hierarchical_metrics(
        records,
        per_sample_results,
        [
            ("task_type", by_meta("task_type")),
            ("source_split", by_meta("source_split")),
            ("lang", by_meta("lang")),
            ("original_category", by_meta("original_category")),
            ("dimension", lambda record: value_list(meta(record).get("dimension"))),
            ("num_choices", lambda record: [str(len(record.get("options") or {}))]),
        ],
    )
