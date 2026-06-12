from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import by_meta, generic_group_metrics


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_source", by_meta("dataset_source")),
            ("by_dimension", by_meta("dimension")),
            ("by_difficulty", by_meta("difficulty")),
        ],
    )
