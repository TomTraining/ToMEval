from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import base_metric_payload


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    del records
    return base_metric_payload(per_sample_results)
