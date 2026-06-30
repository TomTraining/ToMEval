from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import hierarchical_metrics


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # ToMi 无额外切面，仅固定 type 维度（按题型切分）。
    return hierarchical_metrics(records, per_sample_results)
