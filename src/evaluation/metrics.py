from __future__ import annotations

import importlib
from typing import Any, Dict, List


def load_task_metric_fn(dataset_name: str):
    module = importlib.import_module(f"tasks.{dataset_name}.metrics")
    metric_fn = getattr(module, "compute_metrics", None)
    if metric_fn is None:
        raise AttributeError(f"tasks/{dataset_name}/metrics.py must define compute_metrics(records, per_sample_results).")
    return metric_fn


def aggregate_metrics(
    dataset_name: str,
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metric_fn = load_task_metric_fn(dataset_name)
    return metric_fn(records, per_sample_results)
