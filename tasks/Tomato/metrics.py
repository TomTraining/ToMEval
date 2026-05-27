from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.evaluation.task_metrics import base_metric_payload, meta, safe_div, value_list


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = base_metric_payload(per_sample_results)
    slot_stats = [defaultdict(lambda: {"correct": 0, "total": 0}) for _ in range(3)]

    for record, result in zip(records, per_sample_results):
        dims = value_list(meta(record).get("dimension"))
        slot_values = [
            dims[0] if len(dims) >= 1 else "__missing__",
            dims[1] if len(dims) >= 2 else "__missing__",
            dims[2] if len(dims) >= 3 else "__none__",
        ]
        for index, value in enumerate(slot_values):
            stats = slot_stats[index][value]
            stats["total"] += 1
            if result["is_correct"]:
                stats["correct"] += 1

    for index, stats in enumerate(slot_stats, start=1):
        metrics.update(
            {
                f"by_dimension_{index}.{key}": safe_div(value["correct"], value["total"])
                for key, value in sorted(stats.items())
            }
        )
        metrics[f"dimension_{index}_counts"] = {
            key: value["total"] for key, value in sorted(stats.items())
        }
    return metrics
