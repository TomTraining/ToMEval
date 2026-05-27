from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import base_metric_payload, first_value, meta, safe_div


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = base_metric_payload(per_sample_results)
    bu_correct = 0
    bu_total = 0
    bm_correct = 0
    bm_total = 0

    for record, result in zip(records, per_sample_results):
        step = first_value(meta(record).get("step"), default="")
        is_update = step.lower() in {"time_t+1", "time_t1", "t+1", "time_t_1"}
        if is_update:
            bu_total += 1
            if result["is_correct"]:
                bu_correct += 1
        else:
            bm_total += 1
            if result["is_correct"]:
                bm_correct += 1

    bu_acc = safe_div(bu_correct, bu_total)
    bm_acc = safe_div(bm_correct, bm_total)
    metrics.update(
        {
            "BU-Acc": bu_acc,
            "BM-Acc": bm_acc,
            "BREU": (bu_acc + bm_acc) / 2,
            "bu_correct": bu_correct,
            "bu_total": bu_total,
            "bm_correct": bm_correct,
            "bm_total": bm_total,
        }
    )
    return metrics
