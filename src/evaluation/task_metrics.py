from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


def safe_div(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def meta(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("meta", {})
    return value if isinstance(value, dict) else {}


def first_value(value: Any, default: str = "unknown") -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return default
    text = str(value).strip() if value not in (None, "") else ""
    return text or default


def value_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def update_group(stats: Dict[str, Dict[str, int]], key: Any, is_correct: bool) -> None:
    key_str = first_value(key)
    if key_str not in stats:
        stats[key_str] = {"correct": 0, "total": 0}
    stats[key_str]["total"] += 1
    if is_correct:
        stats[key_str]["correct"] += 1


def rate_dict(stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    return {
        key: safe_div(value["correct"], value["total"])
        for key, value in sorted(stats.items())
    }


def count_dict(stats: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    return {key: value["total"] for key, value in sorted(stats.items())}


def flatten_group(stats: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}.{key}": safe_div(value["correct"], value["total"])
        for key, value in sorted(stats.items())
    }


def base_metric_payload(per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(per_sample_results)
    correct = sum(1 for item in per_sample_results if item["is_correct"])
    return {
        "accuracy": safe_div(correct, total),
        "correct": correct,
        "total": total,
        "per_sample_results": per_sample_results,
    }


def generic_group_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
    group_specs: Sequence[Tuple[str, Callable[[Dict[str, Any]], Iterable[str]]]],
) -> Dict[str, Any]:
    # 共享层只保留最基础的分组统计拼装，任务自己的聚合逻辑仍放在 tasks/<dataset>/metrics.py。
    metrics = base_metric_payload(per_sample_results)

    for metric_name, key_fn in group_specs:
        stats: Dict[str, Dict[str, int]] = {}
        for record, result in zip(records, per_sample_results):
            keys = list(key_fn(record))
            if not keys:
                keys = ["unknown"]
            for key in keys:
                update_group(stats, key, result["is_correct"])

        metrics.update(flatten_group(stats, metric_name))
        metrics[metric_name] = rate_dict(stats)
        suffix = metric_name.split(".")[-1]
        metrics[f"{suffix}_counts"] = count_dict(stats)

    return metrics
