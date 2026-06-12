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


def by_meta(key: str) -> Callable[[Dict[str, Any]], List[str]]:
    """返回一个 group key_fn：取 record.meta[key] 的单值（缺失为 "unknown"）。

    配合 generic_group_metrics 使用，等价于
    ``lambda record: [first_value(meta(record).get(key))]``，把各 metrics.py 里
    重复的 by-维度 lambda 收敛成一处。
    """
    return lambda record: [first_value(meta(record).get(key))]


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
    # MCQ 严格模式下没有 \boxed{} 会被标记为 extraction_failed，这里额外统计格式遵循情况。
    extraction_failed = sum(1 for item in per_sample_results if item.get("error_reason") == "extraction_failed")
    return {
        "accuracy": safe_div(correct, total),
        "correct": correct,
        "total": total,
        "extraction_failed": extraction_failed,
        "extraction_failed_rate": safe_div(extraction_failed, total),
        "per_sample_results": per_sample_results,
    }


def group_all_correct(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], str],
    member_fn: Callable[[Dict[str, Any]], str],
    required_members: Sequence[str],
) -> Dict[str, Any]:
    """组级“全对”二级指标:按 key_fn 分组，组内 required_members 指定的成员全部答对才算该组通过。

    用于 FanToM 的 set 级 ALL(同一 snippet 内各题型全对)、TactfulToM 的 Comp∧Just 联合
    (同一对话 comprehension 与 justification 都对)。仅统计“包含全部 required_members”的组，
    避免缺成员的组虚高/虚低。返回 {rate, passed, total}。
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for record, result in zip(records, per_sample_results):
        key = key_fn(record)
        member = member_fn(record)
        bucket = groups.setdefault(key, {"members": {}, "present": set()})
        bucket["present"].add(member)
        if member in required_members:
            prev = bucket["members"].get(member, True)
            bucket["members"][member] = prev and bool(result["is_correct"])

    passed = 0
    total = 0
    for bucket in groups.values():
        if not all(member in bucket["present"] for member in required_members):
            continue
        total += 1
        if all(bucket["members"].get(member, False) for member in required_members):
            passed += 1
    return {"rate": safe_div(passed, total), "passed": passed, "total": total}


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
