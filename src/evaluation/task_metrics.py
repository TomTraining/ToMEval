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


# --------------------------------------------------------------------------- #
# 分层指标（1/2/3 级）                                                          #
# --------------------------------------------------------------------------- #
# 统一模型：一切皆“二级指标（维度）”。一个维度 = 一组 split，每个 split 一条 ACC。
#   - 切分型维度（如 type / ability / task）把整个数据集切成多个 split，展示多项。
#   - 汇总型维度（如 FanToM set_all、Belief_R BREU、TactfulToM joint_comp_just）只有一个
#     split = 整个数据集，展示一项；这类取值无法从样本逐条反推，由任务算好后用
#     add_dimension() 直接挂上。
#   - 任意 split 内部若还能再切，挂三级维度（split["dimensions"]）。
# 输出结构（见 datasets/README.md 的指标规整化约定）：维度值直接是 split 字典，不再多包一层 splits。
#   metrics["dimensions"] = {
#       "<维度名>": {
#           "<split>": {"acc": float, "n": int,
#                       "dimensions": {...三级,可选...}},
#           ...
#       },
#       ...
#   }
# 一级指标仍是顶层 accuracy；每个数据集固定带一个 type 维度（按 prompt_type 切四种题型）。
# 维度声明用 DimSpec：(name, key_fn) 或 (name, key_fn, [子 DimSpec,...]) 带三级嵌套。

# DimSpec：二元 (name, key_fn) 或三元 (name, key_fn, sub_specs)。
DimSpec = Tuple[str, Callable[[Dict[str, Any]], Iterable[str]], Sequence[Any]]


def _unpack_dim_spec(spec: Sequence[Any]) -> Tuple[str, Callable[[Dict[str, Any]], Iterable[str]], Sequence[Any]]:
    name = spec[0]
    key_fn = spec[1]
    sub_specs = spec[2] if len(spec) > 2 else ()
    return name, key_fn, sub_specs or ()


def _prompt_type_key(record: Dict[str, Any]) -> List[str]:
    """固定 type 维度的分组键：取记录的 prompt_type（open / mcq_single / mcq_multi / mcq_grouped）。"""
    return [first_value(record.get("prompt_type"))]


def build_dimension(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], Iterable[str]],
    sub_specs: Sequence[Any] = (),
) -> Dict[str, Any]:
    """构建单个维度：按 key_fn 把样本分到各 split，返回 {key: {acc, n, [dimensions]}}。

    一个样本的 key_fn 可返回多个键（落入多个 split）。sub_specs 非空时，对每个 split 内的
    子样本递归构建三级维度，挂在该 split 的 "dimensions" 下。
    """
    buckets: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for record, result in zip(records, per_sample_results):
        keys = list(key_fn(record))
        if not keys:
            keys = ["unknown"]
        for key in keys:
            buckets.setdefault(first_value(key), []).append((record, result))

    splits: Dict[str, Any] = {}
    for key in sorted(buckets):
        pairs = buckets[key]
        total = len(pairs)
        correct = sum(1 for _, result in pairs if result["is_correct"])
        split: Dict[str, Any] = {"acc": safe_div(correct, total), "n": total}
        if sub_specs:
            sub_records = [pair[0] for pair in pairs]
            sub_results = [pair[1] for pair in pairs]
            child: Dict[str, Any] = {}
            for spec in sub_specs:
                sub_name, sub_key_fn, sub_sub = _unpack_dim_spec(spec)
                child[sub_name] = build_dimension(sub_records, sub_results, sub_key_fn, sub_sub)
            split["dimensions"] = child
        splits[key] = split
    return splits


def hierarchical_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
    dim_specs: Sequence[Any] = (),
    include_type: bool = True,
) -> Dict[str, Any]:
    """拼装分层指标：base_metric_payload + dimensions（固定 type 维度 + dim_specs 声明的各维度）。

    include_type=False 仅用于确实没有题型概念的特殊场景；默认所有数据集都带 type 维度。
    """
    metrics = base_metric_payload(per_sample_results)
    dimensions: Dict[str, Any] = {}
    if include_type:
        dimensions["type"] = build_dimension(records, per_sample_results, _prompt_type_key)
    for spec in dim_specs:
        name, key_fn, sub_specs = _unpack_dim_spec(spec)
        dimensions[name] = build_dimension(records, per_sample_results, key_fn, sub_specs)
    metrics["dimensions"] = dimensions
    return metrics


def make_split(acc: float, n: int, sub_dimensions: Dict[str, Any] = None) -> Dict[str, Any]:
    """构造单个 split 的取值 {"acc": ..., "n": ...[, "dimensions": ...]}。

    acc 既可放 0-1 准确率，也可放任意标量（如 Q4 的 0-10 rubric 均分）；可视化按数据自适应。
    """
    split: Dict[str, Any] = {"acc": float(acc), "n": int(n)}
    if sub_dimensions:
        split["dimensions"] = sub_dimensions
    return split


def add_dimension(metrics: Dict[str, Any], name: str, splits: Dict[str, Any]) -> Dict[str, Any]:
    """把任务自己算好的一个二级维度挂到 metrics["dimensions"][name] 下。

    splits 形如 {split_key: make_split(...)}。汇总型指标（无法从分组 ACC 反推的特有口径，
    如 FanToM set 级 ALL、Belief-R BREU、TactfulToM Comp∧Just）只有一个 "overall" split，
    即整个数据集；切分型指标则有多个 split。统一收进 dimensions，顶层不再留散落的特殊键。
    """
    metrics.setdefault("dimensions", {})[name] = splits
    return metrics
