"""SoMBench 分层指标：维度层级 dim1→dim2→dim3 嵌套为三级，题型/视角/难度变体/长短文本
作为平铺的二级维度；Q4 rubric 平均分（0-10，非 0-1 准确率）作 q4_score 汇总型维度
（overall + 按三级维度切 split）。

注：variant 切面区分 base（dim 1/2/3 全量 284 样本）与 hardest（dim-3 加难版 56 样本，
按模型错误率挑最难子题改写而成，与 base dim-3 一一对应），variant 维度可对比同批题在两难度上的
表现。`qualified` 只在「人工审核合格」样本（meta.review_pass=True）上重算同一套指标，作为
顶层镜像（{accuracy, dimensions, ...}）；v5.3 provenance 标明 FAIL 项已就地修复、裁判结论为
最终权威，故全部样本默认合格，qualified 与全量一致。缺失 review_pass 字段的样本默认视为合格。"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    add_dimension,
    first_value,
    hierarchical_metrics,
    make_split,
    meta,
    safe_div,
)


def _dim_key(key: str):
    return lambda r: [first_value(meta(r).get(key))]


# 二级维度声明：dim1→dim2→dim3 天然层级嵌套，其余切面平铺。
DIM_SPECS: List[Any] = [
    # 维度层级：一级维度 dim1，其下嵌套二级维度 dim2，再嵌套三级维度 dim（dim3）。
    ("dim1", _dim_key("dim1"), [
        ("dim2", _dim_key("dim2"), [
            ("dim3", _dim_key("dim")),
        ]),
    ]),
    ("qtype", _dim_key("qtype")),
    ("perspective", _dim_key("perspective")),
    ("variant", _dim_key("variant")),
    # 长/短文本情景：meta.length_mode 取 long/short。
    ("length", _dim_key("length_mode")),
]


def _is_qualified(record: Dict[str, Any]) -> bool:
    """meta.review_pass 缺失时默认合格，保证未打标数据/其他数据集不受影响。"""
    return bool(meta(record).get("review_pass", True))


def _add_q4_dimension(
    metrics: Dict[str, Any],
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> None:
    """Q4 rubric 平均分（0-10）作 q4_score 汇总型维度：overall split + 按三级维度切 split。"""
    overall: List[float] = []
    by_dim: Dict[str, List[float]] = {}
    for record, result in zip(records, per_sample_results):
        score = result.get("judge_score")
        if record.get("prompt_type") != "open" or score is None:
            continue
        overall.append(float(score))
        dim = first_value(meta(record).get("dim"))
        by_dim.setdefault(dim, []).append(float(score))
    if not overall:
        return
    splits = {"overall": make_split(sum(overall) / len(overall), len(overall))}
    for dim, scores in sorted(by_dim.items()):
        splits[dim] = make_split(sum(scores) / len(scores), len(scores))
    add_dimension(metrics, "q4_score", splits)


def _block(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """一套完整指标（分层准确率 + Q4 平均分维度）。"""
    block = hierarchical_metrics(records, per_sample_results, DIM_SPECS)
    _add_q4_dimension(block, records, per_sample_results)
    return block


def compute_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = _block(records, per_sample_results)

    # qualified 镜像：仅人工审核合格样本（review_pass=True）上重算同一套指标。
    pairs = [
        (r, res) for r, res in zip(records, per_sample_results) if _is_qualified(r)
    ]
    qualified_records = [p[0] for p in pairs]
    qualified_results = [p[1] for p in pairs]

    total = len(records)
    n_qualified = len(qualified_records)
    qualified = _block(qualified_records, qualified_results)
    # 镜像自带的 per_sample_results 与顶层重复，去掉以免膨胀。
    qualified.pop("per_sample_results", None)

    metrics["review_pass_count"] = n_qualified
    metrics["review_fail_count"] = total - n_qualified
    metrics["review_pass_rate"] = safe_div(n_qualified, total)
    metrics["qualified"] = qualified
    return metrics
