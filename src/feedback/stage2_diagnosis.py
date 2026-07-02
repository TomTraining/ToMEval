"""
阶段二：维度分类 + 批量诊断

功能：
1. 读取阶段一输出的 bad_cases.jsonl
2. 按能力维度字段对 bad case 分组
3. 读取 dim_stats.json（Stage 1 生成），按维度内部错误率分配报告配额
   - 若无 dim_stats.json，fallback 到按 bad case 数量比例分配
4. 维度内按 _difficulty 加权采样 bad case，每条生成一个 DimensionDiagnosisReport
5. 保存 dimension_reports.jsonl

输出目录结构：
    data_output/diagnosis_reports/{dataset_name}/
        dimension_reports.jsonl
        dimension_coverage.json
"""

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import runner
from src.llm.llm_utils import extract_json
from .prompts import build_batch_diagnosis_prompt, DATASET_SKILL_REGISTRY  # noqa: F401
from .stage1_load_predictions import get_dimension_key

logger = logging.getLogger(__name__)


# ============ DimensionDiagnosisReport Schema ============

class DimensionDiagnosisReport(BaseModel):
    """维度级诊断报告（来自 K 条 bad case 的跨样本归纳）"""
    dimension: str = Field(description="The dimension/ability being diagnosed")
    common_error_patterns: List[str] = Field(
        description="Abstract error patterns using placeholder Agent A/B/Object X/Location X"
    )
    primary_cognitive_operation: Literal[
        "belief_state_update",
        "perspective_taking",
        "emotion_attribution",
        "recursive_belief",
        "information_access",
        "social_inference",
        "causal_reasoning",
        "temporal_reference",
        "pragmatic_inference",
        "referent_resolution",
        "common_sense_normative",
    ] = Field(description="The primary cognitive operation that failed")
    secondary_cognitive_operations: List[str] = Field(
        default_factory=list,
        description="Secondary cognitive operations that also failed"
    )
    recommended_synthesis_themes: List[str] = Field(
        description="Recommended story themes/contexts for synthesizing new questions"
    )
    difficulty_distribution: Dict[str, float] = Field(
        description="Distribution of difficulties, e.g. {'easy': 0.2, 'medium': 0.5, 'hard': 0.3}"
    )


_FALLBACK_REPORT = {
    "dimension": "unknown",
    "common_error_patterns": [
        "Agent A failed to update belief about Object X after unwitnessed event",
        "Model used world-state truth instead of Agent A's limited perspective",
    ],
    "primary_cognitive_operation": "belief_state_update",
    "secondary_cognitive_operations": [],
    "recommended_synthesis_themes": [
        "everyday household scenarios",
        "workplace interactions",
        "social gatherings",
    ],
    "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
}


def _load_bad_cases_from_file(
    prediction_path: Path,
    data_path: Path,
) -> List[Dict[str, Any]]:
    """从 bad_cases.jsonl（stage1 生成）或旧版 prediction.jsonl + data.jsonl 加载 bad case。"""
    if prediction_path.name == "bad_cases.jsonl" and prediction_path.exists():
        bad_cases = []
        with open(prediction_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                bad_cases.append(json.loads(line))
        return bad_cases

    # 旧版：prediction.jsonl + data.jsonl
    row_map: Dict[int, Dict[str, Any]] = {}
    if data_path.exists():
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                row_map[obj["sample_idx"]] = obj.get("row_data", {})

    samples: Dict[int, List[Dict[str, Any]]] = {}
    with open(prediction_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("sample_idx", rec.get("sample_index", -1))
            if idx < 0:
                continue
            if idx not in samples:
                samples[idx] = []
            samples[idx].append(rec)

    bad_cases: List[Dict[str, Any]] = []
    for sample_idx, recs in sorted(samples.items()):
        any_wrong = any(not r.get("is_correct", True) for r in recs)
        if not any_wrong:
            continue
        first_wrong = next((r for r in recs if not r.get("is_correct", True)), recs[0])
        pred_obj = first_wrong.get("pred", {}) or {}
        content = pred_obj.get("content") or {}
        reasoning = pred_obj.get("reasoning") or ""
        if isinstance(content, dict):
            wrong_answer = content.get("answer") or str(content)
        else:
            wrong_answer = str(content) if content is not None else "(no answer)"
        row_data = row_map.get(sample_idx, {})
        bad_cases.append({
            "sample_idx":       sample_idx,
            "row_data":         row_data,
            "_actual_prompt":   first_wrong.get("prompt", ""),
            "_wrong_answer":    str(wrong_answer).strip() or "(no answer)",
            "_wrong_reasoning": str(reasoning).strip(),
            "_gold_answer":     str(first_wrong.get("gold_answer", "")).strip(),
        })
    return bad_cases


# ============ 报告分配和采样函数 ============

def allocate_reports_by_dimension(
    dim_counts: Dict[str, int],
    total_reports: int,
    dim_totals: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """按维度错误率（或错误数量比例）分配报告配额，总和精确等于 total_reports。

    若提供 dim_totals（每个维度的总样本数），则用维度内部错误率作为权重：
        weight[dim] = dim_counts[dim] / dim_totals[dim]
    否则 fallback 到原逻辑：用 bad case 数量占总 bad case 的比例。

    Args:
        dim_counts: 每个维度的 bad case 数量
        total_reports: 总报告数（target_samples / samples_per_report）
        dim_totals: 每个维度的总样本数（来自 Stage 1 的 dim_stats.json）

    Returns:
        每个维度分配的报告数，总和 = total_reports
    """
    if not dim_counts or total_reports <= 0:
        return {}

    if dim_totals:
        weights = {
            dim: (count / dim_totals[dim]) if dim_totals.get(dim, 0) > 0 else 0.0
            for dim, count in dim_counts.items()
        }
        total_weight = sum(weights.values())
        if total_weight == 0:
            weights = {dim: 1.0 for dim in dim_counts}
            total_weight = float(len(dim_counts))
    else:
        total_bad = sum(dim_counts.values())
        if total_bad == 0:
            weights = {dim: 1.0 for dim in dim_counts}
            total_weight = float(len(dim_counts))
        else:
            weights = {dim: count / total_bad for dim, count in dim_counts.items()}
            total_weight = 1.0

    # 维度数 > 报告数：只挑权重最高的 top-N 维度各 1 份，其余 0 份
    if len(dim_counts) > total_reports:
        ranked = sorted(
            weights.items(),
            key=lambda kv: (-kv[1], -dim_counts.get(kv[0], 0)),
        )
        return {dim: 1 for dim, _ in ranked[:total_reports]}

    # 按权重比例向下取整，每个维度至少 1 份
    raw: Dict[str, int] = {}
    for dim, w in weights.items():
        raw[dim] = max(1, int(total_reports * w / total_weight))

    # 修正舍入差：将剩余配额按权重从大到小依次补给
    diff = total_reports - sum(raw.values())
    for dim in sorted(dim_counts, key=lambda d: -weights.get(d, 0)):
        if diff <= 0:
            break
        raw[dim] += 1
        diff -= 1

    # 若舍入导致超出：从权重最小的维度截减
    while sum(raw.values()) > total_reports:
        progressed = False
        for dim in sorted(raw, key=lambda d: weights.get(d, 0)):
            if sum(raw.values()) <= total_reports:
                break
            if raw[dim] > 1:
                raw[dim] -= 1
                progressed = True
        if not progressed:
            break

    return raw


def sample_bad_cases_weighted(
    bad_cases: List[Dict[str, Any]],
    n_samples: int,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """按 _difficulty 加权采样 bad case（difficulty 越高，被选中概率越大）"""
    import random
    random.seed(random_seed)

    if n_samples >= len(bad_cases):
        return bad_cases

    weights = [float(case.get("_difficulty", 1)) for case in bad_cases]
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0] * len(bad_cases)

    # 不放回加权采样
    population = list(range(len(bad_cases)))
    norm_weights = [w / sum(weights) for w in weights]
    chosen_indices = []
    seen = set()
    attempts = 0
    while len(chosen_indices) < n_samples and attempts < n_samples * 10:
        attempts += 1
        idx = random.choices(population, weights=norm_weights, k=1)[0]
        if idx not in seen:
            seen.add(idx)
            chosen_indices.append(idx)

    # 若仍不足，顺序补充
    if len(chosen_indices) < n_samples:
        for i in range(len(bad_cases)):
            if i not in seen:
                chosen_indices.append(i)
            if len(chosen_indices) >= n_samples:
                break

    return [bad_cases[i] for i in chosen_indices[:n_samples]]


# ============ 主诊断函数 ============

def run_stage2_dimension_diagnosis(
    stage1_dir: str,
    dataset_name: str,
    synthesis_llm_config: Dict[str, Any],
    total_reports: int,
    bad_cases_per_report: int = 1,
    output_dir: str = "data_output/diagnosis_reports",
    batch_size: int = 10,
) -> Path:
    """读取 bad_cases.jsonl，按维度错误率分配报告，每个报告输入 bad_cases_per_report 条 bad case

    逻辑：
      1. 报告数 = target_samples / samples_per_report（由调用方计算传入）
      2. 读取 dim_stats.json（Stage 1 生成），按维度内部错误率分配报告配额
         - 若无 dim_stats.json，fallback 到按 bad case 数量比例分配
      3. 维度内按 _difficulty 加权采样 bad case
      4. bad_cases_per_report 条 bad case → 一个诊断报告

    Args:
        stage1_dir:            bad_cases.jsonl 所在目录
        dataset_name:          数据集名称
        synthesis_llm_config:  诊断模型配置
        total_reports:         该数据集的总报告数（target_samples / samples_per_report）
        bad_cases_per_report:  每个诊断报告输入几条 bad case（默认 1）
        output_dir:            输出根目录

    Returns:
        包含 dimension_reports.jsonl 的输出目录 Path
    """
    stage1_dir = Path(stage1_dir)

    # 优先读新格式 bad_cases.jsonl
    bad_cases_file = stage1_dir / "bad_cases.jsonl"
    if bad_cases_file.exists():
        prediction_path = bad_cases_file
        data_file_path = bad_cases_file  # 同一文件，load 函数会识别
    else:
        prediction_path = stage1_dir / "prediction.jsonl"
        data_file_path = stage1_dir / "data.jsonl"

    if not prediction_path.exists():
        raise FileNotFoundError(f"bad_cases.jsonl or prediction.jsonl not found in: {stage1_dir}")

    # 收集 bad case
    bad_cases = _load_bad_cases_from_file(prediction_path, data_file_path)
    logger.info(f"    📖 加载 {len(bad_cases)} 道错题")

    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_path = out_dir / "dimension_reports.jsonl"

    if not bad_cases:
        logger.warning("    ⚠️  无错题数据，创建空报告文件")
        reports_path.write_text("", encoding="utf-8")
        return out_dir

    # 按维度分组
    dimension_groups: Dict[str, List[Dict[str, Any]]] = {}
    for case in bad_cases:
        row_data = case.get("row_data", {})
        meta = row_data.get("meta", row_data.get("Meta", {}))
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        dim = get_dimension_key(meta, dataset_name)
        if dim not in dimension_groups:
            dimension_groups[dim] = []
        dimension_groups[dim].append(case)

    dim_counts = {dim: len(cases) for dim, cases in dimension_groups.items()}
    logger.info(f"    🏷️  按维度分组: {len(dimension_groups)} 个维度")

    # 显示 top 5 维度
    top_dims = sorted(dim_counts.items(), key=lambda x: -x[1])[:5]
    for dim, count in top_dims:
        logger.info(f"       • {dim}: {count} 道错题")
    if len(dim_counts) > 5:
        logger.info(f"       ... 还有 {len(dim_counts) - 5} 个维度")

    # 尝试读取 Stage 1 生成的 dim_stats.json，用维度内部错误率作为权重
    dim_totals = None
    dim_stats_path = stage1_dir / "dim_stats.json"
    if dim_stats_path.exists():
        with open(dim_stats_path, encoding="utf-8") as f:
            dim_stats = json.load(f)
        dim_totals = {dim: v["total"] for dim, v in dim_stats.items()}
        logger.info(f"    📈 使用维度错误率分配报告配额")

    # 按维度错误率（或 fallback 到 bad case 数量比例）分配报告配额
    allocations = allocate_reports_by_dimension(dim_counts, total_reports, dim_totals=dim_totals)
    logger.info(f"    📊 报告分配: 总计 {sum(allocations.values())} 份")

    # 落盘维度覆盖率
    coverage_path = out_dir / "dimension_coverage.json"
    coverage_path.write_text(
        json.dumps(
            {"dim_counts": dim_counts, "allocations": allocations,
             "total_reports": sum(allocations.values())},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # 按维度采样 bad case，每 bad_cases_per_report 条生成一个报告
    all_prompts: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    # 读取已有报告，跳过已完成的 (dimension, case_idx) 对
    existing_pairs: set = set()
    if reports_path.exists():
        with open(reports_path, encoding="utf-8") as _f:
            for _line in _f:
                if _line.strip():
                    try:
                        _rec = json.loads(_line)
                        _m = _rec.get("_meta", {})
                        existing_pairs.add((_m.get("dimension"), _m.get("case_idx")))
                    except Exception:
                        pass
        if existing_pairs:
            logger.info(f"    ⏭️  已有 {len(existing_pairs)} 份报告，只生成缺失的")

    for dim, n_reports in allocations.items():
        dim_cases = dimension_groups.get(dim, [])
        # 需要采样的总 bad case 数 = 报告数 × 每报告输入条数
        n_to_sample = n_reports * bad_cases_per_report
        sampled_cases = sample_bad_cases_weighted(dim_cases, n_to_sample)

        for report_idx in range(n_reports):
            if (dim, report_idx) in existing_pairs:
                continue
            start = report_idx * bad_cases_per_report
            batch = sampled_cases[start: start + bad_cases_per_report]
            if not batch:
                break
            prompt = build_batch_diagnosis_prompt(
                bad_cases_batch=batch,
                dimension=dim,
                dataset_name=dataset_name,
            )
            all_prompts.append(prompt)
            all_meta.append({
                "dimension": dim,
                "case_idx": report_idx,
                "sample_idx": batch[0].get("sample_idx", -1),
                # 维度键的 __zh 后缀来自 get_dimension_key 的语言拆分，
                # 记录到报告 _meta，供 Stage 3 决定合成语言。
                "lang": "zh" if dim.endswith("__zh") else "en",
            })

    # 批量生成诊断报告
    if not all_prompts:
        logger.info("    ⏭️  所有报告已完成，跳过诊断")
        return out_dir

    logger.info(f"    🤖 调用 LLM 生成 {len(all_prompts)} 份诊断报告（每批 {batch_size} 份）...")
    client = runner.create_llm_client(synthesis_llm_config)

    success_count = 0
    fallback_count = 0
    for batch_start in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[batch_start: batch_start + batch_size]
        batch_meta = all_meta[batch_start: batch_start + batch_size]

        raw_results = client.batch_generate_structure(
            batch_prompts, DimensionDiagnosisReport, desc="诊断中"
        )

        with open(reports_path, "a", encoding="utf-8") as f:
            for meta_info, resp in zip(batch_meta, raw_results):
                result = resp.content if resp else None
                if result is not None:
                    record = result.model_dump()
                    record["_meta"] = meta_info
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success_count += 1
                else:
                    fallback = dict(_FALLBACK_REPORT)
                    fallback["dimension"] = meta_info["dimension"]
                    fallback["_meta"] = meta_info
                    fallback["_is_fallback"] = True
                    f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                    fallback_count += 1

        logger.info(f"    💾 已完成 {min(batch_start + batch_size, len(all_prompts))}/{len(all_prompts)} 份")

    logger.info(f"    ✅ 成功生成 {success_count} 份报告")
    if fallback_count > 0:
        logger.info(f"    ⚠️  {fallback_count} 份使用默认模板（LLM 生成失败）")
    logger.info(f"    💾 保存到: {reports_path}")
    return out_dir

