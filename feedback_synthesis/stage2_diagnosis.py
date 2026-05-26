"""
阶段二：维度分类 + 批量诊断

功能：
1. 读取阶段一输出的 prediction.jsonl + data.jsonl
2. 按能力维度字段对 bad case 分组
3. 每个维度按 ceil(n_c / K) 分配批次，确保所有 bad case 都被覆盖
4. 每批顺序取 K 条 bad case → 合成模型 → DimensionDiagnosisReport
5. 保存 dimension_reports.jsonl

输出目录结构：
    data_output/diagnosis_reports/{dataset_name}/{split}/
        dimension_reports.jsonl

与旧版的区别：
  旧版 (stage2_diagnosis.py) 对每条 bad case 单独诊断，输出 ErrorDiagnosisOutput。
  新版对每个维度批量诊断，输出 DimensionDiagnosisReport（跨样本归纳，更稳健）。
"""

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from .prompts import build_batch_diagnosis_prompt, DATASET_SKILL_REGISTRY  # noqa: F401


def _extract_json(text: str):
    """从文本中提取第一个合法 JSON 对象。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text):
        try:
            return json.loads(m.strip())
        except json.JSONDecodeError:
            continue
    # 找最大括号对
    start = text.find('{')
    if start != -1:
        depth, end = 0, -1
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _batch_generate_json(
    prompts: List[str],
    schema_cls: Type[BaseModel],
    llm_config: Dict[str, Any],
    desc: str = "Generating",
    max_retry: int = 3,
) -> List[Any]:
    """用 ContentClient 批量生成 JSON，解析为 Pydantic 对象。

    在每个 prompt 末尾附加 JSON schema 说明，强制模型输出 JSON。
    失败时返回 None（调用方自行处理 fallback）。
    """
    schema_json = json.dumps(schema_cls.model_json_schema(), ensure_ascii=False, indent=2)
    json_instruction = (
        f"\n\n---\nYou MUST respond with a single valid JSON object that strictly follows "
        f"this JSON Schema (no extra text, no markdown fences):\n{schema_json}"
    )
    augmented_prompts = [p + json_instruction for p in prompts]

    client = runner.create_content_client(llm_config, None)
    results_raw = client.batch_generate(augmented_prompts, desc=desc)

    parsed = []
    for resp in results_raw:
        content_text = resp.content if resp and resp.content else ""
        # content_text 可能是字符串或 Pydantic 对象
        if not isinstance(content_text, str):
            content_text = str(content_text)
        obj = None
        for _ in range(max_retry):
            data = _extract_json(content_text)
            if data is not None:
                try:
                    obj = schema_cls.model_validate(data)
                    break
                except Exception:
                    pass
            break  # raw output 已确定，重试无意义；retry 靠外层调用方
        parsed.append(obj)
    return parsed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
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


# ============ 维度提取函数 ============

def get_dimension_key(meta: Any, dataset_name: str) -> str:
    """从 meta 字段提取维度键（同时兼容小写 meta 和旧版大写 Meta）

    各数据集维度字段映射：
      ToMBench    → meta.ability（预测文件中有此字段）
      SocialIQA / BigToM / SimpleToM / EmoBench → meta.dimension[0]
      HiToM       → str(meta.order)
      FanToM      → meta.question_type，或从 meta.id 解析
    """
    # 兼容：meta 可能是 dict / str / None
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}

    if dataset_name == "ToMBench":
        ability = meta.get("ability", meta.get("Ability", ""))
        return str(ability) if ability else "unknown"

    elif dataset_name == "BigToM":
        # BigToM 按 condition_type 分组（6种：forward_action/backward_belief/true_belief/
        # percept_to_belief/forward_belief/false_belief），比 dimension 更有信息量
        ct = meta.get("condition_type", meta.get("Condition_type", ""))
        if ct:
            return str(ct)
        dim = meta.get("dimension", meta.get("Dimension", ["unknown"]))
        if isinstance(dim, list):
            return str(dim[0]) if dim else "unknown"
        return str(dim) if dim else "unknown"

    elif dataset_name in ("SocialIQA", "SimpleToM", "EmoBench"):
        dim = meta.get("dimension", meta.get("Dimension", ["unknown"]))
        if isinstance(dim, list):
            return str(dim[0]) if dim else "unknown"
        return str(dim) if dim else "unknown"

    elif dataset_name == "HiToM":
        order = meta.get("order", meta.get("Order", None))
        return f"order_{order}" if order is not None else "unknown"

    elif dataset_name == "FanToM":
        # 优先用 question_type 字段
        qt = meta.get("question_type", meta.get("question_Type", ""))
        if qt:
            return str(qt)
        # 回退：从 id 解析 "__<question_type>__"
        meta_id = str(meta.get("id", ""))
        parts = meta_id.split("__")
        return parts[1] if len(parts) > 1 else "unknown"

    else:
        dim = meta.get("dimension", meta.get("Dimension", ["unknown"]))
        if isinstance(dim, list):
            return str(dim[0]) if dim else "unknown"
        return str(dim) if dim else "unknown"


# ============ 批次分配函数 ============

def allocate_batches(
    dim_counts: Dict[str, int],
    samples_per_batch: int,
) -> Dict[str, int]:
    """按维度分配诊断批次，确保每个维度的所有 bad case 都被覆盖

    batches_c = ceil(n_c / K)，每个维度至少 1 批

    Args:
        dim_counts: 每个维度的 bad case 数量
        samples_per_batch: 每批样本数 K

    Returns:
        每个维度分配的批次数
    """
    return {
        dim: max(1, math.ceil(count / samples_per_batch))
        for dim, count in dim_counts.items()
    }


def allocate_batches_capped(
    dim_counts: Dict[str, int],
    max_reports: int,
) -> Dict[str, int]:
    """按维度 bad case 比例分配报告配额，总和精确等于 max_reports。

    错误多的维度获得更多报告。维度数超过 max_reports 时，
    只保留 bad case 最多的前 max_reports 个维度（每维度至少 1 份）。

    Args:
        dim_counts: 每个维度的 bad case 数量
        max_reports: 总报告配额上限（严格遵守）

    Returns:
        每个维度分配的报告数，总和 <= max_reports
    """
    if not dim_counts or max_reports <= 0:
        return {}

    # 若维度数超过 budget，只保留 bad case 最多的 top max_reports 个维度
    if len(dim_counts) > max_reports:
        sorted_dims = sorted(dim_counts.items(), key=lambda x: -x[1])
        dim_counts = dict(sorted_dims[:max_reports])

    n_dims = len(dim_counts)
    total = sum(dim_counts.values())

    # 按比例向下取整，每个维度至少 1 份
    raw: Dict[str, int] = {}
    for dim, count in dim_counts.items():
        raw[dim] = max(1, int(max_reports * count / total))

    # 修正舍入差：将剩余配额按 bad case 数从多到少依次补给
    diff = max_reports - sum(raw.values())
    for dim in sorted(dim_counts, key=lambda d: -dim_counts[d]):
        if diff <= 0:
            break
        raw[dim] += 1
        diff -= 1

    # 若舍入导致超出（理论上不会，但做防御）：从 bad case 最少的维度截减
    if sum(raw.values()) > max_reports:
        for dim in sorted(raw, key=lambda d: dim_counts[d]):
            if sum(raw.values()) <= max_reports:
                break
            if raw[dim] > 1:
                raw[dim] -= 1

    return raw


# ============ 从 Stage1 结果中读取 bad case ============

def load_bad_cases_from_predictions(
    prediction_path: Path,
    data_path: Path,
) -> List[Dict[str, Any]]:
    """从 bad_cases.jsonl（stage1_load_predictions 生成）或旧版 prediction.jsonl + data.jsonl 加载 bad case。

    优先读取 prediction_path 父目录同级的 bad_cases.jsonl（新格式）。
    若不存在，回退旧版 prediction.jsonl + data.jsonl 格式。
    """
    # 尝试新格式：prediction_path 可能就是 bad_cases.jsonl
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

        bad_case = {
            "sample_idx":       sample_idx,
            "row_data":         row_data,
            "_actual_prompt":   first_wrong.get("prompt", ""),
            "_wrong_answer":    str(wrong_answer).strip() or "(no answer)",
            "_wrong_reasoning": str(reasoning).strip(),
            "_gold_answer":     str(first_wrong.get("gold_answer", "")).strip(),
        }
        bad_cases.append(bad_case)

    return bad_cases


# ============ 主诊断函数 ============

def run_stage2_dimension_diagnosis(
    stage1_dir: str,
    dataset_name: str,
    synthesis_llm_config: Dict[str, Any],
    samples_per_batch: int = 5,
    output_dir: str = "data_output/diagnosis_reports",
    max_reports: Optional[int] = None,
) -> Path:
    """读取 bad_cases.jsonl（或旧版 stage1 输出），按维度分组批量诊断，保存 dimension_reports.jsonl

    Args:
        stage1_dir:            bad_cases.jsonl 所在目录（新格式）或旧版 stage1 输出目录
        dataset_name:          数据集名称
        synthesis_llm_config:  诊断模型配置（建议强模型）
        samples_per_batch:     每批样本数 K（当 max_reports 为 None 时用于无上限分配）
        output_dir:            输出根目录
        max_reports:           该数据集的报告配额上限（None = 无上限，用 samples_per_batch 决定批次）

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

    logger.info(f"Stage 2 Dimension Diagnosis: {dataset_name}")
    logger.info(f"  Reading from: {stage1_dir}")
    logger.info(f"  Samples per batch: {samples_per_batch}")

    # 收集 bad case
    bad_cases = load_bad_cases_from_predictions(prediction_path, data_file_path)
    logger.info(f"  Bad cases: {len(bad_cases)}")

    split_name = stage1_dir.name
    out_dir = Path(output_dir) / dataset_name / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_path = out_dir / "dimension_reports.jsonl"

    if not bad_cases:
        logger.warning("  No bad cases found, creating empty reports file")
        reports_path.write_text("", encoding="utf-8")
        return out_dir

    # 按维度分组（兼容新格式 bad_cases.jsonl 中 row_data 用小写 meta，以及旧版大写 Meta）
    dimension_groups: Dict[str, List[Dict[str, Any]]] = {}
    for case in bad_cases:
        row_data = case.get("row_data", {})
        # 优先小写 meta（新格式），回退大写 Meta（旧格式）
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
    logger.info(f"  Dimension groups: {dim_counts}")

    # 按比例分配批次
    if max_reports is not None:
        allocations = allocate_batches_capped(dim_counts, max_reports)
        logger.info(f"  Batch allocations (capped to {max_reports}): {allocations}")
    else:
        allocations = allocate_batches(dim_counts, samples_per_batch)
        logger.info(f"  Batch allocations: {allocations}")

    # 落盘维度覆盖率，便于 ITERATION_LOG 引用
    coverage_path = out_dir / "dimension_coverage.json"
    coverage_path.write_text(
        json.dumps(
            {"dim_counts": dim_counts, "allocations": allocations,
             "n_batches": sum(allocations.values())},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # 构建所有诊断 prompt（顺序切片，确保全覆盖）
    all_prompts: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for dim, n_batches in allocations.items():
        dim_cases = dimension_groups.get(dim, [])
        for batch_idx in range(n_batches):
            start = batch_idx * samples_per_batch
            batch = dim_cases[start: start + samples_per_batch]
            prompt = build_batch_diagnosis_prompt(
                bad_cases_batch=batch,
                dimension=dim,
                dataset_name=dataset_name,
            )
            all_prompts.append(prompt)
            all_meta.append({"dimension": dim, "batch_idx": batch_idx, "k": len(batch)})

    # 创建合成模型客户端（用 ContentClient + JSON 指令）
    logger.info(f"  Running {len(all_prompts)} batch diagnoses in parallel...")
    results = _batch_generate_json(
        all_prompts, DimensionDiagnosisReport, synthesis_llm_config, desc="Diagnosing"
    )

    # 写入 dimension_reports.jsonl
    success_count = 0
    with open(reports_path, "w", encoding="utf-8") as f:
        for meta_info, result in zip(all_meta, results):
            if result is not None:
                record = result.model_dump()
                record["_batch_meta"] = meta_info
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                success_count += 1
            else:
                # Fallback: 用默认报告，并标记维度
                fallback = dict(_FALLBACK_REPORT)
                fallback["dimension"] = meta_info["dimension"]
                fallback["_batch_meta"] = meta_info
                fallback["_is_fallback"] = True
                f.write(json.dumps(fallback, ensure_ascii=False) + "\n")

    logger.info(
        f"Stage 2 done: {success_count}/{len(all_prompts)} reports generated "
        f"({len(all_prompts) - success_count} fallback) → {reports_path}"
    )
    return out_dir

