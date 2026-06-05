"""
从 ToMEval 评估结果加载 bad case。

读取 results/<dataset>/<model>/exp_*/prediction.jsonl（自动取最新 exp_*），
对配置的多个模型取 bad case 并集，每条 bad case 保留：
  - row_data: {story, question, correct_answers, wrong_answers, meta, prompt_type}
  - _actual_prompt: 原始 prompt（来自 prediction）
  - _wrong_answer: 模型给出的错误答案（raw_prediction letter）
  - _wrong_reasoning: 模型 reasoning（若有）
  - _gold_answer: 正确答案 letter（correct_letters[0]）
  - _hard: bool，≥2 个模型均答错 = True（高价值合成目标）

输出：data_output/bad_cases/<dataset_name>/bad_cases.jsonl
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

def _extract_letter(text: Optional[str]) -> str:
    """从模型回复中提取答案字母"""
    if not text:
        return ""
    text = str(text).strip()
    m = re.search(r"\b([A-Oa-o])\b", text)
    if m:
        return m.group(1).upper()
    if text and text[0].isalpha():
        return text[0].upper()
    return text.upper()[:1]


def _is_correct(raw_prediction: Optional[str], correct_letters: List[str]) -> bool:
    letter = _extract_letter(raw_prediction)
    return letter in [c.upper() for c in correct_letters]


def _find_prediction_file(results_root: Path, dataset_name: str, model: str, exp: Optional[str] = None) -> Optional[Path]:
    """在 results/<dataset>/<model>/exp_<tag>/ 下找 prediction.jsonl。

    exp 为 None 或空时自动取最新的 exp_* 目录。
    """
    base = results_root / dataset_name / model
    if not base.exists():
        return None
    if exp:
        tag = exp if exp.startswith("exp_") else f"exp_{exp}"
        pred = base / tag / "prediction.jsonl"
        return pred if pred.exists() else None
    # 自动取最新
    for exp_dir in reversed(sorted(base.glob("exp_*"))):
        pred = exp_dir / "prediction.jsonl"
        if pred.exists():
            return pred
    return None


def _find_metrics_file(pred_file: Path) -> Optional[Path]:
    """返回与 prediction.jsonl 同目录的 metrics.json，不存在则返回 None。"""
    m = pred_file.parent / "metrics.json"
    return m if m.exists() else None


def _load_judge_results(metrics_file: Path) -> Dict[Tuple[str, int], bool]:
    """从 metrics.json 加载 judge model 的判断结果。

    返回 {(sample_id_str, repeat): is_correct} 字典。
    当同一 (sample_id, repeat) 在多个 run 中出现时，任一 run 判为错误即视为错误
    （与 bad case 取并集的语义一致）。
    """
    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)

    results: Dict[Tuple[str, int], bool] = {}
    for run in data.get("all_metrics", []):
        for r in run.get("per_sample_results", []):
            key = (str(r["sample_id"]), int(r.get("repeat", 0)))
            # 任一 run 答错则标记为错误
            if not r.get("is_correct", True):
                results[key] = False
            elif key not in results:
                results[key] = True
    return results


def load_bad_cases_from_predictions(
    dataset_name: str,
    predictions_root: str,
    models: List[Dict[str, Any]],
    max_bad_cases: int = 0,
    output_dir: str = "data_output/bad_cases",
) -> Path:
    """
    读取多个模型的 prediction.jsonl，取 bad case 并集，写入 bad_cases.jsonl。

    Args:
        dataset_name:      数据集名称（ToMBench / BigToM / ...）
        predictions_root:  ToMEval results/ 根目录，布局为 <dataset>/<model>/exp_<tag>/prediction.jsonl
        models:            模型列表，每项 {"name": str, "exp": str|None}
        max_bad_cases:     0=不限制，>0=每数据集最多 N 条 bad case
        output_dir:        输出根目录

    Returns:
        包含 bad_cases.jsonl 的目录 Path
    """
    if not models:
        raise ValueError("models list must be non-empty; configure models under each dataset in config.yaml")

    results_root = Path(predictions_root)

    # sample_index → {model_name: [records]}
    sample_records: Dict[int, Dict[str, List[Dict]]] = {}
    # model_name → judge 查找表 {(sample_id_str, repeat): is_correct}
    judge_tables: Dict[str, Dict[Tuple[str, int], bool]] = {}

    for model_cfg in models:
        model = model_cfg["name"]
        exp = model_cfg.get("exp") or None
        pred_file = _find_prediction_file(results_root, dataset_name, model, exp)
        if pred_file is None:
            logger.warning(f"    ⚠️  找不到预测文件: {model} (exp={exp})")
            continue
        logger.info(f"    📄 读取: {pred_file.relative_to(results_root)}")

        # 加载 judge model 判断结果（metrics.json 与 prediction.jsonl 同目录）
        metrics_file = _find_metrics_file(pred_file)
        if metrics_file is not None:
            judge_tables[model] = _load_judge_results(metrics_file)
            logger.info(f"       ✓ 加载 judge 结果: {metrics_file.name} ({len(judge_tables[model])} 条)")
        else:
            logger.warning(f"       ⚠️  未找到 metrics.json，将用规则判断对错: {pred_file.parent}")

        model_records: Dict[int, List[Dict]] = {}
        with open(pred_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = rec.get("sample_index", rec.get("sample_idx", -1))
                if idx < 0:
                    continue
                if idx not in model_records:
                    model_records[idx] = []
                model_records[idx].append(rec)

        for idx, recs in model_records.items():
            if idx not in sample_records:
                sample_records[idx] = {}
            sample_records[idx][model] = recs

        logger.info(f"       ✓ 加载 {len(model_records)} 个样本")

    # 对每个样本，把所有答错的 (model, repeat) 各自产生一条 bad case
    bad_cases: List[Dict[str, Any]] = []

    def _judge_is_wrong(model_name: str, rec: Dict) -> bool:
        """优先用 metrics.json 的 judge 结果，没有则 fallback 到规则判断。"""
        judge = judge_tables.get(model_name)
        if judge is not None:
            key = (str(rec.get("sample_id", rec.get("sample_index", ""))), int(rec.get("repeat", 0)))
            is_correct = judge.get(key)
            if is_correct is not None:
                return not is_correct
        # fallback：规则判断
        pred_str = str(rec.get("raw_prediction", "") or rec.get("pred", {}).get("content", "") or "")
        return not _is_correct(pred_str, rec.get("correct_letters", []))

    for sample_idx in sorted(sample_records.keys()):
        model_data = sample_records[sample_idx]

        # 取任一模型的第一条记录拿到 row_data
        first_model = next(iter(model_data))
        ref_rec = model_data[first_model][0]

        correct_letters = ref_rec.get("correct_letters", [])
        gold = correct_letters[0] if correct_letters else ""

        row_data = {
            "story":           ref_rec.get("story", ""),
            "question":        ref_rec.get("question", ""),
            "correct_answers": ref_rec.get("correct_answers", []),
            "wrong_answers":   ref_rec.get("wrong_answers", []),
            "meta":            ref_rec.get("meta", {}),
            "prompt_type":     ref_rec.get("prompt_type", "mcq_single"),
        }

        # 统计该 sample 跨所有模型的总答错 repeat 数，作为难度评级
        total_wrong_count = 0
        for model_name, model_recs in model_data.items():
            for r in model_recs:
                if _judge_is_wrong(model_name, r):
                    total_wrong_count += 1

        if total_wrong_count == 0:
            continue

        # 每个答错的 repeat 单独产生一条 bad case，_difficulty = 总答错次数
        for model_name, model_recs in model_data.items():
            for r in model_recs:
                if not _judge_is_wrong(model_name, r):
                    continue
                pred_str = str(r.get("raw_prediction", "") or r.get("pred", {}).get("content", "") or "")
                bad_case = {
                    "sample_idx":       sample_idx,
                    "dataset":          dataset_name,
                    "row_data":         row_data,
                    "_actual_prompt":   r.get("prompt", ""),
                    "_wrong_answer":    _extract_letter(pred_str),
                    "_wrong_reasoning": str(r.get("pred", {}).get("reasoning", "") or "").strip(),
                    "_gold_answer":     gold,
                    "_difficulty":      total_wrong_count,
                    "_model":           model_name,
                    "_repeat":          r.get("repeat", 0),
                }
                bad_cases.append(bad_case)

    total_bad = len(bad_cases)
    logger.info(f"    📊 收集到 {total_bad} 条错误记录（跨模型、跨 repeat）")

    # difficulty 高的优先，同 sample_idx 内按 model/repeat 稳定排序
    bad_cases.sort(key=lambda c: (-c["_difficulty"], c["sample_idx"], c["_model"], c["_repeat"]))

    if max_bad_cases > 0 and len(bad_cases) > max_bad_cases:
        import collections

        by_dim: Dict[str, List[Dict]] = collections.defaultdict(list)
        for bc in bad_cases:
            dim = get_dimension_key(bc["row_data"].get("meta", {}), dataset_name)
            by_dim[dim].append(bc)

        full_dist = dict(collections.Counter(
            get_dimension_key(bc["row_data"].get("meta", {}), dataset_name) for bc in bad_cases
        ))
        logger.info(f"    ⚠️  超出上限 {max_bad_cases}，按维度比例采样...")

        if len(by_dim) > 1:
            total_bad = len(bad_cases)
            dim_quotas: Dict[str, int] = {}
            allocated = 0
            dims_sorted = sorted(by_dim.items())

            for dim, cases in dims_sorted:
                q = max(1, round(max_bad_cases * len(cases) / total_bad))
                dim_quotas[dim] = q
                allocated += q
            diff = allocated - max_bad_cases
            for dim, _ in sorted(dims_sorted, key=lambda x: -len(x[1])):
                if diff == 0:
                    break
                if diff > 0 and dim_quotas[dim] > 1:
                    dim_quotas[dim] -= 1
                    diff -= 1
                elif diff < 0:
                    dim_quotas[dim] += 1
                    diff += 1

            selected: List[Dict[str, Any]] = []
            remainder_pool: List[Dict[str, Any]] = []
            for dim, cases in dims_sorted:
                quota = dim_quotas[dim]
                selected.extend(cases[:quota])
                remainder_pool.extend(cases[quota:])
            remaining = max_bad_cases - len(selected)
            if remaining > 0 and remainder_pool:
                selected.extend(remainder_pool[:remaining])
            bad_cases = selected[:max_bad_cases]
            logger.info(f"       采样后保留 {len(bad_cases)} 条")
        else:
            logger.info(f"       采样后保留 {max_bad_cases} 条（原 {len(bad_cases)} 条）")
            bad_cases = bad_cases[:max_bad_cases]

    # 写输出
    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bad_cases.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for case in bad_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    logger.info(f"    💾 保存到: {out_path}")

    # 计算每个维度的总样本数、答错样本数、错误率，供 Stage 2 按错误率分配报告配额
    dim_total: Dict[str, int] = {}
    dim_wrong: Dict[str, int] = {}
    for sample_idx in sorted(sample_records.keys()):
        model_data = sample_records[sample_idx]
        first_model = next(iter(model_data))
        ref_rec = model_data[first_model][0]
        meta = ref_rec.get("meta", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        dim = get_dimension_key(meta, dataset_name)
        dim_total[dim] = dim_total.get(dim, 0) + 1
        correct_letters = ref_rec.get("correct_letters", [])
        has_wrong = any(
            _judge_is_wrong(model_name, r)
            for model_name, recs in model_data.items()
            for r in recs
        )
        if has_wrong:
            dim_wrong[dim] = dim_wrong.get(dim, 0) + 1

    dim_stats = {
        dim: {
            "total": dim_total[dim],
            "wrong": dim_wrong.get(dim, 0),
            "error_rate": dim_wrong.get(dim, 0) / dim_total[dim] if dim_total[dim] > 0 else 0.0,
        }
        for dim in dim_total
    }
    stats_path = out_dir / "dim_stats.json"
    stats_path.write_text(json.dumps(dim_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 显示 top 5 错误率最高的维度
    top_error_dims = sorted(dim_stats.items(), key=lambda x: -x[1]["error_rate"])[:5]
    logger.info(f"    📈 维度错误率 Top 5:")
    for dim, stats in top_error_dims:
        logger.info(f"       • {dim}: {stats['error_rate']:.1%} ({stats['wrong']}/{stats['total']})")

    return out_dir


def load_bad_cases_all_datasets(
    config: Dict[str, Any],
    max_bad_cases: int = 0,
    only_dataset: str = "",
) -> Dict[str, Path]:
    """对所有配置数据集加载 bad case"""
    predictions_root = config.get("predictions_root", "results")
    output_dir = str(Path(config["output_path"]) / "bad_cases")

    synthesis_datasets = config.get("synthesis_datasets", [])
    results: Dict[str, Path] = {}

    for dataset_info in synthesis_datasets:
        dataset_name = dataset_info["name"]
        if only_dataset and dataset_name != only_dataset:
            continue
        models = dataset_info.get("models", [])
        if not models:
            logger.warning(f"  {dataset_name}: no models configured, skipping")
            continue
        try:
            out_dir = load_bad_cases_from_predictions(
                dataset_name=dataset_name,
                predictions_root=predictions_root,
                models=models,
                max_bad_cases=max_bad_cases,
                output_dir=output_dir,
            )
            results[dataset_name] = out_dir
        except Exception as e:
            logger.error(f"Failed to load bad cases for {dataset_name}: {e}")
            import traceback; traceback.print_exc()

    return results


def get_dimension_key(meta: Any, dataset_name: str) -> str:
    """从 meta 字段提取维度键（兼容 dict/str/None 及大小写字段名）

    各数据集维度字段映射：
      ToMBench    → meta.ability
      BigToM      → meta.condition_type（优先），否则 meta.dimension[0]
      SocialIQA / EmoBench → meta.dimension[0]
      HiToM       → str(meta.order)
      FanToM      → meta.question_type，或从 meta.id 解析

    多语言数据集（ToMBench/EmoBench 含 zh 行）按语言拆分维度组：
    非英文样本的维度键追加 "__{lang}" 后缀，保证诊断/合成的语言闭环干净。
    """
    from src.evaluation.lang import get_sample_lang

    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}

    # zh/en bad case 不混组：诊断报告与后续合成的语言由分组决定。
    lang = get_sample_lang(meta)
    suffix = f"__{lang}" if lang != "en" else ""

    if dataset_name == "ToMBench":
        ability = meta.get("ability", meta.get("Ability", ""))
        return (str(ability) if ability else "unknown") + suffix

    elif dataset_name == "BigToM":
        ct = meta.get("condition_type", meta.get("Condition_type", ""))
        if ct:
            return str(ct)
        dim = meta.get("dimension", meta.get("Dimension", ["unknown"]))
        if isinstance(dim, list):
            return str(dim[0]) if dim else "unknown"
        return str(dim) if dim else "unknown"

    elif dataset_name in ("SocialIQA", "EmoBench"):
        dim = meta.get("dimension", meta.get("Dimension", ["unknown"]))
        if isinstance(dim, list):
            return (str(dim[0]) if dim else "unknown") + suffix
        return (str(dim) if dim else "unknown") + suffix

    elif dataset_name == "HiToM":
        order = meta.get("order", meta.get("Order", None))
        return f"order_{order}" if order is not None else "unknown"

    elif dataset_name == "FanToM":
        qt = meta.get("question_type", meta.get("question_Type", ""))
        if qt:
            return str(qt)
        meta_id = str(meta.get("id", ""))
        parts = meta_id.split("__")
        return parts[1] if len(parts) > 1 else "unknown"

    else:
        dim = meta.get("dimension", meta.get("Dimension", ["unknown"]))
        if isinstance(dim, list):
            return str(dim[0]) if dim else "unknown"
        return str(dim) if dim else "unknown"
