"""
阶段五：难度验证

读取 synth_clean/<dataset>/synthetic_iter*_<model>.parquet，对每条样本用弱模型
(默认 qwen3-8b @ DashScope) 跑 N 次推理。任意一次答错 → 保留；全部答对 → 丢弃。
保留的样本写入同目录 synthetic_iter*_<model>_hard.parquet。

字段约定：合成数据使用与正式评测一致的小写字段 (story / question / answer.{correct,wrong}_answers / meta.id)，
直接复用 src/evaluation/prompts.py 的 build_option_bundle + build_prompt + extract_prediction_value，
确保"训练样本难度"与"评测同款 prompt 下的难度"语义对齐。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from src.evaluation.prompts import (
    build_option_bundle,
    build_prompt,
    extract_prediction_value,
    prompt_type,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _is_correct(
    sample: Dict[str, Any],
    correct_letters: List[str],
    response,
) -> bool:
    pt = prompt_type(sample["answer"])
    pred = extract_prediction_value(pt, response)
    if pt == "mcq_single":
        return pred in correct_letters
    if pt == "mcq_multi":
        return pred is not None and set(pred) == set(correct_letters)
    # open QA：合成数据集几乎不会走这里；做最宽容的文本包含判断
    if pred is None:
        return False
    pred_norm = str(pred).strip().lower()
    return any(
        str(c).strip().lower() in pred_norm
        for c in sample["answer"].get("correct_answers", [])
    )


def filter_by_difficulty(
    parquet_path: Path,
    dataset_name: str,
    llm_config: Dict[str, Any],
    repeats: int = 3,
    keep_when_api_fail: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """对单个 parquet 跑难度过滤，返回 (输出文件路径, 统计)。"""
    parquet_path = Path(parquet_path)
    rows = pq.read_table(str(parquet_path)).to_pylist()
    n = len(rows)
    out_path = parquet_path.with_name(parquet_path.stem + "_hard.parquet")

    if n == 0:
        pq.write_table(pa.table({}), str(out_path))
        return out_path, {"total": 0, "kept": 0, "dropped_all_correct": 0, "dropped_api_fail": 0, "kept_rate": 0.0}

    client = runner.create_content_client(llm_config, None)

    any_wrong = [False] * n
    api_fail_count = [0] * n

    for repeat_idx in range(repeats):
        prompts: List[str] = []
        per_sample_ctx: List[Tuple[Dict[str, Any], Dict[str, str], List[str]]] = []

        for i, row in enumerate(rows):
            sample = {
                "story": row.get("story", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", {"correct_answers": [], "wrong_answers": []}),
            }
            sample_id = (row.get("meta") or {}).get("id") or f"synth_{i}"
            option_map, correct_letters, _, _ = build_option_bundle(
                dataset_name, str(sample_id), sample["answer"], repeat_idx
            )
            prompts.append(build_prompt(sample, option_map))
            per_sample_ctx.append((sample, option_map, correct_letters))

        responses = client.batch_generate(
            prompts, desc=f"Difficulty[{dataset_name}] r{repeat_idx + 1}/{repeats}"
        )

        for i, (resp, (sample, _option_map, correct_letters)) in enumerate(zip(responses, per_sample_ctx)):
            if resp is None or resp.content is None:
                api_fail_count[i] += 1
                continue
            if not _is_correct(sample, correct_letters, resp):
                any_wrong[i] = True

    kept_rows: List[Dict[str, Any]] = []
    dropped_correct = 0
    dropped_apifail = 0
    for i, row in enumerate(rows):
        if any_wrong[i]:
            kept_rows.append(row)
        elif api_fail_count[i] >= repeats:
            if keep_when_api_fail:
                kept_rows.append(row)
            else:
                dropped_apifail += 1
        else:
            dropped_correct += 1

    if kept_rows:
        pq.write_table(pa.Table.from_pylist(kept_rows), str(out_path))
    else:
        # 写一个表头一致的空 parquet，避免下游 glob 拿到一个 0 字节文件
        empty_table = pa.Table.from_pylist([dict(rows[0])]).slice(0, 0) if rows else pa.table({})
        pq.write_table(empty_table, str(out_path))

    stats = {
        "total": n,
        "kept": len(kept_rows),
        "dropped_all_correct": dropped_correct,
        "dropped_api_fail": dropped_apifail,
        "kept_rate": len(kept_rows) / n,
    }
    logger.info(
        f"  {dataset_name}/{parquet_path.name}: kept {len(kept_rows)}/{n} "
        f"({stats['kept_rate']:.1%}); dropped_correct={dropped_correct}, "
        f"api_fail={dropped_apifail}"
    )
    return out_path, stats


def run_stage5_all_datasets(
    config: Dict[str, Any],
    only_dataset: str = "",
    iteration: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """对 synth_clean 下所有匹配 iter 的 parquet 跑难度过滤。"""
    df_cfg = config.get("difficulty_filter", {})
    if not df_cfg.get("enabled", False):
        logger.info("difficulty_filter.enabled=false, skipping Stage 5")
        return {}

    repeats = df_cfg.get("repeats", 3)
    keep_when_api_fail = df_cfg.get("keep_when_api_fail", False)

    output_path = Path(config["output_path"])
    synth_clean_root = output_path / "synth_clean"

    datasets_cfg = config["synthesis_datasets"]

    all_stats: Dict[str, Dict[str, Any]] = {}

    for ds_info in datasets_cfg:
        ds = ds_info["name"]
        if only_dataset and ds != only_dataset:
            continue
        ds_dir = synth_clean_root / ds
        if not ds_dir.exists():
            continue
        for pq_file in sorted(ds_dir.glob(f"synthetic_iter{iteration}_*.parquet")):
            # 跳过自身产物，避免对 _hard.parquet 重复跑
            if pq_file.stem.endswith("_hard"):
                continue
            try:
                _, st = filter_by_difficulty(
                    pq_file, ds, df_cfg, repeats=repeats,
                    keep_when_api_fail=keep_when_api_fail,
                )
                all_stats[f"{ds}/{pq_file.name}"] = st
            except Exception as e:
                logger.error(f"difficulty filter failed for {pq_file}: {e}")
                import traceback
                traceback.print_exc()

    return all_stats
