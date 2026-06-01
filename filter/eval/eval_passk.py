"""V3 Phase B：pass@k 主答（小模型 qwen3-8b，k 次 trial，shuffle 选项）。

输入：synthetic.parquet 形态的 DataFrame（standardized schema：story/question/answer/meta）
输出：DataFrame，每条样本一行，含 sample_id / pass_at_k / bucket / 题型 / correct_letters

bucket：根据 pass_at_k 三分流：
  all_passed   pass_at_k == k
  partial      0 < pass_at_k < k
  all_failed   pass_at_k == 0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from filter.base import load_answer_models
from filter.utils import is_correct_mcq, resolve_sample_id, row_to_sample, write_parquet
from src.evaluation.prompts import (
    build_option_bundle,
    build_prompt,
    prompt_type as compute_prompt_type,
)
from src.llm.content_client import ContentClient

logger = logging.getLogger(__name__)


_BUCKET_ALL_PASSED = "all_passed"
_BUCKET_PARTIAL = "partial"
_BUCKET_ALL_FAILED = "all_failed"


def _bucket_of(pass_at_k: int, k: int) -> str:
    if pass_at_k >= k:
        return _BUCKET_ALL_PASSED
    if pass_at_k <= 0:
        return _BUCKET_ALL_FAILED
    return _BUCKET_PARTIAL


def run_passk_on_df(
    df: pd.DataFrame,
    dataset: str,
    k: int = 3,
    simple_client: Optional[ContentClient] = None,
) -> pd.DataFrame:
    """对 df 跑 pass@k，返回每条样本的 pass_at_k + bucket。

    Args:
        df: standardized 合成样本 DataFrame（含 story/question/answer/meta）
        dataset: 数据集名称（用于 build_option_bundle 的 shuffle 种子）
        k: trial 次数
        simple_client: 可选注入；不传则从 config.yaml 加载 simple

    Returns:
        DataFrame：[sample_id, prompt_type, correct_letters, pass_at_k, bucket]
                   行序与 df 完全对齐（reset_index 之后的位置序）
    """
    if simple_client is None:
        simple_client = load_answer_models()["simple"]

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["sample_id", "prompt_type", "correct_letters", "pass_at_k", "bucket"])

    samples: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    prompt_types: List[str] = []
    correct_letters_per_row: List[List[str]] = []

    df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        sample = row_to_sample(row)
        samples.append(sample)
        sample_ids.append(resolve_sample_id(row, idx))
        prompt_types.append(compute_prompt_type(sample["answer"]))

    pass_counts = [0] * n

    for trial in range(k):
        prompts: List[str] = []
        per_row_correct_letters: List[List[str]] = []
        for i, sample in enumerate(samples):
            option_map, c_letters, _w_letters, _seed = build_option_bundle(
                dataset, sample_ids[i], sample["answer"], trial
            )
            prompts.append(build_prompt(sample, option_map))
            per_row_correct_letters.append(list(c_letters))

        if trial == 0:
            correct_letters_per_row = per_row_correct_letters

        responses = simple_client.batch_generate(
            prompts, desc=f"passk[{dataset}] trial {trial + 1}/{k}"
        )

        for i, resp in enumerate(responses):
            target_letters = per_row_correct_letters[i] or correct_letters_per_row[i]
            if is_correct_mcq(prompt_types[i], resp, target_letters):
                pass_counts[i] += 1

    buckets = [_bucket_of(pc, k) for pc in pass_counts]

    out = pd.DataFrame({
        "sample_id": sample_ids,
        "prompt_type": prompt_types,
        "correct_letters": correct_letters_per_row,
        "pass_at_k": pass_counts,
        "bucket": buckets,
    })
    counts = out["bucket"].value_counts().to_dict()
    logger.info(
        f"[passk] {dataset} n={n} k={k} buckets="
        f"all_passed={counts.get(_BUCKET_ALL_PASSED, 0)}, "
        f"partial={counts.get(_BUCKET_PARTIAL, 0)}, "
        f"all_failed={counts.get(_BUCKET_ALL_FAILED, 0)}"
    )
    return out


def write_passk_parquet(passk_df: pd.DataFrame, out_path: Path) -> None:
    write_parquet(passk_df, out_path, "passk")


__all__ = [
    "run_passk_on_df",
    "write_passk_parquet",
    "_BUCKET_ALL_PASSED",
    "_BUCKET_PARTIAL",
    "_BUCKET_ALL_FAILED",
]
