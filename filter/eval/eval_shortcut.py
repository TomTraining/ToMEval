"""V3 Phase D：三维 shortcut 探测（仅对 partial + answerable 跑）。

三个变体探测：
  no_story    去掉 story 后让小模型答 mcq → majority 答对 = shortcut（题面/选项泄漏）
  no_question 去掉 question 后让小模型答 mcq → majority 答对 = shortcut（story 词汇匹配）
  no_options  去掉 options 改 open QA → majority 答**错** = shortcut（选项排除型）

is_shortcut = (no_story_pass >= ceil(k/2)) OR
              (no_question_pass >= ceil(k/2)) OR
              (no_options_pass < ceil(k/2))

复用：
  - src.evaluation.prompts.OPEN_QA_TEMPLATE / CHOICE_QA_TEMPLATE / build_option_bundle
  - src.evaluation.prompts.extract_prediction_value
  - src.evaluation.judge.judge_repeat（用于 no_options 维度的 open QA 判定）
  - filter.base.load_answer_models / load_judge_client
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from filter.base import load_answer_models, load_judge_client
from filter.utils import is_correct_mcq, resolve_sample_id, row_to_sample, write_parquet
from src.evaluation.judge import judge_repeat
from src.evaluation.prompts import (
    CHOICE_QA_TEMPLATE,
    OPEN_QA_TEMPLATE,
    build_option_bundle,
    prompt_type as compute_prompt_type,
)
from src.llm.content_client import ContentClient

logger = logging.getLogger(__name__)


_STORY_PLACEHOLDER = "(story omitted)"
_QUESTION_PLACEHOLDER = "(question omitted)"


def _options_block(option_map: Dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in option_map.items())


def _answer_instruction(prompt_type: str) -> str:
    if prompt_type == "mcq_multi":
        return "Select every correct option and return a list of option letters."
    return "Select the single best option and return exactly one option letter."


def build_no_story_prompt(sample: Dict[str, Any], option_map: Dict[str, str], prompt_type: str) -> str:
    return CHOICE_QA_TEMPLATE.format(
        story=_STORY_PLACEHOLDER,
        question=sample["question"],
        options_block=_options_block(option_map),
        answer_instruction=_answer_instruction(prompt_type),
    )


def build_no_question_prompt(sample: Dict[str, Any], option_map: Dict[str, str], prompt_type: str) -> str:
    return CHOICE_QA_TEMPLATE.format(
        story=sample["story"],
        question=_QUESTION_PLACEHOLDER,
        options_block=_options_block(option_map),
        answer_instruction=_answer_instruction(prompt_type),
    )


def build_no_options_prompt(sample: Dict[str, Any]) -> str:
    return OPEN_QA_TEMPLATE.format(
        story=sample["story"],
        question=sample["question"],
    )


def _majority_threshold(k: int) -> int:
    return math.ceil(k / 2)


def _run_mcq_dimension(
    samples: List[Dict[str, Any]],
    sample_ids: List[str],
    prompt_types: List[str],
    dataset: str,
    k: int,
    simple_client: ContentClient,
    build_prompt_fn,
    dim_name: str,
) -> List[int]:
    """对一个 MCQ 探测维度（no_story / no_question）跑 k 次，返回每条样本的 pass 计数。"""
    n = len(samples)
    pass_counts = [0] * n

    for trial in range(k):
        prompts: List[str] = []
        per_row_correct_letters: List[List[str]] = []
        per_row_skip: List[bool] = []

        for i, sample in enumerate(samples):
            if not sample["answer"]["wrong_answers"]:
                # 没有 wrong_answers 的样本不是 MCQ，跳过该维度（保留 pass=0 不会触发 shortcut）
                prompts.append("")
                per_row_correct_letters.append([])
                per_row_skip.append(True)
                continue
            option_map, c_letters, _w, _seed = build_option_bundle(
                dataset, sample_ids[i], sample["answer"], trial
            )
            prompts.append(build_prompt_fn(sample, option_map, prompt_types[i]))
            per_row_correct_letters.append(list(c_letters))
            per_row_skip.append(False)

        # batch_generate 不能传空 prompt 给 API；空 prompt 占位的样本要从批里剔除
        active_indices = [i for i, skip in enumerate(per_row_skip) if not skip]
        active_prompts = [prompts[i] for i in active_indices]
        if not active_prompts:
            continue
        responses = simple_client.batch_generate(
            active_prompts, desc=f"shortcut[{dataset}]/{dim_name} trial {trial + 1}/{k}"
        )
        for active_idx, resp in zip(active_indices, responses):
            if is_correct_mcq(prompt_types[active_idx], resp, per_row_correct_letters[active_idx]):
                pass_counts[active_idx] += 1

    return pass_counts


def _run_no_options_dimension(
    samples: List[Dict[str, Any]],
    sample_ids: List[str],
    dataset: str,
    k: int,
    simple_client: ContentClient,
    judge_client,
) -> List[int]:
    """no_options 维度：每条样本 k 次 open QA，judge_repeat 判对错，返回 pass 计数。"""
    n = len(samples)

    # 1) 跑 k 次 open QA 获取小模型回答
    contents_per_trial: List[List[Optional[str]]] = []  # [trial][sample] -> content
    for trial in range(k):
        prompts = [build_no_options_prompt(s) for s in samples]
        responses = simple_client.batch_generate(
            prompts, desc=f"shortcut[{dataset}]/no_options trial {trial + 1}/{k}"
        )
        contents_per_trial.append([
            (resp.content if (resp is not None and resp.content is not None) else None)
            for resp in responses
        ])

    # 2) 把 (sample_id, trial) 的有效回答收集到 records 给 judge_repeat
    records: List[Dict[str, Any]] = []
    record_keys: List[Tuple[int, int]] = []  # (sample_idx, trial)
    for i, sample in enumerate(samples):
        correct_answers = sample["answer"]["correct_answers"]
        wrong_answers = sample["answer"]["wrong_answers"]
        for trial in range(k):
            content = contents_per_trial[trial][i]
            records.append({
                "sample_id": sample_ids[i],
                "repeat": trial,
                "prompt_type": "open",
                "pred": {"content": content},
                "correct_answers": correct_answers,
                "wrong_answers": wrong_answers,
            })
            record_keys.append((i, trial))

    if not records:
        return [0] * n

    # 3) judge 一次性批量判对错
    verdicts = judge_repeat(records, judge_client)

    # 4) 聚合 pass 计数
    pass_counts = [0] * n
    for (i, _trial), verdict in zip(record_keys, verdicts):
        if verdict.get("is_correct"):
            pass_counts[i] += 1
    return pass_counts


def run_shortcut_on_df(
    df: pd.DataFrame,
    dataset: str,
    k: int = 3,
    threshold: str = "majority",
    simple_client: Optional[ContentClient] = None,
    judge_client=None,
    dimensions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """对 partial+answerable 子集跑三维 shortcut 探测。

    Args:
        df: 待探测子集
        dataset: 数据集名（shuffle 种子用）
        k: 每维度 trial 次数
        threshold: "majority" / "any" / "all"
        simple_client: qwen3-8b client；不传则 load
        judge_client: StructureClient；不传则 load
        dimensions: 启用的探测维度；None=全部三维

    Returns:
        DataFrame: [sample_id, no_story_pass, no_question_pass, no_options_pass, is_shortcut]
                   行序与 df 对齐
    """
    if simple_client is None:
        simple_client = load_answer_models()["simple"]
    if judge_client is None:
        judge_client = load_judge_client("strong")
    if dimensions is None:
        dimensions = ["no_story", "no_question", "no_options"]

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=[
            "sample_id", "no_story_pass", "no_question_pass", "no_options_pass", "is_shortcut",
        ])

    df = df.reset_index(drop=True)
    samples: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    prompt_types: List[str] = []
    for idx, row in df.iterrows():
        sample = row_to_sample(row)
        samples.append(sample)
        sample_ids.append(resolve_sample_id(row, idx))
        prompt_types.append(compute_prompt_type(sample["answer"]))

    # 三个维度独立跑
    no_story_pass = (
        _run_mcq_dimension(samples, sample_ids, prompt_types, dataset, k, simple_client,
                           build_no_story_prompt, "no_story")
        if "no_story" in dimensions else [0] * n
    )
    no_question_pass = (
        _run_mcq_dimension(samples, sample_ids, prompt_types, dataset, k, simple_client,
                           build_no_question_prompt, "no_question")
        if "no_question" in dimensions else [0] * n
    )
    no_options_pass = (
        _run_no_options_dimension(samples, sample_ids, dataset, k, simple_client, judge_client)
        if "no_options" in dimensions else [k] * n  # 该维度没启用 → 默认"全过"，不会反向触发 shortcut
    )

    # 阈值合并
    if threshold == "majority":
        thr = _majority_threshold(k)
    elif threshold == "any":
        thr = 1
    elif threshold == "all":
        thr = k
    else:
        raise ValueError(f"unknown threshold: {threshold}")

    is_shortcut: List[bool] = []
    for i in range(n):
        cond = False
        if "no_story" in dimensions and no_story_pass[i] >= thr:
            cond = True
        if "no_question" in dimensions and no_question_pass[i] >= thr:
            cond = True
        if "no_options" in dimensions and no_options_pass[i] < thr:
            # 反向：该答对却答不对 = shortcut
            cond = True
        is_shortcut.append(cond)

    out = pd.DataFrame({
        "sample_id": sample_ids,
        "no_story_pass": no_story_pass,
        "no_question_pass": no_question_pass,
        "no_options_pass": no_options_pass,
        "is_shortcut": is_shortcut,
    })
    n_sc = sum(1 for s in is_shortcut if s)
    logger.info(
        f"[shortcut] {dataset} n={n} k={k} threshold={threshold}({thr}) "
        f"shortcut={n_sc} medium={n - n_sc}"
    )
    return out


def write_shortcut_parquet(sc_df: pd.DataFrame, out_path: Path) -> None:
    write_parquet(sc_df, out_path, "shortcut")


__all__ = [
    "run_shortcut_on_df",
    "write_shortcut_parquet",
    "build_no_story_prompt",
    "build_no_question_prompt",
    "build_no_options_prompt",
]
