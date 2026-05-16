from __future__ import annotations

import json
from typing import Any, Dict, List

from .judge_schema import QAJudgeResult


def judge_prompt(record: Dict[str, Any]) -> str:
    # 始终使用模型的完整原始输出，避免字母提取失败影响正确率。
    model_output = (record.get("pred") or {}).get("content") or ""

    if record["prompt_type"] == "open":
        return f"""You are grading an open-ended QA response.

Story:
{record["story"]}

Question:
{record["question"]}

Accepted correct answers:
{json.dumps(record["correct_answers"], ensure_ascii=False)}

Model response:
{model_output}

Output ONLY a JSON object: {{"is_correct": true}} or {{"is_correct": false}}
Mark is_correct as true if the model response semantically matches at least one accepted correct answer. Minor wording differences are acceptable."""

    # 选择题：同时展示选项字母和对应文本，让 judge 理解语义而非强依赖字母匹配。
    options: Dict[str, str] = record.get("options") or {}
    correct_letters: List[str] = record.get("correct_letters") or []
    options_block = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    correct_display = ", ".join(
        f"{letter}. {options.get(letter, '')}" for letter in correct_letters
    )

    return f"""You are grading a multiple-choice QA response.

Story:
{record["story"]}

Question:
{record["question"]}

Options:
{options_block}

Correct answer(s): {correct_display}

Model response:
{model_output}

Output ONLY a JSON object: {{"is_correct": true}} or {{"is_correct": false}}
Mark is_correct as true if the model response correctly identifies the answer, whether expressed as a letter, the option text, or a paraphrase.
For single-choice, exactly one correct option must be chosen. For multi-choice, all correct options must be identified."""


def judge_repeat(records: List[Dict[str, Any]], judge_client: Any) -> List[Dict[str, Any]]:
    # 先给每个样本放一个默认错误结果，后面只覆盖真正拿到 judge 输出的样本。
    per_sample_results: List[Dict[str, Any]] = [
        {
            "is_correct": False,
            "error_reason": "content_none",
        }
        for _ in records
    ]

    prompts: List[str] = []
    prompt_indices: List[int] = []
    for index, record in enumerate(records):
        # 以模型原始输出是否为空作为判断依据（不依赖字母提取结果）。
        model_output = (record.get("pred") or {}).get("content")
        has_prediction = model_output not in (None, "")
        if not has_prediction:
            continue
        prompts.append(judge_prompt(record))
        prompt_indices.append(index)

    if prompts:
        # create 模式能正确传入 extra_body（含 enable_thinking: false），
        # parse 模式会覆盖 vLLM chat template 导致 thinking 被意外开启、token 耗尽。
        judge_results = judge_client.batch_generate_structure(prompts, QAJudgeResult, mode="create", desc="Judging")
        for index, response in zip(prompt_indices, judge_results):
            content = response.content
            if content is None:
                per_sample_results[index] = {
                    "is_correct": False,
                    "error_reason": "judge_error",
                }
                continue
            per_sample_results[index] = {
                "is_correct": bool(content.is_correct),
                "error_reason": None if content.is_correct else "wrong_answer",
            }

    # 回填 sample_id 和 repeat，保证后续 metric 聚合时不丢样本身份。
    for record, result in zip(records, per_sample_results):
        result["sample_id"] = record["sample_id"]
        result["repeat"] = record["repeat"]
    return per_sample_results
