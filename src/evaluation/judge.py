from __future__ import annotations

import json
from typing import Any, Dict, List

from .judge_schema import QAJudgeResult
from .lang import get_sample_lang
from .prompts import extract_prediction_from_text


def judge_prompt(record: Dict[str, Any]) -> str:
    # 只有 open 题走 LLM judge，MCQ 一律走 rule_judge_mcq 规则判分。
    assert record["prompt_type"] == "open", f"judge_prompt only supports open, got {record['prompt_type']}"

    # 始终使用模型的完整原始输出，避免提取失败影响正确率。
    model_output = (record.get("pred") or {}).get("content") or ""

    # storyless 版本：不再把 story / question 输入给 judge，
    # 只给 correct / wrong 答案 + 模型预测，要求 judge 做"参照式"对比判定。
    correct_answers: List[str] = record.get("correct_answers") or []
    wrong_answers: List[str] = record.get("wrong_answers") or []

    # 中文样本用中文措辞的 judge prompt，JSON 输出契约（{"is_correct": ...}）不变。
    if get_sample_lang(record.get("meta")) == "zh":
        return f"""你需要把模型的回答与参考答案对比，判定其是否正确。

可接受的正确答案：
{json.dumps(correct_answers, ensure_ascii=False)}

已知的错误答案（不能匹配这些）：
{json.dumps(wrong_answers, ensure_ascii=False)}

模型回答：
{model_output}

只输出一个 JSON 对象：{{"is_correct": true}} 或 {{"is_correct": false}}
仅当模型回答在语义上至少匹配一个正确答案时，is_correct 才为 true。
如果模型回答匹配了已知错误答案、与正确答案矛盾或与问题无关，is_correct 为 false。
允许轻微的措辞差异。"""

    return f"""You are grading a QA response by comparing it against reference answers.

Accepted correct answers:
{json.dumps(correct_answers, ensure_ascii=False)}

Known wrong answers (must NOT match these):
{json.dumps(wrong_answers, ensure_ascii=False)}

Model response:
{model_output}

Output ONLY a JSON object: {{"is_correct": true}} or {{"is_correct": false}}
Mark is_correct as true ONLY if the model response semantically matches at least one accepted correct answer.
Mark is_correct as false if the model response matches a known wrong answer, contradicts the correct answers, or is irrelevant.
Minor wording differences are acceptable."""


def rule_judge_mcq(record: Dict[str, Any]) -> Dict[str, Any]:
    # MCQ 规则判分：从模型原始输出中提取 \boxed{} 答案，和正确字母直接比对。
    model_output = (record.get("pred") or {}).get("content")
    if model_output in (None, ""):
        return {"is_correct": False, "error_reason": "content_none", "extracted": None}

    # extractor 跟随记录里 stamp 的协议（direct/cot/legacy），单独跑 stage=metric 也能正确分派。
    extractor = record.get("extractor", "legacy")
    extracted = extract_prediction_from_text(record["prompt_type"], str(model_output), extractor)
    if extracted is None:
        # 严格模式：没有 \boxed{} 或 boxed 内无字母，直接判错并标记提取失败。
        return {"is_correct": False, "error_reason": "extraction_failed", "extracted": None}

    correct_letters: List[str] = record.get("correct_letters") or []
    if record["prompt_type"] == "mcq_multi":
        is_correct = set(extracted) == set(correct_letters)
    else:
        is_correct = extracted in correct_letters
    return {
        "is_correct": is_correct,
        "error_reason": None if is_correct else "wrong_answer",
        "extracted": extracted,
    }


def judge_repeat(records: List[Dict[str, Any]], judge_client: Any) -> List[Dict[str, Any]]:
    # 先给每个样本放一个默认错误结果，后面只覆盖真正拿到判分结果的样本。
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
        # MCQ 走规则判分，不消耗 judge API。
        if record["prompt_type"] in ("mcq_single", "mcq_multi"):
            per_sample_results[index] = rule_judge_mcq(record)
            continue

        # open 题：以模型原始输出是否为空作为判断依据。
        model_output = (record.get("pred") or {}).get("content")
        has_prediction = model_output not in (None, "")
        if not has_prediction:
            continue
        prompts.append(judge_prompt(record))
        prompt_indices.append(index)

    if prompts:
        if judge_client is None:
            raise ValueError("Open QA records present but judge_client is None.")
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
