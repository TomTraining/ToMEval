"""SocialMind 自定义 body prompt。

Q4 开放分析题走 open_judge=rubric 判分，rubric 的「额外校准规则」对答案的
长度/单要点/反堆砌/不得复述参考答案有硬性约束（见 q4_judge_prompts.json 与
open_judge._build_rubric_prompt）。这些约束必须在生成阶段就告知被测模型，
否则判分规则会单方面惩罚模型而模型并不知情，故此处为 open 题定制生成 prompt。

Q1-Q3 为 MCQ，沿用通用 build_prompt（\\boxed{} 协议不变）。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import build_prompt as _generic_build_prompt
from src.evaluation.prompts import prompt_type
from src.evaluation.types import StandardizedSample

# open(Q4)生成 prompt：约束与 rubric「额外校准规则」一一对应
# （≤300字 / 单要点 / 不复述参考答案 / 反堆砌），让被测模型在生成阶段就知情。
OPEN_QA_TEMPLATE_ZH = """请根据下面的故事回答问题。

故事：
{story}

问题：
{question}

请直接回答问题，答案不超过300字。
不要复制或改写参考答案；只能依据故事本身推理。
使用简洁、基于证据的句子。
每句话最多表达一个独立要点。
避免堆砌松散相关的概念或理论术语。
只输出答案文本。"""


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    # 只有 open(Q4)定制；MCQ 一律回退通用实现，保持 \\boxed{} 协议与选项排版不变。
    if prompt_type(sample["answer"]) != "open":
        return _generic_build_prompt(sample, option_map, include_instruction)

    # open 题不带 \\boxed{} 指令（答题格式由 rubric 而非字母决定），与通用 open 模板一致，
    # 故 include_instruction 在 open 分支下不影响 body。
    lang = get_sample_lang(sample.get("meta"))
    if lang != "zh":
        # SocialMind 目前全为中文样本；非中文样本回退通用实现以防意外语言。
        return _generic_build_prompt(sample, option_map, include_instruction)
    return OPEN_QA_TEMPLATE_ZH.format(story=sample["story"], question=sample["question"])
