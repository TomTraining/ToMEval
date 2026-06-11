"""TactfulToM 原论文 prompt(忠实复刻 nii-cl/tactful-tom 的 get_original_results.py)。

- system prompt 复刻官方 "You are an expert in social reasoning." + 题型相关的答案格式要求；
  官方按 binary/mcq/list 给不同格式，这里统一换成本框架 \\boxed{}(list 题→mcq_multi 多字母 boxed)。
- user body 复刻官方 "# Context: … / # Question: … / # Options: …"。
  (官方对部分题型还会插入 information_prompt；我们标准化数据未单独保留该字段，question 文本已含所需信息。)
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

SYS = "You are an expert in social reasoning."

BODY = """# Context:
{context}

# Question:
{question}

# Options:
{options_block}"""


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    return SYS + " " + boxed_directive(lang, prompt_type, reasoning_for(protocol))


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    lang = get_sample_lang(sample.get("meta"))
    body = BODY.format(
        context=sample["story"],
        question=sample["question"],
        options_block=render_options_block(option_map or {}),
    )
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, prompt_type(sample["answer"]), reasoning=False)
    return body.rstrip()
