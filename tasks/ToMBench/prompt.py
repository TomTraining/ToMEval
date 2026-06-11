"""ToMBench 原论文 prompt(忠实复刻 zhchen18/ToMBench 的 prompts.py)。

- system prompt 复刻官方 SystemEvaluatePrompt(direct / cot 两个变体，中英双语)，
  仅把官方的 [[答案序号]] 格式换成本框架的 \\boxed{};保留"必须从选项中选一个、信息不足
  也要随机选一个"的硬约束。
- user body 复刻官方 UserEvaluatePrompt：英文 [Story]/[Question]/[Candidate Answers]，
  中文 [故事]/[问题]/[答案选项]。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

SYS_EN_DIRECT = """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer.
Note:
(1) Please output the most likely answer in the format \\boxed{X}, for example, if the most likely answer option is 'A. Handbag', then output \\boxed{A};
(2) You must choose one of the given answer options as the most likely answer, regardless of whether the story provides enough information. If you think there is not enough information in the story to choose an answer, please randomly output one of the given option letters;
(3) Please only output the most likely answer based on the given information, and do not output any other content."""

SYS_EN_COT = """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer.
Note:
(1) Please first think step by step and analyze the answer to the question, and finally output the most likely answer in the format \\boxed{X}, for example, if the most likely answer option is 'A. Handbag', then output \\boxed{A};
(2) You must choose one of the given answer options as the most likely answer, regardless of whether the story provides enough information. If you think there is not enough information in the story to choose an answer, please randomly output one of the given option letters;
(3) Again, you must first output the step-by-step reasoning, and finally output the most likely answer. You should not directly output the answer."""

SYS_ZH_DIRECT = """下面给你提供一段故事，一个问题和若干答案选项，请你根据故事内容和给定的问题，按照常理推测，选择一个最可能的答案选项。
注意：
（1）请把最可能的答案以 \\boxed{X} 的格式输出，例如，最可能的答案选项为“A. 手提包”，则输出 \\boxed{A}；
（2）请必须从给定的答案选项中选择一个作为最可能的答案，无论故事中是否提供足够的信息，如果你认为故事里没有足够的信息选出答案，请随机输出其中一个选项字母；
（3）请只输出在给定的信息下最可能的答案，不要输出其他内容。"""

SYS_ZH_COT = """下面给你提供一段故事，一个问题和若干答案选项，请你根据故事内容和给定的问题，按照常理推测，选择一个最可能的答案选项。
注意：
（1）请先一步步思考，对问题的答案进行推理分析，最后把最可能的答案以 \\boxed{X} 的格式输出，例如，最可能的答案选项为“A. 手提包”，则输出 \\boxed{A}；
（2）请必须从给定的答案选项中选择一个作为最可能的答案，无论故事中是否提供足够的信息，如果你认为故事里没有足够的信息选出答案，请随机输出其中一个选项字母；
（3）再次强调，你必须先给出一步步推理的结果，最后再输出最可能的答案。你不应该直接输出答案。"""

BODY_EN = """[Story]
{story}

[Question]
{question}

[Candidate Answers]
{options_block}"""

BODY_ZH = """[故事]
{story}

[问题]
{question}

[答案选项]
{options_block}"""


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    reasoning = reasoning_for(protocol)
    if lang == "zh":
        return SYS_ZH_COT if reasoning else SYS_ZH_DIRECT
    return SYS_EN_COT if reasoning else SYS_EN_DIRECT


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    lang = get_sample_lang(sample.get("meta"))
    options_block = render_options_block(option_map or {})
    template = BODY_ZH if lang == "zh" else BODY_EN
    body = template.format(
        story=sample["story"], question=sample["question"], options_block=options_block
    )
    # 协议模式(include_instruction=False)由 system prompt 承载 \boxed 指令；
    # legacy 模式(无协议)在 body 末尾补一条，保证仍输出标准 \boxed。
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, "mcq_single", reasoning=False)
    return body.rstrip()
