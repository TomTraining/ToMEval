"""ExploreToM prompt(Sclar et al. 2025, arXiv:2412.12175)。

程序生成的对抗式 ToM 故事，问角色对物体位置/知晓状态的（高阶）信念。
官方为 free-form 评测：二元题（yes/no、knows/不knows）走单选，位置题为开放短答。
保留 Story / Question 结构，答案格式统一用本框架 \\boxed{}（开放题也把最终答案放进 \\boxed{}，f1 判分抽取）。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

BODY_MCQ = """Story:
{story}

Question:
{question}

Options:
{options_block}"""

BODY_OPEN = """Story:
{story}

Question:
{question}"""


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    return boxed_directive(lang, prompt_type, reasoning_for(protocol))


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    lang = get_sample_lang(sample.get("meta"))
    pt = prompt_type(sample["answer"])
    if option_map:
        body = BODY_MCQ.format(
            story=sample["story"],
            question=sample["question"],
            options_block=render_options_block(option_map),
        )
    else:
        body = BODY_OPEN.format(story=sample["story"], question=sample["question"])
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, pt, reasoning=False)
    return body.rstrip()
