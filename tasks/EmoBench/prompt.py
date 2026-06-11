"""EmoBench 原论文 prompt(忠实复刻 Sahandfer/EmoBench 的 prompts.yaml / response.yaml）。

- system prompt 复刻官方 "# Instructions ... take the perspective ... # Output"，
  把官方 JSON(answer_q1/answer_q2)输出格式换成本框架的 \\boxed{}。
- body 复刻官方结构：EU 用 ## Scenario / ## Question 1 / ## Choices for Question 1 /
  ## Question 2 / ## Choices for Question 2(中文 ## 场景 / ## 问题 1 / ## 问题 1 的选项 …)；
  EA 用 ## Scenario / ## Question / ## Choices。
- EU 两问在预测前由 prepare_samples 合并成 mcq_grouped(两问都对才得分)。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, build_option_bundle, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

# ----- 官方 system prompt 框架(# Instructions ... # Output) -----
SYS_EN = (
    "# Instructions\n\n"
    "In this task, you are presented with a scenario, a question, and multiple choices. "
    "Carefully analyze the scenario and take the perspective of the individual involved. "
    "Then, select the option that best reflects their perspective or emotional response.\n\n"
    "# Output\n"
)
SYS_ZH = (
    "# 说明\n\n"
    "在这个任务中，你会面临一个场景、一个问题和多个选项。"
    "仔细分析场景，并从相关人员的角度进行思考。"
    "然后，选择最能反映他们的观点或情绪反应的选项。\n\n"
    "# 输出\n"
)


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    reasoning = reasoning_for(protocol)
    base = SYS_ZH if lang == "zh" else SYS_EN
    return base + boxed_directive(lang, prompt_type, reasoning)


# ----- EU grouped 样本合并(同前) -----


def prepare_samples(samples: List[StandardizedSample]) -> List[StandardizedSample]:
    """同 paired_group_id 的 EU 两问合并成一条 mcq_grouped 样本；EA 单问原样保留。"""
    groups: "OrderedDict[str, List[StandardizedSample]]" = OrderedDict()
    for sample in samples:
        meta = sample.get("meta") or {}
        key = meta.get("paired_group_id") or sample["sample_id"]
        groups.setdefault(key, []).append(sample)

    out: List[StandardizedSample] = []
    for key, members in groups.items():
        if len(members) < 2:
            out.append(members[0])
            continue
        members = sorted(
            members,
            key=lambda s: 0 if (s.get("meta") or {}).get("question_subtype") == "emotion" else 1,
        )
        sub_questions: List[Dict[str, Any]] = []
        for index, member in enumerate(members):
            option_map, correct_letters, _wrong, _seed = build_option_bundle(
                "EmoBench", f"{key}|{index}", member["answer"], 0, shuffle=True
            )
            sub_questions.append(
                {
                    "question": member["question"],
                    "subtype": (member.get("meta") or {}).get("question_subtype"),
                    "options": option_map or {},
                    "correct_letters": correct_letters,
                }
            )
        merged_meta = dict(members[0].get("meta") or {})
        merged_meta["prompt_type_override"] = "mcq_grouped"
        merged_meta["sub_questions"] = sub_questions
        merged_meta["question_subtype"] = "emotion+cause"
        out.append(
            {
                "sample_id": key,
                "story": members[0]["story"],
                "question": " || ".join(member["question"] for member in members),
                "answer": {"correct_answers": [], "wrong_answers": []},
                "meta": merged_meta,
            }
        )
    return out


# ----- body(官方 ## 结构) -----


def _grouped_body(sample: StandardizedSample, lang: str) -> str:
    subs: List[Dict[str, Any]] = (sample.get("meta") or {}).get("sub_questions") or []
    if lang == "zh":
        parts = [f"## 场景\n{sample['story']}"]
        for index, sub in enumerate(subs, start=1):
            parts.append(f"## 问题 {index}\n{sub['question']}")
            parts.append(f"## 问题 {index} 的选项\n{render_options_block(sub.get('options') or {})}")
    else:
        parts = [f"## Scenario\n{sample['story']}"]
        for index, sub in enumerate(subs, start=1):
            parts.append(f"## Question {index}\n{sub['question']}")
            parts.append(f"## Choices for Question {index}\n{render_options_block(sub.get('options') or {})}")
    return "\n\n".join(parts)


def _single_body(sample: StandardizedSample, lang: str, option_map: Optional[Dict[str, str]]) -> str:
    options_block = render_options_block(option_map or {})
    if lang == "zh":
        return f"## 场景\n{sample['story']}\n\n## 问题\n{sample['question']}\n\n## 选项\n{options_block}"
    return f"## Scenario\n{sample['story']}\n\n## Question\n{sample['question']}\n\n## Choices\n{options_block}"


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    meta = sample.get("meta") or {}
    lang = get_sample_lang(meta)
    if meta.get("prompt_type_override") == "mcq_grouped":
        body = _grouped_body(sample, lang)
        ptype = "mcq_grouped"
    else:
        body = _single_body(sample, lang, option_map)
        ptype = prompt_type(sample["answer"])
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, ptype, reasoning=False)
    return body.rstrip()
