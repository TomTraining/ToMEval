"""
你的实现文件 —— **只需要改这一个文件**(通常只改 predict 一个函数)。

任务:实现 `predict(sample, model)`,对一道题目给出答案。下面提供一个开箱即用的
单轮 baseline(调一次模型),可直接跑,也可作为你自己策略(多轮、工具、投票……)的起点。

提交要求:你的仓库根目录必须有本文件(solution.py),且其中定义可调用的 `predict`。
提交前请用 `python selftest.py` 自测,确认四种题型都能跑通、返回格式合规。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sample:一道题目(不含任何标准答案),字段:
  · sample_id   : str            样本 id(你不必处理,原样即可)
  · prompt_type : str            题型:mcq_single | mcq_multi | mcq_grouped | open
  · lang        : str            "en" 或 "zh"
  · story       : str            故事/背景(少数题型可能为空串)
  · question    : str            问题
  · options     : dict[str,str]  选项字母→文本,如 {"A":"...","B":"..."};open 题无此字段
  · sub_questions: list          仅 mcq_grouped:每项含自身 question 与 options,按顺序作答

model:提供给你的统一模型后端(每条请求下发,请勿写死):
  · api_url    : str   OpenAI 兼容 base_url
  · api_key    : str   调用用的 key
  · model_name : str   统一模型名

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出(严格格式,格式不符该题直接判错;不会对你的返回值做二次提取/纠正)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  · mcq_single  → 单个大写字母字符串,如 "A"
  · mcq_multi   → 大写字母数组,升序、去重、至少一个,如 ["A","C"](禁止字符串 "A,C")
  · mcq_grouped → 大写字母数组,每个子问一个、顺序对应 sub_questions,如 ["A","B"]
  · open        → 非空文本字符串

约束:MCQ 必须使用 options 里给定的字母作答(选项顺序已被打乱,返回选项文本或自编编号会判错)。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

Prediction = Union[str, List[str]]


# ---------------------------------------------------------------------------
# 默认单轮 baseline —— 可直接用,也可整段替换成你自己的策略。
# ---------------------------------------------------------------------------

_ANSWER_TAG = "ANSWER:"  # 要求模型在最后一行用固定前缀给答案,便于稳健提取。


def _build_prompt(sample: Dict[str, Any]) -> str:
    """把题目排成纯文本 prompt(含答题指令)。选项字母是给定的,必须原样使用。"""
    ptype = sample.get("prompt_type", "open")
    lines = [sample.get("story", ""), "", sample.get("question", "")]

    if ptype == "mcq_grouped":
        # grouped:选项在每个子问自带的 options 里,逐个列出。
        for i, sub in enumerate(sample.get("sub_questions") or [], start=1):
            lines += ["", f"[Q{i}] {sub.get('question', '')}"]
            lines += [f"{k}. {v}" for k, v in (sub.get("options") or {}).items()]
        n = len(sample.get("sub_questions") or [])
        directive = (
            f"End your reply with a line `{_ANSWER_TAG} X,Y` giving one option letter per "
            f"sub-question in order ({n} letters, comma-separated)."
        )
    else:
        options = sample.get("options")
        if options:
            lines += ["", "Options:"] + [f"{k}. {v}" for k, v in options.items()]
        if ptype == "mcq_single":
            directive = f"End your reply with a line `{_ANSWER_TAG} X` (X = the single best option letter)."
        elif ptype == "mcq_multi":
            directive = f"End your reply with a line `{_ANSWER_TAG} X,Y` listing every correct option letter."
        else:
            directive = f"End your reply with a line `{_ANSWER_TAG} <your answer>`."

    lines += ["", directive]
    return "\n".join(lines)


def _answer_segment(text: str) -> str:
    """取最后一个 `ANSWER:` 行之后的内容;没有则退回全文。"""
    matches = list(re.finditer(r"(?im)^\s*ANSWER:\s*(.*)$", text))
    return matches[-1].group(1).strip() if matches else text.strip()


def _to_prediction(raw: str, sample: Dict[str, Any]) -> Prediction:
    """把模型自由文本收敛成契约要求的严格格式(约束到 options 的合法字母)。

    提取不到有效字母时退回第一个合法选项,保证 MCQ 永不违约;open 题返回非空文本。
    """
    prompt_type = sample.get("prompt_type", "open")
    segment = _answer_segment(raw)

    if prompt_type == "open":
        return segment or (raw.strip() or "(no answer)")

    if prompt_type == "mcq_grouped":
        # 逐子问抽一个字母(约束到各自 options),顺序对应 sub_questions,长度=子问数。
        picked: List[str] = []
        # 按逗号切分 ANSWER 段,和子问一一对应;不足则从整段兜底。
        parts = [p.strip() for p in segment.split(",")]
        for i, sub in enumerate(sample.get("sub_questions") or []):
            valid = list((sub.get("options") or {}).keys())
            valid_set = set(valid)
            token = parts[i].upper() if i < len(parts) else segment.upper()
            letter = next((c for c in token if c in valid_set), None)
            if letter is None:
                letter = next((c for c in segment.upper() if c in valid_set), None)
            picked.append(letter or (valid[0] if valid else "A"))
        return picked

    valid = list((sample.get("options") or {}).keys())
    valid_set = set(valid)
    picked = []
    for ch in segment.upper():
        if ch in valid_set and ch not in picked:
            picked.append(ch)

    if prompt_type == "mcq_single":
        return picked[0] if picked else (valid[0] if valid else "A")

    # mcq_multi:升序去重字母数组,至少一个。
    if not picked:
        picked = [valid[0]] if valid else ["A"]
    return sorted(set(picked))


def predict(sample: Dict[str, Any], model: Dict[str, Any]) -> Prediction:
    """默认单轮实现:渲染题目 → 调一次模型 → 收敛成严格 prediction。

    参赛方可自由重写本函数(多轮、工具、自洽投票……),只要:
      1. 用传入的 `model` 凭证访问模型(勿写死);
      2. 返回值严格符合题型格式(见本文件顶部)。
    """
    client = OpenAI(api_key=model["api_key"], base_url=model["api_url"], timeout=600.0)

    completion = client.chat.completions.create(
        model=model["model_name"],
        messages=[{"role": "user", "content": _build_prompt(sample)}],
        temperature=0.0,
        max_tokens=2048,
        extra_body={"enable_thinking": False},
    )
    raw = completion.choices[0].message.content or ""
    return _to_prediction(raw, sample)
