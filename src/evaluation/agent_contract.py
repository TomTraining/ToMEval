"""
智能体评测契约(单一权威源)—— 错误码、HTTP 映射、重试策略、prediction 严格校验/归一化。

AgentClient、参考 mock、conformance 校验器都从本模块引用,避免"各写各的"导致契约漂移。
schema 文件在 docs/agent_schema/,本模块是其运行时对应物(二者必须一致)。

契约详见 docs/agent_eval.md。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 题型
# ---------------------------------------------------------------------------

PROMPT_TYPES = ("mcq_single", "mcq_multi", "mcq_grouped", "open")

# ---------------------------------------------------------------------------
# 错误码 → (典型 HTTP 状态码, 是否可重试)
#
# 可重试(retryable=True):框架退避后重试,重试耗尽才判错。
# 不可重试(False):直接判该样本为错(content_none),不再重试。
# ---------------------------------------------------------------------------

ERROR_CODES: Dict[str, Tuple[int, bool]] = {
    "MODEL_TIMEOUT": (504, True),              # agent 调模型超时
    "OVERLOADED": (429, True),                 # agent 过载,要求限速
    "MODEL_ERROR": (502, True),                # agent 调模型返回错误(可能瞬时)
    "INTERNAL": (500, False),                  # agent 内部错误(非模型)
    "UNSUPPORTED_PROMPT_TYPE": (422, False),   # agent 不支持该题型
    "INVALID_REQUEST": (400, False),           # 请求不符合契约(理论上不会发生)
}

# 网络层(连接失败/读超时,拿不到 HTTP 响应)统一按这个可重试码处理。
NETWORK_ERROR_CODE = "MODEL_TIMEOUT"

# HTTP 状态码 → 错误码(当 agent 没在 body 里给结构化 error 时的兜底推断)。
_HTTP_TO_CODE: Dict[int, str] = {
    400: "INVALID_REQUEST",
    422: "UNSUPPORTED_PROMPT_TYPE",
    429: "OVERLOADED",
    500: "INTERNAL",
    502: "MODEL_ERROR",
    503: "OVERLOADED",
    504: "MODEL_TIMEOUT",
}


def is_retryable(code: Optional[str]) -> bool:
    """错误码是否可重试;未知码保守地判为不可重试。"""
    if code is None:
        return False
    entry = ERROR_CODES.get(code)
    return bool(entry and entry[1])


def code_for_http_status(status: int) -> str:
    """把 HTTP 状态码映射成错误码(body 无结构化 error 时的兜底)。"""
    if status in _HTTP_TO_CODE:
        return _HTTP_TO_CODE[status]
    # 其余 5xx 视为可重试的模型错误,4xx 视为不可重试的内部错误。
    return "MODEL_ERROR" if 500 <= status < 600 else "INTERNAL"


# ---------------------------------------------------------------------------
# 重试策略(固化,写进文档,参赛方据此了解框架如何对待其服务)
# ---------------------------------------------------------------------------

MAX_RETRIES = 3                       # 可重试错误最多重试次数(首次调用之外)
BACKOFF_BASE_SECONDS = 1.0            # 指数退避基数:1s, 2s, 4s ...
PER_SAMPLE_TIMEOUT_SECONDS = 300.0    # 单样本(含所有重试)总超时


def backoff_seconds(attempt: int) -> float:
    """第 attempt 次重试前的退避秒数(attempt 从 1 起):1s, 2s, 4s ...。"""
    return BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))


# ---------------------------------------------------------------------------
# prediction 严格单一格式:校验 + 归一化
#
# validate_prediction:严格判定 agent 返回的 prediction 是否符合该题型的唯一合法格式。
#   返回 (ok, reason)。ok=False 时 reason 是违约说明(供 conformance 校验器报错)。
# normalize_prediction:运行时把合法 prediction 归一成判分层要用的形态。
#   —— mcq_single/grouped 归一成字母(或字母列表);mcq_multi 归一成排序字母列表;open 原样。
#   —— 非法格式返回 None(判分层落进 content_none / format_violation)。
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"^[A-Z]$")


def validate_prediction(prompt_type: str, prediction: Any) -> Tuple[bool, Optional[str]]:
    """严格校验:每种题型只接受唯一格式(见 docs/agent_eval.md §5)。"""
    if prompt_type == "mcq_single":
        if isinstance(prediction, str) and _LETTER_RE.match(prediction):
            return True, None
        return False, "mcq_single 只接受单个大写字母,如 \"A\""

    if prompt_type == "mcq_multi":
        if not isinstance(prediction, list):
            return False, "mcq_multi 只接受大写字母数组,如 [\"A\",\"C\"](禁止字符串 \"A,C\")"
        if not prediction:
            return False, "mcq_multi 至少含一个字母"
        if any(not (isinstance(x, str) and _LETTER_RE.match(x)) for x in prediction):
            return False, "mcq_multi 每个元素须为单个大写字母"
        if len(set(prediction)) != len(prediction):
            return False, "mcq_multi 不允许重复字母"
        if prediction != sorted(prediction):
            return False, "mcq_multi 须升序排列"
        return True, None

    if prompt_type == "mcq_grouped":
        if not isinstance(prediction, list) or not prediction:
            return False, "mcq_grouped 只接受非空大写字母数组,顺序对应各子问"
        if any(not (isinstance(x, str) and _LETTER_RE.match(x)) for x in prediction):
            return False, "mcq_grouped 每个元素须为单个大写字母"
        return True, None

    if prompt_type == "open":
        if isinstance(prediction, str) and prediction.strip():
            return True, None
        return False, "open 只接受非空字符串"

    return False, f"未知 prompt_type: {prompt_type!r}"


def normalize_prediction(prompt_type: str, prediction: Any) -> Optional[Any]:
    """运行时归一:合法→判分层形态;非法→None。

    判分层(prompts.extract_prediction_from_text 的 agent 分支)约定:
      - mcq_single:单字母字符串 "A"
      - mcq_multi :字母列表 ["A","C"]
      - mcq_grouped:字母列表(顺序对应子问)
      - open      :字符串
    """
    ok, _ = validate_prediction(prompt_type, prediction)
    if not ok:
        return None
    if prompt_type == "open":
        return prediction
    return prediction  # 已是契约要求的严格形态,直接透传


__all__ = [
    "PROMPT_TYPES",
    "ERROR_CODES",
    "NETWORK_ERROR_CODE",
    "is_retryable",
    "code_for_http_status",
    "MAX_RETRIES",
    "BACKOFF_BASE_SECONDS",
    "PER_SAMPLE_TIMEOUT_SECONDS",
    "backoff_seconds",
    "validate_prediction",
    "normalize_prediction",
]
