"""Open QA 判分模式(open_judge)注册表 —— 类比 protocols.py 的可插拔设计。

open 题(无干扰项、唯一正确答案)有三种判分模式,按数据集在 tasks/<DS>/config.yaml
的 `open_judge` 字段选择(默认 llm_simple,保持历史行为):

| mode       | 需要 judge model | 需要 rubric | 判定方式                                          |
|------------|------------------|-------------|---------------------------------------------------|
| f1         | 否               | 否          | token/char 级 F1 对比 correct_answers,阈值二值化  |
| llm_simple | 是 (judge1)      | 否          | 二元 LLM judge(参照式对比,输出 is_correct)        |
| rubric     | 是 (judge1[+2])  | 是          | 数据集 rubric prompt 打总分,多 judge 取平均,过阈值 |

设计要点:
- 配置位置在数据集 config.yaml —— 判分方式是数据集答案格式的内在属性
  (短答案→f1、社会认知长答案→rubric)。
- 上下文 OpenJudgeContext 由 build_open_ctx() 一次性构建并校验,judge.py /
  voting.py / pipeline.py 都只认这个上下文,不再硬编码任一模式。
- f1/rubric 的阈值各用独立 key(f1_threshold / open_threshold),量纲不同不混用。
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .judge_schema import QAJudgeResult
from .lang import get_sample_lang
from .prompts import extract_final_answer
from .storage import pred_content

OPEN_JUDGE_MODES = ("f1", "llm_simple", "rubric")
DEFAULT_OPEN_JUDGE = "llm_simple"

# 各模式的默认阈值。
_DEFAULT_F1_THRESHOLD = 0.5
_DEFAULT_RUBRIC_THRESHOLD = 7.0
_DEFAULT_RUBRIC_MAX_SCORE = 10

# open(Q4)答案长度上限:生成 prompt 约束 ≤2000 字符,rubric 判分时同步截断到前 2000 字符。
_OPEN_ANSWER_MAX_CHARS = 2000


def needs_judge_model(mode: str) -> bool:
    return mode in ("llm_simple", "rubric")


def needs_rubric(mode: str) -> bool:
    return mode == "rubric"


def _check_mode(mode: str) -> None:
    if mode not in OPEN_JUDGE_MODES:
        raise ValueError(f"Unknown open_judge mode: {mode!r}; expected one of {OPEN_JUDGE_MODES}")


class RubricResult(BaseModel):
    """通用 rubric 打分输出:单一总分 + 理由(D1-D5 等细则写在 prompt 正文里)。"""
    score: int = Field(description="总分(0 到 max_score)")
    reason: str = Field(default="", description="打分依据")


@dataclass
class OpenJudgeContext:
    """open 判分上下文:承载模式 + judge 客户端 + 阈值 + rubric 配置。"""
    mode: str
    judge_clients: List[Any] = field(default_factory=list)
    threshold: Optional[float] = None
    rubric: Optional[Dict[str, Any]] = None  # {prompts, key_field, max_score, fallback}


# --------------------------------------------------------------------------- #
# 上下文构建 + 校验                                                             #
# --------------------------------------------------------------------------- #
def build_open_ctx(
    task_config: Dict[str, Any],
    mode: str,
    task_dir: Optional[Path] = None,
) -> OpenJudgeContext:
    """按数据集 config 构建 open 判分上下文,并集中做配置校验。

    task_dir:数据集任务目录(tasks/<DS>),用于解析 rubric.prompts_file 相对路径。
    """
    _check_mode(mode)
    dataset = task_config.get("dataset", "?")

    if mode == "f1":
        threshold = float(task_config.get("f1_threshold", _DEFAULT_F1_THRESHOLD))
        return OpenJudgeContext(mode=mode, judge_clients=[], threshold=threshold)

    # llm_simple / rubric 都需要 judge1。
    from src import runner

    judge1_cfg = task_config.get("judge1")
    if not judge1_cfg:
        raise ValueError(
            f"数据集 {dataset} 的 open_judge={mode} 需在 tasks/{dataset}/config.yaml 配置 judge1"
            f"(至少 model_name/api_key/api_url)。"
        )
    clients = [runner.create_llm_client(dict(judge1_cfg))]
    judge2_cfg = task_config.get("judge2")
    if judge2_cfg:
        clients.append(runner.create_llm_client(dict(judge2_cfg)))

    if mode == "llm_simple":
        return OpenJudgeContext(mode=mode, judge_clients=clients, threshold=None)

    # rubric:需要 rubric 块 + 加载 prompts_file。
    rubric_cfg = task_config.get("rubric")
    if not isinstance(rubric_cfg, dict):
        raise ValueError(
            f"数据集 {dataset} 的 open_judge=rubric 需在 config.yaml 配置 rubric 块"
            f"(prompts_file/key_field/max_score)。"
        )
    prompts: Dict[str, Any] = {}
    prompts_file = rubric_cfg.get("prompts_file")
    if prompts_file:
        base = Path(task_dir) if task_dir else Path(f"tasks/{dataset}")
        prompts_path = base / prompts_file
        if not prompts_path.exists():
            raise FileNotFoundError(f"rubric prompts 文件不存在: {prompts_path}")
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    rubric = {
        "prompts": prompts,
        "key_field": rubric_cfg.get("key_field", ""),
        "max_score": int(rubric_cfg.get("max_score", _DEFAULT_RUBRIC_MAX_SCORE)),
        "fallback": rubric_cfg.get("fallback_prompt", ""),
    }
    threshold = float(task_config.get("open_threshold", _DEFAULT_RUBRIC_THRESHOLD))
    return OpenJudgeContext(mode=mode, judge_clients=clients, threshold=threshold, rubric=rubric)


def ctx_from_judge_client(judge_client: Any) -> OpenJudgeContext:
    """向后兼容:旧调用方只给 judge_client 时,按 llm_simple 构造上下文。"""
    clients = [judge_client] if judge_client is not None else []
    return OpenJudgeContext(mode="llm_simple", judge_clients=clients, threshold=None)


# --------------------------------------------------------------------------- #
# 分派入口                                                                      #
# --------------------------------------------------------------------------- #
def judge_open(open_records: List[Dict[str, Any]], ctx: OpenJudgeContext) -> List[Dict[str, Any]]:
    """对 open 记录按模式判分,返回与 open_records 等长、对齐的 per_sample_results。

    每条结果至少含 is_correct/error_reason;f1/rubric 额外含 judge_score 等。
    sample_id / repeat 的回填由上层 judge_repeat 统一处理。
    """
    _check_mode(ctx.mode)
    if not open_records:
        return []
    if ctx.mode == "f1":
        return _judge_f1(open_records, ctx.threshold)
    if ctx.mode == "llm_simple":
        return _judge_llm_simple(open_records, ctx.judge_clients)
    return _judge_rubric(open_records, ctx)


# --------------------------------------------------------------------------- #
# 模式 1:F1(免 judge model)                                                   #
# --------------------------------------------------------------------------- #
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_en(text: str) -> List[str]:
    """SQuAD 式归一化:小写、去标点、去冠词,按空白切 token。"""
    text = text.lower()
    text = _ARTICLES_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    return text.split()


def _normalize_zh(text: str) -> List[str]:
    """中文 char 级:去空白与标点后逐字。"""
    text = re.sub(r"\s+", "", text)
    text = text.translate(_PUNCT_TABLE)
    return [c for c in text if c.strip()]


def _f1_score(pred: str, gold: str, lang: str) -> float:
    tokenize = _normalize_zh if lang == "zh" else _normalize_en
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1(pred: str, golds: List[str], lang: str) -> float:
    """对所有可接受答案取最大 F1。"""
    if not golds:
        return 0.0
    return max(_f1_score(pred, gold, lang) for gold in golds)


def _judge_f1(records: List[Dict[str, Any]], threshold: Optional[float]) -> List[Dict[str, Any]]:
    thr = _DEFAULT_F1_THRESHOLD if threshold is None else float(threshold)
    results: List[Dict[str, Any]] = []
    for record in records:
        content = pred_content(record)
        if content in (None, ""):
            results.append({"is_correct": False, "error_reason": "content_none", "judge_score": 0.0})
            continue
        # 只用最终答案算 F1:reasoning 协议(cot)下整段推理都在 content 里,直接拿全文比对
        # 会被推理文字稀释到接近 0。extract_final_answer 优先抽 \boxed{},抽不到回退全文。
        lang = get_sample_lang(record.get("meta"))
        score = max_f1(extract_final_answer(content), record.get("correct_answers") or [], lang)
        is_correct = score >= thr
        results.append({
            "is_correct": is_correct,
            "error_reason": None if is_correct else "below_threshold",
            "judge_score": score,
        })
    return results


# --------------------------------------------------------------------------- #
# 模式 2:llm_simple(二元 LLM judge,即历史行为)                               #
# --------------------------------------------------------------------------- #
def _judge_llm_simple(records: List[Dict[str, Any]], judge_clients: List[Any]) -> List[Dict[str, Any]]:
    from .judge import judge_prompt  # 复用现有 storyless 参照式 prompt

    if not judge_clients:
        raise ValueError("open_judge=llm_simple 但没有可用的 judge client。")
    judge_client = judge_clients[0]

    results: List[Dict[str, Any]] = [
        {"is_correct": False, "error_reason": "content_none"} for _ in records
    ]
    prompts: List[str] = []
    prompt_indices: List[int] = []
    for index, record in enumerate(records):
        content = pred_content(record)
        if content in (None, ""):
            continue
        prompts.append(judge_prompt(record))
        prompt_indices.append(index)

    if prompts:
        # create 模式才能正确传入 extra_body(含 enable_thinking: false);parse 模式会覆盖
        # vLLM chat template 导致 thinking 被意外开启、token 耗尽。
        responses = judge_client.batch_generate_structure(prompts, QAJudgeResult, mode="create", desc="Judging")
        for index, response in zip(prompt_indices, responses):
            content = response.content
            if content is None:
                results[index] = {"is_correct": False, "error_reason": "judge_error"}
                continue
            results[index] = {
                "is_correct": bool(content.is_correct),
                "error_reason": None if content.is_correct else "wrong_answer",
            }
    return results


# --------------------------------------------------------------------------- #
# 模式 3:rubric(数据集 rubric prompt + judge model,多 judge 取平均)          #
# --------------------------------------------------------------------------- #
# 跨维度通用的额外校准规则，优先级高于各 per-dim 细则；插在 rubric 正文与题面之间。
# 与评测模型生成 prompt 的约束(≤2000字符 / 单要点 / 不复制参考答案 / 反堆砌)一一对应，
# 用判分侧惩罚兜住生成侧没遵守约束的情况(罗列碰答案、堆砌、复述参考答案)。
_RUBRIC_EXTRA_CALIBRATION = """## 额外校准规则（优先级高于各维度细则）
- 参考答案要点只用于判分校准，不是逐字匹配模板；不得因被测答案贴近参考答案措辞而自动给高分。
- 若被测答案主要是在复述、改写或堆砌参考答案/题干信息，但缺少自主的情境证据链与机制解释，总分最高6分。
- 若被测答案明显依靠罗列覆盖大量可能点来碰答案，总分最高7分；严重堆砌且缺乏取舍时最高6分。
- 若一句话同时塞入多个独立判断点，导致语义边界模糊或无法判断每个点是否成立，相关维度最高1分。
- 给高分必须基于清晰、可核验、情境特异的单点论证；泛泛术语、概念堆叠、模糊归纳不能替代具体分析。"""


def _build_rubric_prompt(record: Dict[str, Any], rubric: Dict[str, Any]) -> str:
    key_field = rubric.get("key_field") or ""
    key = ""
    if key_field:
        meta = record.get("meta") or {}
        key = str(meta.get(key_field) or "")
    entry = rubric["prompts"].get(key) if key else None
    head = (entry.get("prompt") if isinstance(entry, dict) else None) or rubric.get("fallback") or ""

    max_score = rubric.get("max_score", _DEFAULT_RUBRIC_MAX_SCORE)
    reference = "\n".join(record.get("correct_answers") or [])
    # 生成 prompt 已约束答案 ≤2000 字符；判分侧同样把被测答案截断到前 2000 字符，
    # 既与生成约束对齐，也避免个别超长输出撑爆 judge 上下文。
    model_output = (pred_content(record) or "")[:_OPEN_ANSWER_MAX_CHARS]
    return (
        f"{head}\n\n"
        f"{_RUBRIC_EXTRA_CALIBRATION}\n\n"
        f"【问题】{record.get('question', '')}\n\n"
        f"【参考答案要点】\n{reference}\n\n"
        f"【被测答案】\n{model_output}\n\n"
        f"## 输出格式\n"
        f'只输出一个 JSON 对象:{{"score": 总分(0-{max_score}), "reason": "打分依据"}}'
    )


def _judge_rubric(records: List[Dict[str, Any]], ctx: OpenJudgeContext) -> List[Dict[str, Any]]:
    rubric = ctx.rubric or {}
    threshold = _DEFAULT_RUBRIC_THRESHOLD if ctx.threshold is None else float(ctx.threshold)
    judges = ctx.judge_clients
    if not judges:
        raise ValueError("open_judge=rubric 但没有可用的 judge client。")

    prompts = [_build_rubric_prompt(record, rubric) for record in records]

    # 每个 judge 跑全量,逐样本取平均。
    per_judge: List[List[Optional[RubricResult]]] = []
    for judge_client in judges:
        responses = judge_client.batch_generate_structure(
            prompts, RubricResult, mode="create", desc="Judging (rubric)"
        )
        per_judge.append([r.content for r in responses])

    results: List[Dict[str, Any]] = []
    for index in range(len(records)):
        objs = [per_judge[j][index] for j in range(len(judges))]
        vals = [float(o.score) for o in objs if o is not None]
        avg = (sum(vals) / len(vals)) if vals else None
        result: Dict[str, Any] = {"judge_score": avg}
        for j, obj in enumerate(objs):
            result[f"judge{j + 1}_score"] = (float(obj.score) if obj is not None else None)
        first = next((o for o in objs if o is not None), None)
        if first is not None:
            result["judge_reason"] = first.reason
        if avg is None:
            result["is_correct"] = False
            result["error_reason"] = "judge_error"
        else:
            result["is_correct"] = avg >= threshold
            result["error_reason"] = None if result["is_correct"] else "below_threshold"
        results.append(result)
    return results


__all__ = [
    "OPEN_JUDGE_MODES",
    "DEFAULT_OPEN_JUDGE",
    "OpenJudgeContext",
    "RubricResult",
    "needs_judge_model",
    "needs_rubric",
    "build_open_ctx",
    "ctx_from_judge_client",
    "judge_open",
    "max_f1",
]
