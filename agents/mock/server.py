"""
最小参考 agent —— 单轮问答,也是参赛方模板。

自包含:除 openai 外只用标准库。文件内含少量辅助函数(把结构化样本渲染成 prompt、
把模型自由文本收敛成契约要求的 prediction、拼响应信封),核心策略只有"调一次模型"。
参赛方可用任意语言/框架重写,只要满足契约(见 docs/agent_eval.md)。

模型访问:调 model 的连接信息(api_url/api_key/model_name,我们部署的统一后端)由框架随
每条 /predict 请求体的 `model` 字段下发;本服务只需监听 PORT(参赛方自行启动、把 URL 交给
框架)。
"""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from openai import OpenAI

PORT = int(os.environ.get("PORT", "8100"))


def _model_client(model: Optional[Dict[str, Any]]) -> "OpenAI":
    """按请求体 `model` 字段构造 OpenAI 客户端(api_url/api_key 我们部署的统一后端)。"""
    model = model or {}
    return OpenAI(
        api_key=model.get("api_key", ""),
        base_url=model.get("api_url", ""),
        timeout=600.0,
    )


# --- 契约辅助:渲染样本 / 收敛 prediction / 拼响应信封 --------------------------

# 提示模型在最后一行用固定前缀给答案,便于稳健提取(避免从正文里瞎抓字母)。
_ANSWER_TAG = "ANSWER:"


def render(sample: Dict[str, Any]) -> str:
    """把结构化样本排成纯文本 prompt。选项字母由评测框架给定,必须原样使用。"""
    lines = [sample.get("story", ""), "", sample.get("question", "")]
    options = sample.get("options")
    if options:
        lines.append("")
        lines.append("Options:")
        for letter, text in options.items():
            lines.append(f"{letter}. {text}")
    return "\n".join(lines)


def answer_directive(prompt_type: str) -> str:
    """追加到 prompt 末尾的答题指令:要求模型用 `ANSWER:` 行给出答案。"""
    if prompt_type == "mcq_single":
        return f"\n\nEnd your reply with a line `{_ANSWER_TAG} X` where X is the single best option letter."
    if prompt_type == "mcq_multi":
        return (
            f"\n\nEnd your reply with a line `{_ANSWER_TAG} X,Y` listing every correct "
            f"option letter, comma-separated."
        )
    if prompt_type == "mcq_grouped":
        return (
            f"\n\nEnd your reply with a line `{_ANSWER_TAG} X,Y` giving one option letter "
            f"per sub-question, in order, comma-separated."
        )
    return f"\n\nEnd your reply with a line `{_ANSWER_TAG} <your answer>`."


def _answer_segment(text: str) -> str:
    """取最后一个 `ANSWER:` 行之后的内容;没有则退回全文(尽量别违约)。"""
    matches = list(re.finditer(r"(?im)^\s*ANSWER:\s*(.*)$", text))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


def to_prediction(raw: str, prompt_type: str, options: Optional[Dict[str, str]]) -> Any:
    """把模型自由文本收敛成契约要求的严格格式。

    从 `ANSWER:` 行提取,并约束到 options 的合法字母(避免把正文里的字母误当选项)。
    提取不到时退回第一个合法选项(保证永不违约);open 题返回非空文本。
    """
    segment = _answer_segment(raw)
    if prompt_type == "open":
        return segment or (raw.strip() or "(no answer)")

    valid = list((options or {}).keys())
    valid_set = set(valid)
    picked: List[str] = []
    seen = set()
    for ch in segment.upper():
        if ch in valid_set and ch not in seen:
            seen.add(ch)
            picked.append(ch)

    if prompt_type == "mcq_single":
        return picked[0] if picked else (valid[0] if valid else "A")

    # mcq_multi / mcq_grouped:字母数组。
    if not picked:
        picked = [valid[0]] if valid else ["A"]
    if prompt_type == "mcq_multi":
        picked = sorted(set(picked))  # 契约:升序、去重
    return picked


def ok_response(sample_id: Any, prediction: Any) -> Dict[str, Any]:
    return {"sample_id": sample_id, "prediction": prediction}


def error_response(sample_id: Any, code: str, retryable: bool, message: str = "") -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "retryable": retryable}
    if message:
        err["message"] = message[:200]
    return {"sample_id": sample_id, "error": err}


# --- 答题策略:单轮调一次模型 -------------------------------------------------


def predict(sample: dict):
    """单轮:渲染样本 + 答题指令 → 调一次模型 → 收敛成严格 prediction。"""
    ptype = sample.get("prompt_type", "open")
    model = sample.get("model") or {}
    client = _model_client(model)
    prompt = render(sample) + answer_directive(ptype)
    completion = client.chat.completions.create(
        model=model.get("model_name", ""),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
        # qwen3 系模型非流式调用要求显式关闭 thinking,否则后端返回 400。
        # 对不认此参数的后端无害(OpenAI 兼容层会忽略未知字段)。
        extra_body={"enable_thinking": False},
    )
    raw = completion.choices[0].message.content or ""
    return to_prediction(raw, ptype, sample.get("options") or {})


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, obj: dict) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, error_response(None, "INVALID_REQUEST", False, "not found"))

    def do_POST(self):  # noqa: N802
        if self.path != "/predict":
            self._send(404, error_response(None, "INVALID_REQUEST", False, "not found"))
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            sample = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._send(400, error_response(None, "INVALID_REQUEST", False, "bad json"))
            return
        sample_id = sample.get("sample_id")
        try:
            prediction = predict(sample)
        except Exception as error:  # noqa: BLE001 —— 调模型失败:MODEL_ERROR(可重试)
            self._send(502, error_response(sample_id, "MODEL_ERROR", True, str(error)))
            return
        self._send(200, ok_response(sample_id, prediction))

    def log_message(self, *args):  # 静音默认访问日志
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[mock] listening on 0.0.0.0:{PORT} (model 凭证随每条请求下发)")
    server.serve_forever()
