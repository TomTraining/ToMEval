"""
多轮参考 agent —— 验证模型代理的 model_calls 累加(每样本调 2 次模型)。

与 agents/mock 的唯一区别:predict() 内部调 2 次模型,模拟"先分析、再定答案"的
两步智能体。用来确认代理记到的 model_calls = 样本数 × 2(而非样本数)。

契约与 agents/mock 完全一致(GET /health、POST /predict),参赛方可任意语言重写。
"""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import OpenAI

# 占位符由评测框架替换成代理的连接信息(LLM_API_URL 指向代理,不是真实后端)。
API_URL = os.environ.get("LLM_API_URL", "")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "")
PORT = int(os.environ.get("PORT", "8100"))

_client = OpenAI(api_key=API_KEY, base_url=API_URL, timeout=600.0)


def _render(sample: dict) -> str:
    """把结构化样本排成纯文本。选项字母由评测框架给定,必须原样使用。"""
    lines = [sample.get("story", ""), "", sample.get("question", "")]
    options = sample.get("options")
    if options:
        lines.append("")
        lines.append("Options:")
        for letter, text in options.items():
            lines.append(f"{letter}. {text}")
    return "\n".join(lines)


def _postprocess(prediction: str, prompt_type: str) -> str:
    """把模型自由文本收敛成契约要求的形态(mcq 抽字母;open 原样)。"""
    text = (prediction or "").strip()
    if prompt_type in ("mcq_single", "mcq_grouped"):
        m = re.search(r"[A-Za-z]", text)
        return m.group(0).upper() if m else ""
    if prompt_type == "mcq_multi":
        letters = []
        seen = set()
        for token in re.findall(r"[A-Za-z]", text):
            up = token.upper()
            if up not in seen:
                seen.add(up)
                letters.append(up)
        return ",".join(letters)
    return text


def predict(sample: dict) -> str:
    """两步智能体:第 1 次调模型做分析,第 2 次带着分析定最终答案。

    这里每样本调 2 次模型,用于验证代理的 model_calls 累加(应为样本数 × 2)。
    """
    body = _render(sample)
    ptype = sample.get("prompt_type", "open")

    # 第 1 次:让模型先分析角色心理状态(不下结论)。
    analysis = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": f"{body}\n\nBriefly analyze the characters' mental states. Do NOT give the final answer yet."},
        ],
        temperature=0.0,
        max_tokens=512,
    ).choices[0].message.content or ""

    # 第 2 次:带着分析给最终答案。
    if ptype == "mcq_single":
        directive = "Now answer with only the letter of the single best option."
    elif ptype == "mcq_multi":
        directive = "Now answer with every correct option letter, comma-separated (e.g. A,C)."
    elif ptype == "mcq_grouped":
        directive = "Now for each question in order, answer with its option letter."
    else:
        directive = "Now answer the question directly."

    final = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": body},
            {"role": "assistant", "content": analysis},
            {"role": "user", "content": directive},
        ],
        temperature=0.0,
        max_tokens=2048,
    ).choices[0].message.content or ""

    return _postprocess(final, ptype)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, obj: dict) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/predict":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            sample = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._send_json(400, {"error": "bad json"})
            return
        sample_id = sample.get("sample_id")
        try:
            prediction = predict(sample)
        except Exception as error:  # noqa: BLE001 —— agent 内部错误,回空预测让评测判错即可
            self._send_json(200, {"sample_id": sample_id, "prediction": None, "error": str(error)})
            return
        self._send_json(200, {"sample_id": sample_id, "prediction": prediction})

    def log_message(self, *args):  # 静音默认访问日志
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[mock-multicall-agent] listening on 127.0.0.1:{PORT}, model={MODEL}, api_url={API_URL}")
    server.serve_forever()
