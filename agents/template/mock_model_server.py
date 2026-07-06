"""
模拟 model 服务 —— 一个假的 OpenAI 兼容接口,供本地自测 predict 用(无需真实模型/网络)。

它只实现 `POST /v1/chat/completions`,读取最后一条 user message,粗略识别其中的答题指令
(ANSWER: 行、Options 里的字母、题型线索),**返回一个能被大多数 predict 正确解析的合法答案**。
它不理解题意、答案是随机/固定的,只用来验证「predict 能把模型输出收敛成契约要求的格式」这条链路。

用法:
  · 由 selftest.py 自动在后台线程起(默认),参赛方一般不用直接碰。
  · 也可单独起来手测:PORT=9001 python mock_model_server.py
    然后把 model={"api_url":"http://127.0.0.1:9001/v1","api_key":"x","model_name":"mock"} 传给 predict。

如果你想用**真实模型**自测(自己的 vLLM 或我们给的后端),不必用本文件:
直接给 selftest.py 设 MODEL_API_URL / MODEL_API_KEY / MODEL_NAME 环境变量即可(见 selftest.py)。
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional


def _letters_in_prompt(text: str) -> List[str]:
    """从 prompt 的 Options 段抽出出现过的选项字母(形如 `A.` / `A、` / `A)`)。"""
    seen = []
    for m in re.finditer(r"(?m)^\s*([A-Z])[\.\)、\:]", text):
        ch = m.group(1)
        if ch not in seen:
            seen.append(ch)
    return seen


def _fake_answer(prompt: str) -> str:
    """根据 prompt 里的线索,伪造一段「以 ANSWER: 行结尾」的模型输出。

    - 有选项字母:随机挑 1~2 个合法字母(mcq_multi 可能多选)。
    - 无选项字母(open):返回一句非空文本。
    默认实现的 predict 会从 ANSWER: 行提取,这样能覆盖到最常见的解析路径。
    """
    letters = _letters_in_prompt(prompt)
    low = prompt.lower()

    if not letters:
        # open 题:给一句非空文本。
        return "Reasoning about the scenario.\nANSWER: This is a mock free-form answer."

    # 多选线索:prompt 里出现 "select all" / "X,Y" / "every correct" 之类。
    multi = any(k in low for k in ("select all", "every correct", "x,y", "comma-separated"))
    grouped = "sub-question" in low or "per sub" in low

    if grouped:
        # 每个子问给一个字母(这里简单地对每个字母各挑一次;predict 会按子问 options 再校准)。
        picks = [random.choice(letters) for _ in range(max(1, len(letters)))]
        return "Analysis...\nANSWER: " + ",".join(picks)

    if multi:
        k = random.randint(1, len(letters))
        picks = sorted(random.sample(letters, k))
        return "Analysis...\nANSWER: " + ",".join(picks)

    # 单选:挑一个。
    return f"Analysis...\nANSWER: {random.choice(letters)}"


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, obj: dict) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._send(400, {"error": {"message": "bad json"}})
            return

        messages = body.get("messages") or []
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content") or ""
                break

        answer = _fake_answer(user_text)
        completion = {
            "id": "mock-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8)),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send(200, completion)

    def do_GET(self):  # noqa: N802
        # 便于探活。
        self._send(200, {"status": "ok"})

    def log_message(self, *args):  # 静音默认访问日志
        return


def serve(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9001"))
    srv = serve(port)
    print(f"[mock-model] listening on http://127.0.0.1:{port}/v1 (fake OpenAI chat.completions)")
    srv.serve_forever()
