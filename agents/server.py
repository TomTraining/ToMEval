"""
统一 Agent 运行时(评测方所有,参赛方**看不到也不需要改动本文件**)。

本文件封装评测框架的 HTTP 契约(POST /predict、GET /health)、请求体拆分、
并发、异常兜底与错误信封。它从参赛方提交的仓库里**动态加载 `solution.py`**,
调用其中的 `predict(sample, model) -> str | list`。

参赛方交付物:一个仓库,根目录必须含 `solution.py`,其中定义 `predict` 函数。
`solution.py` 可 import 同仓库内的其它模块(仓库根目录会被加入 sys.path)。

启动(评测方执行):
    SOLUTION_DIR=/path/to/参赛方仓库 PORT=8100 python agents/server.py
其中 SOLUTION_DIR 缺省为当前工作目录(即在参赛方仓库目录下直接跑本 server)。

请求体拆分:
  · model —— 评测方部署的统一模型后端 {api_url, api_key, model_name},随每条请求下发。
  · sample —— 其余字段即题目 {sample_id, prompt_type, lang, story, question, options?/sub_questions?}。
predict 的返回值原样作为 prediction 回传;格式校验由框架完成(格式不符 → 该题判错)。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Tuple

PORT = int(os.environ.get("PORT", "8100"))

# 请求体里属于「模型连接信息」的字段名;其余字段组成 sample。
_MODEL_KEY = "model"


# ---------------------------------------------------------------------------
# 动态加载参赛方的 solution.py（fail-fast:缺文件/缺 predict/非可调用 → 启动即报错退出）
# ---------------------------------------------------------------------------

def _load_predict() -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
    """从 SOLUTION_DIR（缺省 cwd）加载 solution.py，返回其 predict 函数。"""
    solution_dir = os.path.abspath(os.environ.get("SOLUTION_DIR", os.getcwd()))
    solution_path = os.path.join(solution_dir, "solution.py")

    if not os.path.isfile(solution_path):
        raise SystemExit(
            f"[agent] 找不到 solution.py:{solution_path}\n"
            f"  请把 SOLUTION_DIR 指向参赛方仓库根目录(须含 solution.py),"
            f"或在该目录下启动本 server。"
        )

    # 让 solution.py 能 import 同仓库内的其它模块。
    if solution_dir not in sys.path:
        sys.path.insert(0, solution_dir)

    spec = importlib.util.spec_from_file_location("solution", solution_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[agent] 无法加载 solution.py:{solution_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 —— solution.py 顶层执行报错
        traceback.print_exc()
        raise SystemExit(f"[agent] 加载 solution.py 失败:{error}") from error

    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise SystemExit(
            f"[agent] solution.py 未定义可调用的 predict 函数:{solution_path}\n"
            f"  须实现:def predict(sample: dict, model: dict) -> str | list"
        )
    return predict


def _split_body(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """把请求体拆成 (sample, model)。

    model 是评测方下发的后端凭证 {api_url, api_key, model_name};其余字段即题目 sample。
    """
    model = body.get(_MODEL_KEY) or {}
    sample = {k: v for k, v in body.items() if k != _MODEL_KEY}
    return sample, model


def _error(sample_id: Any, code: str, retryable: bool, message: str = "") -> Dict[str, Any]:
    """按契约拼错误信封:{sample_id, error:{code, retryable, message?}}。"""
    err: Dict[str, Any] = {"code": code, "retryable": retryable}
    if message:
        err["message"] = message[:200]
    return {"sample_id": sample_id, "error": err}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, obj: Dict[str, Any]) -> None:
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
            self._send(404, _error(None, "INVALID_REQUEST", False, "not found"))

    def do_POST(self):  # noqa: N802
        if self.path != "/predict":
            self._send(404, _error(None, "INVALID_REQUEST", False, "not found"))
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._send(400, _error(None, "INVALID_REQUEST", False, "bad json"))
            return

        sample, model = _split_body(body)
        sample_id = sample.get("sample_id")

        try:
            prediction = PREDICT(sample, model)
        except Exception as error:  # noqa: BLE001
            # predict 内部异常(通常是调模型失败)→ 可重试的 MODEL_ERROR,框架退避后重试。
            # 打印堆栈便于调试;正式评测时框架会把该样本重试/判错,服务不会崩。
            traceback.print_exc()
            self._send(502, _error(sample_id, "MODEL_ERROR", True, str(error)))
            return

        # prediction 原样回传;格式校验由框架完成(格式不符 → 该题判错)。
        self._send(200, {"sample_id": sample_id, "prediction": prediction})

    def log_message(self, *args):  # 静音默认访问日志
        return


# 启动时即加载 solution.predict(缺失/报错则 fail-fast,不带病起服务)。
PREDICT = _load_predict()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[agent] listening on 0.0.0.0:{PORT}  (solution: {os.environ.get('SOLUTION_DIR', os.getcwd())}/solution.py)")
    server.serve_forever()
