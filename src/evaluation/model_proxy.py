"""
模型代理(Model Proxy)—— 智能体评测里模型访问的统一入口 + 效率记账。

agent 内部代码自写、可多轮/多次调 model,但它拿到的 LLM_API_URL 指向的是**本代理**,
不是真实后端(vLLM / 外部 API)。代理负责:

1. 转发:把 agent 的 OpenAI 兼容请求转发给真实后端(真实 api_key 只有代理知道)。
2. 记账:从后端响应的 usage 累加 prompt/completion tokens 和调用次数(线程安全)。
   —— 这是效率的权威口径,agent 伪造不了(它连 usage 都看不到)。
3. 抹账:回给 agent 的响应是标准 OpenAI 格式,但**剥掉 usage 字段**,只留 content + reasoning。
4. 防偷换模型:强制把请求体的 model 改成我们配置的统一 model,agent 传什么都不作数。
5. 并发限流:信号量限制同时打到后端的请求数。
6. 只支持非流式:收到 stream=true 直接 400(非流式才能稳定拿到 usage)。

效率只记录、不进排名(见 docs/agent_eval_plan.md)。全部标准库,零新依赖。
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProxyUsage:
    """代理累计的效率计数(线程安全由 ModelProxy 的锁保护)。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0          # 成功从后端拿到响应的 chat 调用数
    failed_calls: int = 0         # 后端非 2xx 或转发异常
    rejected_calls: int = 0       # 被代理挡下(stream / 非法请求)
    swapped_model_attempts: int = 0  # agent 试图用非统一 model 的次数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_calls": self.model_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "swapped_model_attempts": self.swapped_model_attempts,
        }


class ModelProxy:
    """本地 OpenAI 兼容代理:转发 → 记账 → 抹 usage → 回 agent。"""

    def __init__(
        self,
        backend_url: str,
        backend_key: str,
        model_name: str,
        host: str = "127.0.0.1",
        port: int = 0,
        max_concurrency: int = 16,
        request_timeout: float = 1200.0,
    ):
        # backend_url 形如 https://api.tokenkey.dev/v1 或 http://127.0.0.1:8000/v1。
        self.backend_url = backend_url.rstrip("/")
        self.backend_key = backend_key
        self.model_name = model_name
        self.host = host
        self.port = port  # 0 = 自动选空闲端口
        self.request_timeout = request_timeout

        self.usage = ProxyUsage()
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max(1, int(max_concurrency)))
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -----------------------------------------------------------------------
    # 生命周期
    # -----------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """给 agent 的 LLM_API_URL(OpenAI 约定带 /v1)。"""
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> str:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        # port=0 时由系统分配,回填真实端口。
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            f"[proxy] 监听 {self.base_url} → 后端 {self.backend_url}(统一 model={self.model_name})"
        )
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.usage.to_dict()

    # -----------------------------------------------------------------------
    # 转发 + 记账(由 handler 调用)
    # -----------------------------------------------------------------------

    def handle_chat(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """处理一次 chat/completions:强制 model、拒流式、转发、记账、抹 usage。"""
        # 只支持非流式:流式响应默认不带 usage,记不准。
        if body.get("stream"):
            with self._lock:
                self.usage.rejected_calls += 1
            return 400, {
                "error": {
                    "message": "streaming is not supported in this evaluation proxy; set stream=false",
                    "type": "invalid_request_error",
                }
            }

        # 防偷换模型:强制统一 model,记录 agent 的尝试。
        requested = body.get("model")
        if requested and requested != self.model_name:
            with self._lock:
                self.usage.swapped_model_attempts += 1
            logger.warning(f"[proxy] agent 请求 model={requested!r},已强制改回 {self.model_name!r}")
        body["model"] = self.model_name

        status, payload = self._forward(body)

        if status == 200 and isinstance(payload, dict):
            self._account(payload.get("usage"))
            # 抹掉 usage:agent 只看到 content + reasoning,看不到 token 账。
            payload.pop("usage", None)
        else:
            with self._lock:
                self.usage.failed_calls += 1
        return status, payload

    def _forward(self, body: Dict[str, Any]) -> Tuple[int, Any]:
        """把请求转发到真实后端(真实 key 只在这里注入)。并发受信号量限制。"""
        url = f"{self.backend_url}/chat/completions"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.backend_key}",
            },
            method="POST",
        )
        with self._sem:
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    text = response.read().decode("utf-8")
                    return response.status, json.loads(text)
            except urllib.error.HTTPError as error:  # 后端返回了错误状态码
                try:
                    detail = error.read().decode("utf-8")
                    return error.code, json.loads(detail)
                except Exception:  # noqa: BLE001
                    return error.code, {"error": {"message": f"backend error {error.code}"}}
            except Exception as error:  # noqa: BLE001 —— 网络/超时等
                logger.warning(f"[proxy] 转发后端失败: {error}")
                return 502, {"error": {"message": f"proxy forward failed: {error}"}}

    def _account(self, usage: Optional[Dict[str, Any]]) -> None:
        """从后端响应的 usage 累加计数。usage 缺失时只记一次成功调用。"""
        with self._lock:
            self.usage.model_calls += 1
            if isinstance(usage, dict):
                self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.usage.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                total = usage.get("total_tokens")
                if total is None:
                    total = (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)
                self.usage.total_tokens += int(total or 0)


# ---------------------------------------------------------------------------
# HTTP handler(闭包持有 proxy 实例)
# ---------------------------------------------------------------------------


def _make_handler(proxy: ModelProxy):
    class _ProxyHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, obj: Any) -> None:
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            # 只对 chat/completions 做记账与强制;其余 POST 一律拒绝(收口模型访问面)。
            if not self.path.endswith("/chat/completions"):
                self._send_json(404, {"error": {"message": f"unsupported path {self.path}"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except Exception:  # noqa: BLE001
                self._send_json(400, {"error": {"message": "invalid JSON body"}})
                return
            status, payload = proxy.handle_chat(body)
            self._send_json(status, payload)

        def do_GET(self):  # noqa: N802
            # /v1/models:回一个只含统一 model 的列表(有些 SDK 会探测),不透传后端。
            if self.path.endswith("/models"):
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": proxy.model_name, "object": "model", "owned_by": "eval-proxy"}],
                })
                return
            self._send_json(404, {"error": {"message": f"unsupported path {self.path}"}})

        def log_message(self, *args):  # 静音默认访问日志
            return

    return _ProxyHandler


def find_free_port(host: str = "127.0.0.1") -> int:
    """返回一个当前空闲的端口(bind 到 0 让系统分配)。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]
