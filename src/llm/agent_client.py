"""
Agent Client - 黑盒智能体评测后端

参赛方提交一个 HTTP 服务(内部代码自写、可多轮/多次调 model),我们逐条把结构化样本
POST 到其 /predict,收回一条预测。模型访问经我们的代理(见 model_proxy.py),agent 拿到
的 LLM_API_URL 指向代理而非真实后端。

契约(权威定义在 src/evaluation/agent_contract.py + docs/agent_schema/*.json,说明见
docs/agent_eval.md):
- 请求:{sample_id, prompt_type, lang, story, question, options?/sub_questions?}
  · options 由 build_option_bundle 生成(带 shuffle);correct_letters 绝不下发。
- 响应:{sample_id, prediction} 或 {sample_id, error}
  · prediction 严格单一格式:mcq_single→"A";mcq_multi→["A","C"](升序去重);
    mcq_grouped→["A","B"];open→非空字符串。违约按答错处理并计入 format_violation。
  · error:{code, retryable, message?}。retryable=true 的错误 + 网络层错误 → 退避重试;
    重试耗尽 / 不可重试 → content=None(judge 判 content_none)。

签名对齐 ContentClient.batch_generate(prompts, desc, system_prompts),额外接一个与 prompts
等长的 payloads(结构化样本);agent 只吃 payloads,prompts(渲染文本)仅作兜底/调试。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.evaluation import agent_contract as contract

from .client import LLMResponse, LLMUsage


class AgentClient:
    """把评测样本 POST 给参赛 agent 的 HTTP 服务,收回预测。"""

    def __init__(
        self,
        base_url: str,
        predict_path: str = "/predict",
        timeout: float = 60.0,
        max_workers: int = 16,
        max_retries: int = contract.MAX_RETRIES,
        per_sample_timeout: float = contract.PER_SAMPLE_TIMEOUT_SECONDS,
    ):
        # base_url 形如 http://127.0.0.1:8100;predict_path 拼在其后。
        self.base_url = base_url.rstrip("/")
        self.predict_path = predict_path if predict_path.startswith("/") else f"/{predict_path}"
        self.predict_url = f"{self.base_url}{self.predict_path}"
        # timeout:单次 HTTP 请求超时;per_sample_timeout:含所有重试的单样本总超时。
        self.timeout = timeout
        self.per_sample_timeout = per_sample_timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        # 与 LLMClient 接口对齐:效率主口径走代理记账,这里只记调用成功/失败/违约计数。
        self.usage: LLMUsage = LLMUsage()
        self.format_violations: int = 0   # prediction 违反严格格式的样本数
        self._lock = threading.Lock()
        self.model = "agent"

    # -----------------------------------------------------------------------
    # 单样本预测(含重试)
    # -----------------------------------------------------------------------

    def _post_once(self, request_body: Dict[str, Any]) -> Tuple[int, Any]:
        """发一次 HTTP POST。返回 (status, parsed_body_or_None)。

        status:HTTP 状态码;网络层失败用 -1 表示(拿不到响应)。
        parsed_body:解析后的 JSON(dict)或 None(解析失败/网络失败)。
        """
        data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.predict_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
            try:
                return response.status, json.loads(body)
            except (json.JSONDecodeError, ValueError):
                return response.status, None
        except urllib.error.HTTPError as error:
            # agent 返回了非 2xx:尽量解析 body 里的结构化 error。
            try:
                detail = json.loads(error.read().decode("utf-8"))
            except Exception:  # noqa: BLE001
                detail = None
            return error.code, detail
        except Exception:  # noqa: BLE001 —— 连接失败/读超时等网络层错误
            return -1, None

    def _predict_one(self, payload: Dict[str, Any]) -> LLMResponse:
        """对一条样本作答(含退避重试),归一化成 LLMResponse。

        payload:predict_records 构造的语义样本,直接作为请求体发给 agent。
        content 归一化:合法 prediction → 判分层形态;非法/错误/超时 → None。
        """
        prompt_type = payload.get("prompt_type", "open")
        sample_id = payload.get("sample_id")
        request_body = dict(payload)

        deadline = time.time() + self.per_sample_timeout
        attempt = 0
        last_reason = "unknown"
        while True:
            status, body = self._post_once(request_body)
            content, reason, retryable = self._interpret(prompt_type, status, body, sample_id)
            if content is not None:
                with self._lock:
                    self.usage.total_calls += 1
                    self.usage.successful_calls += 1
                return LLMResponse(content=content, reasoning="")

            last_reason = reason
            # 不可重试,或重试次数/总时间耗尽 → 放弃,判错。
            if not retryable or attempt >= self.max_retries or time.time() >= deadline:
                break
            attempt += 1
            wait = min(contract.backoff_seconds(attempt), max(0.0, deadline - time.time()))
            logging.warning(
                f"[AgentClient] sample_id={sample_id} 第{attempt}次重试(原因={reason}),退避 {wait:.1f}s"
            )
            time.sleep(wait)

        logging.warning(f"[AgentClient] sample_id={sample_id} 最终失败(原因={last_reason}),判错。")
        with self._lock:
            self.usage.total_calls += 1
            self.usage.failed_calls += 1
        return LLMResponse(content=None, reasoning="")

    def _interpret(
        self,
        prompt_type: str,
        status: int,
        body: Any,
        sample_id: Any,
    ) -> Tuple[Optional[str], str, bool]:
        """把一次 HTTP 结果翻译成 (content, reason, retryable)。

        content:合法预测归一化后的字符串(判分层用);无法得到有效预测时为 None。
        reason :诊断串(重试/最终失败日志用)。
        retryable:是否可重试(仅在 content 为 None 时有意义)。
        """
        # 网络层失败:拿不到响应,按可重试超时处理。
        if status == -1:
            return None, "network_error", True

        # 非 2xx:优先用 body 里的结构化 error.code 判断可重试性,否则按 HTTP 状态码兜底推断。
        if not (200 <= status < 300):
            code = None
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                code = body["error"].get("code")
            if code is None:
                code = contract.code_for_http_status(status)
            return None, f"http_{status}:{code}", contract.is_retryable(code)

        # 2xx 但 body 解析失败或不是对象:协议破坏,不可重试,判错。
        if not isinstance(body, dict):
            return None, "bad_response_body", False

        # sample_id 回带校验(不一致只告警,不判错——以我方 index 为准)。
        if body.get("sample_id") not in (None, sample_id):
            logging.warning(
                f"[AgentClient] 响应 sample_id={body.get('sample_id')!r} 与请求 {sample_id!r} 不一致。"
            )

        # 2xx 里也可能带 error(agent 明确表示这条答不出):按其 retryable。
        if isinstance(body.get("error"), dict):
            err = body["error"]
            code = err.get("code")
            return None, f"error:{code}", bool(err.get("retryable")) and contract.is_retryable(code)

        # 正常路径:严格校验 prediction。
        prediction = body.get("prediction")
        if prediction is None:
            return None, "no_prediction", False
        ok, why = contract.validate_prediction(prompt_type, prediction)
        if not ok:
            # 格式违约:按答错处理(content=None → content_none),单列计数,不重试。
            with self._lock:
                self.format_violations += 1
            logging.warning(f"[AgentClient] sample_id={sample_id} prediction 违约: {why}")
            return None, f"format_violation:{why}", False

        normalized = contract.normalize_prediction(prompt_type, prediction)
        # 归一后交给判分层:mcq_multi/grouped 是列表,序列化成字符串由 extractor='agent' 再解析。
        if isinstance(normalized, list):
            return ",".join(str(x) for x in normalized), "ok", False
        return str(normalized), "ok", False

    # -----------------------------------------------------------------------
    # 批量预测(签名对齐 ContentClient.batch_generate)
    # -----------------------------------------------------------------------

    def batch_generate(
        self,
        prompts: List[str],
        desc: str = "Generating",
        system_prompts: Optional[List[str]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> List[LLMResponse]:
        """并发把 payloads 逐条 POST 给 agent。

        payloads 与 prompts 等长、一一对应,是发给 agent 的结构化样本;为 None 时报错
        (agent 模式必须由 predict_records 传入结构化样本)。system_prompts 在 agent 模式无意义。
        """
        if payloads is None:
            raise ValueError("AgentClient.batch_generate requires structured `payloads` (agent 模式).")
        if len(payloads) != len(prompts):
            raise ValueError(
                f"payloads length ({len(payloads)}) must match prompts length ({len(prompts)})"
            )

        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = [executor.submit(self._predict_one, payload) for payload in payloads]
            results: List[LLMResponse] = []
            for future in tqdm(futures, total=len(futures), desc=desc, miniters=100):
                results.append(future.result())
            return results

    # -----------------------------------------------------------------------
    # 健康探活 —— 由 launcher 在发题前轮询
    # -----------------------------------------------------------------------

    def health_ok(self, health_path: str = "/health") -> bool:
        url = f"{self.base_url}{health_path if health_path.startswith('/') else '/' + health_path}"
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                return 200 <= response.status < 300
        except Exception:  # noqa: BLE001
            return False

    def wait_healthy(self, timeout: float = 120.0, interval: float = 2.0, health_path: str = "/health") -> bool:
        """轮询 /health 直到就绪或超时。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.health_ok(health_path):
                return True
            time.sleep(interval)
        return False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AgentClient":
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8100)
        base_url = config.get("base_url") or f"http://{host}:{port}"
        return cls(
            base_url=base_url,
            predict_path=config.get("predict_path", "/predict"),
            timeout=config.get("request_timeout", 60.0),
            max_workers=config.get("max_workers", 16),
            max_retries=config.get("max_retries", contract.MAX_RETRIES),
            per_sample_timeout=config.get("predict_timeout", contract.PER_SAMPLE_TIMEOUT_SECONDS),
        )

    def get_usage(self) -> LLMUsage:
        return self.usage

    def __repr__(self) -> str:
        return f"AgentClient(base_url='{self.base_url}')"
