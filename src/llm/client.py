"""
LLM 客户端基类

LLMClient: 基类，包含通用功能
- ContentClient: 负责文本生成
- StructureClient: 负责结构化对象生成
"""

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import openai

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LLMUsage:
    """累计 Token 使用统计和延迟。

    Attributes:
        total_prompt_tokens: 累计 prompt token 数
        total_completion_tokens: 累计 completion token 数
        total_tokens: 累计总 token 数
        total_latency: 累计延迟（秒）
        total_calls: 累计调用次数
        successful_calls: 成功调用次数
        failed_calls: 失败调用次数
    """
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0


@dataclass
class LLMResponse:
    """LLM 输出响应，包含内容和推理过程。

    Attributes:
        content: 生成的内容，可能是字符串 (文本生成) 或 Pydantic 模型 (结构化输出)
        reasoning: 推理过程/思考链内容
    """
    content: Any
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Base LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    LLM 客户端基类

    提供通用功能：
    - 配置管理
    - OpenAI 客户端延迟初始化
    - 使用统计（线程安全）
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_url: str,
        temperature: float = 0.6,
        max_tokens: int = 32768,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 2,
        enable_thinking: bool = False,
        max_workers: int = 32,
        system_prompt: str = "",
    ):
        """Initialize LLM client.

        Args:
            model_name: Model identifier
            api_key: API authentication key
            api_url: API base URL
            temperature: Sampling temperature (default: 0.6)
            max_tokens: Maximum completion tokens (default: 32768)
            top_p: Nucleus sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 20)
            presence_penalty: Presence penalty (default: 2)
            enable_thinking: Enable thinking mode (default: True)
            max_workers: Max threads for batch operations (default: 32)
            system_prompt: System prompt to use in all messages (default: "")
        """
        # 统一配置日志（抑制第三方库的冗余日志）
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        # 保留 OpenAI SDK 的重试日志，方便诊断问题

        self.model = model_name
        self.api_key = api_key
        self.api_url = api_url

        # 延迟初始化客户端
        self._client = None

        # Generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt

        # Batch processing
        self.max_workers = max_workers

        # Usage tracking (thread-safe)
        self.usage: LLMUsage = LLMUsage()
        self._lock = threading.Lock()

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                timeout=1200.0,  # 设置超时时间
            )
        return self._client

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """从配置字典创建客户端实例"""
        return cls(
            model_name=config["model_name"],
            api_key=config["api_key"],
            api_url=config["api_url"],
            temperature=config.get("temperature", 0.6),
            max_tokens=config.get("max_tokens", 32768),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 20),
            presence_penalty=config.get("presence_penalty", 2),
            enable_thinking=config.get("enable_thinking", True),
            max_workers=config.get("max_workers", 32),
            system_prompt=config.get("system_prompt", ""),
        )

    # -----------------------------------------------------------------------
    # Usage Tracking (Thread-Safe)
    # -----------------------------------------------------------------------

    def _track_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency: float = 0.0,
        success: bool = True,
    ) -> None:
        """Update global usage statistics in a thread-safe manner."""
        with self._lock:
            self.usage.total_prompt_tokens += prompt_tokens
            self.usage.total_completion_tokens += completion_tokens
            self.usage.total_tokens += total_tokens
            self.usage.total_latency += latency
            self.usage.total_calls += 1
            if success:
                self.usage.successful_calls += 1
            else:
                self.usage.failed_calls += 1

    def get_usage(self) -> LLMUsage:
        """获取使用统计信息"""
        with self._lock:
            return self.usage

    def reset_usage(self) -> None:
        """重置使用统计"""
        with self._lock:
            self.usage = LLMUsage()

    @staticmethod
    def _extract_reasoning(message) -> str:
        """兼容新旧两种 reasoning 字段：reasoning（新）/ reasoning_content（旧，如 deepseek-reasoner）"""
        return (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
            or ""
        )

    @staticmethod
    def _split_think_content(content: Any, reasoning: str = "") -> Tuple[Any, str]:
        """把 content 里内联的 <think> 思考内容剥离为 reasoning，其余作为真正 content。

        部分模型在 enable_thinking 时不走独立的 reasoning 字段，而是把思考过程
        以 <think>...</think> 内联在 content 里。这里统一抽出来，兼容三种形态：
          - 成对 <think>...</think>
          - 仅 </think>（开头的 <think> 被 chat template 省略，常见于 Qwen 系列）
          - 仅 <think>（被 max_tokens 截断，没有闭合标签）

        已有的 reasoning（模型单独返回的）保留在前，内联抽取的追加其后。
        content 不是字符串（如结构化对象）时原样返回。
        """
        if not isinstance(content, str) or ("<think>" not in content and "</think>" not in content):
            return content, reasoning

        think_parts = []

        def _collect(match: "re.Match") -> str:
            think_parts.append(match.group(1))
            return ""

        # 1) 成对 <think>...</think>
        text = re.sub(r"<think>(.*?)</think>", _collect, content, flags=re.DOTALL)
        # 2) 残留单独 </think>：其前为思考，其后为正文
        if "</think>" in text:
            head, _, tail = text.partition("</think>")
            think_parts.append(head)
            text = tail
        # 3) 残留单独 <think>（未闭合/被截断）：其后全部视为思考
        if "<think>" in text:
            head, _, tail = text.partition("<think>")
            think_parts.append(tail)
            text = head

        inline = "\n".join(p.strip() for p in think_parts if p and p.strip())
        merged = "\n".join(x for x in [(reasoning or "").strip(), inline] if x)
        return text.strip(), merged

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"