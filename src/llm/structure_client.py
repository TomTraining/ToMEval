"""
Structure Client - 负责结构化对象生成
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Type

from pydantic import BaseModel
from tqdm import tqdm


class _ProbeSchema(BaseModel):
    ok: bool

from .client import LLMClient, LLMResponse
from .llm_utils import build_extra_body, extract_json, format_schema_for_prompt

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class StructureClient(LLMClient):
    """
    负责结构化对象生成的 LLM 客户端

    提供:
    - generate_structure(): 单次结构化输出
    - batch_generate_structure(): 批量并行结构化输出

    支持两种模式：
    1. parse 模式：chat.completions.parse() - 直接返回 Pydantic 对象
    2. create 模式：chat.completions.create() + json_object response_format + prompt 引导
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
        enable_thinking: bool = True,
        max_workers: int = 32,
        system_prompt: str = "",
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_url=api_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            enable_thinking=enable_thinking,
            max_workers=max_workers,
            system_prompt=system_prompt,
        )
        self._use_parse_mode: bool = None  # None: 未检测, True: 用 parse, False: 用 create

    def _generate_with_parse(
        self,
        prompt: str,
        response_object: Type[BaseModel],
        max_retry: int = 5,
        timeout: int = 1200,
    ) -> LLMResponse:
        """使用 parse API 的原生结构化输出。"""
        extra_body = build_extra_body(self.top_k, self.enable_thinking)

        # 构建消息列表，如果配置了 system_prompt 则添加到开头
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retry):
            try:
                start = time.time()
                response = self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    presence_penalty=self.presence_penalty,
                    response_format=response_object,
                    extra_body=extra_body,
                    timeout=timeout,
                )
                latency = time.time() - start
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens

                message = response.choices[0].message
                result = message.parsed
                reasoning = self._extract_reasoning(message)
                self._track_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency=latency,
                    success=True,
                )
                return LLMResponse(content=result, reasoning=reasoning)

            except Exception as e:
                logging.warning(f"[StructureClient] parse mode attempt {attempt + 1}/{max_retry} failed: {type(e).__name__}: {e}")

        logging.error(f"[StructureClient] parse mode all {max_retry} attempts exhausted")
        self._track_usage(success=False)
        return LLMResponse(content=None, reasoning=None)

    def _generate_with_create(
        self,
        prompt: str,
        response_object: Type[BaseModel],
        max_retry: int = 5,
    ) -> LLMResponse:
        """降级模式：使用 json_object response_format + prompt 引导 + 解析验证。"""
        import json

        # 构建 schema 描述
        schema_desc = format_schema_for_prompt(response_object)
        # 增强提示词
        enhanced_prompt = f"""{prompt}

---
Response Format:
{schema_desc}

IMPORTANT: Respond with a valid JSON object that matches the schema above.
Output ONLY the JSON object, no additional text or markdown formatting."""

        extra_body = build_extra_body(self.top_k, self.enable_thinking)

        # 构建消息列表，如果配置了 system_prompt 则添加到开头
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": enhanced_prompt})

        for attempt in range(max_retry):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    presence_penalty=self.presence_penalty,
                    extra_body=extra_body,
                    timeout=1200,
                )

                latency = time.time() - start
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens

                message = response.choices[0].message
                content = message.content or ""
                reasoning = self._extract_reasoning(message)
                # thinking 模式下，先把内联 <think> 思考剥离为 reasoning，
                # 同时避免思考文本里的花括号污染后续 JSON 提取
                if self.enable_thinking:
                    content, reasoning = self._split_think_content(content, reasoning)
                # 提取 JSON
                json_data = extract_json(content)
                if json_data is None:
                    raise ValueError(f"Failed to extract valid JSON")

                # 用 Pydantic 验证（不符合就重试）
                result = response_object.model_validate(json_data)
                self._track_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency=latency,
                    success=True,
                )
                return LLMResponse(content=result, reasoning=reasoning)

            except Exception as e:
                logging.warning(f"[StructureClient] create mode attempt {attempt + 1}/{max_retry} failed: {type(e).__name__}")

        logging.error(f"[StructureClient] create mode all {max_retry} attempts exhausted")
        self._track_usage(success=False)
        return LLMResponse(content=None, reasoning=None)

    def generate_structure(
        self,
        prompt: str,
        response_object: Type[BaseModel],
        max_retry: int = 5,
        mode: str = "auto",
    ) -> LLMResponse:
        """调用 LLM，返回 Pydantic 对象（自动适配不同模型）。

        Args:
            prompt: 提示词
            response_object: 必须提供的 Pydantic 模型类
            max_retry: 最大重试次数

        Returns:
            LLMResponse，其中 content 是 response_object 的实例，失败时返回空实例
        """
        if mode == "create":
            return self._generate_with_create(prompt, response_object, max_retry)
        if mode == "parse":
            return self._generate_with_parse(prompt, response_object, max_retry)

        if self._use_parse_mode is None:
            # 未检测时，默认尝试 parse
            return self._generate_with_parse(prompt, response_object, max_retry)
        elif self._use_parse_mode:
            return self._generate_with_parse(prompt, response_object, max_retry)
        else:
            return self._generate_with_create(prompt, response_object, max_retry)

    def batch_generate_structure(
        self,
        prompts: List[str],
        response_object: Type[BaseModel],
        mode: str = "auto",
        desc: str = "Judging",
    ) -> List[Any]:
        """
        批量调用 LLM，返回 Pydantic 对象列表。

        先用第一个 prompt 测试 parse 是否可用，然后批量处理。

        Args:
            prompts: 提示词列表
            response_object: 必须提供的 Pydantic 模型类
            mode: 模式选择 ("auto", "parse", "create")
            desc: 进度条描述

        Returns:
            LLMResponse 列表
        """
        if not prompts:
            return []

        # 仅在本 session 第一次调用时探测 parse 模式
        if mode == "auto" and self._use_parse_mode is None:
            logging.info("[StructureClient] Testing parse mode with first prompt...")
            probe_prompt = 'Reply with {"ok": true}.'
            probe_result = self._generate_with_parse(probe_prompt, _ProbeSchema, max_retry=1, timeout=1200)
            if probe_result.content is None:
                self._use_parse_mode = False
                logging.warning("[StructureClient] Parse mode failed, switching to create mode for all requests")
            else:
                self._use_parse_mode = True
                logging.info("[StructureClient] Parse mode works, using it for all requests")
        results = []
        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = [
                executor.submit(self.generate_structure, p, response_object, 5, mode)
                for p in prompts
            ]

            for future in tqdm(
                futures,
                total=len(futures),
                desc=desc,
            ):
                results.append(future.result())
        return results
