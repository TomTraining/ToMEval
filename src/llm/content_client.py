"""
Content Client - 负责文本生成
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from tqdm import tqdm

from .client import LLMClient, LLMResponse
from .llm_utils import build_extra_body

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class ContentClient(LLMClient):
    """
    负责文本生成的 LLM 客户端

    提供:
    - generate(): 单次文本生成
    - batch_generate(): 批量并行文本生成
    """

    def generate(
        self,
        prompt: str,
        max_retry: int = 5,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text from a single prompt with retry.

        Args:
            prompt: User prompt content
            max_retry: Maximum retry attempts
            system_prompt: 本次调用使用的 system prompt；为 None 时回退到 self.system_prompt
                （协议评测下按样本传不同的 system prompt）。

        Returns:
            LLMResponse with generated content and reasoning
        """
        extra_body = build_extra_body(self.top_k, self.enable_thinking)

        # 构建消息列表，如果配置了 system_prompt 则添加到开头
        effective_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        messages = []
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retry):
            try:
                start = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    messages=messages,
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
                text = message.content or ""
                reasoning = self._extract_reasoning(message)
                # thinking 模式下，把内联在 content 里的 <think> 思考剥离为 reasoning
                if self.enable_thinking:
                    text, reasoning = self._split_think_content(text, reasoning)
                if text:
                    self._track_usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency=latency,
                        success=True,
                    )
                    return LLMResponse(content=text, reasoning=reasoning)

            except Exception as e:
                logging.warning(f"[ContentClient] generate attempt {attempt + 1} failed: {e}")

        logging.error(f"[ContentClient] generate all {max_retry} attempts exhausted, returning None.")
        self._track_usage(success=False)
        return LLMResponse(content=None, reasoning="")

    def batch_generate(
        self,
        prompts: List[str],
        desc: str = "Generating",
        system_prompts: Optional[List[str]] = None,
    ) -> List[LLMResponse]:
        """Generate text from multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            desc: Description for progress bar
            system_prompts: 与 prompts 等长、一一对应的 system prompt 列表；为 None 时
                所有调用回退到 self.system_prompt（协议评测下按样本传不同 system prompt）。

        Returns:
            List of LLMResponse
        """
        if system_prompts is not None and len(system_prompts) != len(prompts):
            raise ValueError(
                f"system_prompts length ({len(system_prompts)}) must match prompts length ({len(prompts)})"
            )

        with ThreadPoolExecutor(self.max_workers) as executor:
            if system_prompts is None:
                futures = [executor.submit(self.generate, p) for p in prompts]
            else:
                futures = [
                    executor.submit(self.generate, p, system_prompt=sp)
                    for p, sp in zip(prompts, system_prompts)
                ]

            results = []
            for future in tqdm(
                futures,
                total=len(futures),
                desc=desc,
                miniters=100,
            ):
                results.append(future.result())

            return results