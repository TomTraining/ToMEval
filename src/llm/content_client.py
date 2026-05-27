"""
Content Client - 负责文本生成
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

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
    ) -> LLMResponse:
        """Generate text from a single prompt with retry.

        Args:
            prompt: User prompt content
            max_retry: Maximum retry attempts

        Returns:
            LLMResponse with generated content and reasoning
        """
        extra_body = build_extra_body(self.top_k, self.enable_thinking)

        # 构建消息列表，如果配置了 system_prompt 则添加到开头
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
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
    ) -> List[LLMResponse]:
        """Generate text from multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            desc: Description for progress bar

        Returns:
            List of LLMResponse，顺序与输入 prompts 严格一致。
        """
        n = len(prompts)
        if n == 0:
            return []

        # 按 prompt 长度倒序提交：长样本先进入 vLLM prefill，降低尾延迟与队列空转。
        # 调度顺序不改变单条请求的返回内容，最终结果按原始 index 写回，对调用方完全透明。
        submit_order = sorted(range(n), key=lambda i: len(prompts[i]), reverse=True)

        results: List[LLMResponse] = [None] * n  # type: ignore[list-item]
        with ThreadPoolExecutor(self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate, prompts[i]): i
                for i in submit_order
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=n,
                desc=desc,
                miniters=100,
            ):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results