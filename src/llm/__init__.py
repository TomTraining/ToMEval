"""
LLM 模块

统一的 LLM 调用接口，只支持两个核心功能：
- 文本生成: generate(), batch_generate()
- 结构化输出: generate_structure(), batch_generate_structure()
"""

from .client import LLMClient, LLMUsage, LLMResponse
from .content_client import ContentClient
from .structure_client import StructureClient

__all__ = [
    "LLMClient",
    "LLMUsage",
    "LLMResponse",
    "ContentClient",
    "StructureClient",
]
