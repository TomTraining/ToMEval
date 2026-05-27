"""
LLM 工具函数

提取自 LLMClient 的通用工具函数。
"""

import json
import re
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取 JSON。

    尝试多种方式提取 JSON：
    1. 直接解析整个文本
    2. 提取 ```json ... ``` 代码块
    3. 提取第一个 {} 大括号内容

    Args:
        text: 包含 JSON 的文本

    Returns:
        解析后的 JSON 字典，失败返回 None
    """
    text = text.strip()

    # 尝试 1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试 2: 提取 ```json ... ``` 代码块
    matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # 尝试 3: 提取 {} 大括号内容
    matches = re.findall(r'\{[\s\S]*?\}', text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def format_schema_for_prompt(response_object: Type[BaseModel]) -> str:
    """将 Pydantic Schema 格式化为可读的描述。

    Args:
        response_object: Pydantic 模型类

    Returns:
        格式化后的 Schema 描述
    """
    schema_dict = response_object.model_json_schema()
    properties = schema_dict.get("properties", {})
    required = schema_dict.get("required", [])

    lines = ["Schema:"]
    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "any")
        is_required = field_name in required
        enum = field_schema.get("enum")

        if enum:
            type_desc = f"one of: {', '.join(repr(v) for v in enum)}"
        else:
            type_desc = field_type

        desc = field_schema.get("description", "")
        req_str = " (required)" if is_required else " (optional)"

        lines.append(f"  {field_name}: {type_desc}{req_str} - {desc}")

    return "\n".join(lines)


def build_extra_body(top_k: int, enable_thinking: bool) -> Dict[str, Any]:
    """构建 API 调用的 extra_body 参数。

    Args:
        top_k: Top-k 采样参数
        enable_thinking: 是否启用思考模式

    Returns:
        extra_body 字典
    """
    extra_body: Dict[str, Any] = {"top_k": top_k}
    if not enable_thinking:
        extra_body["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return extra_body
