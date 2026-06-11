"""按数据集动态加载自定义 prompt 逻辑。

与 metrics.load_task_metric_fn 同范式:约定每个数据集可选地在
`tasks/<dataset>/prompt.py` 里提供两个钩子,缺省则回退到通用实现:

- build_prompt(sample, option_map, include_instruction=True) -> str
    复现该数据集原论文的 body prompt(story/question/options 排版 + 措辞)。
    签名与 src.evaluation.prompts.build_prompt 完全一致;答题"格式提示"仍由
    system_prompt_for 经 system prompt 注入,这里只负责正文。

- prepare_samples(samples) -> samples
    在预测前对样本做整体变换(如 EmoBench 把 emotion+cause 两行并成一条
    多问 grouped 样本)。不提供则样本原样进入预测。
"""

from __future__ import annotations

import importlib
from typing import Callable, Optional


def _load_task_attr(dataset_name: str, attr: str) -> Optional[Callable]:
    try:
        module = importlib.import_module(f"tasks.{dataset_name}.prompt")
    except ModuleNotFoundError as exc:
        # 仅当“prompt 模块本身不存在”时回退默认;模块内部 import 失败必须暴露,
        # 避免把真实的依赖错误误判成“无自定义 prompt”。
        missing = exc.name or ""
        if missing in (f"tasks.{dataset_name}.prompt", f"tasks.{dataset_name}"):
            return None
        raise
    return getattr(module, attr, None)


def load_task_prompt_builder(dataset_name: str) -> Optional[Callable]:
    """返回数据集自定义的 build_prompt;无则 None(调用方用 prompts.build_prompt)。"""
    return _load_task_attr(dataset_name, "build_prompt")


def load_task_prepare_samples(dataset_name: str) -> Optional[Callable]:
    """返回数据集自定义的 prepare_samples(samples)->samples;无则 None(不做预处理)。"""
    return _load_task_attr(dataset_name, "prepare_samples")


def load_task_system_prompt_builder(dataset_name: str) -> Optional[Callable]:
    """返回数据集自定义的 build_system_prompt(sample, protocol, lang, prompt_type)->str。

    用于忠实复刻官方 system prompt(ToMBench/EmoBench/BigToM 等),只把答案格式换成 \\boxed。
    无则 None,调用方回退到通用 protocols.system_prompt_for。
    """
    return _load_task_attr(dataset_name, "build_system_prompt")
