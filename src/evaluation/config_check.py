"""实验配置关键采样参数的标准一致性校验。

标准测试对 llm / judge 的四个关键采样参数（temperature、enable_thinking、
system_prompt、max_tokens）有**固定的标准值**。无论用户用哪个 --experiment-config
（哪怕是默认的 experiment_config.yaml 本身），只要这些参数偏离固定标准值，就先
交互询问用户是否确认使用与标准测试不同的参数，避免无意间用了非标准设置跑评测。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# 标准测试固定的关键采样参数（偏离这些值即视为非标准，需用户确认）。
# 取自标准测试设定（experiment_config.yaml 的基准值）。
STANDARD_PARAMS: Dict[str, Dict[str, Any]] = {
    "llm": {
        "temperature": 0.6,
        "enable_thinking": True,
        "system_prompt": "",
        "max_tokens": 32768,
    },
    "judge": {
        "temperature": 0.0,
        "enable_thinking": False,
        "system_prompt": "",
        "max_tokens": 32768,
    },
}

# 用户配置缺省某字段时的“生效默认值”，与 LLMClient.from_config 的 .get 默认保持一致，
# 这样比对的是实际生效值，而非仅 raw yaml 里写没写。
_EFFECTIVE_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.6,
    "enable_thinking": True,
    "system_prompt": "",
    "max_tokens": 32768,
}

# 子进程跳过重复询问的环境变量（run_all 批量时父进程确认一次即可）
ACK_ENV = "TOMEVAL_CONFIG_DIFF_ACK"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _effective(section: Dict[str, Any], key: str) -> Any:
    if not isinstance(section, dict):
        section = {}
    return section.get(key, _EFFECTIVE_DEFAULTS[key])


def diff_against_standard(user_config_path: str) -> List[Tuple[str, str, Any, Any]]:
    """返回 [(section, key, 标准值, 用户值)] 差异列表。

    对用户配置 llm / judge 两个 section 的四个关键参数，按生效值与固定标准值比对。
    配置文件不存在时返回空列表。
    """
    user_path = Path(user_config_path)
    if not user_path.exists():
        return []

    user = _load_yaml(user_path)

    diffs: List[Tuple[str, str, Any, Any]] = []
    for section, std_params in STANDARD_PARAMS.items():
        usr_sec = user.get(section, {})
        for key, std_val in std_params.items():
            usr_val = _effective(usr_sec, key)
            if std_val != usr_val:
                diffs.append((section, key, std_val, usr_val))
    return diffs


def _fmt(value: Any) -> str:
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def confirm_config_against_standard(user_config_path: str, assume_yes: bool = False) -> None:
    """若关键采样参数偏离固定标准值，交互询问是否继续。

    - 已通过 assume_yes 或环境变量 ACK 确认过 → 直接放行
    - 无差异 → 直接放行
    - 有差异：
        * 交互终端 → 打印差异并 input 询问，回答非 y 则中止
        * 非交互（无 tty）且未显式确认 → 中止并提示用 --allow-config-diff
      确认通过后写入环境变量，供 run_all 的子进程跳过重复询问。
    """
    if assume_yes or os.environ.get(ACK_ENV) == "1":
        return

    diffs = diff_against_standard(user_config_path)
    if not diffs:
        return

    lines = [
        "",
        "⚠️  以下关键参数与标准测试的固定值不一致：",
        f"    配置: {Path(user_config_path).resolve()}",
        "",
        f"    {'参数':<26}{'标准值':<28}{'你的值'}",
    ]
    for section, key, std_val, usr_val in diffs:
        name = f"{section}.{key}"
        lines.append(f"    {name:<24}{_fmt(std_val):<28}{_fmt(usr_val)}")
    lines.append("")
    print("\n".join(lines), file=sys.stderr)

    if not sys.stdin or not sys.stdin.isatty():
        print(
            "非交互环境下检测到与标准测试不同的参数，已中止。\n"
            "如确认要使用这些非标准参数，请加 --allow-config-diff 重新运行。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    answer = input("是否使用与标准测试不同的参数继续？[y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("已取消。请将上述参数改回标准值，或确认后重试。", file=sys.stderr)
        raise SystemExit(2)

    # 确认通过：标记环境变量，供 run_all 的子进程跳过重复询问
    os.environ[ACK_ENV] = "1"


__all__ = [
    "STANDARD_PARAMS",
    "ACK_ENV",
    "diff_against_standard",
    "confirm_config_against_standard",
]
