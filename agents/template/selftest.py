"""
predict 自测工具 —— 提交前在本地验证你的 predict 是否产出**要求的严格格式**。

它做三件事:
  1. 起一个模拟 model 服务(假 OpenAI 接口,见 mock_model_server.py),或用你指定的真实模型;
  2. 用 mock_samples.json 里的四种题型样本逐条调用你的 predict(sample, model);
  3. 按题型严格校验返回值,打印每条 PASS/FAIL 与汇总。

用法:
    # 默认:用内置模拟 model(无需真实模型/网络),最省事
    python selftest.py

    # 用真实模型(例如你自己 vLLM 部署的模型):设三个环境变量即可
    MODEL_API_URL=http://127.0.0.1:8000/v1 MODEL_API_KEY=x MODEL_NAME=qwen3-8b \
        python selftest.py

全部 PASS 不代表答案正确(模拟 model 是随机的),只代表你的 predict **格式合规、能跑通**。
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

# 参赛方实现。selftest 与 solution.py 同目录,直接 import。
from solution import predict

_HERE = Path(__file__).resolve().parent
_LETTER_RE = re.compile(r"^[A-Z]$")


# ---------------------------------------------------------------------------
# 严格格式校验(与提交后线上校验规则一致)
# ---------------------------------------------------------------------------

def validate(prompt_type: str, prediction: Any, options_letters: List[str]) -> Tuple[bool, str]:
    """返回 (ok, reason)。ok=False 时 reason 说明为何违约。"""
    if prompt_type == "mcq_single":
        if not (isinstance(prediction, str) and _LETTER_RE.match(prediction)):
            return False, 'mcq_single 只接受单个大写字母字符串,如 "A"'
        if options_letters and prediction not in options_letters:
            return False, f"字母 {prediction!r} 不在本题选项 {options_letters} 内"
        return True, ""

    if prompt_type == "mcq_multi":
        if not isinstance(prediction, list):
            return False, 'mcq_multi 只接受大写字母数组,如 ["A","C"](禁止字符串 "A,C")'
        if not prediction:
            return False, "mcq_multi 至少含一个字母"
        if any(not (isinstance(x, str) and _LETTER_RE.match(x)) for x in prediction):
            return False, "mcq_multi 每个元素须为单个大写字母"
        if len(set(prediction)) != len(prediction):
            return False, "mcq_multi 不允许重复字母"
        if prediction != sorted(prediction):
            return False, "mcq_multi 须升序排列"
        if options_letters and any(x not in options_letters for x in prediction):
            return False, f"含选项外字母(本题选项 {options_letters})"
        return True, ""

    if prompt_type == "mcq_grouped":
        if not isinstance(prediction, list) or not prediction:
            return False, "mcq_grouped 只接受非空大写字母数组,顺序对应各子问"
        if any(not (isinstance(x, str) and _LETTER_RE.match(x)) for x in prediction):
            return False, "mcq_grouped 每个元素须为单个大写字母"
        return True, ""

    if prompt_type == "open":
        if isinstance(prediction, str) and prediction.strip():
            return True, ""
        return False, "open 只接受非空字符串"

    return False, f"未知 prompt_type: {prompt_type!r}"


def _expected_len_hint(sample: dict) -> Optional[int]:
    """mcq_grouped 期望的答案长度(子问数),用于额外提示(非硬性校验)。"""
    if sample.get("prompt_type") == "mcq_grouped":
        return len(sample.get("sub_questions") or [])
    return None


# ---------------------------------------------------------------------------
# 模拟 model:后台线程起一个假 OpenAI 接口(除非指定了真实模型)
# ---------------------------------------------------------------------------

def _start_mock_model() -> Tuple[str, Any]:
    """在后台线程起 mock_model_server,返回 (api_url, server)。"""
    import mock_model_server

    # 端口 0 让 OS 分配空闲端口,避免冲突。
    server = mock_model_server.serve(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}/v1", server


def _resolve_model() -> Tuple[dict, Optional[Any]]:
    """决定用哪个 model:真实(环境变量)或内置模拟。返回 (model_dict, mock_server_or_None)。"""
    url = os.environ.get("MODEL_API_URL")
    if url:
        model = {
            "api_url": url,
            "api_key": os.environ.get("MODEL_API_KEY", "x"),
            "model_name": os.environ.get("MODEL_NAME", "model"),
        }
        print(f"[selftest] 使用真实模型: {url} (model={model['model_name']})")
        return model, None

    api_url, server = _start_mock_model()
    print(f"[selftest] 使用内置模拟 model: {api_url}(随机合法答案,仅验证格式与链路)")
    return {"api_url": api_url, "api_key": "mock", "model_name": "mock"}, server


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    samples = json.loads((_HERE / "mock_samples.json").read_text(encoding="utf-8"))
    model, mock_server = _resolve_model()

    passed = 0
    failed = 0
    print("\n" + "=" * 70)
    for i, sample in enumerate(samples, 1):
        ptype = sample.get("prompt_type", "?")
        sid = sample.get("sample_id", f"#{i}")
        options_letters = list((sample.get("options") or {}).keys())

        try:
            prediction = predict(sample, model)
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {sid} ({ptype}): predict 抛异常: {error!r}")
            continue

        ok, reason = validate(ptype, prediction, options_letters)
        hint = ""
        exp_len = _expected_len_hint(sample)
        if ok and exp_len is not None and isinstance(prediction, list) and len(prediction) != exp_len:
            # 长度不符不判 FAIL(实际子问数以线上题目为准),但给出提示。
            hint = f"  (提示:mcq_grouped 期望 {exp_len} 个字母,当前 {len(prediction)} 个)"

        if ok:
            passed += 1
            print(f"[PASS] {sid} ({ptype}): {json.dumps(prediction, ensure_ascii=False)}{hint}")
        else:
            failed += 1
            print(f"[FAIL] {sid} ({ptype}): {json.dumps(prediction, ensure_ascii=False)}\n        → {reason}")

    print("=" * 70)
    print(f"结果: {passed} PASS / {failed} FAIL  (共 {len(samples)} 条)")
    if failed == 0:
        print("✓ 格式全部合规。注意:模拟 model 答案是随机的,PASS 只代表格式与链路 OK,不代表答对。")
    else:
        print("✗ 有样本未通过,请按上面的 → 说明修正 predict 的返回格式。")

    if mock_server is not None:
        mock_server.shutdown()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
