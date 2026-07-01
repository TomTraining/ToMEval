"""
Agent 生命周期管理 + 模型代理(效率记账)。

职责:
1. 起模型代理(ModelProxy):agent 拿到的 LLM_API_URL 指向本代理,不是真实后端
   (vLLM / 外部 API)。代理转发请求、记 token/调用数、抹掉 usage 后回给 agent。
2. 起 agent 进程:把 agent 环境变量里的占位符替换成**代理**的连接信息
   (LLM_API_URL → 代理地址;LLM_API_KEY 为占位 key;LLM_MODEL → 统一 model),
   并指定 agent 监听端口 PORT。
3. 轮询 agent /health 直到就绪,才允许发题。
4. 评完关掉 agent 进程、停掉代理;代理累计计数即当前 agent 的全部消耗
   (agent 只能经代理访问模型,伪造不了)。

效率只做记录、不进排名(见 docs/agent_eval_plan.md)。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# agent 侧读取模型连接信息的环境变量名(占位符);可由 agent_config.env_names 覆盖。
_DEFAULT_ENV_NAMES = {
    "api_url": "LLM_API_URL",
    "api_key": "LLM_API_KEY",
    "model": "LLM_MODEL",
    "port": "PORT",
}

# 代理不校验 key(真实 key 只在代理→后端那段注入),给 agent 一个占位 key 即可。
_PROXY_PLACEHOLDER_KEY = "eval-proxy-key"


def _substitute_env(
    agent_config: Dict[str, Any],
    proxy_base_url: str,
    model_name: str,
    port: int,
) -> Dict[str, str]:
    """构造 agent 进程的环境变量:占位符 → 代理连接信息。

    关键:LLM_API_URL 指向**代理**,不是真实后端。agent 只能经代理访问模型,
    因此偷换不了模型、也看不到 token 账。额外允许 agent_config.env 透传自定义变量。
    """
    env = dict(os.environ)
    env_names = {**_DEFAULT_ENV_NAMES, **(agent_config.get("env_names") or {})}
    env[env_names["api_url"]] = proxy_base_url
    env[env_names["api_key"]] = _PROXY_PLACEHOLDER_KEY
    env[env_names["model"]] = str(model_name)
    env[env_names["port"]] = str(port)
    for key, value in (agent_config.get("env") or {}).items():
        env[str(key)] = str(value)
    return env


@contextmanager
def launch_agent(agent_config: Dict[str, Any], llm_config: Dict[str, Any]):
    """上下文管理器:起代理 → 起 agent → 等 /health → yield (client, proxy) → 退出时清理。

    llm_config:真实后端连接(api_url/api_key/model_name)—— 只有代理知道,不下发给 agent。

    agent_config 关键字段:
      start_command : list[str] 或 str,启动 agent HTTP 服务的命令(缺省则假定 agent
                      已在外部起好、仅探活;此时占位符替换不生效,需 agent 自己连代理)。
      cwd           : 启动命令工作目录(可选)。
      host/port     : agent 监听地址(默认 127.0.0.1:8100)。
      health_path   : 健康检查路径(默认 /health)。
      health_timeout: 等待就绪最长秒数(默认 120)。
      max_concurrency: 代理并发上限(默认取 llm.max_workers 或 16)。
      predict_path/predict_timeout/max_retry/max_workers: 透传给 AgentClient。
    """
    from src.llm import AgentClient

    from .model_proxy import ModelProxy

    host = agent_config.get("host", "127.0.0.1")
    port = int(agent_config.get("port", 8100))
    client = AgentClient.from_config({**agent_config, "host": host, "port": port})

    # 代理:统一模型访问入口 + 记账。端口自动分配(port=0)。
    max_concurrency = int(
        agent_config.get("max_concurrency", llm_config.get("max_workers", 16))
    )
    proxy = ModelProxy(
        backend_url=str(llm_config.get("api_url", "")),
        backend_key=str(llm_config.get("api_key", "")),
        model_name=str(llm_config.get("model_name", "")),
        max_concurrency=max_concurrency,
        request_timeout=float(agent_config.get("predict_timeout", 1200.0)),
    )

    process: Optional[subprocess.Popen] = None
    start_command = agent_config.get("start_command")
    try:
        proxy_base_url = proxy.start()

        if start_command:
            env = _substitute_env(agent_config, proxy_base_url, proxy.model_name, port)
            shell = isinstance(start_command, str)
            logger.info(f"[agent] 启动命令: {start_command}")
            process = subprocess.Popen(
                start_command,
                shell=shell,
                cwd=agent_config.get("cwd"),
                env=env,
                start_new_session=True,  # 独立进程组,便于整组回收
            )
        else:
            logger.info(
                "[agent] 未配置 start_command,假定 agent 已在外部启动,仅做健康探活。"
                f"(agent 需自行把模型请求打到代理 {proxy_base_url})"
            )

        health_timeout = float(agent_config.get("health_timeout", 120.0))
        health_path = agent_config.get("health_path", "/health")
        logger.info(f"[agent] 等待 {client.base_url}{health_path} 就绪(≤{health_timeout:.0f}s)...")
        if not client.wait_healthy(timeout=health_timeout, health_path=health_path):
            raise RuntimeError(f"agent 在 {health_timeout:.0f}s 内未就绪: {client.base_url}{health_path}")
        logger.info("[agent] 就绪,开始评测。")

        yield client, proxy
    finally:
        if process is not None and process.poll() is None:
            logger.info("[agent] 关闭 agent 进程...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        proxy.stop()
