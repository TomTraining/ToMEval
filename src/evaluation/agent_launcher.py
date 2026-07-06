"""
本地拉起统一 Agent 运行时(agents/server.py),供 agent 评测使用。

仅当 agent 配置里给了 `solution_dir`(参赛方仓库根目录,含 solution.py)时启用:
框架在评测前把 server.py 跑起来,注入 SOLUTION_DIR + PORT,轮询 /health 就绪后发题,
评完关掉。正式评测远程端点时**不配 solution_dir**,走 api_url 直连,本模块不参与。

两种运行时(由 agent.runtime 选择,缺省 local,向后兼容):
  · local  —— subprocess 直接起 `python agents/server.py`(轻量,依赖跑在评测机本机环境)。
  · docker —— 把参赛方仓库连同环境构建成镜像,再叠加我方 server.py 当 ENTRYPOINT 跑容器。
              环境彻底交给参赛方(交 Dockerfile 或 requirements.txt),HTTP 契约/并发/错误兜底
              仍由我方 server.py 负责。见 launch_docker_agent。

不论哪种运行时,server.py 的角色、契约、以及「model 凭证随请求体下发」都完全一致;本模块只
决定「怎么把 server 跑起来 / 收拾干净」,不碰契约。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 统一运行时相对仓库根的位置。
_SERVER_REL_PATH = "agents/server.py"
# docker 运行时:包裹层 Dockerfile(以参赛方镜像为底叠加 server.py)。
_RUNTIME_DOCKERFILE_REL = "agents/Dockerfile.runtime"
# 容器内 server 监听端口(与 Dockerfile.runtime 的 ENV PORT 一致)。
_CONTAINER_PORT = 8100
# 参赛方没交 Dockerfile 时兜底用的基础镜像。
_DEFAULT_BASE_IMAGE = "python:3.11-slim"


def _port_from_base_url(base_url: str, default: int = 8100) -> int:
    """从 agent.api_url 解析端口(本地自测时 server 监听此端口)。"""
    try:
        parsed = urlparse(base_url)
        if parsed.port:
            return int(parsed.port)
    except Exception:  # noqa: BLE001
        pass
    return default


def _repo_root() -> Path:
    """仓库根目录 = 本文件的上上级(src/evaluation/ → 仓库根)。"""
    return Path(__file__).resolve().parents[2]


@contextmanager
def launch_local_agent(agent_config: Dict[str, Any], client: Any):
    """按 agent.runtime 分流拉起统一运行时:local(subprocess)或 docker(容器)。

    保留历史函数名(pipeline 直接 import 它);内部据 runtime 选择实现。
    """
    runtime = str(agent_config.get("runtime", "local")).lower()
    if runtime == "docker":
        with launch_docker_agent(agent_config, client):
            yield
    elif runtime == "local":
        with _launch_subprocess_agent(agent_config, client):
            yield
    else:
        raise ValueError(f"未知 agent.runtime={runtime!r}(可选 local / docker)。")


# ---------------------------------------------------------------------------
# local 运行时:subprocess 直接起 server.py(原实现,保持不变)
# ---------------------------------------------------------------------------

@contextmanager
def _launch_subprocess_agent(agent_config: Dict[str, Any], client: Any):
    """local 运行时:起 agents/server.py → 等 /health → yield → 退出时清理。

    agent_config 关键字段:
      solution_dir : 参赛方仓库根目录(含 solution.py);必有,否则不该调用本函数。
      api_url      : 决定 server 监听端口(从 URL 解析,缺省 8100)。
      health_path  : 健康检查路径(默认 /health)。
      health_timeout: 等就绪最长秒数(默认 120)。
    client:已建好的 AgentClient,用其 wait_healthy 探活。
    """
    solution_dir = os.path.abspath(str(agent_config["solution_dir"]))
    if not os.path.isfile(os.path.join(solution_dir, "solution.py")):
        raise RuntimeError(f"solution_dir 下找不到 solution.py:{solution_dir}")

    server_path = _repo_root() / _SERVER_REL_PATH
    if not server_path.is_file():
        raise RuntimeError(f"找不到统一运行时:{server_path}")

    port = _port_from_base_url(agent_config.get("api_url", ""), default=8100)
    env = dict(os.environ)
    env["SOLUTION_DIR"] = solution_dir
    env["PORT"] = str(port)

    process: Optional[subprocess.Popen] = None
    try:
        logger.info(f"[agent] 本地拉起 {server_path} (SOLUTION_DIR={solution_dir}, PORT={port})")
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            env=env,
            start_new_session=True,  # 独立进程组,便于整组回收
        )

        health_timeout = float(agent_config.get("health_timeout", 120.0))
        health_path = agent_config.get("health_path", "/health")
        logger.info(f"[agent] 等待 {client.base_url}{health_path} 就绪(≤{health_timeout:.0f}s)...")
        if not client.wait_healthy(timeout=health_timeout, health_path=health_path):
            # 进程可能已因加载 solution.py 失败而退出;附带退出码便于排查。
            code = process.poll()
            raise RuntimeError(
                f"agent 在 {health_timeout:.0f}s 内未就绪: {client.base_url}{health_path}"
                + (f"(server 进程已退出,exit={code})" if code is not None else "")
            )
        logger.info("[agent] 就绪,开始评测。")

        yield
    finally:
        if process is not None and process.poll() is None:
            logger.info("[agent] 关闭 server 进程...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


# ---------------------------------------------------------------------------
# docker 运行时:构建镜像(参赛方环境 + 我方 server.py)→ 起容器 → 探活 → 清理
# ---------------------------------------------------------------------------

def _run(cmd: list, *, timeout: Optional[float] = None, capture: bool = False) -> subprocess.CompletedProcess:
    """跑一条 docker 子命令;capture=True 时把 stdout/stderr 收进结果(供报错时打印)。"""
    logger.info("[agent][docker] $ %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        timeout=timeout,
        text=True,
        capture_output=capture,
        check=False,
    )


def _ensure_docker_available() -> None:
    """docker 不可用时给出明确的排障提示(而不是让后续命令抛难懂的错)。"""
    try:
        proc = _run(["docker", "info"], timeout=30.0, capture=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "未找到 docker 命令。请先安装 Docker(如 colima:`brew install colima docker` 后 "
            "`colima start`,或安装 Docker Desktop)。"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("`docker info` 超时:docker 守护进程可能未就绪。") from error
    if proc.returncode != 0:
        raise RuntimeError(
            "docker 守护进程未运行。若用 colima 请先 `colima start`;"
            f"若用 Docker Desktop 请先启动它。\n原始错误:\n{(proc.stderr or '').strip()[-500:]}"
        )


def _default_dockerfile_text() -> str:
    """参赛方没交 Dockerfile 时的兜底:python:3.11-slim + 装 requirements.txt。

    与 agents/template/Dockerfile 等价(约定:仓库拷到 /agent,不写 ENTRYPOINT)。
    """
    return (
        f"FROM {_DEFAULT_BASE_IMAGE}\n"
        "WORKDIR /agent\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY . /agent\n"
    )


def _build_base_image(solution_dir: str, tag: str, build_timeout: float) -> None:
    """构建参赛方镜像(第一层)。有 Dockerfile 用之;否则用兜底 Dockerfile。

    兜底时把临时 Dockerfile 写到 solution_dir 内(构建结束即删),这样 COPY 上下文
    仍是参赛方仓库,requirements.txt 也在其中。
    """
    dockerfile_in_repo = os.path.join(solution_dir, "Dockerfile")
    if os.path.isfile(dockerfile_in_repo):
        logger.info("[agent][docker] 使用参赛方 Dockerfile 构建环境镜像。")
        proc = _run(
            ["docker", "build", "-t", tag, solution_dir],
            timeout=build_timeout,
            capture=True,
        )
        _check_build(proc, "参赛方 Dockerfile")
        return

    # 无 Dockerfile:要求有 requirements.txt,用兜底 Dockerfile。
    if not os.path.isfile(os.path.join(solution_dir, "requirements.txt")):
        raise RuntimeError(
            f"{solution_dir} 下既无 Dockerfile 也无 requirements.txt,无法构建镜像。"
            "参赛方须至少交其一(仅 pip 依赖时交 requirements.txt 即可)。"
        )
    logger.info("[agent][docker] 参赛方未交 Dockerfile,用默认基础镜像 + requirements.txt 兜底。")
    tmp_dockerfile = None
    try:
        # 写到仓库内的临时 Dockerfile(唯一名,避免与参赛方文件冲突)。
        fd, tmp_dockerfile = tempfile.mkstemp(
            prefix=".tomeval_runtime_", suffix=".Dockerfile", dir=solution_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_default_dockerfile_text())
        proc = _run(
            ["docker", "build", "-t", tag, "-f", tmp_dockerfile, solution_dir],
            timeout=build_timeout,
            capture=True,
        )
        _check_build(proc, "默认兜底 Dockerfile")
    finally:
        if tmp_dockerfile and os.path.isfile(tmp_dockerfile):
            os.remove(tmp_dockerfile)


def _build_runtime_image(base_tag: str, runtime_tag: str, build_timeout: float) -> None:
    """构建运行时镜像(第二层):以 base 为底叠加 agents/server.py。"""
    repo_root = _repo_root()
    dockerfile = repo_root / _RUNTIME_DOCKERFILE_REL
    if not dockerfile.is_file():
        raise RuntimeError(f"找不到运行时包裹层 Dockerfile:{dockerfile}")
    agents_dir = repo_root / "agents"  # 构建上下文:含 server.py
    proc = _run(
        [
            "docker", "build",
            "-t", runtime_tag,
            "--build-arg", f"BASE_IMAGE={base_tag}",
            "-f", str(dockerfile),
            str(agents_dir),
        ],
        timeout=build_timeout,
        capture=True,
    )
    _check_build(proc, "运行时包裹层")


def _check_build(proc: subprocess.CompletedProcess, what: str) -> None:
    """构建失败时抛错并附上 docker 输出尾部,便于定位参赛方镜像问题。"""
    if proc.returncode != 0:
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[-1500:]
        raise RuntimeError(f"[agent][docker] 构建失败({what}),退出码 {proc.returncode}:\n{tail}")


@contextmanager
def launch_docker_agent(agent_config: Dict[str, Any], client: Any):
    """docker 运行时:构建两层镜像 → 起容器 → 等 /health → yield → 退出时停并删容器。

    构建两层:
      1) 参赛方镜像:以 solution_dir 为上下文,用其 Dockerfile(或兜底 Dockerfile)构建;
         环境由参赛方决定,仓库(含 solution.py)拷进 /agent。
      2) 运行时镜像:以第一层为底,叠加 agents/server.py 当 ENTRYPOINT(agents/Dockerfile.runtime)。
    随后 docker run -d,把宿主端口(从 api_url 解析)映射到容器的 8100,并加资源/进程数上限。
    评完只删容器、留镜像(下次构建走缓存,更快)。

    agent_config 关键字段:
      solution_dir     : 参赛方仓库根目录(含 solution.py,可选 Dockerfile/requirements.txt)。
      api_url          : 决定宿主映射端口(从 URL 解析,缺省 8100)。
      health_path      : 健康检查路径(默认 /health)。
      health_timeout   : 等就绪最长秒数(默认 120)。
      build_timeout    : 单次 docker build 超时秒数(默认 600)。
      docker_memory    : 容器内存上限(默认 4g)。
      docker_cpus      : 容器 CPU 上限(默认 "2")。
      docker_pids_limit: 容器进程数上限(默认 512)。
      docker_run_as_root: True 时不加 --user(少数镜像必须 root 才能跑);默认非 root。
    """
    _ensure_docker_available()

    solution_dir = os.path.abspath(str(agent_config["solution_dir"]))
    if not os.path.isfile(os.path.join(solution_dir, "solution.py")):
        raise RuntimeError(f"solution_dir 下找不到 solution.py:{solution_dir}")

    host_port = _port_from_base_url(agent_config.get("api_url", ""), default=8100)
    build_timeout = float(agent_config.get("build_timeout", 600.0))
    # 镜像/容器名:用端口区分,避免并行评测多个 agent 撞名。
    base_tag = f"tomeval-agent-base:{host_port}"
    runtime_tag = f"tomeval-agent-runtime:{host_port}"
    container_name = f"tomeval-agent-{host_port}"

    # 起容器前先清掉可能残留的同名容器(上次异常退出没删干净)。
    _run(["docker", "rm", "-f", container_name], timeout=30.0, capture=True)

    logger.info("[agent][docker] 构建参赛方环境镜像 %s ...", base_tag)
    _build_base_image(solution_dir, base_tag, build_timeout)
    logger.info("[agent][docker] 叠加运行时 server.py → %s ...", runtime_tag)
    _build_runtime_image(base_tag, runtime_tag, build_timeout)

    run_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", f"{host_port}:{_CONTAINER_PORT}",
        # 资源/进程数上限:防单个参赛方镜像拖垮评测机(防滥用熔断的一环)。
        "--memory", str(agent_config.get("docker_memory", "4g")),
        "--cpus", str(agent_config.get("docker_cpus", "2")),
        "--pids-limit", str(agent_config.get("docker_pids_limit", 512)),
    ]
    if not agent_config.get("docker_run_as_root", False):
        # 非 root 运行,降低容器逃逸/写宿主的风险。
        run_cmd += ["--user", "65534:65534"]
    # 第一版:默认桥接网络(容器需连我方 model 后端)。
    # TODO(安全收紧):后续改为出口白名单,只放行 model 后端 host,堵死题目/数据外传。
    run_cmd.append(runtime_tag)

    started = False
    try:
        proc = _run(run_cmd, timeout=60.0, capture=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"[agent][docker] 启动容器失败,退出码 {proc.returncode}:\n"
                f"{(proc.stderr or '').strip()[-1000:]}"
            )
        started = True
        container_id = (proc.stdout or "").strip()[:12]
        logger.info("[agent][docker] 容器已起 %s(%s),映射 %s→%s",
                    container_name, container_id, host_port, _CONTAINER_PORT)

        health_timeout = float(agent_config.get("health_timeout", 120.0))
        health_path = agent_config.get("health_path", "/health")
        logger.info(f"[agent][docker] 等待 {client.base_url}{health_path} 就绪(≤{health_timeout:.0f}s)...")
        if not client.wait_healthy(timeout=health_timeout, health_path=health_path):
            logs = _container_logs_tail(container_name)
            raise RuntimeError(
                f"agent 容器在 {health_timeout:.0f}s 内未就绪: {client.base_url}{health_path}\n"
                f"容器日志尾部:\n{logs}"
            )
        logger.info("[agent][docker] 就绪,开始评测。")

        yield
    finally:
        if started:
            logger.info("[agent][docker] 停止并删除容器 %s(镜像保留以复用缓存)...", container_name)
            _run(["docker", "stop", "-t", "10", container_name], timeout=30.0, capture=True)
            _run(["docker", "rm", "-f", container_name], timeout=30.0, capture=True)


def _container_logs_tail(container_name: str, lines: int = 50) -> str:
    """取容器日志尾部,附在未就绪报错里便于排查(参赛方 solution.py 顶层报错等)。"""
    try:
        proc = _run(
            ["docker", "logs", "--tail", str(lines), container_name],
            timeout=15.0,
            capture=True,
        )
        return ((proc.stdout or "") + (proc.stderr or "")).strip()[-1500:] or "(无日志输出)"
    except Exception as error:  # noqa: BLE001
        return f"(取容器日志失败:{error})"
