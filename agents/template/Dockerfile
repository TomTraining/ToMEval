# ============================================================================
# 参赛方环境声明(可选交付物)。
#
# 你只需在这里把「运行 solution.py 所需的环境」装好即可 —— 系统包、Python 依赖等。
# HTTP 服务入口由评测框架接管,所以:
#   · 不要写 ENTRYPOINT / CMD(写了也会被框架覆盖);
#   · 不需要在这里起任何 server,你的职责始终只是 solution.py 里的 predict()。
#
# 约定(务必遵守):
#   · 把整个仓库拷到 /agent(框架从 /agent 加载 solution.py)。
#   · 依赖写进 requirements.txt(与本文件同目录)。
#
# 如果你只有 pip 依赖、没有特殊系统包,可以不交本文件:框架会用等价的默认镜像
# (python:3.11-slim + 安装 requirements.txt)兜底。需要特殊环境时才自定义本文件。
# ============================================================================

FROM python:3.11-slim

WORKDIR /agent

# 先拷依赖清单并安装,利用 Docker 层缓存(改代码不必重装依赖)。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷入整个仓库(含 solution.py)。
COPY . /agent
