FROM python:3.10-slim

# 从官方镜像复制 uv 二进制
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先复制依赖声明，利用 Docker 层缓存
COPY pyproject.toml .

# uv sync 自动创建 .venv 并安装所有依赖
RUN uv sync --no-dev

# 复制项目代码和模型权重
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
