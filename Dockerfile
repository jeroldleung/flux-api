FROM nvidia/cuda:12.8.1-base-ubuntu22.04

WORKDIR /app

COPY . .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv sync --frozen --no-cache --no-dev

CMD ["uv", "run", "api.py"]