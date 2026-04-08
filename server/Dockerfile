ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Install OpenSCAD and Xvfb for headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    openscad \
    xvfb \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV DISPLAY=:99

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Xvfb for headless OpenSCAD, then run FastAPI server
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 &>/dev/null & cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
