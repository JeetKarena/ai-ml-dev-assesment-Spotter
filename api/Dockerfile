FROM python:3.13-slim-bookworm

# 1. Consolidate Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 2. Security & Size: --no-install-recommends minimizes the OS attack surface
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Stability: Use a virtual environment to isolate dependencies
RUN python -m venv /opt/venv

# 4. Build Time: Copy ONLY requirements first to maximize Docker layer caching
COPY requirements.txt .

# 5. Build Time: BuildKit cache mount prevents re-downloading wheels on every build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    pip check

# 6. Copy application code
COPY . .

EXPOSE 8000

# 7. Scalability & Stability: Dynamic workers + 'exec' for graceful shutdowns
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]