FROM python:3.13-slim-bookworm

# 1. Removed PIP_NO_CACHE_DIR=1 (Replaced by BuildKit cache mounts below)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 2. Build Time: Use BuildKit cache mounts for APT to speed up OS package downloads
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Layer Caching: Copy ONLY requirements first
COPY requirements.txt .

# 4. Build Time: Use BuildKit cache mount for PIP. 
# This reuses downloaded wheels across builds without bloating the final image size.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt && \
    /opt/venv/bin/pip check

# 5. Copy application code
COPY . .

CMD ["python", "src/train.py"]