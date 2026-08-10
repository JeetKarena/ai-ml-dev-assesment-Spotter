# -----------------------------------------------------------------------------
# Freight Rate Prediction Assessment
# Production-ready development image
# -----------------------------------------------------------------------------

FROM python:3.13-slim-bookworm

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Force stdout/stderr to be unbuffered
ENV PYTHONUNBUFFERED=1

# Disable pip cache
ENV PIP_NO_CACHE_DIR=1

# Prevent pip version check
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project
COPY . .

# Add project to Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "src/train.py"]