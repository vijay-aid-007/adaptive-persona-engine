# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Persona Engine — Production Dockerfile
# Multi-stage build: builder → runtime
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: builder
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG CACHE_BUST=3
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/       ./src/
COPY data/      ./data/
COPY scripts/   ./scripts/

# Create non-root user FIRST
RUN useradd --create-home --shell /bin/bash appuser

# Create models dir and give appuser ownership BEFORE switching user
RUN mkdir -p ./models && chown -R appuser:appuser ./models && chmod 755 ./models

# Now switch to non-root user
USER appuser

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", "python scripts/prestart.py && uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 1"]