# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Persona Engine — Production Dockerfile
# Multi-stage build: builder → runtime
# Final image: ~350MB, no dev dependencies
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: builder
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


ARG CACHE_BUST=2
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
RUN mkdir -p ./models
COPY scripts/   ./scripts/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Train model on first start if not present, then serve
CMD ["sh", "-c", "python scripts/prestart.py && uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 1"]
