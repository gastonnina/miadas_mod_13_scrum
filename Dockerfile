# ── Stage 1: install dependencies ──────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# Install cron (used in the CMD stage)
RUN apt-get update \
    && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

# Copy only what pip needs so the layer is cached on dep changes
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "pandas>=2.2.0" \
    "pyarrow>=15.0.0" \
    "numpy>=1.26.0" \
    "scikit-learn>=1.4.0"

# ── Stage 2: copy application code ─────────────────────────────────────────
FROM base AS app

WORKDIR /app

# Pipeline source
COPY notebooks/sprint_02_pipeline/scripts/ /app/scripts/

# Checkpoint file (volume-mounted at runtime to persist state between runs)
COPY notebooks/sprint_02_pipeline/pipeline_state.json /app/pipeline_state.json

# ── Runtime volumes (mount from host) ──────────────────────────────────────
# /app/data/raw       → read-only raw parquet files
# /app/data/processed → output master_table.parquet
VOLUME ["/app/data/raw", "/app/data/processed"]

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PROJECT_ROOT=/app \
    RAW_DIR=/app/data/raw \
    PROCESSED_DIR=/app/data/processed \
    PIPELINE_STATE_FILE=/app/pipeline_state.json

# ── Cron schedule ────────────────────────────────────────────────────────────
# Default: 1st day of Jan, Apr, Jul, Oct at 02:00 UTC (every 3 months).
# Override PIPELINE_CRON at build time with --build-arg or at runtime via env.
ARG PIPELINE_CRON="0 2 1 1,4,7,10 *"
RUN echo "${PIPELINE_CRON} cd /app && python -m scripts.pipeline >> /var/log/pipeline.log 2>&1" \
    | crontab - \
    && touch /var/log/pipeline.log

# Expose log for `docker logs` tailing
CMD ["sh", "-c", "cron && tail -f /var/log/pipeline.log"]
