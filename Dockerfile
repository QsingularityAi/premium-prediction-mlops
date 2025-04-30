# Multi-stage build for premium prediction model API
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Second stage - clean production image
FROM python:3.9-slim AS runtime

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g appuser -s /bin/bash -m appuser && \
    mkdir -p /app/data /app/models /app/logs /app/artifacts && \
    chown -R appuser:appuser /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser *.py /app/

# Expose the port the app runs on
EXPOSE 8000

# Create required directories with proper permissions
RUN mkdir -p /tmp/prometheus_multiproc_dir && \
    chown -R appuser:appuser /tmp/prometheus_multiproc_dir

# Set environment variables for the application
ENV PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    ARTIFACT_DIR=/app/artifacts \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python", "-m", "src.api.app"]

