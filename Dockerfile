# GENESIS-AI Production Dockerfile
# Multi-stage build for optimized deployment

FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash genesis

# Copy Python packages from builder
COPY --from=builder /root/.local /home/genesis/.local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/
COPY .streamlit/ ./.streamlit/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Create necessary directories
RUN mkdir -p artifacts/models artifacts/eval logs \
    && chown -R genesis:genesis /app

# Switch to non-root user
USER genesis

# Set Python path
ENV PATH=/home/genesis/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - runs both services
CMD ["sh", "-c", "python -m uvicorn genesis_ai.inference.service:app --host 0.0.0.0 --port 8000 & streamlit run src/genesis_ai/app/main.py --server.port 8501 --server.address 0.0.0.0 & wait"]

