# NAVIQ Railway Deployment Dockerfile
FROM python:3.9-slim

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
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p artifacts/models artifacts/eval logs

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port (Railway will set PORT env var)
EXPOSE $PORT

# Command for Railway deployment
CMD streamlit run src/genesis_ai/app/competition_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true

