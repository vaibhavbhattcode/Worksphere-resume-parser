# Multi-stage Dockerfile for Resume Parser
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-production.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY resume_parser_production.py .
COPY .env.production .env

# Create temp directory
RUN mkdir -p temp

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health')"

# Run with gunicorn
CMD ["gunicorn", "resume_parser_production:app", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120"]
