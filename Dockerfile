# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt \
    && python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords

# Copy application code
COPY . .

# Copy and set permissions for startup and healthcheck scripts
COPY start.sh healthcheck.sh ./
RUN chmod +x start.sh healthcheck.sh

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check using the dedicated script
HEALTHCHECK --interval=90s --timeout=10s --start-period=45s --retries=3 \
    CMD ./healthcheck.sh

# Start command using bash to execute the startup script
CMD ["/bin/bash", "./start.sh"]