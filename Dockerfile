# Single source of truth for building and running the app (Railway builds from this file).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

RUN adduser --disabled-password --gecos '' appuser

# Copy only what the app serves; secrets and local files can never end up in the image
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser static/ static/
COPY --chown=appuser:appuser index.html iufp_chat.html ./

USER appuser

EXPOSE 8000

# Railway injects PORT; single worker is enough for this traffic (the app is async)
CMD ["sh", "-c", "exec uvicorn src.chat_api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-server-header"]
