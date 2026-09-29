# IUFP Chat Assistant

RAG chatbot for IUFP (UK university applications and student visas). FastAPI app that answers
questions from IUFP's PDF guides, using OpenAI for embeddings and chat, and PostgreSQL + pgvector
for retrieval.

```
S3 (PDF guides) --python -m src.ingest--> Postgres + pgvector <-- FastAPI app (/chat) <-- browser
```

## Hosting

| Piece | Where | Notes |
|---|---|---|
| App | Railway (Hobby) | Built from `Dockerfile`; one async worker |
| Database | Railway Postgres 17 + pgvector | Private network from the app |
| Source documents | AWS S3 | Read only by the ingestion command |

### Railway settings (set once in the dashboard)

The `Dockerfile` defines the build and start command. Service settings live in the Railway
dashboard, not in the repo:

- **Healthcheck path:** `/health` (returns 503 when the database is unreachable)
- **Healthcheck timeout:** `120` seconds (startup checks the database and OpenAI)
- **Restart policy:** On failure
- **Variables:** see below; `DATABASE_URL` should reference the Postgres service
  (`${{Postgres.DATABASE_URL}}`) so traffic stays on the private network

## Configuration

Set as environment variables (Railway) or in a local `.env`. Unknown keys are ignored, so check
spelling if a setting seems to have no effect.

| Variable | Required | Default |
|---|---|---|
| `DATABASE_URL` | yes | |
| `OPENAI_API_KEY` | yes | |
| `ADMIN_API_KEY` | for `/stats` and `/chat/history` | |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | | `text-embedding-3-small` / `1536` (must match stored vectors) |
| `CHAT_MODEL` | | `gpt-4o-mini` |
| `ALLOWED_ORIGINS` | | `http://localhost:3000,http://localhost:8080` |
| `TRUSTED_PROXY_HOPS` | | `1` (Railway's proxy) |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_PERIOD` | | `100` per `3600` seconds, per client IP |
| `OPENAI_TIMEOUT_SECONDS` / `OPENAI_MAX_RETRIES` | | `20` / `2` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME` | for ingestion | |

See `.env.example` for a template.

## Updating the knowledge base

Upload or edit PDFs in the S3 bucket, then from a machine with `.env` pointing at the database's
**public** URL:

```bash
pip install -r requirements.txt
python -m src.ingest --dry-run   # what would change + estimated embedding cost
python -m src.ingest             # embed only new or edited documents
python -m src.ingest --prune     # also remove documents deleted from S3
```

Unchanged documents are skipped, so routine runs cost nothing. Edited documents are replaced
atomically. After changing how documents are split (`MAX_CHUNK_SIZE`, `CHUNK_OVERLAP` or the chunker
code), run once with `--force` so every document is re-chunked (under $0.01 for the current guides).

## Running locally

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.chat_api:app --reload
```

Or with Docker, matching production:

```bash
docker build -t iufp-api .
docker run --env-file .env -p 8000:8000 iufp-api
```

Open http://localhost:8000.
