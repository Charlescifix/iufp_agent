# IUFP RAG Chat API

A production-ready RAG (Retrieval-Augmented Generation) chatbot system for the International Union of Financial Professionals, featuring enterprise-grade security, hybrid search, and comprehensive logging.

## Architecture Overview

```
[ S3 Bucket (PDFs) ] 
         │
         ▼
[ Ingestion Service ] ──▶ [ Text Extraction & Chunking ] ──▶ [ Embedding Service ] 
                                                     │
                                                     ▼
                                             [ PostgreSQL + pgvector ]
                                                     │
                                                     ▼
[ Chat API (FastAPI) ] ◀── [ Hybrid Retriever (k-NN + BM25) ]
         │                                      │
         ▼                                      ▼
[ Frontend Chat UI ]                    [ LLM (GPT-4-Turbo) ]
```

## Features

### 🔒 Security (99% Implementation)
- **API Key Authentication** with rate limiting
- **JWT Token Support** for session management  
- **Input Validation** preventing injection attacks
- **Path Traversal Protection** for file operations
- **Comprehensive Audit Logging** for all operations
- **CORS Protection** with configurable origins
- **SSL/TLS Enforcement** for database connections

### 🚀 Performance & Scalability  
- **Hybrid Search** combining vector similarity (k-NN) and keyword matching (BM25)
- **Batch Processing** for efficient embedding operations
- **Connection Pooling** for database operations
- **Async/Await** throughout for maximum throughput
- **Caching** for BM25 index with automatic updates

### 📊 Monitoring & Observability
- **Structured Logging** with request tracing
- **Health Check Endpoints** for system monitoring
- **Performance Metrics** and statistics API
- **Error Handling** with detailed logging
- **Chat History** storage and retrieval

## Setup Instructions

### 1. Environment Configuration

Copy the `.env` file and configure your credentials:

```bash
# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=eu-west-2
S3_BUCKET_NAME=iufp-knowledge-base

# PostgreSQL + pgvector Configuration  
DATABASE_URL=postgresql://username:password@host:port/database

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4-turbo

# Security Configuration
SECRET_KEY=your-very-long-random-secret-key-here-minimum-32-chars
ADMIN_API_KEY=your-admin-api-key-here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Database Setup

Ensure PostgreSQL with pgvector extension is available:

```sql
-- Connect to your PostgreSQL database
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Document Ingestion

```python
import asyncio
from src.ingestion import sync_s3_documents
from src.chunker import process_all_documents  
from src.embedder import embed_document_chunks
from src.vectorstore import store_document_with_embeddings

async def ingest_documents():
    # 1. Download PDFs from S3
    files = await sync_s3_documents()
    
    # 2. Process documents into chunks
    chunks = await process_all_documents()
    
    # 3. Create embeddings
    embeddings = await embed_document_chunks(chunks)
    
    # 4. Store in vector database
    await store_document_with_embeddings(chunks, embeddings)

# Run ingestion
asyncio.run(ingest_documents())
```

### 5. Start the API Server

```bash
# Development
python -m uvicorn src.chat_api:app --reload --host 0.0.0.0 --port 8000

# Production  
python -m uvicorn src.chat_api:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Usage

### Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "X-API-Key: your-admin-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the IUFP certification requirements?",
    "session_id": "optional-session-id",
    "max_results": 10,
    "include_sources": true
  }'
```

**Response:**
```json
{
  "message_id": "uuid-generated-id",
  "response": "Based on the IUFP documentation...",
  "sources": [
    {
      "chunk_id": "chunk-uuid",
      "document_name": "IUFP_Certification_Guide.pdf", 
      "relevance_score": 0.89,
      "text_snippet": "The certification requirements include..."
    }
  ],
  "session_id": "session-uuid",
  "processing_time": 1.23,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

### System Statistics  

```bash
curl "http://localhost:8000/stats" \
  -H "X-API-Key: your-admin-api-key-here"
```

### Chat History

```bash
curl "http://localhost:8000/chat/history/session-id" \
  -H "X-API-Key: your-admin-api-key-here"
```

## Configuration Options

### Retrieval Configuration

```python
from src.retriever import RetrievalConfig

config = RetrievalConfig(
    vector_weight=0.7,        # Weight for semantic similarity
    bm25_weight=0.3,         # Weight for keyword matching
    max_results=10,          # Maximum results to return
    min_score_threshold=0.1, # Minimum relevance score
    enable_query_expansion=True,  # Expand queries with synonyms
    enable_reranking=True    # Re-rank results for better relevance
)
```

### Security Settings

- **Rate Limiting**: 100 requests per hour per IP
- **API Key Rotation**: Update `ADMIN_API_KEY` in environment
- **JWT Expiration**: 24 hours (configurable)
- **CORS Origins**: Configure in `ALLOWED_ORIGINS`

## Monitoring & Maintenance

### Log Files
- Application logs: Structured JSON format
- Security events: Separate security audit trail
- Performance metrics: Response times and throughput

### Health Monitoring
- Database connectivity
- Vector store performance  
- Embedding service status
- Rate limiting metrics

### Maintenance Tasks
- **Index Updates**: BM25 index auto-updates every hour
- **Session Cleanup**: Expired sessions cleaned automatically
- **Database Optimization**: Run `VACUUM ANALYZE` periodically

## Security Best Practices

1. **API Keys**: Rotate regularly and use strong, unique keys
2. **Database**: Use SSL connections and principle of least privilege
3. **Network**: Deploy behind reverse proxy with rate limiting
4. **Monitoring**: Set up alerts for security events
5. **Updates**: Keep dependencies updated for security patches

## Troubleshooting

### Common Issues

**Database Connection Error:**
```bash
# Check PostgreSQL status
pg_isready -h your-host -p 5432

# Verify pgvector extension
psql -d your_database -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

**OpenAI API Errors:**  
- Check API key validity and rate limits
- Monitor token usage in logs
- Verify model availability

**Search Results Poor Quality:**
- Adjust vector/BM25 weights in RetrievalConfig
- Check document ingestion quality
- Review embedding model performance

## Performance Tuning

### Database Optimization
```sql
-- Create additional indexes for better performance
CREATE INDEX CONCURRENTLY idx_chunks_embedding_hnsw 
ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM document_chunks 
ORDER BY embedding <-> '[vector_here]' LIMIT 10;
```

### Application Scaling
- Use multiple worker processes
- Configure connection pooling
- Implement Redis for session storage
- Set up load balancing for high availability

## Development

### Project Structure
```
iufp_agent/
├── src/
│   ├── ingestion.py       # S3 document ingestion
│   ├── chunker.py         # PDF text extraction & chunking  
│   ├── embedder.py        # OpenAI embedding service
│   ├── vectorstore.py     # PostgreSQL + pgvector integration
│   ├── retriever.py       # Hybrid k-NN + BM25 search
│   ├── chat_api.py        # FastAPI endpoints
│   ├── config.py          # Configuration management
│   └── logger.py          # Structured logging
├── data/
│   ├── raw/               # Downloaded PDFs
│   └── chunks/            # Processed text chunks
├── .env                   # Environment configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Testing
```bash
# Run tests
pytest tests/ -v

# Test API endpoints
python -m pytest tests/test_api.py

# Performance testing
python -m pytest tests/test_performance.py --benchmark
```

## License

Private - IUFP Internal Use Only

## Support

For technical support or questions:
- Check logs in structured JSON format
- Review health check endpoint status  
- Monitor system statistics via API
- Contact development team with specific error messages