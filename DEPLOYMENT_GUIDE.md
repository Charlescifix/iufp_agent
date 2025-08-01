# IUFP Chatbot Deployment Guide

## Overview
This is an advanced RAG (Retrieval-Augmented Generation) chatbot for IUFP with the following features:
- FastAPI backend with security middleware
- PostgreSQL + pgvector for embeddings storage
- Hybrid retrieval (vector + BM25)
- Comprehensive logging and monitoring
- Chat history tracking with user IP logging

## Database Configuration

### User Chat Logs Storage
Chat conversations are stored in PostgreSQL database in the `chat_messages` table:
- **Table**: `chat_messages`
- **Fields**: 
  - `message_id` (primary key)
  - `session_id` (for conversation grouping)
  - `user_message` (user input)
  - `bot_response` (AI response)
  - `sources` (array of document chunk IDs used)
  - `created_at` (timestamp)
  - `processing_time` (response time in seconds)
  - `user_ip` (client IP address for monitoring)

### Document Storage
Document chunks are stored in `document_chunks` table with vector embeddings for semantic search.

## Local Testing

### Prerequisites
1. Docker and Docker Compose
2. Python 3.11+ with virtual environment
3. OpenAI API key

### Setup Steps
1. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database settings
   ```

2. **Database Setup**
   ```bash
   docker-compose up -d  # Starts PostgreSQL with pgvector
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run API Server**
   ```bash
   python -m uvicorn src.chat_api:app --host 0.0.0.0 --port 8000
   ```

5. **Run Web Interface**
   ```bash
   python serve_chat.py  # Serves on port 3000
   ```

## Railway Deployment

### Configuration Files Created
- `railway.toml` - Railway configuration
- `nixpacks.toml` - Build configuration  
- `Procfile` - Process definition

### Environment Variables Required
```
DATABASE_URL=postgresql://username:password@host:port/database
OPENAI_API_KEY=your_openai_api_key_here
SECRET_KEY=generate-secure-random-key-minimum-32-chars
ADMIN_API_KEY=generate-secure-admin-api-key
DEBUG=false
LOG_LEVEL=INFO
```

### Deployment Steps
1. **Railway Setup**
   ```bash
   railway login
   railway init
   ```

2. **Database**
   - Add PostgreSQL plugin in Railway dashboard
   - Ensure pgvector extension is available

3. **Environment Variables**
   - Set all required variables in Railway dashboard
   - Use Railway's generated DATABASE_URL

4. **Deploy**
   ```bash
   railway up
   ```

## Monitoring & Logging

### Built-in Monitoring Features
1. **Structured Logging** (JSON format with structlog)
   - Request/response logging
   - Performance metrics
   - Security events
   - Error tracking

2. **Health Endpoints**
   - `/health` - Service health check
   - `/stats` - API statistics (admin only)

3. **Security Features**
   - Rate limiting
   - API key authentication
   - Input validation
   - IP tracking

### Monitoring the Chatbot

#### 1. Application Logs
- All logs are in structured JSON format
- Key log types:
  - `chat_request_processed` - Successful interactions
  - `security_event` - Security-related events
  - `database_operation` - Database activities
  - `function_call`/`function_result` - Function execution tracking

#### 2. Performance Metrics
- Response times logged for each request
- Database query performance
- Embedding generation time
- Retrieval performance stats

#### 3. Usage Analytics
- User session tracking
- Message frequency analysis
- Popular query patterns
- Source document usage

#### 4. Error Monitoring
- API errors with full stack traces
- Database connection issues
- OpenAI API failures
- Rate limiting violations

### Production Monitoring Setup

#### Recommended Tools
1. **Log Aggregation**: ElasticSearch + Kibana or Datadog
2. **Metrics**: Prometheus + Grafana
3. **Alerting**: Set up alerts for:
   - High error rates
   - Slow response times
   - Database connection failures
   - API rate limit breaches

#### Example Monitoring Queries
```json
# High error rate alert
{
  "query": "level:error",
  "time_range": "5m",
  "threshold": "> 10 events"
}

# Slow response alert  
{
  "query": "processing_time:>5.0",
  "time_range": "1m",
  "threshold": "> 5 events"
}
```

## Security Considerations

### API Security
- API key authentication required
- Rate limiting (100 requests/hour by default)
- Input validation and sanitization
- CORS protection
- SQL injection prevention

### Data Privacy
- User IP addresses are logged for monitoring
- Chat history is stored indefinitely (consider retention policy)
- No personal data collection beyond IP addresses

### Production Recommendations
1. Use strong, unique API keys
2. Enable SSL/TLS in production
3. Implement proper database backup strategy
4. Regular security updates
5. Monitor for suspicious activities

## Troubleshooting

### Common Issues
1. **Database Connection Errors**
   - Check DATABASE_URL format
   - Ensure pgvector extension is installed
   - Verify network connectivity

2. **OpenAI API Errors**
   - Validate API key format
   - Check rate limits and quotas
   - Monitor token usage

3. **Performance Issues**
   - Check database connection pool settings
   - Monitor embedding generation time
   - Optimize chunk sizes and retrieval parameters

### Debug Mode
Set `DEBUG=true` in environment to enable:
- Verbose SQL logging
- Detailed error traces
- API documentation at `/docs`

## Maintenance

### Regular Tasks
1. **Database Maintenance**
   - Monitor storage usage
   - Clean old chat logs if needed
   - Update pgvector extension

2. **Performance Optimization**
   - Analyze slow queries
   - Update embeddings for new documents
   - Tune retrieval parameters

3. **Security Updates**
   - Update dependencies regularly
   - Rotate API keys periodically
   - Monitor security logs

This chatbot is production-ready with comprehensive logging, security, and monitoring capabilities built-in.