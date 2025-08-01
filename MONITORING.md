# IUFP Chatbot Monitoring & Logging Guide

## Log Monitoring

### Railway Platform Logs
1. **Access Logs**: Visit [Railway Dashboard](https://railway.app/) → Your Project → Deployments → View Logs
2. **Real-time Logs**: Railway provides live log streaming in the deployment view
3. **Log Retention**: Railway keeps logs for the past 7 days

### Log Levels
- **INFO**: Normal operations, chat requests, component initialization
- **WARNING**: Non-critical issues, search failures (uses fallback)
- **ERROR**: Serious issues, failed requests, component failures

### Key Log Events to Monitor

#### Chat Operations
```
INFO: Chat request processed successfully
WARNING: Search failed, using fallback
ERROR: Chat request failed with detailed trace
```

#### System Health
```
INFO: All components initialized successfully
ERROR: Failed to initialize components
INFO: Database connection established
```

#### Security Events
```
WARNING: chat_request_failed
WARNING: http_error
WARNING: unhandled_exception
```

### Monitoring Commands

#### View Recent Logs (if using railway CLI):
```bash
railway logs --follow
```

#### Check Health Status:
```
GET https://iufp.up.railway.app/health
```

### Performance Metrics
- **Response Time**: Logged with each successful chat request
- **Token Usage**: OpenAI token consumption tracked
- **Database Performance**: Connection and query timing

### Alert Thresholds
- **Error Rate**: >5% of requests failing
- **Response Time**: >10s average response time
- **Database Issues**: Connection failures or timeouts

### Troubleshooting Common Issues

#### 500 Internal Server Error
1. Check Railway logs for detailed error traces
2. Verify database connectivity (`/health` endpoint)
3. Check OpenAI API key validity and quota

#### Slow Responses
1. Monitor OpenAI API response times
2. Check database query performance
3. Review vector search complexity

#### Chat Not Working
1. Verify all environment variables are set
2. Check database connection
3. Confirm OpenAI API access

### Log Analysis Tips
- Filter logs by log level for specific issues
- Search for error patterns using Railway's log search
- Monitor processing times for performance trends
- Track user session IDs for debugging user-specific issues

### Emergency Response
If the chatbot is completely down:
1. Check Railway deployment status
2. Review recent deployment logs
3. Verify environment variables (DATABASE_URL, OPENAI_API_KEY)
4. Use `/health` endpoint to identify failing components