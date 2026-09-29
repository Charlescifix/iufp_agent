import asyncio
import time
import uuid
from datetime import datetime
from collections import OrderedDict
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
import hashlib
import secrets
import json

from fastapi import FastAPI, HTTPException, Depends, Request, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from openai import AsyncOpenAI
import structlog

from .config import settings
from .logger import get_logger, log_function_call, log_function_result, log_security_event, setup_logging
from .retriever import HybridRetriever, RetrievalConfig, RetrievalResult
from .vectorstore import PostgreSQLVectorStore, ChatMessage


# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    max_results: Optional[int] = Field(settings.max_retrieval_results, ge=1, le=20, description="Maximum search results")
    include_sources: Optional[bool] = Field(True, description="Include source citations")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        # Additional message validation
        if not v.strip():
            raise ValueError("Message cannot be empty")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            '<script', 'javascript:', 'data:text/html', 'vbscript:',
            '\x00', '\uffff'  # Null bytes and invalid Unicode
        ]
        
        v_lower = v.lower()
        for pattern in suspicious_patterns:
            if pattern in v_lower:
                raise ValueError("Invalid characters detected in message")
        
        return v.strip()


class SourceCitation(BaseModel):
    chunk_id: str
    document_name: str
    relevance_score: float
    text_snippet: str = Field(..., max_length=500)


class ChatResponse(BaseModel):
    message_id: str
    response: str
    sources: List[SourceCitation] = []
    session_id: str
    processing_time: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"
    components: Dict[str, str]


class StatsResponse(BaseModel):
    total_messages: int
    active_sessions: int
    avg_response_time: float
    system_stats: Dict[str, Any]


# Security Classes
class ChatAPISecurityError(Exception):
    pass


class RetrievalUnavailableError(Exception):
    """Raised when the knowledge base could not be searched at all.

    Kept separate from "nothing relevant was found": answering with no context
    tells the user IUFP has no information on something its guides may cover.
    """
    pass


# Previous turns of a session replayed to the model, so follow-ups have context
HISTORY_TURN_LIMIT = 4


# The system prompt tells the model to open with this exact phrase when CONTEXT has no answer
NO_ANSWER_PHRASE = "I don't have that information in IUFP's guides."


def is_no_answer(response_text: str) -> bool:
    normalized = response_text.strip().replace("’", "'").lower()
    return normalized.startswith(NO_ANSWER_PHRASE.lower().rstrip("."))


def get_client_ip(request: Request) -> str:
    """Resolve the real client IP behind a reverse proxy.

    Uses the X-Forwarded-For entry appended by our own proxy (counting
    trusted_proxy_hops from the right). The leftmost entries are client-supplied
    and can be spoofed, so they are never trusted.
    """
    hops = settings.trusted_proxy_hops
    forwarded_for = request.headers.get("x-forwarded-for")
    if hops > 0 and forwarded_for:
        hosts = [h.strip() for h in forwarded_for.split(",") if h.strip()]
        if len(hosts) >= hops:
            return hosts[-hops]
    return get_remote_address(request)


class SecurityManager:
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Rate limiting
        self.limiter = Limiter(key_func=get_client_ip)
        
        # Session management
        self.active_sessions = {}
        self.session_cleanup_interval = 3600  # 1 hour
        
    def create_session_id(self) -> str:
        """Create secure session ID"""
        return secrets.token_urlsafe(32)
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        if not api_key or not settings.admin_api_key:
            return False
        
        # Constant-time comparison so the key can't be guessed via response timing
        return secrets.compare_digest(api_key.encode(), settings.admin_api_key.encode())
    
    def log_security_event(self, event_type: str, request: Request, details: Dict[str, Any]):
        """Log security events with request context"""
        log_security_event(
            event_type,
            {
                **details,
                "client_ip": get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "timestamp": datetime.utcnow().isoformat()
            },
            "WARNING"
        )


# Global instances
security_manager = SecurityManager()
logger = get_logger(__name__)

# API Key authentication
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


# Dependency functions
async def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Validate API key"""
    if not api_key or not security_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key



# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    logger.info("IUFP RAG Chat API starting up...")
    
    # Initialise components
    try:
        # Test database connection
        vector_store = PostgreSQLVectorStore()
        await vector_store.get_document_stats()

        # Initialise retriever (shares the vector store's engine and connection pool)
        retriever = HybridRetriever(vector_store=vector_store)
        
        # Store in app state
        app.state.vector_store = vector_store
        app.state.retriever = retriever
        app.state.openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries
        )
        app.state.health_state = HealthcheckState(settings.healthcheck_cache_ttl_seconds)
        app.state.chat_service = ChatService(vector_store, retriever, app.state.openai_client)
        
        logger.info("All components initialised successfully")
        
    except Exception as e:
        logger.error("Failed to initialise components", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("IUFP RAG Chat API shutting down...")
    await app.state.openai_client.close()
    await app.state.retriever.embedding_service.close()
    app.state.vector_store.close()


# Create FastAPI app
app = FastAPI(
    title="IUFP RAG Chat API",
    description="Secure RAG-based chat API with hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=settings.get_cors_methods(),
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your domain
)

app.add_middleware(SlowAPIMiddleware)

# Rate limiting
app.state.limiter = security_manager.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)




class TTLResponseCache:
    """Small in-memory TTL+LRU cache for repeated queries."""
    def __init__(self, ttl_seconds: int, max_entries: int):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

    def _evict_expired(self) -> None:
        now = time.time()
        expired_keys = [k for k, (exp, _) in self._store.items() if exp <= now]
        for k in expired_keys:
            self._store.pop(k, None)

    def get(self, key: str) -> Optional[str]:
        self._evict_expired()
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at <= time.time():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: str) -> None:
        self._evict_expired()
        self._store[key] = (time.time() + self.ttl_seconds, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


class HealthcheckState:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self.last_run = 0.0
        self.last_response: Optional[HealthResponse] = None

    def get_cached(self) -> Optional[HealthResponse]:
        if self.last_response and (time.time() - self.last_run) < self.ttl_seconds:
            return self.last_response
        return None

    def set(self, response: HealthResponse) -> None:
        self.last_response = response
        self.last_run = time.time()


# Chat API Implementation
class ChatService:
    def __init__(self, vector_store: PostgreSQLVectorStore, retriever: HybridRetriever, openai_client: AsyncOpenAI):
        self.vector_store = vector_store
        self.retriever = retriever
        self.openai_client = openai_client
        self.logger = get_logger(__name__)
        self.response_cache = TTLResponseCache(
            ttl_seconds=settings.response_cache_ttl_seconds,
            max_entries=settings.response_cache_max_entries
        )
    
    def _create_cache_key(self, query: str, context_chunks: List[RetrievalResult],
                          history: Optional[List[Dict[str, str]]] = None) -> str:
        normalized_query = " ".join(query.lower().split())
        context_signature = "|".join([f"{c.chunk_id}:{round(c.hybrid_score, 3)}" for c in context_chunks])
        # The conversation is part of the input, so two sessions asking the same
        # question after different turns must not share a cached answer
        history_signature = "|".join(f"{m['role']}:{m['content']}" for m in (history or []))
        return hashlib.sha256(
            f"{normalized_query}::{context_signature}::{history_signature}".encode("utf-8")
        ).hexdigest()

    async def _load_history(self, session_id: str) -> List[Dict[str, str]]:
        """Recent turns of this session, as OpenAI messages.

        History is a nicety, not a requirement: if it can't be read, the current
        question is still answerable on its own.
        """
        if not session_id:
            return []

        try:
            past = await self.vector_store.get_chat_history(session_id, HISTORY_TURN_LIMIT)
        except Exception as e:
            self.logger.warning("Could not load chat history, answering without it", error=str(e))
            return []

        messages: List[Dict[str, str]] = []
        for turn in past:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.bot_response})
        return messages

    async def generate_response(self, query: str, context_chunks: List[RetrievalResult],
                                history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using OpenAI with context"""
        log_function_call(self.logger, "generate_response", query_length=len(query), context_count=len(context_chunks))
        cache_key = self._create_cache_key(query, context_chunks, history)
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            self.logger.info("Response cache hit", query_length=len(query))
            return cached_response

        try:
            # Build context from retrieved chunks
            context_text = "\n\n".join([
                f"Source: {chunk.document_name}\n{chunk.text}"
                for chunk in context_chunks
            ]) if context_chunks else "(No relevant IUFP documents were found for this message.)"

            # Create system prompt
            system_prompt = f"""You are IUFP's AI assistant, helping with UK university applications and student visas.

ACCURACY RULES (most important):
- Answer ONLY from the IUFP documents in CONTEXT. Do not use outside knowledge.
- Every fact, condition and eligibility rule you state must be written in CONTEXT. Do not fill gaps with general knowledge about UK visas, even if you believe it is true.
- Earlier turns are shown only so you can tell what the user is referring to. Never treat them as a source of facts; every fact still has to be in CONTEXT.
- For "can I / am I allowed" questions, only say yes or no if CONTEXT states it directly. A fee that mentions something (e.g. dependants) is not evidence that it is allowed.
- If CONTEXT does not contain the answer, start your reply with exactly "{NO_ANSWER_PHRASE}" and suggest visiting www.iufp.org.uk or booking a consultation. Never guess.
- If CONTEXT answers only part of the question, answer that part and say the rest isn't covered in IUFP's guides.
- Never invent fees, amounts, dates, deadlines or requirements. Use figures exactly as they appear in CONTEXT, with the label CONTEXT gives them.
- Never calculate, multiply or combine figures to produce a new amount; quote only amounts written in CONTEXT.
- When giving fees or financial requirements, add that UK visa rules change and the user should confirm current figures on gov.uk.
- Questions about UK study, student visas, dependants, fees or living in the UK as a student are in scope: if CONTEXT doesn't cover them, say it isn't in IUFP's guides rather than calling them off-topic.
- For topics unrelated to UK study or student visas, politely say in one sentence that you can only help with those.
- If the message is only a greeting or thanks, reply in one short sentence and offer help. Don't add sign-off lines like "let me know" to other answers.

RESPONSE FORMATTING:
- Keep answers under 150 words. A step-by-step process may run to 250 words so that no step is left out.
- Open with one or two plain sentences that answer the question directly.
- Use "- " for bullet lists and "1. ", "2. " for ordered steps. Keep each item to one line; don't nest lists.
- Use **bold** sparingly for key terms and figures, not whole lines.
- Separate paragraphs and lists with one blank line. No headings.
- Be direct and actionable, with a helpful, professional tone.

CONTEXT:
{context_text}"""
            
            # Generate response
            response = await self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *(history or []),
                    {"role": "user", "content": query}
                ],
                max_tokens=settings.max_output_tokens,
                temperature=0.2
            )
            
            assistant_response = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                self.logger.warning("Response hit max_output_tokens and was cut off",
                                    max_output_tokens=settings.max_output_tokens)

            self.logger.info(
                "Response generated successfully",
                query_length=len(query),
                response_length=len(assistant_response),
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
            
            self.response_cache.set(cache_key, assistant_response)
            log_function_result(self.logger, "generate_response", result=f"Generated {len(assistant_response)} chars")
            return assistant_response
            
        except Exception as e:
            log_function_result(self.logger, "generate_response", error=e)
            raise
    
    async def process_chat_message(self, request: ChatRequest, client_ip: str) -> ChatResponse:
        """Process chat message end-to-end"""
        log_function_call(self.logger, "process_chat_message", 
                         message_length=len(request.message), session_id=request.session_id)
        
        start_time = time.time()
        message_id = str(uuid.uuid4())
        
        try:
            # Generate session ID if not provided
            session_id = request.session_id or security_manager.create_session_id()
            
            # Retrieve relevant context
            retrieval_config = RetrievalConfig(
                max_results=request.max_results or settings.max_retrieval_results,
                vector_weight=0.7,
                bm25_weight=0.3
            )
            
            search_results = []
            try:
                # The retriever applies min_relevance_score itself, before it cuts to
                # max_results, so a weak chunk can't take a usable one's slot
                search_results = await self.retriever.search(request.message, retrieval_config)
                self.logger.info(f"Retrieved {len(search_results)} relevant search results")
            except Exception as search_error:
                # Falling through with no context would tell the user the guides
                # don't cover a question they may well cover, so report the outage
                self.logger.error(f"Search failed, cannot ground an answer: {str(search_error)}")
                raise RetrievalUnavailableError(str(search_error)) from search_error
            
            # Earlier turns, so a follow-up like "what about dependants?" makes sense
            history = await self._load_history(session_id)

            # Generate response (works with empty search results too)
            response_text = await self.generate_response(request.message, search_results, history)
            
            # Create source citations
            # A "not in IUFP's guides" reply used none of the retrieved chunks, so citing them would mislead
            sources = []
            if request.include_sources and not is_no_answer(response_text):
                for result in search_results:
                    source = SourceCitation(
                        chunk_id=result.chunk_id,
                        document_name=result.document_name,
                        relevance_score=result.hybrid_score,
                        text_snippet=result.text[:500]  # Truncate snippet
                    )
                    sources.append(source)
            
            processing_time = time.time() - start_time
            
            # Store chat message
            chat_message = ChatMessage(
                message_id=message_id,
                session_id=session_id,
                user_message=request.message,
                bot_response=response_text,
                sources=[s.chunk_id for s in sources],
                created_at=datetime.utcnow().isoformat(),
                processing_time=processing_time
            )
            
            # The reply is already generated and the user is waiting on it; the
            # audit row is worth less than the answer
            try:
                await self.vector_store.store_chat_message(chat_message, client_ip)
            except Exception as store_error:
                self.logger.error("Could not store chat message",
                                  message_id=message_id, error=str(store_error))
            
            # Create response
            response = ChatResponse(
                message_id=message_id,
                response=response_text,
                sources=sources,
                session_id=session_id,
                processing_time=processing_time,
                timestamp=datetime.utcnow().isoformat()
            )
            
            self.logger.info(
                "Chat message processed successfully",
                message_id=message_id,
                session_id=session_id,
                processing_time=processing_time,
                sources_count=len(sources)
            )
            
            log_function_result(self.logger, "process_chat_message", result=f"Processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            log_function_result(self.logger, "process_chat_message", error=e)
            raise


# Static file serving for images and assets (only the static/ folder, never the project root)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pages are re-fetched on every load; without this, browsers can keep showing an old chat UI
# inside the iufp.org.uk iframe after a deploy
NO_CACHE_HEADERS = {"Cache-Control": "no-cache"}

# Static file serving
@app.get("/")
async def serve_index_page():
    """Serve the index landing page"""
    return FileResponse("index.html", headers=NO_CACHE_HEADERS)

@app.get("/chat")
async def serve_chat_interface():
    """Serve the chat interface"""
    return FileResponse("iufp_chat.html", headers=NO_CACHE_HEADERS)

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
@security_manager.limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_period}second")
async def chat_endpoint(
    request_data: ChatRequest,
    request: Request
):
    """Main chat endpoint with security and rate limiting"""
    client_ip = get_client_ip(request)
    
    try:
        # Process message
        response = await app.state.chat_service.process_chat_message(request_data, client_ip)
        
        # Log successful request
        logger.info(
            "Chat request processed successfully",
            client_ip=client_ip,
            message_id=response.message_id,
            processing_time=response.processing_time
        )
        
        return response
        
    except RetrievalUnavailableError as e:
        logger.error("Chat request failed: knowledge base unreachable",
                     client_ip=client_ip, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="I couldn't reach IUFP's guides just now. Please try again in a moment."
        )

    except Exception as e:
        # Log detailed error for debugging
        import traceback
        error_trace = traceback.format_exc()
        
        logger.error(
            "Chat request failed with detailed trace",
            client_ip=client_ip,
            error=str(e),
            error_type=type(e).__name__,
            error_trace=error_trace
        )
        
        security_manager.log_security_event(
            "chat_request_failed",
            request,
            {"error": str(e), "error_type": type(e).__name__, "message_preview": request_data.message[:100]}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint. Returns 503 when the database is unreachable, so Railway
    won't promote a deploy that can't serve chats."""
    def as_http(health: HealthResponse):
        if health.status == "healthy":
            return health
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=health.model_dump())

    try:
        cached = app.state.health_state.get_cached()
        if cached:
            return as_http(cached)
        # Check database (the retriever shares this store, so one query covers both)
        db_status = "healthy"
        try:
            await app.state.vector_store.ping()
        except Exception:
            db_status = "unhealthy"
        retriever_status = db_status

        # Overall status
        overall_status = "healthy" if db_status == "healthy" else "degraded"
        
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            components={
                "database": db_status,
                "retriever": retriever_status,
                "api": "healthy"
            }
        )
        app.state.health_state.set(response)
        return as_http(response)

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(api_key: str = Depends(get_api_key)):
    """Get API statistics (admin only)"""
    try:
        # Get database stats
        db_stats = await app.state.vector_store.get_document_stats()
        
        # Get retrieval stats
        retrieval_stats = await app.state.retriever.get_retrieval_stats()
        
        # Calculate basic stats (simplified)
        stats = StatsResponse(
            total_messages=0,  # Would query from database
            active_sessions=len(security_manager.active_sessions),
            avg_response_time=0.5,  # Would calculate from stored data
            system_stats={
                **db_stats,
                **retrieval_stats
            }
        )
        
        return stats
        
    except Exception as e:
        logger.error("Stats request failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get statistics"
        )


@app.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    api_key: str = Depends(get_api_key)
):
    """Get chat history for a session"""
    try:
        # Validate session ID
        if not session_id or len(session_id) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID"
            )
        
        # Get history
        history = await app.state.vector_store.get_chat_history(session_id, limit)
        
        return {"session_id": session_id, "messages": history}
        
    except HTTPException:
        # An invalid session ID is the caller's error; don't relabel it a 500
        raise
    except Exception as e:
        logger.error("Chat history request failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat history"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with logging"""
    client_ip = get_client_ip(request)
    
    if exc.status_code >= 400:
        security_manager.log_security_event(
            "http_error",
            request,
            {"status_code": exc.status_code, "detail": exc.detail}
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    client_ip = get_client_ip(request)
    
    logger.error("Unhandled exception", client_ip=client_ip, error=str(exc))
    
    security_manager.log_security_event(
        "unhandled_exception",
        request,
        {"error": str(exc)}
    )
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "chat_api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )