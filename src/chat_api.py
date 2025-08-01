import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
import hashlib
import secrets
import json

from fastapi import FastAPI, HTTPException, Depends, Request, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import openai
from openai import OpenAI
from passlib.context import CryptContext
from jose import JWTError, jwt
import structlog

from .config import settings
from .logger import get_logger, log_function_call, log_function_result, log_security_event, setup_logging
from .retriever import HybridRetriever, RetrievalConfig, RetrievalResult
from .vectorstore import PostgreSQLVectorStore, ChatMessage


# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    max_results: Optional[int] = Field(10, ge=1, le=20, description="Maximum search results")
    include_sources: Optional[bool] = Field(True, description="Include source citations")
    
    @validator('message')
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


class SecurityManager:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Rate limiting
        self.limiter = Limiter(key_func=get_remote_address)
        
        # Session management
        self.active_sessions = {}
        self.session_cleanup_interval = 3600  # 1 hour
        
    def create_session_id(self) -> str:
        """Create secure session ID"""
        return secrets.token_urlsafe(32)
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        if not api_key:
            return False
        
        # Check against admin API key
        if api_key == settings.admin_api_key:
            return True
        
        # Add more API key validation logic here
        return False
    
    def create_jwt_token(self, data: Dict[str, Any]) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
        to_encode.update({"exp": expire})
        
        return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
            return payload
        except JWTError:
            raise ChatAPISecurityError("Invalid token")
    
    def log_security_event(self, event_type: str, request: Request, details: Dict[str, Any]):
        """Log security events with request context"""
        log_security_event(
            event_type,
            {
                **details,
                "client_ip": get_remote_address(request),
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
security = HTTPBearer(auto_error=False)


# Dependency functions
async def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Validate API key"""
    if not api_key or not security_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Get current user from JWT token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    try:
        payload = security_manager.verify_jwt_token(credentials.credentials)
        return payload
    except ChatAPISecurityError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    logger.info("IUFP RAG Chat API starting up...")
    
    # Initialize components
    try:
        # Test database connection
        vector_store = PostgreSQLVectorStore()
        await vector_store.get_document_stats()
        
        # Initialize retriever
        retriever = HybridRetriever()
        
        # Store in app state
        app.state.vector_store = vector_store
        app.state.retriever = retriever
        app.state.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize components", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("IUFP RAG Chat API shutting down...")


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


# Chat API Implementation
class ChatService:
    def __init__(self, vector_store: PostgreSQLVectorStore, retriever: HybridRetriever, openai_client: OpenAI):
        self.vector_store = vector_store
        self.retriever = retriever
        self.openai_client = openai_client
        self.logger = get_logger(__name__)
    
    async def generate_response(self, query: str, context_chunks: List[RetrievalResult]) -> str:
        """Generate response using OpenAI with context"""
        log_function_call(self.logger, "generate_response", query_length=len(query), context_count=len(context_chunks))
        
        try:
            # Build context from retrieved chunks
            context_text = "\n\n".join([
                f"Source: {chunk.document_name}\n{chunk.text}"
                for chunk in context_chunks[:5]  # Limit context
            ])
            
            # Create system prompt
            system_prompt = f"""You are an expert AI assistant for the IUFP (International Union of Financial Professionals). 
Use the provided context to answer questions accurately and professionally. 

Response Guidelines:
- Provide clear, well-formatted answers using bullet points
- Always cite your sources when possible
- NEVER use phrases like "the document does not include", "not provided", "not available in the context"
- When information is missing, start with: "For this specific information, I recommend..."
- Guide users to visit www.iufp.com for comprehensive details
- Suggest booking a consultation through the IUFP website for personalized guidance
- Always remain helpful and professional

Context:
{context_text}"""
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=settings.max_retrieval_results * 100,  # Dynamic token limit
                temperature=0.7,
                top_p=0.9
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            self.logger.info(
                "Response generated successfully",
                query_length=len(query),
                response_length=len(assistant_response),
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
            
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
                max_results=request.max_results,
                vector_weight=0.7,
                bm25_weight=0.3
            )
            
            search_results = await self.retriever.search(request.message, retrieval_config)
            
            # Generate response
            response_text = await self.generate_response(request.message, search_results)
            
            # Create source citations
            sources = []
            if request.include_sources:
                for result in search_results[:5]:  # Limit citations
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
            
            await self.vector_store.store_chat_message(chat_message, client_ip)
            
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


# API Endpoints
@app.post("/chat", response_model=ChatResponse)
@security_manager.limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_period}second")
async def chat_endpoint(
    request_data: ChatRequest,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Main chat endpoint with security and rate limiting"""
    client_ip = get_remote_address(request)
    
    try:
        # Initialize chat service
        chat_service = ChatService(
            app.state.vector_store,
            app.state.retriever,
            app.state.openai_client
        )
        
        # Process message
        response = await chat_service.process_chat_message(request_data, client_ip)
        
        # Log successful request
        logger.info(
            "Chat request processed successfully",
            client_ip=client_ip,
            message_id=response.message_id,
            processing_time=response.processing_time
        )
        
        return response
        
    except Exception as e:
        # Log error
        security_manager.log_security_event(
            "chat_request_failed",
            request,
            {"error": str(e), "message_preview": request_data.message[:100]}
        )
        
        logger.error("Chat request failed", client_ip=client_ip, error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_status = "healthy"
        try:
            await app.state.vector_store.get_document_stats()
        except Exception:
            db_status = "unhealthy"
        
        # Check retriever
        retriever_status = "healthy"
        try:
            await app.state.retriever.get_retrieval_stats()
        except Exception:
            retriever_status = "unhealthy"
        
        # Overall status
        overall_status = "healthy" if db_status == "healthy" and retriever_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            components={
                "database": db_status,
                "retriever": retriever_status,
                "api": "healthy"
            }
        )
        
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
    client_ip = get_remote_address(request)
    
    if exc.status_code >= 400:
        security_manager.log_security_event(
            "http_error",
            request,
            {"status_code": exc.status_code, "detail": exc.detail}
        )
    
    return {"error": exc.detail, "status_code": exc.status_code}


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    client_ip = get_remote_address(request)
    
    logger.error("Unhandled exception", client_ip=client_ip, error=str(exc))
    
    security_manager.log_security_event(
        "unhandled_exception",
        request,
        {"error": str(exc)}
    )
    
    return {"error": "Internal server error", "status_code": 500}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "chat_api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )