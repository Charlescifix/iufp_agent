import asyncio
import json
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import uuid

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Integer, Float, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, ARRAY
try:
    import pgvector
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    # Fallback: use PostgreSQL arrays for vectors
    from sqlalchemy import ARRAY, Float
import numpy as np

from .config import settings
from .logger import get_logger, log_function_call, log_function_result, log_security_event
from .embedder import EmbeddingResult
from .chunker import DocumentChunk


@dataclass
class SearchResult:
    """Data class for search results"""
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    score: float
    metadata: Optional[Dict] = None


@dataclass
class ChatMessage:
    """Data class for chat messages"""
    message_id: str
    session_id: str
    user_message: str
    bot_response: str
    sources: List[str]
    created_at: str
    processing_time: Optional[float] = None
    token_count: Optional[int] = None


class VectorStoreSecurityError(Exception):
    pass


Base = declarative_base()


class DocumentChunkEntity(Base):
    """SQLAlchemy model for document chunks"""
    __tablename__ = 'document_chunks'
    
    chunk_id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    document_name = Column(String, nullable=False)
    page_number = Column(Integer)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)
    source_hash = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    section_title = Column(String)
    chunk_metadata = Column(Text)  # JSON string
    
    # Vector embedding
    embedding = Column(Vector(settings.embedding_dimension))
    
    __table_args__ = (
        Index('idx_document_chunks_document_id', 'document_id'),
        Index('idx_document_chunks_created_at', 'created_at'),
        Index('idx_document_chunks_embedding_vector', 'embedding', postgresql_using='ivfflat'),
    )


class ChatMessageEntity(Base):
    """SQLAlchemy model for chat messages"""
    __tablename__ = 'chat_messages'
    
    message_id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    sources = Column(ARRAY(String))
    created_at = Column(DateTime, nullable=False)
    processing_time = Column(Float)
    token_count = Column(Integer)
    user_ip = Column(String)
    
    __table_args__ = (
        Index('idx_chat_messages_session_id', 'session_id'),
        Index('idx_chat_messages_created_at', 'created_at'),
    )


class PostgreSQLVectorStore:
    def __init__(self):
        self.logger = get_logger(__name__)
        self._validate_configuration()
        self._setup_database_connection()
        self._ensure_pgvector_extension()
        self._create_tables()
        
    def _validate_configuration(self) -> None:
        """Validate PostgreSQL configuration"""
        log_function_call(self.logger, "_validate_configuration")
        
        if not settings.database_url:
            error = VectorStoreSecurityError("Database URL not configured")
            log_security_event(
                "missing_database_url",
                {"service": "PostgreSQLVectorStore"},
                "ERROR"
            )
            log_function_result(self.logger, "_validate_configuration", error=error)
            raise error
        
        # Validate database URL format
        if not settings.database_url.startswith('postgresql://'):
            error = VectorStoreSecurityError("Invalid PostgreSQL URL format")
            log_security_event(
                "invalid_database_url",
                {"service": "PostgreSQLVectorStore"},
                "ERROR"
            )
            log_function_result(self.logger, "_validate_configuration", error=error)
            raise error
        
        log_function_result(self.logger, "_validate_configuration")
    
    def _setup_database_connection(self) -> None:
        """Setup database connection with security configuration"""
        log_function_call(self.logger, "_setup_database_connection")
        
        try:
            # Create SQLAlchemy engine with security settings
            # Parse SSL mode from database URL or use default
            ssl_mode = "disable" if "sslmode=disable" in settings.database_url else "prefer"
            connect_args = {
                "application_name": "iufp_rag_system",
            }
            
            # Only set sslmode if not already in the URL
            if "sslmode=" not in settings.database_url:
                connect_args["sslmode"] = ssl_mode
            
            self.engine = create_engine(
                settings.database_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=settings.debug,
                connect_args=connect_args
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                self.logger.info("Database connection established", postgres_version=version[:50])
            
            log_function_result(self.logger, "_setup_database_connection")
            
        except Exception as e:
            error = VectorStoreSecurityError(f"Failed to connect to database: {str(e)}")
            log_security_event(
                "database_connection_failed",
                {"error": str(e)},
                "ERROR"
            )
            log_function_result(self.logger, "_setup_database_connection", error=error)
            raise error
    
    def _ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension is installed"""
        log_function_call(self.logger, "_ensure_pgvector_extension")
        
        try:
            with self.engine.connect() as conn:
                # Check if pgvector extension exists
                result = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                ))
                
                if not result.fetchone()[0]:
                    # Try to create extension (requires superuser privileges)
                    try:
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        conn.commit()
                        self.logger.info("pgvector extension created successfully")
                    except Exception as e:
                        error = VectorStoreSecurityError(
                            f"pgvector extension not available. Please install it manually: {str(e)}"
                        )
                        log_security_event(
                            "pgvector_extension_missing",
                            {"error": str(e)},
                            "ERROR"
                        )
                        log_function_result(self.logger, "_ensure_pgvector_extension", error=error)
                        raise error
                else:
                    self.logger.debug("pgvector extension already installed")
            
            log_function_result(self.logger, "_ensure_pgvector_extension")
            
        except Exception as e:
            if not isinstance(e, VectorStoreSecurityError):
                error = VectorStoreSecurityError(f"Failed to check pgvector extension: {str(e)}")
                log_function_result(self.logger, "_ensure_pgvector_extension", error=error)
                raise error
            raise
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist"""
        log_function_call(self.logger, "_create_tables")
        
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            self.logger.info("Database tables created/verified successfully")
            log_function_result(self.logger, "_create_tables")
            
        except Exception as e:
            error = VectorStoreSecurityError(f"Failed to create database tables: {str(e)}")
            log_function_result(self.logger, "_create_tables", error=error)
            raise error
    
    def _validate_chunk_data(self, chunk: DocumentChunk, embedding: List[float]) -> None:
        """Validate chunk and embedding data"""
        log_function_call(self.logger, "_validate_chunk_data", chunk_id=chunk.chunk_id)
        
        # Validate chunk data
        if not chunk.chunk_id or not chunk.document_id or not chunk.text:
            error = VectorStoreSecurityError("Invalid chunk data: missing required fields")
            log_function_result(self.logger, "_validate_chunk_data", error=error)
            raise error
        
        # Validate embedding
        if not embedding or len(embedding) != settings.embedding_dimension:
            error = VectorStoreSecurityError(
                f"Invalid embedding dimension: {len(embedding)} (expected: {settings.embedding_dimension})"
            )
            log_function_result(self.logger, "_validate_chunk_data", error=error)
            raise error
        
        # Validate text length
        if len(chunk.text) > 100000:  # 100KB limit
            error = VectorStoreSecurityError(f"Text too long: {len(chunk.text)} chars")
            log_security_event(
                "text_length_exceeded",
                {"chunk_id": chunk.chunk_id, "length": len(chunk.text)},
                "WARNING"
            )
            log_function_result(self.logger, "_validate_chunk_data", error=error)
            raise error
        
        log_function_result(self.logger, "_validate_chunk_data")
    
    async def store_chunk_with_embedding(self, chunk: DocumentChunk, embedding: List[float]) -> None:
        """Store document chunk with its embedding"""
        log_function_call(self.logger, "store_chunk_with_embedding", chunk_id=chunk.chunk_id)
        
        try:
            # Validate input
            self._validate_chunk_data(chunk, embedding)
            
            # Create database session
            with self.SessionLocal() as session:
                # Check if chunk already exists
                existing = session.query(DocumentChunkEntity).filter_by(chunk_id=chunk.chunk_id).first()
                
                if existing:
                    self.logger.debug("Chunk already exists, updating", chunk_id=chunk.chunk_id)
                    # Update existing chunk
                    existing.text = chunk.text
                    existing.char_count = chunk.char_count
                    existing.word_count = chunk.word_count
                    existing.embedding = embedding
                    existing.chunk_metadata = json.dumps(chunk.metadata) if chunk.metadata else None
                else:
                    # Create new chunk entity
                    chunk_entity = DocumentChunkEntity(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        char_count=chunk.char_count,
                        word_count=chunk.word_count,
                        source_hash=chunk.source_hash,
                        created_at=datetime.fromisoformat(chunk.created_at),
                        section_title=chunk.section_title,
                        chunk_metadata=json.dumps(chunk.metadata) if chunk.metadata else None,
                        embedding=embedding
                    )
                    session.add(chunk_entity)
                
                session.commit()
                
                self.logger.info(
                    "Chunk stored successfully",
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id
                )
            
            log_function_result(self.logger, "store_chunk_with_embedding")
            
        except Exception as e:
            log_function_result(self.logger, "store_chunk_with_embedding", error=e)
            raise
    
    async def store_chunks_batch(self, chunks_with_embeddings: List[Tuple[DocumentChunk, List[float]]]) -> None:
        """Store multiple chunks with embeddings in batch"""
        log_function_call(self.logger, "store_chunks_batch", batch_size=len(chunks_with_embeddings))
        
        if not chunks_with_embeddings:
            self.logger.warning("Empty batch provided for storage")
            return
        
        try:
            # Validate all chunks first
            for chunk, embedding in chunks_with_embeddings:
                self._validate_chunk_data(chunk, embedding)
            
            with self.SessionLocal() as session:
                chunk_entities = []
                
                for chunk, embedding in chunks_with_embeddings:
                    # Check if chunk exists
                    existing = session.query(DocumentChunkEntity).filter_by(chunk_id=chunk.chunk_id).first()
                    
                    if existing:
                        # Update existing
                        existing.text = chunk.text
                        existing.char_count = chunk.char_count
                        existing.word_count = chunk.word_count
                        existing.embedding = embedding
                        existing.chunk_metadata = json.dumps(chunk.metadata) if chunk.metadata else None
                    else:
                        # Create new entity
                        chunk_entity = DocumentChunkEntity(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            document_name=chunk.document_name,
                            page_number=chunk.page_number,
                            chunk_index=chunk.chunk_index,
                            text=chunk.text,
                            char_count=chunk.char_count,
                            word_count=chunk.word_count,
                            source_hash=chunk.source_hash,
                            created_at=datetime.fromisoformat(chunk.created_at),
                            section_title=chunk.section_title,
                            chunk_metadata=json.dumps(chunk.metadata) if chunk.metadata else None,
                            embedding=embedding
                        )
                        chunk_entities.append(chunk_entity)
                
                # Batch insert new entities
                if chunk_entities:
                    session.add_all(chunk_entities)
                
                session.commit()
                
                self.logger.info(
                    "Batch chunks stored successfully",
                    total_chunks=len(chunks_with_embeddings),
                    new_chunks=len(chunk_entities)
                )
            
            log_function_result(self.logger, "store_chunks_batch", result=f"Stored {len(chunks_with_embeddings)} chunks")
            
        except Exception as e:
            log_function_result(self.logger, "store_chunks_batch", error=e)
            raise
    
    async def similarity_search(self, query_embedding: List[float], limit: int = 10, document_id: Optional[str] = None) -> List[SearchResult]:
        """Perform similarity search using vector embeddings"""
        log_function_call(self.logger, "similarity_search", limit=limit, document_id=document_id)
        
        try:
            # Validate input
            if not query_embedding or len(query_embedding) != settings.embedding_dimension:
                error = VectorStoreSecurityError(f"Invalid query embedding dimension: {len(query_embedding)}")
                log_function_result(self.logger, "similarity_search", error=error)
                raise error
            
            if limit <= 0 or limit > 100:
                error = VectorStoreSecurityError(f"Invalid limit: {limit} (must be 1-100)")
                log_function_result(self.logger, "similarity_search", error=error)
                raise error
            
            with self.SessionLocal() as session:
                # Build query with similarity search
                query = session.query(
                    DocumentChunkEntity.chunk_id,
                    DocumentChunkEntity.document_id,
                    DocumentChunkEntity.document_name,
                    DocumentChunkEntity.text,
                    DocumentChunkEntity.chunk_metadata,
                    DocumentChunkEntity.embedding.cosine_distance(query_embedding).label('distance')
                )
                
                # Filter by document if specified
                if document_id:
                    query = query.filter(DocumentChunkEntity.document_id == document_id)
                
                # Order by similarity and limit
                results = query.order_by('distance').limit(limit).all()
                
                # Convert to SearchResult objects
                search_results = []
                for result in results:
                    # Convert distance to similarity score (1 - distance)
                    similarity_score = max(0.0, 1.0 - result.distance)
                    
                    search_result = SearchResult(
                        chunk_id=result.chunk_id,
                        document_id=result.document_id,
                        document_name=result.document_name,
                        text=result.text,
                        score=similarity_score,
                        metadata=json.loads(result.chunk_metadata) if result.chunk_metadata else None
                    )
                    search_results.append(search_result)
                
                self.logger.info(
                    "Similarity search completed",
                    results_count=len(search_results),
                    limit=limit,
                    document_id=document_id
                )
                
                log_function_result(self.logger, "similarity_search", result=f"Found {len(search_results)} results")
                return search_results
            
        except Exception as e:
            log_function_result(self.logger, "similarity_search", error=e)
            raise
    
    async def store_chat_message(self, chat_message: ChatMessage, user_ip: Optional[str] = None) -> None:
        """Store chat message in database"""
        log_function_call(self.logger, "store_chat_message", message_id=chat_message.message_id)
        
        try:
            # Validate input
            if not chat_message.message_id or not chat_message.session_id:
                error = VectorStoreSecurityError("Invalid chat message: missing required fields")
                log_function_result(self.logger, "store_chat_message", error=error)
                raise error
            
            # Validate message length
            if len(chat_message.user_message) > 10000 or len(chat_message.bot_response) > 50000:
                error = VectorStoreSecurityError("Message too long")
                log_security_event(
                    "message_length_exceeded",
                    {"message_id": chat_message.message_id},
                    "WARNING"
                )
                log_function_result(self.logger, "store_chat_message", error=error)
                raise error
            
            with self.SessionLocal() as session:
                chat_entity = ChatMessageEntity(
                    message_id=chat_message.message_id,
                    session_id=chat_message.session_id,
                    user_message=chat_message.user_message,
                    bot_response=chat_message.bot_response,
                    sources=chat_message.sources,
                    created_at=datetime.fromisoformat(chat_message.created_at),
                    processing_time=chat_message.processing_time,
                    token_count=chat_message.token_count,
                    user_ip=user_ip
                )
                
                session.add(chat_entity)
                session.commit()
                
                self.logger.info(
                    "Chat message stored successfully",
                    message_id=chat_message.message_id,
                    session_id=chat_message.session_id
                )
            
            log_function_result(self.logger, "store_chat_message")
            
        except Exception as e:
            log_function_result(self.logger, "store_chat_message", error=e)
            raise
    
    async def get_chat_history(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Retrieve chat history for a session"""
        log_function_call(self.logger, "get_chat_history", session_id=session_id, limit=limit)
        
        try:
            # Validate input
            if not session_id:
                error = VectorStoreSecurityError("Session ID required")
                log_function_result(self.logger, "get_chat_history", error=error)
                raise error
            
            if limit <= 0 or limit > 100:
                error = VectorStoreSecurityError(f"Invalid limit: {limit}")
                log_function_result(self.logger, "get_chat_history", error=error)
                raise error
            
            with self.SessionLocal() as session:
                results = session.query(ChatMessageEntity)\
                    .filter_by(session_id=session_id)\
                    .order_by(ChatMessageEntity.created_at.desc())\
                    .limit(limit)\
                    .all()
                
                # Convert to ChatMessage objects
                chat_messages = []
                for result in results:
                    chat_message = ChatMessage(
                        message_id=result.message_id,
                        session_id=result.session_id,
                        user_message=result.user_message,
                        bot_response=result.bot_response,
                        sources=result.sources or [],
                        created_at=result.created_at.isoformat(),
                        processing_time=result.processing_time,
                        token_count=result.token_count
                    )
                    chat_messages.append(chat_message)
                
                # Reverse to get chronological order
                chat_messages.reverse()
                
                self.logger.info(
                    "Chat history retrieved",
                    session_id=session_id,
                    message_count=len(chat_messages)
                )
                
                log_function_result(self.logger, "get_chat_history", result=f"Retrieved {len(chat_messages)} messages")
                return chat_messages
            
        except Exception as e:
            log_function_result(self.logger, "get_chat_history", error=e)
            raise
    
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        log_function_call(self.logger, "delete_document", document_id=document_id)
        
        try:
            if not document_id:
                error = VectorStoreSecurityError("Document ID required")
                log_function_result(self.logger, "delete_document", error=error)
                raise error
            
            with self.SessionLocal() as session:
                deleted_count = session.query(DocumentChunkEntity)\
                    .filter_by(document_id=document_id)\
                    .delete()
                
                session.commit()
                
                self.logger.info(
                    "Document deleted successfully",
                    document_id=document_id,
                    deleted_chunks=deleted_count
                )
                
                log_function_result(self.logger, "delete_document", result=f"Deleted {deleted_count} chunks")
                return deleted_count
            
        except Exception as e:
            log_function_result(self.logger, "delete_document", error=e)
            raise
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        log_function_call(self.logger, "get_document_stats")
        
        try:
            with self.SessionLocal() as session:
                # Count total chunks
                total_chunks = session.query(DocumentChunkEntity).count()
                
                # Count unique documents
                unique_documents = session.query(DocumentChunkEntity.document_id).distinct().count()
                
                # Get storage info
                result = session.execute(text("""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('document_chunks')) as table_size,
                        pg_size_pretty(pg_total_relation_size('chat_messages')) as chat_table_size
                """))
                storage_info = result.fetchone()
                
                stats = {
                    'total_chunks': total_chunks,
                    'unique_documents': unique_documents,
                    'table_size': storage_info[0] if storage_info else 'Unknown',
                    'chat_table_size': storage_info[1] if storage_info else 'Unknown',
                    'embedding_dimension': settings.embedding_dimension
                }
                
                self.logger.info("Database stats retrieved", **stats)
                log_function_result(self.logger, "get_document_stats", result=f"{total_chunks} chunks, {unique_documents} docs")
                return stats
            
        except Exception as e:
            log_function_result(self.logger, "get_document_stats", error=e)
            raise


# Convenience functions for external use
async def create_vector_store() -> PostgreSQLVectorStore:
    """Create and initialize vector store"""
    return PostgreSQLVectorStore()


async def store_document_with_embeddings(chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> None:
    """Store document chunks with their embeddings"""
    vector_store = PostgreSQLVectorStore()
    
    # Combine chunks with embeddings
    chunks_with_embeddings = []
    for chunk in chunks:
        # Find matching embedding
        embedding_result = next((e for e in embeddings if e.chunk_id == chunk.chunk_id), None)
        if embedding_result:
            chunks_with_embeddings.append((chunk, embedding_result.embedding))
    
    await vector_store.store_chunks_batch(chunks_with_embeddings)