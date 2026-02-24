import asyncio
import hashlib
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import openai
from openai import OpenAI
import structlog

from .config import settings
from .logger import get_logger, log_function_call, log_function_result, log_security_event
from .chunker import DocumentChunk


@dataclass
class EmbeddingResult:
    """Data class for embedding results with metadata"""
    chunk_id: str
    embedding: List[float]
    model: str
    embedding_dimension: int
    processing_time: float
    created_at: str
    token_count: Optional[int] = None
    cost_estimate: Optional[float] = None


class EmbeddingSecurityError(Exception):
    pass


class EmbeddingService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self._validate_configuration()
        self._setup_openai_client()
        
        # Rate limiting and cost tracking
        self.requests_per_minute = 500  # OpenAI rate limit
        self.tokens_per_minute = 300000  # OpenAI rate limit
        self.current_requests = 0
        self.current_tokens = 0
        self.last_reset_time = time.time()
        
    def _validate_configuration(self) -> None:
        """Validate OpenAI configuration and security settings"""
        log_function_call(self.logger, "_validate_configuration")
        
        if not settings.openai_api_key:
            error = EmbeddingSecurityError("OpenAI API key not configured")
            log_security_event(
                "missing_api_key",
                {"service": "EmbeddingService"},
                "ERROR"
            )
            log_function_result(self.logger, "_validate_configuration", error=error)
            raise error
        
        if not settings.openai_api_key.startswith(('sk-', 'sk-proj-')):
            error = EmbeddingSecurityError("Invalid OpenAI API key format")
            log_security_event(
                "invalid_api_key_format",
                {"service": "EmbeddingService"},
                "ERROR"
            )
            log_function_result(self.logger, "_validate_configuration", error=error)
            raise error
        
        # Validate embedding model
        allowed_models = {
            'text-embedding-ada-002',
            'text-embedding-3-small', 
            'text-embedding-3-large'
        }
        if settings.embedding_model not in allowed_models:
            error = EmbeddingSecurityError(f"Embedding model {settings.embedding_model} not allowed")
            log_security_event(
                "invalid_embedding_model",
                {"model": settings.embedding_model, "allowed": list(allowed_models)},
                "WARNING"
            )
            log_function_result(self.logger, "_validate_configuration", error=error)
            raise error
        
        log_function_result(self.logger, "_validate_configuration")
    
    def _setup_openai_client(self) -> None:
        """Setup OpenAI client with security configuration"""
        log_function_call(self.logger, "_setup_openai_client")
        
        try:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=30.0,  # 30 second timeout
                max_retries=3
            )
            
            # Test API connection
            self._test_api_connection()
            
            self.logger.info(
                "OpenAI client initialised successfully",
                model=settings.embedding_model,
                dimension=settings.embedding_dimension
            )
            log_function_result(self.logger, "_setup_openai_client")
            
        except Exception as e:
            error = EmbeddingSecurityError(f"Failed to initialise OpenAI client: {str(e)}")
            log_function_result(self.logger, "_setup_openai_client", error=error)
            raise error
    
    def _test_api_connection(self) -> None:
        """Test OpenAI API connection with minimal request"""
        log_function_call(self.logger, "_test_api_connection")
        
        try:
            # Test with minimal text
            response = self.client.embeddings.create(
                input="test",
                model=settings.embedding_model
            )
            
            if not response.data or len(response.data) == 0:
                raise EmbeddingSecurityError("Invalid API response")
            
            actual_dimension = len(response.data[0].embedding)
            if actual_dimension != settings.embedding_dimension:
                self.logger.warning(
                    "Embedding dimension mismatch",
                    expected=settings.embedding_dimension,
                    actual=actual_dimension
                )
            
            self.logger.debug("API connection test successful")
            log_function_result(self.logger, "_test_api_connection")
            
        except openai.OpenAIError as e:
            error = EmbeddingSecurityError(f"OpenAI API test failed: {str(e)}")
            log_security_event(
                "api_connection_failed",
                {"error": str(e), "model": settings.embedding_model},
                "ERROR"
            )
            log_function_result(self.logger, "_test_api_connection", error=error)
            raise error
    
    def _validate_text_input(self, text: str) -> None:
        """Validate text input for security and size limits"""
        log_function_call(self.logger, "_validate_text_input", text_length=len(text))
        
        if not text or not text.strip():
            error = EmbeddingSecurityError("Empty text provided for embedding")
            log_function_result(self.logger, "_validate_text_input", error=error)
            raise error
        
        # Check text length limits (OpenAI has ~8191 token limit)
        max_chars = 32000  # Conservative estimate ~8000 tokens
        if len(text) > max_chars:
            error = EmbeddingSecurityError(f"Text too long: {len(text)} chars (max: {max_chars})")
            log_security_event(
                "text_length_exceeded",
                {"length": len(text), "max_length": max_chars},
                "WARNING"
            )
            log_function_result(self.logger, "_validate_text_input", error=error)
            raise error
        
        # Check for potentially malicious content
        suspicious_patterns = [
            '\x00',  # Null bytes
            '\uffff',  # Invalid Unicode
            '<script',  # Script injection attempts
            'javascript:',  # JavaScript injection
        ]
        
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                error = EmbeddingSecurityError(f"Suspicious content detected: {pattern}")
                log_security_event(
                    "suspicious_content",
                    {"pattern": pattern, "text_preview": text[:100]},
                    "WARNING"
                )
                log_function_result(self.logger, "_validate_text_input", error=error)
                raise error
        
        log_function_result(self.logger, "_validate_text_input")
    
    def _check_rate_limits(self, estimated_tokens: int) -> None:
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset counters every minute
        if current_time - self.last_reset_time >= 60:
            self.current_requests = 0
            self.current_tokens = 0
            self.last_reset_time = current_time
        
        # Check request rate limit
        if self.current_requests >= self.requests_per_minute:
            error = EmbeddingSecurityError("Request rate limit exceeded")
            log_security_event(
                "rate_limit_exceeded",
                {"limit_type": "requests", "current": self.current_requests, "limit": self.requests_per_minute},
                "WARNING"
            )
            raise error
        
        # Check token rate limit
        if self.current_tokens + estimated_tokens > self.tokens_per_minute:
            error = EmbeddingSecurityError("Token rate limit exceeded")
            log_security_event(
                "rate_limit_exceeded",
                {"limit_type": "tokens", "current": self.current_tokens, "limit": self.tokens_per_minute},
                "WARNING"
            )
            raise error
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough estimation: ~4 characters per token for English text
        return max(1, len(text) // 4)
    
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate estimated cost for embedding request"""
        # OpenAI pricing (as of 2024)
        pricing_per_1k_tokens = {
            'text-embedding-ada-002': 0.0001,
            'text-embedding-3-small': 0.00002,
            'text-embedding-3-large': 0.00013
        }
        
        rate = pricing_per_1k_tokens.get(model, 0.0001)
        return (tokens / 1000) * rate
    
    async def create_embedding(self, text: str, chunk_id: Optional[str] = None) -> EmbeddingResult:
        """Create embedding for text with comprehensive security validation"""
        log_function_call(self.logger, "create_embedding", text_length=len(text), chunk_id=chunk_id)
        
        start_time = time.time()
        
        try:
            # Security validation
            self._validate_text_input(text)
            
            # Estimate tokens and check rate limits
            estimated_tokens = self._estimate_tokens(text)
            self._check_rate_limits(estimated_tokens)
            
            # Create embedding request
            response = self.client.embeddings.create(
                input=text,
                model=settings.embedding_model
            )
            
            # Update rate limiting counters
            self.current_requests += 1
            actual_tokens = response.usage.total_tokens if response.usage else estimated_tokens
            self.current_tokens += actual_tokens
            
            # Extract embedding data
            embedding_data = response.data[0]
            embedding_vector = embedding_data.embedding
            
            # Validate embedding
            if not embedding_vector or len(embedding_vector) != settings.embedding_dimension:
                error = EmbeddingSecurityError(f"Invalid embedding dimension: {len(embedding_vector)}")
                log_function_result(self.logger, "create_embedding", error=error)
                raise error
            
            # Calculate metrics
            processing_time = time.time() - start_time
            cost_estimate = self._calculate_cost(actual_tokens, settings.embedding_model)
            
            # Create result object
            result = EmbeddingResult(
                chunk_id=chunk_id or hashlib.md5(text.encode()).hexdigest()[:16],
                embedding=embedding_vector,
                model=settings.embedding_model,
                embedding_dimension=len(embedding_vector),
                processing_time=processing_time,
                token_count=actual_tokens,
                cost_estimate=cost_estimate,
                created_at=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            self.logger.info(
                "Embedding created successfully",
                chunk_id=result.chunk_id,
                model=settings.embedding_model,
                tokens=actual_tokens,
                cost=cost_estimate,
                processing_time=processing_time
            )
            
            log_function_result(self.logger, "create_embedding", result=f"Dimension: {len(embedding_vector)}")
            return result
            
        except openai.OpenAIError as e:
            error = EmbeddingSecurityError(f"OpenAI API error: {str(e)}")
            log_security_event(
                "openai_api_error",
                {"error": str(e), "model": settings.embedding_model},
                "ERROR"
            )
            log_function_result(self.logger, "create_embedding", error=error)
            raise error
        except Exception as e:
            log_function_result(self.logger, "create_embedding", error=e)
            raise
    
    async def create_embeddings_batch(self, texts: List[str], chunk_ids: Optional[List[str]] = None) -> List[EmbeddingResult]:
        """Create embeddings for multiple texts with batch optimisation"""
        log_function_call(self.logger, "create_embeddings_batch", batch_size=len(texts))
        
        if not texts:
            self.logger.warning("Empty text list provided for batch embedding")
            return []
        
        # Validate batch size (OpenAI allows up to 2048 inputs per request)
        max_batch_size = 100  # Conservative limit for safety
        if len(texts) > max_batch_size:
            error = EmbeddingSecurityError(f"Batch size too large: {len(texts)} (max: {max_batch_size})")
            log_security_event(
                "batch_size_exceeded",
                {"batch_size": len(texts), "max_size": max_batch_size},
                "WARNING"
            )
            log_function_result(self.logger, "create_embeddings_batch", error=error)
            raise error
        
        start_time = time.time()
        results = []
        
        try:
            # Validate all texts first
            for i, text in enumerate(texts):
                try:
                    self._validate_text_input(text)
                except EmbeddingSecurityError as e:
                    self.logger.error(f"Text validation failed for index {i}", error=str(e))
                    continue
            
            # Estimate total tokens
            total_estimated_tokens = sum(self._estimate_tokens(text) for text in texts)
            self._check_rate_limits(total_estimated_tokens)
            
            # Create batch embedding request
            response = self.client.embeddings.create(
                input=texts,
                model=settings.embedding_model
            )
            
            # Update rate limiting
            self.current_requests += 1
            actual_tokens = response.usage.total_tokens if response.usage else total_estimated_tokens
            self.current_tokens += actual_tokens
            
            # Process results
            for i, embedding_data in enumerate(response.data):
                chunk_id = chunk_ids[i] if chunk_ids and i < len(chunk_ids) else hashlib.md5(texts[i].encode()).hexdigest()[:16]
                
                result = EmbeddingResult(
                    chunk_id=chunk_id,
                    embedding=embedding_data.embedding,
                    model=settings.embedding_model,
                    embedding_dimension=len(embedding_data.embedding),
                    processing_time=(time.time() - start_time) / len(texts),  # Average per item
                    token_count=actual_tokens // len(texts),  # Average per item
                    cost_estimate=self._calculate_cost(actual_tokens, settings.embedding_model) / len(texts),
                    created_at=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                results.append(result)
            
            total_time = time.time() - start_time
            self.logger.info(
                "Batch embeddings created successfully",
                batch_size=len(texts),
                total_tokens=actual_tokens,
                total_time=total_time,
                avg_time_per_item=total_time / len(texts)
            )
            
            log_function_result(self.logger, "create_embeddings_batch", result=f"Created {len(results)} embeddings")
            return results
            
        except Exception as e:
            log_function_result(self.logger, "create_embeddings_batch", error=e)
            raise
    
    async def process_document_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddingResult]:
        """Process document chunks to create embeddings"""
        log_function_call(self.logger, "process_document_chunks", chunk_count=len(chunks))
        
        if not chunks:
            self.logger.warning("No chunks provided for embedding")
            return []
        
        try:
            # Extract texts and chunk IDs
            texts = [chunk.text for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            # Process in smaller batches if needed
            batch_size = 50  # Process 50 chunks at a time
            all_results = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                
                self.logger.debug(f"Processing batch {i//batch_size + 1}", batch_size=len(batch_texts))
                
                batch_results = await self.create_embeddings_batch(batch_texts, batch_ids)
                all_results.extend(batch_results)
                
                # Small delay between batches to be respectful of rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            self.logger.info(
                "Document chunks processed successfully",
                total_chunks=len(chunks),
                total_embeddings=len(all_results)
            )
            
            log_function_result(self.logger, "process_document_chunks", result=f"Processed {len(all_results)} chunks")
            return all_results
            
        except Exception as e:
            log_function_result(self.logger, "process_document_chunks", error=e)
            raise
    
    def save_embeddings_to_json(self, embeddings: List[EmbeddingResult], output_path: str) -> None:
        """Save embeddings to JSON file with security validation"""
        log_function_call(self.logger, "save_embeddings_to_json", count=len(embeddings), output_path=output_path[:50])
        
        try:
            # Validate output path
            import os
            abs_output_path = os.path.abspath(output_path)
            allowed_base = os.path.abspath("data")
            if not abs_output_path.startswith(allowed_base):
                error = EmbeddingSecurityError(f"Output path outside allowed directory: {output_path}")
                log_security_event(
                    "path_traversal_attempt",
                    {"output_path": output_path, "allowed_base": allowed_base},
                    "ERROR"
                )
                log_function_result(self.logger, "save_embeddings_to_json", error=error)
                raise error
            
            # Create directory if needed
            os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
            
            # Convert to serializable format
            data = {
                'metadata': {
                    'total_embeddings': len(embeddings),
                    'model': settings.embedding_model,
                    'dimension': settings.embedding_dimension,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_cost_estimate': sum(e.cost_estimate or 0 for e in embeddings)
                },
                'embeddings': [asdict(embedding) for embedding in embeddings]
            }
            
            # Save to JSON
            with open(abs_output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(abs_output_path)
            self.logger.info(
                "Embeddings saved successfully",
                output_path=abs_output_path,
                count=len(embeddings),
                file_size=file_size
            )
            
            log_function_result(self.logger, "save_embeddings_to_json", result=abs_output_path)
            
        except Exception as e:
            log_function_result(self.logger, "save_embeddings_to_json", error=e)
            raise


# Convenience functions for external use
async def create_single_embedding(text: str) -> EmbeddingResult:
    """Create embedding for single text"""
    service = EmbeddingService()
    return await service.create_embedding(text)


async def embed_document_chunks(chunks: List[DocumentChunk]) -> List[EmbeddingResult]:
    """Create embeddings for document chunks"""
    service = EmbeddingService()
    return await service.process_document_chunks(chunks)