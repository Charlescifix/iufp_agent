import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import re
import math
from collections import Counter, defaultdict
import json

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from .config import settings
from .logger import get_logger, log_function_call, log_function_result, log_security_event
from .vectorstore import PostgreSQLVectorStore, SearchResult
from .embedder import EmbeddingService, EmbeddingResult


@dataclass
class RetrievalResult:
    """Enhanced search result with hybrid scoring"""
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    vector_score: float
    bm25_score: float
    hybrid_score: float
    rank: int
    metadata: Optional[Dict] = None


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval"""
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    max_results: int = 10
    min_score_threshold: float = 0.1
    enable_query_expansion: bool = True
    enable_reranking: bool = True


class RetrievalSecurityError(Exception):
    pass


class HybridRetriever:
    def __init__(self):
        self.logger = get_logger(__name__)
        self._setup_components()
        self._initialize_nltk()
        
        # Retrieval configuration
        self.config = RetrievalConfig()
        
        # BM25 index cache
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        self.last_index_update = 0
        self.index_cache_duration = 3600  # 1 hour
        
        # Query preprocessing
        self.stemmer = PorterStemmer()
        self.stop_words = set()
        
    def _setup_components(self) -> None:
        """Initialize vector store and embedding service"""
        log_function_call(self.logger, "_setup_components")
        
        try:
            self.vector_store = PostgreSQLVectorStore()
            self.embedding_service = EmbeddingService()
            
            self.logger.info("Hybrid retriever components initialized successfully")
            log_function_result(self.logger, "_setup_components")
            
        except Exception as e:
            error = RetrievalSecurityError(f"Failed to initialize retriever components: {str(e)}")
            log_function_result(self.logger, "_setup_components", error=error)
            raise error
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK components with error handling"""
        log_function_call(self.logger, "_initialize_nltk")
        
        try:
            # Expect NLTK data to be pre-baked into the runtime image.
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('english'))
            
            self.logger.debug("NLTK components initialized successfully")
            log_function_result(self.logger, "_initialize_nltk")
            
        except Exception as e:
            self.logger.warning("NLTK initialization failed, using fallback", error=str(e))
            # Fallback stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
            log_function_result(self.logger, "_initialize_nltk")
    
    def _validate_query(self, query: str) -> None:
        """Validate search query for security"""
        log_function_call(self.logger, "_validate_query", query_length=len(query))
        
        if not query or not query.strip():
            error = RetrievalSecurityError("Empty query provided")
            log_function_result(self.logger, "_validate_query", error=error)
            raise error
        
        # Check query length
        if len(query) > 5000:
            error = RetrievalSecurityError(f"Query too long: {len(query)} chars (max: 5000)")
            log_security_event(
                "query_length_exceeded",
                {"query_length": len(query), "max_length": 5000},
                "WARNING"
            )
            log_function_result(self.logger, "_validate_query", error=error)
            raise error
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script.*?>.*?</script>',  # Script injection
            r'javascript:',  # JavaScript URLs
            r'data:.*?base64',  # Data URLs
            r'\x00',  # Null bytes
        ]
        
        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                error = RetrievalSecurityError(f"Suspicious pattern detected in query")
                log_security_event(
                    "suspicious_query_pattern",
                    {"pattern": pattern, "query_preview": query[:100]},
                    "WARNING"
                )
                log_function_result(self.logger, "_validate_query", error=error)
                raise error
        
        log_function_result(self.logger, "_validate_query")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        if not text:
            return []
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        # Remove stopwords and stem
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                try:
                    stemmed = self.stemmer.stem(token)
                    processed_tokens.append(stemmed)
                except:
                    processed_tokens.append(token)
        
        return processed_tokens
    
    async def _update_bm25_index(self, force_update: bool = False) -> None:
        """Update BM25 index from database"""
        log_function_call(self.logger, "_update_bm25_index", force_update=force_update)
        
        current_time = time.time()
        
        # Check if update is needed
        if (not force_update and 
            self.bm25_index is not None and 
            (current_time - self.last_index_update) < self.index_cache_duration):
            self.logger.debug("BM25 index is up to date, skipping update")
            return
        
        try:
            # Get all documents from vector store
            with self.vector_store.SessionLocal() as session:
                from .vectorstore import DocumentChunkEntity
                
                results = session.query(DocumentChunkEntity).all()
                
                if not results:
                    self.logger.warning("No documents found for BM25 indexing")
                    self.bm25_index = None
                    self.bm25_documents = []
                    self.bm25_metadata = []
                    return
                
                # Preprocess documents for BM25
                documents = []
                metadata = []
                
                for result in results:
                    processed_tokens = self._preprocess_text(result.text)
                    if processed_tokens:  # Only include non-empty documents
                        documents.append(processed_tokens)
                        metadata.append({
                            'chunk_id': result.chunk_id,
                            'document_id': result.document_id,
                            'document_name': result.document_name,
                            'text': result.text,
                            'metadata': json.loads(result.metadata) if result.metadata else None
                        })
                
                if documents:
                    # Create BM25 index
                    self.bm25_index = BM25Okapi(documents)
                    self.bm25_documents = documents
                    self.bm25_metadata = metadata
                    self.last_index_update = current_time
                    
                    self.logger.info(
                        "BM25 index updated successfully",
                        document_count=len(documents),
                        avg_tokens=sum(len(doc) for doc in documents) / len(documents)
                    )
                else:
                    self.logger.warning("No valid documents for BM25 indexing after preprocessing")
                    self.bm25_index = None
                    self.bm25_documents = []
                    self.bm25_metadata = []
            
            log_function_result(self.logger, "_update_bm25_index", result=f"Indexed {len(self.bm25_documents)} documents")
            
        except Exception as e:
            log_function_result(self.logger, "_update_bm25_index", error=e)
            raise
    
    async def _vector_search(self, query: str, limit: int) -> List[SearchResult]:
        """Perform vector similarity search"""
        log_function_call(self.logger, "_vector_search", query_length=len(query), limit=limit)
        
        try:
            # Create query embedding
            embedding_result = await self.embedding_service.create_embedding(query)
            query_embedding = embedding_result.embedding
            
            # Perform similarity search
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                limit=limit
            )
            
            self.logger.debug(f"Vector search returned {len(results)} results")
            log_function_result(self.logger, "_vector_search", result=f"{len(results)} results")
            return results
            
        except Exception as e:
            log_function_result(self.logger, "_vector_search", error=e)
            raise
    
    async def _bm25_search(self, query: str, limit: int) -> List[Dict]:
        """Perform BM25 search"""
        log_function_call(self.logger, "_bm25_search", query_length=len(query), limit=limit)
        
        try:
            # Update BM25 index if needed
            await self._update_bm25_index()
            
            if not self.bm25_index or not self.bm25_documents:
                self.logger.warning("BM25 index not available")
                return []
            
            # Preprocess query
            query_tokens = self._preprocess_text(query)
            if not query_tokens:
                self.logger.warning("Query preprocessing resulted in empty tokens")
                return []
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top results with scores
            scored_results = []
            for i, score in enumerate(scores):
                if i < len(self.bm25_metadata) and score > 0:
                    result = self.bm25_metadata[i].copy()
                    result['bm25_score'] = float(score)
                    scored_results.append(result)
            
            # Sort by score and limit
            scored_results.sort(key=lambda x: x['bm25_score'], reverse=True)
            top_results = scored_results[:limit]
            
            self.logger.debug(f"BM25 search returned {len(top_results)} results")
            log_function_result(self.logger, "_bm25_search", result=f"{len(top_results)} results")
            return top_results
            
        except Exception as e:
            log_function_result(self.logger, "_bm25_search", error=e)
            raise
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        if not self.config.enable_query_expansion:
            return query
        
        # Simple query expansion (can be enhanced with word embeddings)
        expansion_map = {
            'ai': 'artificial intelligence machine learning',
            'ml': 'machine learning artificial intelligence',
            'nlp': 'natural language processing text analysis',
            'api': 'application programming interface endpoint',
            'db': 'database storage data',
            'ui': 'user interface frontend design',
            'ux': 'user experience usability design'
        }
        
        expanded_query = query
        query_lower = query.lower()
        
        for term, expansion in expansion_map.items():
            if term in query_lower:
                expanded_query += f" {expansion}"
        
        return expanded_query
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _combine_results(self, vector_results: List[SearchResult], bm25_results: List[Dict]) -> List[RetrievalResult]:
        """Combine and rank vector and BM25 results"""
        log_function_call(self.logger, "_combine_results", 
                         vector_count=len(vector_results), bm25_count=len(bm25_results))
        
        # Create lookup maps
        vector_map = {result.chunk_id: result for result in vector_results}
        bm25_map = {result['chunk_id']: result for result in bm25_results}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(vector_map.keys()) | set(bm25_map.keys())
        
        combined_results = []
        
        for chunk_id in all_chunk_ids:
            vector_result = vector_map.get(chunk_id)
            bm25_result = bm25_map.get(chunk_id)
            
            # Get scores (default to 0 if not found)
            vector_score = vector_result.score if vector_result else 0.0
            bm25_score = bm25_result['bm25_score'] if bm25_result else 0.0
            
            # Get document info (prefer vector result as it has more metadata)
            if vector_result:
                document_id = vector_result.document_id
                document_name = vector_result.document_name
                text = vector_result.text
                metadata = vector_result.metadata
            else:
                document_id = bm25_result['document_id']
                document_name = bm25_result['document_name']
                text = bm25_result['text']
                metadata = bm25_result.get('metadata')
            
            combined_results.append({
                'chunk_id': chunk_id,
                'document_id': document_id,
                'document_name': document_name,
                'text': text,
                'vector_score': vector_score,
                'bm25_score': bm25_score,
                'metadata': metadata
            })
        
        # Normalize scores separately
        vector_scores = [r['vector_score'] for r in combined_results]
        bm25_scores = [r['bm25_score'] for r in combined_results]
        
        normalized_vector = self._normalize_scores(vector_scores)
        normalized_bm25 = self._normalize_scores(bm25_scores)
        
        # Calculate hybrid scores and create final results
        final_results = []
        for i, result in enumerate(combined_results):
            # Weighted combination of normalized scores
            hybrid_score = (
                self.config.vector_weight * normalized_vector[i] +
                self.config.bm25_weight * normalized_bm25[i]
            )
            
            # Apply minimum score threshold
            if hybrid_score >= self.config.min_score_threshold:
                retrieval_result = RetrievalResult(
                    chunk_id=result['chunk_id'],
                    document_id=result['document_id'],
                    document_name=result['document_name'],
                    text=result['text'],
                    vector_score=result['vector_score'],
                    bm25_score=result['bm25_score'],
                    hybrid_score=hybrid_score,
                    rank=0,  # Will be set after sorting
                    metadata=result['metadata']
                )
                final_results.append(retrieval_result)
        
        # Sort by hybrid score and assign ranks
        final_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        # Limit results
        final_results = final_results[:self.config.max_results]
        
        self.logger.info(
            "Results combined successfully",
            total_unique_chunks=len(all_chunk_ids),
            above_threshold=len(final_results),
            final_count=len(final_results)
        )
        
        log_function_result(self.logger, "_combine_results", result=f"Combined to {len(final_results)} results")
        return final_results
    
    async def search(self, query: str, config: Optional[RetrievalConfig] = None) -> List[RetrievalResult]:
        """Perform hybrid search combining vector and BM25 results"""
        log_function_call(self.logger, "search", query_length=len(query))
        
        start_time = time.time()
        
        try:
            # Validate query
            self._validate_query(query)
            
            # Use provided config or default
            if config:
                self.config = config
            
            # Expand query if enabled
            if self.config.enable_query_expansion:
                expanded_query = self._expand_query(query)
                if expanded_query != query:
                    self.logger.debug("Query expanded", original=query[:100], expanded=expanded_query[:100])
                    query = expanded_query
            
            # Perform both searches concurrently
            search_limit = min(self.config.max_results * 2, 50)  # Get more results for better ranking
            
            vector_task = asyncio.create_task(self._vector_search(query, search_limit))
            bm25_task = asyncio.create_task(self._bm25_search(query, search_limit))
            
            vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
            
            # Combine and rank results
            final_results = self._combine_results(vector_results, bm25_results)
            
            # Re-ranking (if enabled)
            if self.config.enable_reranking and len(final_results) > 1:
                final_results = await self._rerank_results(query, final_results)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                "Hybrid search completed",
                query_length=len(query),
                vector_results=len(vector_results),
                bm25_results=len(bm25_results),
                final_results=len(final_results),
                processing_time=processing_time
            )
            
            log_function_result(self.logger, "search", result=f"Found {len(final_results)} results in {processing_time:.2f}s")
            return final_results
            
        except Exception as e:
            log_function_result(self.logger, "search", error=e)
            raise
    
    async def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank results using additional signals"""
        log_function_call(self.logger, "_rerank_results", result_count=len(results))
        
        try:
            # Simple re-ranking based on query term matches and document freshness
            query_terms = set(self._preprocess_text(query))
            
            for result in results:
                # Calculate additional features
                text_tokens = set(self._preprocess_text(result.text))
                
                # Term overlap ratio
                if query_terms and text_tokens:
                    overlap_ratio = len(query_terms & text_tokens) / len(query_terms)
                else:
                    overlap_ratio = 0.0
                
                # Adjust hybrid score with additional features
                rerank_boost = overlap_ratio * 0.1  # Small boost for exact term matches
                result.hybrid_score = min(1.0, result.hybrid_score + rerank_boost)
            
            # Re-sort by adjusted scores
            results.sort(key=lambda x: x.hybrid_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            self.logger.debug("Results re-ranked successfully")
            log_function_result(self.logger, "_rerank_results")
            return results
            
        except Exception as e:
            self.logger.warning("Re-ranking failed, using original results", error=str(e))
            log_function_result(self.logger, "_rerank_results", error=e)
            return results
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        log_function_call(self.logger, "get_retrieval_stats")
        
        try:
            # Get database stats
            db_stats = await self.vector_store.get_document_stats()
            
            # Get BM25 index info
            bm25_stats = {
                'bm25_documents': len(self.bm25_documents),
                'last_update': self.last_index_update,
                'cache_duration': self.index_cache_duration,
                'index_available': self.bm25_index is not None
            }
            
            # Combine stats
            stats = {
                **db_stats,
                **bm25_stats,
                'config': asdict(self.config)
            }
            
            self.logger.info("Retrieval stats collected", **{k: v for k, v in stats.items() if k != 'config'})
            log_function_result(self.logger, "get_retrieval_stats")
            return stats
            
        except Exception as e:
            log_function_result(self.logger, "get_retrieval_stats", error=e)
            raise


# Convenience functions for external use
async def create_hybrid_retriever() -> HybridRetriever:
    """Create and initialize hybrid retriever"""
    return HybridRetriever()


async def search_documents(query: str, max_results: int = 10) -> List[RetrievalResult]:
    """Search documents using hybrid retrieval"""
    retriever = HybridRetriever()
    config = RetrievalConfig(max_results=max_results)
    return await retriever.search(query, config)