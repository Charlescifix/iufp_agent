import asyncio
import time
from collections import OrderedDict
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import re

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
    # Equal weights: the IDF keyword score is what surfaces exact names (e.g. a degree
    # in a long list) whose chunk embeds too diffusely to win on similarity alone
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    max_results: int = 10
    min_score_threshold: float = 0.1
    # Absolute cosine similarity a chunk needs to count as being about the question,
    # judged before the max_results cut so a weak chunk can't take a good one's slot
    min_vector_score: float = settings.min_relevance_score
    # A chunk matching a rare query term (summed IDF >= 3.5, i.e. a word in ~2 of 77
    # chunks, such as a person's name) is judged against a lower similarity floor:
    # short "Who is X?" questions embed far from the long table chunk that answers them
    keyword_rescue_min_score: float = 3.5
    keyword_rescue_min_vector_score: float = 0.2
    enable_reranking: bool = True


class RetrievalSecurityError(Exception):
    pass


class HybridRetriever:
    def __init__(self, vector_store: Optional[PostgreSQLVectorStore] = None):
        self.logger = get_logger(__name__)
        self._setup_components(vector_store)

        # Default retrieval configuration (per-request configs are passed to search())
        self.config = RetrievalConfig()

        # LRU of normalised query -> embedding; repeated questions skip the embedding API call
        self._embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        
    def _setup_components(self, vector_store: Optional[PostgreSQLVectorStore]) -> None:
        """Initialise vector store and embedding service"""
        log_function_call(self.logger, "_setup_components")
        
        try:
            # Reuse the caller's store so the app keeps a single engine and connection pool
            self.vector_store = vector_store or PostgreSQLVectorStore()
            self.embedding_service = EmbeddingService()
            
            self.logger.info("Hybrid retriever components initialised successfully")
            log_function_result(self.logger, "_setup_components")
            
        except Exception as e:
            error = RetrievalSecurityError(f"Failed to initialise retriever components: {str(e)}")
            log_function_result(self.logger, "_setup_components", error=error)
            raise error
    
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
    
    async def _vector_search(self, query_embedding: List[float], limit: int) -> List[SearchResult]:
        """Perform vector similarity search"""
        log_function_call(self.logger, "_vector_search", limit=limit)

        try:
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
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Embed a query, reusing cached embeddings for repeated questions."""
        cache_key = " ".join(query.lower().split())
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            self._embedding_cache.move_to_end(cache_key)
            return cached

        embedding_result = await self.embedding_service.create_embedding(query)
        self._embedding_cache[cache_key] = embedding_result.embedding
        while len(self._embedding_cache) > settings.query_embedding_cache_size:
            self._embedding_cache.popitem(last=False)
        return embedding_result.embedding

    async def _score_keyword_only_hits(self, query_embedding: List[float],
                                       chunk_ids: List[str]) -> Dict[str, float]:
        """Real similarity for chunks that only keyword search returned.

        Losing these scores costs recall, not correctness: they fall back to 0.0
        and are filtered out exactly as they were before.
        """
        if not chunk_ids:
            return {}

        try:
            return await self.vector_store.similarity_scores_for_chunks(query_embedding, chunk_ids)
        except Exception as e:
            self.logger.warning("Could not score keyword-only chunks", error=str(e))
            return {}

    async def _fts_search(self, query: str, limit: int) -> List[Dict]:
        """Perform full-text search via PostgreSQL (no in-memory index)."""
        log_function_call(self.logger, "_fts_search", query_length=len(query), limit=limit)

        try:
            results = await self.vector_store.fts_search(query, limit)
            self.logger.debug(f"FTS search returned {len(results)} results")
            log_function_result(self.logger, "_fts_search", result=f"{len(results)} results")
            return results
        except Exception as e:
            log_function_result(self.logger, "_fts_search", error=e)
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _combine_results(self, vector_results: List[SearchResult], bm25_results: List[Dict],
                         config: RetrievalConfig,
                         keyword_only_scores: Optional[Dict[str, float]] = None) -> List[RetrievalResult]:
        """Combine and rank vector and BM25 results"""
        keyword_only_scores = keyword_only_scores or {}
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
            
            # Get scores. A chunk that only keyword search returned still has a real
            # similarity to the query; 0.0 is the fallback when it couldn't be read
            vector_score = (vector_result.score if vector_result
                            else keyword_only_scores.get(chunk_id, 0.0))
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
        
        # Drop chunks that aren't about the question before anything is ranked or
        # truncated, so every max_results slot goes to usable context. Judged on
        # absolute similarity, not the per-query normalised score.
        candidate_count = len(combined_results)
        combined_results = [r for r in combined_results
                            if r['vector_score'] >= config.min_vector_score
                            or (r['bm25_score'] >= config.keyword_rescue_min_score
                                and r['vector_score'] >= config.keyword_rescue_min_vector_score)]
        below_relevance = candidate_count - len(combined_results)

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
                config.vector_weight * normalized_vector[i] +
                config.bm25_weight * normalized_bm25[i]
            )
            
            # Apply minimum score threshold
            if hybrid_score >= config.min_score_threshold:
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
        final_results = final_results[:config.max_results]
        
        self.logger.info(
            "Results combined successfully",
            total_unique_chunks=len(all_chunk_ids),
            below_relevance=below_relevance,
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
            
            # Per-request config; never stored on self, so concurrent requests can't clash
            config = config or self.config
            
            # Perform both searches concurrently
            search_limit = min(config.max_results * 2, 50)  # Get more results for better ranking
            
            # Embedded once here: the vector search needs it, and so does scoring
            # the chunks that only keyword search returns
            query_embedding = await self._get_query_embedding(query)

            vector_task = asyncio.create_task(self._vector_search(query_embedding, search_limit))
            fts_task = asyncio.create_task(self._fts_search(query, search_limit))

            vector_results, bm25_results = await asyncio.gather(vector_task, fts_task)

            # Keyword search surfaces chunks ranked below the vector top-N — exact
            # terms, acronyms, figures. Scoring them is what lets one survive the gate.
            vector_ids = {result.chunk_id for result in vector_results}
            keyword_only_ids = [result['chunk_id'] for result in bm25_results
                                if result['chunk_id'] not in vector_ids]
            keyword_only_scores = await self._score_keyword_only_hits(query_embedding, keyword_only_ids)

            # Combine and rank results
            final_results = self._combine_results(vector_results, bm25_results, config,
                                                  keyword_only_scores)
            
            # Re-ranking (if enabled)
            if config.enable_reranking and len(final_results) > 1:
                final_results = await self._rerank_results(query, final_results)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                "Hybrid search completed",
                query_length=len(query),
                vector_results=len(vector_results),
                bm25_results=len(bm25_results),
                keyword_only_scored=len(keyword_only_scores),
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
            # Simple re-ranking: boost results that share words with the query
            query_terms = set(re.sub(r'[^\w\s]', ' ', query.lower()).split())

            for result in results:
                text_terms = set(re.sub(r'[^\w\s]', ' ', result.text.lower()).split())
                overlap_ratio = len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0
                rerank_boost = overlap_ratio * 0.1
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
            
            stats = {
                **db_stats,
                'keyword_search': 'postgresql_fts',
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
    """Create and initialise hybrid retriever"""
    return HybridRetriever()


async def search_documents(query: str, max_results: int = 10) -> List[RetrievalResult]:
    """Search documents using hybrid retrieval"""
    retriever = HybridRetriever()
    config = RetrievalConfig(max_results=max_results)
    return await retriever.search(query, config)