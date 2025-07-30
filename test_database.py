#!/usr/bin/env python3
"""
Database test suite for PostgreSQL with pgvector
Tests database connection, table creation, and vector operations
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime
from typing import List
import uuid

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import settings
from src.vectorstore import PostgreSQLVectorStore, SearchResult, ChatMessage
from src.chunker import DocumentChunk
from src.logger import setup_logging

# Test configuration
TEST_DATABASE_URL = os.getenv('TEST_DATABASE_URL', settings.database_url)

class TestPostgreSQLVectorStore:
    """Test suite for PostgreSQL vector store"""
    
    @pytest.fixture(scope="session")
    def event_loop(self):
        """Create an instance of the default event loop for the test session"""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture(scope="session")
    async def vector_store(self):
        """Create vector store instance for testing"""
        setup_logging()
        if not TEST_DATABASE_URL:
            pytest.skip("Database URL not configured")
        
        store = PostgreSQLVectorStore()
        yield store
    
    @pytest.fixture
    def sample_chunk(self):
        """Create sample document chunk for testing"""
        return DocumentChunk(
            chunk_id=f"test_chunk_{uuid.uuid4()}",
            document_id=f"test_doc_{uuid.uuid4()}",
            document_name="test_document.pdf",
            page_number=1,
            chunk_index=0,
            text="This is a test document chunk for testing purposes.",
            char_count=54,
            word_count=10,
            source_hash="test_hash_123",
            created_at=datetime.now().isoformat(),
            section_title="Test Section",
            metadata={"test": True}
        )
    
    @pytest.fixture
    def sample_embedding(self):
        """Create sample embedding vector"""
        # Create a vector with the correct dimension
        return [0.1] * settings.embedding_dimension
    
    @pytest.fixture
    def sample_chat_message(self):
        """Create sample chat message for testing"""
        return ChatMessage(
            message_id=f"msg_{uuid.uuid4()}",
            session_id=f"session_{uuid.uuid4()}",
            user_message="Test user message",
            bot_response="Test bot response",
            sources=["test_source_1", "test_source_2"],
            created_at=datetime.now().isoformat(),
            processing_time=1.5,
            token_count=100
        )

    async def test_database_connection(self, vector_store):
        """Test database connection"""
        stats = await vector_store.get_document_stats()
        assert isinstance(stats, dict)
        assert 'total_chunks' in stats
        assert 'unique_documents' in stats
        print(f"Database connection successful - {stats['total_chunks']} chunks in store")

    async def test_store_single_chunk(self, vector_store, sample_chunk, sample_embedding):
        """Test storing a single chunk with embedding"""
        await vector_store.store_chunk_with_embedding(sample_chunk, sample_embedding)
        
        # Verify chunk was stored by searching for it
        results = await vector_store.similarity_search(sample_embedding, limit=1)
        assert len(results) >= 1
        
        # Find our test chunk
        test_result = next((r for r in results if r.chunk_id == sample_chunk.chunk_id), None)
        assert test_result is not None
        assert test_result.document_id == sample_chunk.document_id
        assert test_result.text == sample_chunk.text
        print(f"Single chunk stored successfully: {sample_chunk.chunk_id}")

    async def test_store_batch_chunks(self, vector_store):
        """Test storing multiple chunks in batch"""
        chunks_with_embeddings = []
        
        for i in range(3):
            chunk = DocumentChunk(
                chunk_id=f"batch_test_{uuid.uuid4()}",
                document_id=f"batch_doc_{uuid.uuid4()}",
                document_name=f"batch_document_{i}.pdf",
                page_number=1,
                chunk_index=i,
                text=f"This is batch test chunk number {i}.",
                char_count=len(f"This is batch test chunk number {i}."),
                word_count=7,
                source_hash=f"batch_hash_{i}",
                created_at=datetime.now().isoformat(),
                section_title=f"Batch Section {i}",
                metadata={"batch_test": True, "index": i}
            )
            embedding = [0.1 + i * 0.1] * settings.embedding_dimension
            chunks_with_embeddings.append((chunk, embedding))
        
        await vector_store.store_chunks_batch(chunks_with_embeddings)
        
        # Verify all chunks were stored
        stats = await vector_store.get_document_stats()
        print(f"Batch chunks stored successfully - Total chunks: {stats['total_chunks']}")

    async def test_similarity_search(self, vector_store, sample_embedding):
        """Test similarity search functionality"""
        results = await vector_store.similarity_search(sample_embedding, limit=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.chunk_id
            assert result.document_id
            assert result.text
            assert 0 <= result.score <= 1
        
        print(f"Similarity search returned {len(results)} results")

    async def test_filtered_search(self, vector_store, sample_chunk, sample_embedding):
        """Test similarity search with document filter"""
        # First ensure our test chunk exists
        await vector_store.store_chunk_with_embedding(sample_chunk, sample_embedding)
        
        # Search with document filter
        results = await vector_store.similarity_search(
            sample_embedding, 
            limit=10, 
            document_id=sample_chunk.document_id
        )
        
        # All results should be from the specified document
        for result in results:
            assert result.document_id == sample_chunk.document_id
        
        print(f"Filtered search returned {len(results)} results for document {sample_chunk.document_id}")

    async def test_chat_message_storage(self, vector_store, sample_chat_message):
        """Test storing and retrieving chat messages"""
        # Store chat message
        await vector_store.store_chat_message(sample_chat_message, user_ip="127.0.0.1")
        
        # Retrieve chat history
        history = await vector_store.get_chat_history(sample_chat_message.session_id, limit=10)
        
        assert len(history) >= 1
        
        # Find our test message
        test_message = next((m for m in history if m.message_id == sample_chat_message.message_id), None)
        assert test_message is not None
        assert test_message.user_message == sample_chat_message.user_message
        assert test_message.bot_response == sample_chat_message.bot_response
        assert test_message.sources == sample_chat_message.sources
        
        print(f"Chat message stored and retrieved successfully: {sample_chat_message.message_id}")

    async def test_document_deletion(self, vector_store):
        """Test document deletion functionality"""
        # Create a test document with multiple chunks
        test_doc_id = f"delete_test_{uuid.uuid4()}"
        chunks_with_embeddings = []
        
        for i in range(3):
            chunk = DocumentChunk(
                chunk_id=f"delete_chunk_{i}_{uuid.uuid4()}",
                document_id=test_doc_id,
                document_name="delete_test.pdf",
                page_number=1,
                chunk_index=i,
                text=f"This chunk will be deleted {i}.",
                char_count=30,
                word_count=6,
                source_hash=f"delete_hash_{i}",
                created_at=datetime.now().isoformat(),
                section_title=f"Delete Section {i}",
                metadata={"delete_test": True}
            )
            embedding = [0.2 + i * 0.1] * settings.embedding_dimension
            chunks_with_embeddings.append((chunk, embedding))
        
        # Store the chunks
        await vector_store.store_chunks_batch(chunks_with_embeddings)
        
        # Delete the document
        deleted_count = await vector_store.delete_document(test_doc_id)
        assert deleted_count == 3
        
        # Verify chunks are deleted
        results = await vector_store.similarity_search(
            chunks_with_embeddings[0][1], 
            limit=10, 
            document_id=test_doc_id
        )
        assert len(results) == 0
        
        print(f"Document deletion successful - {deleted_count} chunks deleted")

    async def test_database_stats(self, vector_store):
        """Test database statistics retrieval"""
        stats = await vector_store.get_document_stats()
        
        required_keys = ['total_chunks', 'unique_documents', 'table_size', 'embedding_dimension']
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats['total_chunks'], int)
        assert isinstance(stats['unique_documents'], int)
        assert stats['embedding_dimension'] == settings.embedding_dimension
        
        print(f"Database stats: {stats}")

    async def test_error_handling(self, vector_store):
        """Test error handling for invalid inputs"""
        
        # Test invalid embedding dimension
        with pytest.raises(Exception):
            invalid_embedding = [0.1] * (settings.embedding_dimension - 1)  # Wrong dimension
            await vector_store.similarity_search(invalid_embedding)
        
        # Test invalid limit
        with pytest.raises(Exception):
            valid_embedding = [0.1] * settings.embedding_dimension
            await vector_store.similarity_search(valid_embedding, limit=0)
        
        # Test missing required fields
        with pytest.raises(Exception):
            invalid_chunk = DocumentChunk(
                chunk_id="",  # Empty chunk_id should fail
                document_id="test",
                document_name="test.pdf",
                page_number=1,
                chunk_index=0,
                text="test",
                char_count=4,
                word_count=1,
                source_hash="test",
                created_at=datetime.now().isoformat()
            )
            valid_embedding = [0.1] * settings.embedding_dimension
            await vector_store.store_chunk_with_embedding(invalid_chunk, valid_embedding)
        
        print("Error handling tests passed")


def run_database_tests():
    """Run all database tests"""
    print("PostgreSQL Database Test Suite")
    print("=" * 50)
    
    if not TEST_DATABASE_URL:
        print("ERROR: Database URL not configured")
        print("Please set DATABASE_URL environment variable or configure in .env file")
        return False
    
    # Run tests
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    result = pytest.main(pytest_args)
    return result == 0


if __name__ == "__main__":
    success = run_database_tests()
    if success:
        print("\nAll database tests passed!")
    else:
        print("\nSome tests failed. Please check the output above.")
    
    sys.exit(0 if success else 1)