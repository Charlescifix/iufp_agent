#!/usr/bin/env python3
"""
IUFP RAG Chatbot Quick Start Script
===================================

This script helps you get started with the IUFP RAG chatbot system.
It will:
1. Check your environment setup
2. Test database connectivity
3. Process sample documents
4. Start the API server

Usage:
    python quickstart.py [command]

Commands:
    setup    - Check environment and setup
    ingest   - Process documents from data/raw/
    test     - Test the system components
    serve    - Start the API server
    all      - Run all steps in sequence
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import settings
from src.logger import setup_logging, get_logger
from src.ingestion import sync_s3_documents
from src.chunker import process_all_documents
from src.embedder import embed_document_chunks
from src.vectorstore import store_document_with_embeddings, PostgreSQLVectorStore
from src.retriever import HybridRetriever, RetrievalConfig
from src.chat_api import app

logger = get_logger(__name__)


class RAGSetupError(Exception):
    pass


class QuickStart:
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def check_environment(self) -> Dict[str, bool]:
        """Check if all required environment variables are set"""
        print("[INFO] Checking environment configuration...")
        
        required_vars = {
            'OPENAI_API_KEY': settings.openai_api_key,
            'DATABASE_URL': settings.database_url,
            'SECRET_KEY': settings.secret_key,
            'ADMIN_API_KEY': settings.admin_api_key
        }
        
        optional_vars = {
            'AWS_ACCESS_KEY_ID': settings.aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': settings.aws_secret_access_key,
            'S3_BUCKET_NAME': settings.s3_bucket_name
        }
        
        status = {}
        all_good = True
        
        print("\n[REQUIRED] Environment Variables:")
        for var, value in required_vars.items():
            is_set = bool(value and value.strip())
            status[var] = is_set
            icon = "[OK]" if is_set else "[MISSING]"
            print(f"  {icon} {var}: {'Set' if is_set else 'Not set'}")
            if not is_set:
                all_good = False
        
        print("\n[OPTIONAL] Environment Variables (for S3 ingestion):")
        for var, value in optional_vars.items():
            is_set = bool(value and value.strip())
            status[var] = is_set
            icon = "[OK]" if is_set else "[WARN]"
            print(f"  {icon} {var}: {'Set' if is_set else 'Not set'}")
        
        if not all_good:
            print("\n[ERROR] Missing required environment variables!")
            print("Please copy .env.example to .env and configure your settings.")
            raise RAGSetupError("Environment not properly configured")
        
        print("\n[OK] Environment configuration looks good!")
        return status
    
    def check_directories(self) -> None:
        """Ensure required directories exist"""
        print("\n[INFO] Checking directories...")
        
        directories = [
            "data",
            "data/raw",
            "data/chunks"
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"  [OK] Created directory: {directory}")
            else:
                print(f"  [OK] Directory exists: {directory}")
    
    async def test_database(self) -> bool:
        """Test database connectivity"""
        print("\n[INFO] Testing database connection...")
        
        try:
            vector_store = PostgreSQLVectorStore()
            stats = await vector_store.get_document_stats()
            
            print(f"  [OK] Database connected successfully!")
            print(f"     - Total chunks: {stats['total_chunks']}")
            print(f"     - Unique documents: {stats['unique_documents']}")
            print(f"     - Table size: {stats['table_size']}")
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] Database connection failed: {e}")
            print("     Please ensure PostgreSQL is running and pgvector extension is installed.")
            return False
    
    async def test_openai(self) -> bool:
        """Test OpenAI API connectivity"""
        print("\n[INFO] Testing OpenAI API connection...")
        
        try:
            from src.embedder import EmbeddingService
            
            service = EmbeddingService()
            result = await service.create_embedding("test connection")
            
            print(f"  [OK] OpenAI API connected successfully!")
            print(f"     - Model: {result.model}")
            print(f"     - Embedding dimension: {result.embedding_dimension}")
            print(f"     - Cost estimate: ${result.cost_estimate:.6f}")
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] OpenAI API connection failed: {e}")
            print("     Please check your API key and network connection.")
            return False
    
    def check_sample_documents(self) -> List[str]:
        """Check for sample documents to process"""
        print("\n[INFO] Checking for documents to process...")
        
        raw_dir = "data/raw"
        if not os.path.exists(raw_dir):
            print(f"  [WARN] Directory {raw_dir} doesn't exist")
            return []
        
        supported_extensions = {'.pdf', '.txt'}
        documents = []
        
        for filename in os.listdir(raw_dir):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                documents.append(os.path.join(raw_dir, filename))
        
        if documents:
            print(f"  [OK] Found {len(documents)} documents to process:")
            for doc in documents:
                size = os.path.getsize(doc) / 1024  # KB
                print(f"     - {os.path.basename(doc)} ({size:.1f} KB)")
        else:
            print(f"  [WARN] No supported documents found in {raw_dir}")
            print(f"     Supported formats: {', '.join(supported_extensions)}")
        
        return documents
    
    async def ingest_documents(self, force: bool = False) -> bool:
        """Process and ingest documents"""
        print("\n[INFO] Starting document ingestion...")
        
        documents = self.check_sample_documents()
        if not documents:
            print("  [WARN] No documents to process. Add PDF or TXT files to data/raw/")
            return False
        
        try:
            # Step 1: Process documents into chunks
            print("  [INFO] Processing documents into chunks...")
            chunks = await process_all_documents("data/raw")
            
            if not chunks:
                print("  [ERROR] No chunks created from documents")
                return False
            
            print(f"  [OK] Created {len(chunks)} chunks from {len(set(c.document_id for c in chunks))} documents")
            
            # Step 2: Create embeddings
            print("  [INFO] Creating embeddings...")
            embeddings = await embed_document_chunks(chunks)
            
            if not embeddings:
                print("  [ERROR] No embeddings created")
                return False
            
            print(f"  [OK] Created {len(embeddings)} embeddings")
            
            # Step 3: Store in vector database
            print("  [INFO] Storing in vector database...")
            await store_document_with_embeddings(chunks, embeddings)
            
            print("  [OK] Documents ingested successfully!")
            
            # Show final stats
            vector_store = PostgreSQLVectorStore()
            stats = await vector_store.get_document_stats()
            print(f"     - Database now contains {stats['total_chunks']} total chunks")
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] Document ingestion failed: {e}")
            self.logger.error("Document ingestion failed", error=str(e))
            return False
    
    async def test_retrieval(self) -> bool:
        """Test the retrieval system"""
        print("\n[INFO] Testing retrieval system...")
        
        try:
            retriever = HybridRetriever()
            config = RetrievalConfig(max_results=3)
            
            # Test query
            test_query = "What is artificial intelligence?"
            print(f"  [INFO] Testing query: '{test_query}'")
            
            results = await retriever.search(test_query, config)
            
            if results:
                print(f"  [OK] Retrieval successful! Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"     {i}. {result.document_name} (score: {result.hybrid_score:.3f})")
                    print(f"        {result.text[:100]}...")
            else:
                print("  [WARN] No results found. This might be normal if no relevant documents are ingested.")
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] Retrieval test failed: {e}")
            return False
    
    async def test_chat_api(self) -> bool:
        """Test a simple chat interaction"""
        print("\n[INFO] Testing chat functionality...")
        
        try:
            # Import here to avoid circular imports
            from src.chat_api import ChatService
            from src.chat_api import ChatRequest
            from openai import OpenAI
            
            # Initialize components
            vector_store = PostgreSQLVectorStore()
            retriever = HybridRetriever()
            openai_client = OpenAI(api_key=settings.openai_api_key)
            
            chat_service = ChatService(vector_store, retriever, openai_client)
            
            # Test request
            request = ChatRequest(
                message="Hello, can you tell me about AI?",
                max_results=3,
                include_sources=True
            )
            
            print(f"  [INFO] Testing chat with message: '{request.message}'")
            
            response = await chat_service.process_chat_message(request, "127.0.0.1")
            
            print(f"  [OK] Chat test successful!")
            print(f"     - Response length: {len(response.response)} characters")
            print(f"     - Sources found: {len(response.sources)}")
            print(f"     - Processing time: {response.processing_time:.2f}s")
            print(f"     - Response preview: {response.response[:150]}...")
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] Chat test failed: {e}")
            return False
    
    def start_server(self) -> None:
        """Start the FastAPI server"""
        print("\n[INFO] Starting API server...")
        print("   Server URL: http://localhost:8000")
        print("   API Documentation: http://localhost:8000/docs")
        print("   Health Check: http://localhost:8000/health")
        print("\n   Press Ctrl+C to stop the server")
        
        import uvicorn
        uvicorn.run(
            "src.chat_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    
    async def run_setup(self) -> bool:
        """Run full setup and validation"""
        print("IUFP RAG Chatbot Quick Start")
        print("=" * 50)
        
        try:
            # Check environment
            self.check_environment()
            self.check_directories()
            
            # Test connections
            db_ok = await self.test_database()
            openai_ok = await self.test_openai()
            
            if not (db_ok and openai_ok):
                print("\n[ERROR] Setup failed. Please fix the issues above and try again.")
                return False
            
            print("\n[OK] All checks passed! System is ready.")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Setup failed: {e}")
            return False
    
    async def run_full_pipeline(self) -> None:
        """Run the complete pipeline"""
        print("[INFO] Running Complete RAG Pipeline")
        print("=" * 50)
        
        # Setup
        if not await self.run_setup():
            return
        
        # Ingest documents
        print("\n" + "=" * 50)
        await self.ingest_documents()
        
        # Test retrieval
        print("\n" + "=" * 50)
        await self.test_retrieval()
        
        # Test chat
        print("\n" + "=" * 50)
        await self.test_chat_api()
        
        print("\n" + "=" * 50)
        print("[OK] Pipeline completed successfully!")
        print("\nYou can now:")
        print("  1. Start the API server: python quickstart.py serve")
        print("  2. Test the API at: http://localhost:8000/docs")
        print("  3. Use the chat endpoint with your API key")


async def main():
    setup_logging()
    qs = QuickStart()
    
    if len(sys.argv) < 2:
        command = "all"
    else:
        command = sys.argv[1].lower()
    
    if command == "setup":
        await qs.run_setup()
    elif command == "ingest":
        await qs.ingest_documents()
    elif command == "test":
        await qs.test_retrieval()
        await qs.test_chat_api()
    elif command == "serve":
        qs.start_server()
    elif command == "all":
        await qs.run_full_pipeline()
    else:
        print("Unknown command. Available commands: setup, ingest, test, serve, all")


if __name__ == "__main__":
    asyncio.run(main())